# cxr.py — score-SDE trainer for Chest X-rays with PC sampling, EMA, schedules
import argparse, os, math, json, functools
from datetime import datetime
from collections import Counter

import jax
import jax.numpy as jnp
import numpy as np
import optax
import tensorflow as tf
import tqdm
import torch
from torch.utils.data import DataLoader

from flax import struct
from flax.training.train_state import TrainState
from flax.serialization import to_bytes, from_bytes

# --- Local dataset ---
from datasets.ChestXRay import ChestXrayDataset

# --- Project modules (equations / sampling / model / loss) ---
# VE schedule always exists; VPSDE is optional (we try-import below)
from diffusion.equations import marginal_prob_std, diffusion_coeff  # VE  :contentReference[oaicite:5]{index=5}
from diffusion.sampling import (
    ode_sampler, Euler_Maruyama_sampler, pc_sampler                  #     :contentReference[oaicite:6]{index=6}
)
from models.cxr_unet import ScoreNet                                 #     :contentReference[oaicite:7]{index=7}
from train.train_score_sde import get_train_step_fn                  #     :contentReference[oaicite:8]{index=8}

# --- Optional: Weights & Biases ---
try:
    import wandb
    _WANDB_AVAILABLE = True
except Exception:
    wandb = None
    _WANDB_AVAILABLE = False


# ---------------- Utils ----------------

def n_local_devices():
    return jax.local_device_count()

def tile_to_multiple(x_np, multiple):
    B = x_np.shape[0]
    if B % multiple == 0:
        return x_np
    pad = multiple - (B % multiple)
    reps = (pad + B - 1) // B + 1
    tiled = np.concatenate([x_np] * reps, axis=0)
    return tiled[:B + pad]

def to_pmap_batch(x_np, H, W, C):
    devices = n_local_devices()
    x_np = tile_to_multiple(x_np, devices)
    return x_np.reshape(devices, -1, H, W, C)

def make_grid_torch(imgs_tensor, nrow=None):
    from torchvision.utils import make_grid
    N = imgs_tensor.shape[0]
    if nrow is None:
        nrow = int(math.sqrt(max(1, N)))
    return make_grid(imgs_tensor, nrow=nrow)

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p

def tree_ema_update(ema, new, decay):
    return jax.tree_map(lambda e, p: e * decay + (1.0 - decay) * p, ema, new)
# ---------------- CLI ----------------

def parse_args():
    p = argparse.ArgumentParser("JAX SDE Chest X-ray trainer (overfit/tiny/full)")

    # Data
    p.add_argument("--data_root", default="../datasets/cleaned")
    p.add_argument("--task", choices=["TB", "PNEUMONIA"], default="TB")
    p.add_argument("--split", choices=["train", "val", "test"], default="train")
    p.add_argument("--img_size", type=int, default=256)
    p.add_argument("--class_filter", type=int, default=1, help="Optional: keep a class index only (e.g., 1)")

    # Debug/overfit
    p.add_argument("--overfit_one", action="store_true")
    p.add_argument("--overfit_k", type=int, default=0)
    p.add_argument("--repeat_len", type=int, default=16384, help="Length of RepeatOne dataset in of1 mode")
    p.add_argument("--eval_mse_to_target", action="store_true")
    p.add_argument("--sample_every", type=int, default=1)

    # Model
    p.add_argument("--channels", type=str, default="64,128,256,512")
    p.add_argument("--embed_dim", type=int, default=256)

    # SDE schedule
    p.add_argument("--sde", choices=["VE", "VPSDE"], default="VE")
    p.add_argument("--sigma_max", type=float, default=25.0, help="VE sigma_max (ignored for VPSDE)")

    # Sampler
    p.add_argument("--sampler", choices=["pc", "em", "ode"], default="pc")
    p.add_argument("--num_steps", type=int, default=500)
    p.add_argument("--snr", type=float, default=0.16, help="SNR for Langevin corrector (PC)")
    p.add_argument("--eps", type=float, default=1e-3, help="Final time for samplers")

    # Optimizer / schedule / hygiene
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--min_lr", type=float, default=1e-5)
    p.add_argument("--warmup_steps", type=int, default=1000)
    p.add_argument("--schedule", choices=["const", "cosine"], default="cosine")
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_per_device", type=int, default=4)
    p.add_argument("--sample_batch_size", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)

    # EMA
    p.add_argument("--ema_decay", type=float, default=0.9995)
    p.add_argument("--ema_update_every", type=int, default=1, help="Update EMA every N steps")
    p.add_argument("--use_ema_for_sampling", action="store_true")

    # Experiment/run management
    p.add_argument("--output_root", default="runs")
    p.add_argument("--exp_name", default="cxr_sde")
    p.add_argument("--run_name", default=None)
    p.add_argument("--resume_dir", default=None, help="Resume from an existing run dir (loads last.flax if present)")

    # Checkpointing (paths are auto-derived under run_dir; these are fallbacks/overrides)
    p.add_argument("--ckpt_path", default=None)
    p.add_argument("--samples_dir", default=None)

    # WandB
    p.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    p.add_argument("--wandb_project", default="cxr-sde")
    p.add_argument("--wandb_entity", default=None)
    p.add_argument("--wandb_tags", default="", help="Comma-separated tags")
    p.add_argument("--wandb_id", default=None, help="Set to resume a specific W&B run id")

    p.add_argument("--postfix_every_steps", type=int, default=25, help="How often tqdm.set_postfix runs (steps)")
    p.add_argument("--log_every_steps", type=int, default=100, help="How often to log metrics (steps)")

    # --- Composition flags (optional; if set, we run composition-only flow) ---
    p.add_argument("--compose_run_a", default=None, help="Run name under runs/ to use as Model A")
    p.add_argument("--compose_run_b", default=None, help="Run name under runs/ to use as Model B")
    p.add_argument("--compose_alpha", type=float, default=0.5, help="Fixed kappa in [0,1] for 'fixed' mode")
    p.add_argument("--compose_mode", choices=["fixed", "sum", "normsum"], default="fixed")
    p.add_argument("--compose_use_ema", action="store_true", help="Use EMA params from checkpoints (recommended)")
    p.add_argument("--compose_sampler", choices=["pc", "em", "ode"], default="pc")
    p.add_argument("--compose_num_steps", type=int, default=500)
    p.add_argument("--compose_batch_size", type=int, default=8)

    return p.parse_args()


# ---------- Composition helpers ----------

def _parse_channels(s: str):
    return tuple(int(c.strip()) for c in str(s).split(",") if c.strip())

def _latest_subdir(path: str):
    if not os.path.isdir(path):
        raise FileNotFoundError(f"No run directory: {path}")
    subs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    if not subs:
        # allow runs written directly to runs/<name>/ (no timestamp layer)
        return path
    return os.path.join(path, sorted(subs)[-1])

def _resolve_run_dir(output_root: str, run_name: str):
    base = os.path.join(output_root, run_name)
    return _latest_subdir(base)

def _latest_ckpt(ckpt_dir: str):
    last = os.path.join(ckpt_dir, "last.flax")
    if tf.io.gfile.exists(last):
        return last
    # fallback: newest ep*.flax
    eps = [p for p in tf.io.gfile.listdir(ckpt_dir) if p.startswith("ep") and p.endswith(".flax")]
    if not eps:
        raise FileNotFoundError(f"No checkpoints in {ckpt_dir}")
    eps.sort()
    return os.path.join(ckpt_dir, eps[-1])

def _load_config(run_dir: str):
    meta_path = os.path.join(run_dir, "run_meta.json")
    if not tf.io.gfile.exists(meta_path):
        raise FileNotFoundError(f"Missing run_meta.json in {run_dir}")
    with tf.io.gfile.GFile(meta_path, "r") as f:
        return json.load(f)

def _build_sde_from_cfg(cfg):
    """Make (marginal_prob_std_fn, diffusion_coeff_fn, sde_label) from a run's cfg."""
    # Prefer the stored schedule for exact compatibility with that run
    sde = cfg.get("sde", "VE")
    try:
        from diffusion.equations import vpsde_marginal_prob_std, vpsde_diffusion_coeff
        _has_vpsde = True
    except Exception:
        _has_vpsde = False

    if sde == "VPSDE" and _has_vpsde:
        return vpsde_marginal_prob_std, vpsde_diffusion_coeff, "VPSDE(cosine)"
    else:
        from diffusion.equations import marginal_prob_std, diffusion_coeff
        sigma_max = float(cfg.get("sigma_max", 25.0))
        return functools.partial(marginal_prob_std, sigma=sigma_max), \
               functools.partial(diffusion_coeff,   sigma=sigma_max), f"VE(σmax={sigma_max:g})"

def _instantiate_model_from_cfg(cfg, marginal_prob_std_fn):
    ch = _parse_channels(cfg.get("channels", "64,128,256,512"))
    emb = int(cfg.get("embed_dim", 256))
    return ScoreNet(marginal_prob_std_fn, channels=ch, embed_dim=emb), ch, emb

def _load_params_tuple(path, state_template, ema_template, ema_decay_default):
    with tf.io.gfile.GFile(path, "rb") as f:
        blob = f.read()
    # New format: (TrainState, ema_params, ema_decay)
    try:
        ts, ema, decay = from_bytes((state_template, ema_template, ema_decay_default), blob)
        return ts.params, ema, decay
    except Exception:
        # Old format: TrainState only
        ts = from_bytes(state_template, blob)
        return ts.params, ts.params, ema_decay_default

def _load_model_from_run(output_root, run_name):
    """Return (score_model, params, cfg, run_dir). Uses EMA if present."""
    run_dir = _resolve_run_dir(output_root, run_name)
    cfg = _load_config(run_dir)
    mstd, dcoeff, sde_label = _build_sde_from_cfg(cfg)

    model, ch, emb = _instantiate_model_from_cfg(cfg, mstd)

    ckpt_dir = os.path.join(run_dir, "ckpts")
    ckpt = _latest_ckpt(ckpt_dir)

    # Build templates for deserialization
    # minimal fake batch to init shapes
    H = int(cfg.get("img_size", 256))
    C = 1
    per_dev = max(1, int(cfg.get("batch_per_device", 4)))
    batch = per_dev * max(1, n_local_devices())
    fake_x = jnp.ones((batch, H, H, C), dtype=jnp.float32)
    fake_t = jnp.ones((batch,), dtype=jnp.float32)
    params = model.init({'params': jax.random.PRNGKey(0)}, fake_x, fake_t)
    state_tmpl = TrainState.create(apply_fn=model.apply, params=params, tx=optax.adam(1e-4))
    ema_tmpl = params

    params_model, ema_params, decay = _load_params_tuple(ckpt, state_tmpl, ema_tmpl, cfg.get("ema_decay", 0.9995))
    params_to_use = ema_params  # prefer EMA for sampling
    return model, params_to_use, cfg, run_dir, (mstd, dcoeff, sde_label)

def _assert_compat(cfg_a, cfg_b):
    keys = ["img_size", "channels", "embed_dim", "sde", "sigma_max"]
    mismatches = []
    for k in keys:
        if str(cfg_a.get(k)) != str(cfg_b.get(k)):
            mismatches.append((k, cfg_a.get(k), cfg_b.get(k)))
    if mismatches:
        msg = "Incompatible runs for composition:\n" + "\n".join([f"  {k}: A={a} vs B={b}" for k,a,b in mismatches])
        raise ValueError(msg)

def _make_composed_pmap_score_fn(score_model):
    """Return pmap over (params_a, params_b, x, t, alpha, mode_idx)."""
    def score_fn(params_a, params_b, x, t, alpha, mode_idx):
        s_a = score_model.apply(params_a, x, t)
        s_b = score_model.apply(params_b, x, t)
        # mode: 0=fixed, 1=sum, 2=normsum
        def _fixed(_):
            return s_b + alpha * (s_a - s_b)
        def _sum(_):
            return s_a + s_b
        def _normsum(_):
            return (s_a + s_b) / jnp.sqrt(2.0)
        return jax.lax.switch(mode_idx, (_fixed, _sum, _normsum), operand=None)
    return jax.pmap(score_fn, in_axes=(None, None, 0, 0, None, None))


def main():
    args = parse_args()

    # ---- Choose noise schedule (VE vs VPSDE) ----
    try:
        from diffusion.equations import vpsde_marginal_prob_std, vpsde_diffusion_coeff  # optional
        _has_vpsde = True
    except Exception:
        _has_vpsde = False

    if args.sde == "VE" or not _has_vpsde:
        if args.sde != "VE":
            print("[warn] VPSDE schedule not found in diffusion.equations; falling back to VE.")
        sigma = float(args.sigma_max)
        marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=sigma)
        diffusion_coeff_fn   = functools.partial(diffusion_coeff,   sigma=sigma)
        sde_label = f"VE(σmax={sigma:g})"
    else:
        marginal_prob_std_fn = vpsde_marginal_prob_std
        diffusion_coeff_fn   = vpsde_diffusion_coeff
        sde_label = "VPSDE(cosine)"

    # ---- Mode string and channels ----
    mode = "full"
    if args.overfit_one:
        mode = "of1"
    elif args.overfit_k > 0:
        mode = f"tiny{args.overfit_k}"

    channels = tuple(int(c.strip()) for c in args.channels.split(",") if c.strip())

    # ---- Run directory ----
    H = W = int(args.img_size)
    per_dev = max(1, args.batch_per_device)
    ndev = n_local_devices()
    slurm_id = os.environ.get("SLURM_JOB_ID")
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")

    exp_slug = (
        f"{args.exp_name}"
        f"-{args.task.lower()}-{args.split}"
        f"-cxr{H}-{mode}"
        f"-ch{'x'.join(map(str,channels))}"
        f"-{args.sde.lower()}-{args.sampler}"
        f"-lr{args.lr:g}-b{per_dev}x{ndev}"
    )
    if slurm_id:
        exp_slug += f"-slurm{slurm_id}"

    if args.resume_dir:
        run_dir = args.resume_dir
        print(f"[info] Resuming into: {run_dir}")
    else:
        base_run_name = args.run_name or exp_slug
        run_dir = os.path.join(args.output_root, base_run_name, ts)

    ckpt_dir    = ensure_dir(os.path.join(run_dir, "ckpts"))
    samples_dir = ensure_dir(args.samples_dir or os.path.join(run_dir, "samples"))
    meta_path   = os.path.join(run_dir, "run_meta.json")
    ckpt_latest = os.path.join(ckpt_dir, "last.flax")

    # ---- Persist config ----
    cfg_dump = dict(vars(args))
    cfg_dump.update({
        "exp_slug": exp_slug,
        "run_dir": run_dir,
        "ckpt_latest": ckpt_latest,
        "samples_dir": samples_dir,
        "sde_label": sde_label,
    })
    ensure_dir(run_dir)
    with open(meta_path, "w") as f:
        json.dump(cfg_dump, f, indent=2, sort_keys=True)

    # ---- RNG, dataset & loader ----
    rng = jax.random.PRNGKey(args.seed)
    C = 1

    ds = ChestXrayDataset(
        root_dir=args.data_root, task=args.task, split=args.split,
        img_size=args.img_size, class_filter=args.class_filter
    )
    label_counts = Counter(ds.labels)
    ds_size = len(ds)

    if args.overfit_one:
        first_img, _ = ds[0]  # (1,H,W) in [-1,1]
        class RepeatOne(torch.utils.data.Dataset):
            def __init__(self, img, length):
                self.img, self.length = img.clone(), int(length)
            def __len__(self): return self.length
            def __getitem__(self, idx): return self.img, 0
        train_ds = RepeatOne(first_img, length=max(args.repeat_len, ds_size))
        target_np = first_img.numpy()
    elif args.overfit_k > 0:
        class FirstK(torch.utils.data.Dataset):
            def __init__(self, base, k): self.base, self.k = base, int(k)
            def __len__(self): return min(self.k, len(self.base))
            def __getitem__(self, i): return self.base[i]
        train_ds = FirstK(ds, args.overfit_k)
        target_np = None
    else:
        train_ds = ds
        target_np = None

    batch_size = per_dev * ndev
    loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=8, drop_last=True, pin_memory=True
    )

    # ---- Model ----
    score_model = ScoreNet(
        marginal_prob_std_fn,
        channels=channels,
        embed_dim=args.embed_dim,
    )
    fake_x = jnp.ones((batch_size, H, W, C), dtype=jnp.float32)
    fake_t = jnp.ones((batch_size,), dtype=jnp.float32)
    params = score_model.init({'params': rng}, fake_x, fake_t)

    # ---- Optimizer (AdamW + clip + schedule) ----
    steps_per_epoch = max(1, len(loader))
    total_steps = args.epochs * steps_per_epoch
    if args.schedule == "cosine":
        lr_schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=args.lr,
            warmup_steps=args.warmup_steps,
            decay_steps=max(1, total_steps - args.warmup_steps),
            end_value=args.min_lr,
        )
    else:
        lr_schedule = args.lr

    tx = optax.chain(
        optax.clip_by_global_norm(args.grad_clip) if args.grad_clip and args.grad_clip > 0 else optax.identity(),
        optax.adamw(learning_rate=lr_schedule, weight_decay=args.weight_decay),
    )

    host_state = TrainState.create(
        apply_fn=score_model.apply,
        params=params,
        tx=tx,
    )
    ema_params = host_state.params  # start EMA at init params

    # ---- Resume from checkpoint if requested or present ----
    resume_ckpt = None
    if args.resume_dir and tf.io.gfile.exists(ckpt_latest):
        resume_ckpt = ckpt_latest
    elif args.ckpt_path and tf.io.gfile.exists(args.ckpt_path):
        resume_ckpt = args.ckpt_path

    if resume_ckpt:
        print(f"[info] Loading checkpoint from {resume_ckpt}")
        with tf.io.gfile.GFile(resume_ckpt, "rb") as f:
            blob = f.read()
        # Try new format first: (TrainState, ema_params, ema_decay)
        try:
            host_state, ema_params, loaded_decay = from_bytes(
                (host_state, ema_params, args.ema_decay), blob
            )
            # keep CLI value if user set it; otherwise adopt loaded
            if "ema_decay" not in vars(args) or args.ema_decay is None:
                args.ema_decay = float(loaded_decay)
            print("[info] Loaded (TrainState, EMA, decay).")
        except Exception:
            # Fallback: plain TrainState (older runs). Use params as EMA.
            try:
                host_state = from_bytes(host_state, blob)
                ema_params = host_state.params
                print("[warn] Loaded plain TrainState; initializing EMA from params.")
            except Exception as e:
                print(f"[warn] Could not deserialize checkpoint in any known format: {e}")
                print("[warn] Starting fresh.")
    else:
        print("[info] No checkpoint; fresh training.")

    # ---- Replicate & get pmapped step ----
    state = jax.device_put_replicated(host_state, jax.local_devices())
    train_step_fn = get_train_step_fn(score_model, marginal_prob_std_fn)

    # ---- Weights & Biases init ----
    use_wandb = bool(args.wandb and _WANDB_AVAILABLE)
    if args.wandb and not _WANDB_AVAILABLE:
        print("[warn] wandb requested, but not installed. Proceeding without online logging.")
    run_name = args.run_name or exp_slug
    wandb_run = None
    if use_wandb:
        wandb_tags = [t.strip() for t in args.wandb_tags.split(",") if t.strip()]
        wandb_config = {**vars(args),
                        "exp_slug": exp_slug,
                        "run_dir": run_dir,
                        "ds_size": ds_size,
                        "label_counts": dict(label_counts),
                        "n_local_devices": ndev,
                        "effective_batch": batch_size,
                        "sde_label": sde_label}
        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=run_name,
            group=args.exp_name,
            job_type=mode,
            tags=wandb_tags,
            config=wandb_config,
            id=args.wandb_id,
            resume="allow" if args.wandb_id else None,
            dir=run_dir,
        )
        wandb.define_metric("train/step")
        wandb.define_metric("train/*", step_metric="train/step")
        wandb.define_metric("epoch/*", step_metric="epoch/idx")
        table = wandb.Table(columns=["label", "count"])
        for k, v in sorted(label_counts.items()):
            table.add_data(str(k), int(v))
        wandb.log({"dataset/summary": table, "epoch/idx": 0})

    # ---- Train ----
    global_step = 0
    running_loss = 0.0
    running_count = 0

    # Host copies for EMA updates/sampling/checkpoint
    host_params_last = jax.tree_map(lambda x: np.array(x), host_state.params)
    ema_params = jax.tree_map(lambda x: np.array(x), ema_params)

    for epoch in tqdm.trange(args.epochs, desc="epochs"):
        losses = []
        inner = tqdm.tqdm(loader, desc=f"epoch {epoch+1}/{args.epochs}", leave=False)
        for x, _ in inner:
            # Torch (B,1,H,W) in [-1,1] -> JAX NHWC [0,1]
            x = x.permute(0, 2, 3, 1).contiguous()
            x = (x + 1.0) * 0.5
            x_np = x.numpy()
            x_np = to_pmap_batch(x_np, H, W, C)

            rng, *step_rng = jax.random.split(rng, n_local_devices() + 1)
            step_rng = jnp.asarray(step_rng)
            loss, state = train_step_fn(step_rng, x_np, state)
            loss_val = float(jax.device_get(loss)[0])
            losses.append(loss_val)

            # Host pull params for EMA (every N steps to reduce overhead)
            global_step += 1
            if (global_step % max(1, args.ema_update_every)) == 0:
                host_params_step = jax.device_get(jax.tree_map(lambda v: v[0], state.params))
                ema_params = tree_ema_update(ema_params, host_params_step, args.ema_decay)
                host_params_last = host_params_step

            running_loss += loss_val
            running_count += 1
            if (global_step % max(1, args.postfix_every_steps)) == 0:
                inner.set_postfix(loss=f"{loss_val:.4f}")

            if use_wandb and (global_step % max(1, args.log_every_steps) == 0):
                mean_loss = running_loss / max(1, running_count)
                wandb.log({"train/loss": mean_loss, "train/step": global_step})
                running_loss = 0.0
                running_count = 0

        # Save checkpoint (include EMA)
        host_state_to_save = jax.device_get(jax.tree_map(lambda v: v[0], state))
        payload_bytes = to_bytes((host_state_to_save, ema_params, args.ema_decay))

        ep_path = os.path.join(ckpt_dir, f"ep{epoch + 1:04d}.flax")
        with tf.io.gfile.GFile(ep_path, "wb") as f:
            f.write(payload_bytes)
        with tf.io.gfile.GFile(ckpt_latest, "wb") as f:
            f.write(payload_bytes)

        avg_loss = float(np.mean(losses)) if len(losses) else float("nan")
        print(f"[epoch {epoch+1}] avg loss: {avg_loss:.6f}")

        if use_wandb:
            wandb.log({"epoch/avg_loss": avg_loss, "epoch/idx": epoch+1,
                       "ckpt/last_path": ckpt_latest, "ckpt/epoch_path": ep_path})

        # --- Periodic sampling ---
        if ((epoch + 1) % max(1, args.sample_every)) == 0:
            params_for_sampling = ema_params if args.use_ema_for_sampling else host_params_last
            images, eval_dict = sample_and_log(
                rng_key=rng,
                score_model=score_model,
                params=params_for_sampling,
                H=H, W=W, img_size=args.img_size,
                batch_size=args.sample_batch_size,
                out_dir=samples_dir,
                epoch=epoch+1,
                target_np=target_np,
                sampler_name=args.sampler,
                num_steps=args.num_steps,
                snr=args.snr,
                eps=args.eps,
                marginal_prob_std_fn=marginal_prob_std_fn,
                diffusion_coeff_fn=diffusion_coeff_fn,
            )
            if use_wandb:
                wandb_imgs = [wandb.Image(img, caption=cap) for cap, img in images]
                log_payload = {"samples/grid": wandb_imgs, "epoch/idx": epoch+1}
                log_payload.update({f"eval/{k}": v for k, v in eval_dict.items()})
                wandb.log(log_payload)

    print(f"[done] run dir: {run_dir}")
    if use_wandb:
        wandb.finish()


# ---------------- Sampling helper ----------------

def sample_and_log(rng_key,
                   score_model,
                   params,
                   H, W, img_size, batch_size,
                   out_dir, epoch, target_np=None,
                   sampler_name="pc", num_steps=500, snr=0.16, eps=1e-3,
                   marginal_prob_std_fn=None, diffusion_coeff_fn=None):
    """Returns (images_to_log, eval_metrics) where images_to_log is a list of (caption, image-array)."""
    import matplotlib.pyplot as plt
    from torchvision.utils import save_image

    # Choose sampler
    sampler_name = (sampler_name or "pc").lower()
    if sampler_name in ("pc", "predictor-corrector"):
        def _run(rng):
            return pc_sampler(rng, score_model, params, marginal_prob_std_fn, diffusion_coeff_fn,
                              batch_size=batch_size, img_size=img_size, num_steps=num_steps,
                              snr=snr, eps=eps)
    elif sampler_name in ("em", "euler", "euler-maruyama"):
        def _run(rng):
            return Euler_Maruyama_sampler(rng, score_model, params, marginal_prob_std_fn, diffusion_coeff_fn,
                                          batch_size=batch_size, num_steps=num_steps, eps=eps, img_size=img_size)
    elif sampler_name in ("ode", "pf-ode", "probability-flow-ode"):
        def _run(rng):
            return ode_sampler(rng, score_model, params, marginal_prob_std_fn, diffusion_coeff_fn,
                               batch_size=batch_size, img_size=img_size, eps=eps)
    else:
        raise ValueError(f"Unknown sampler: {sampler_name}")

    rng_key, step_rng = jax.random.split(rng_key)
    samples = _run(step_rng)
    samples = jnp.clip(samples, 0.0, 1.0)
    samples = jnp.transpose(samples.reshape((-1, H, W, 1)), (0, 3, 1, 2))
    samples_t = torch.tensor(np.asarray(samples))

    eval_metrics = {}
    if (target_np is not None):
        target01 = (target_np + 1.0) * 0.5  # [-1,1] -> [0,1]
        target01 = np.expand_dims(target01, axis=0)     # (1,1,H,W)
        diff = samples_t.numpy() - target01
        mse = np.mean(diff * diff, axis=(1, 2, 3))
        eval_metrics["mse_best"] = float(np.min(mse))
        eval_metrics["mse_mean"] = float(np.mean(mse))
        print(f"[eval] epoch {epoch} MSE-to-target: best={eval_metrics['mse_best']:.6f}, "
              f"mean={eval_metrics['mse_mean']:.6f}")

    grid = make_grid_torch(samples_t)
    grid_np = grid.permute(1, 2, 0).numpy()
    grid_np = np.clip(grid_np, 0.0, 1.0)

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    ensure_dir(out_dir)
    out_matplotlib = os.path.join(out_dir, f"grid_ep{epoch:03d}_{ts}.png")
    out_torchvision = os.path.join(out_dir, f"grid_ep{epoch:03d}_{ts}_tv.png")

    # Matplotlib save
    plt.figure(figsize=(6, 6))
    plt.axis("off")
    plt.imshow(grid_np, vmin=0., vmax=1.)
    plt.tight_layout()
    plt.savefig(out_matplotlib, bbox_inches="tight", pad_inches=0)
    plt.close()

    # Torchvision save
    save_image(grid, out_torchvision)

    print(f"[saved] {out_matplotlib}\n[saved] {out_torchvision}")
    return [("sample_grid", grid_np)], eval_metrics


if __name__ == "__main__":
    main()
