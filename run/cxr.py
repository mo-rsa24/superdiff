import argparse, os, math, json
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

from flax.training.train_state import TrainState
from flax.serialization import to_bytes, from_bytes

# --- Project modules (JAX) ---
# Use 1-arg wrappers (bind sigma) so UNet can call marginal_prob_std(t) directly.
import functools
from diffusion.equations import marginal_prob_std, diffusion_coeff
from diffusion.sampling import ode_sampler
from models.cxr_unet import ScoreNet
from train.train_score_sde import get_train_step_fn

sigma = 25.0
marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=sigma)
diffusion_coeff_fn   = functools.partial(diffusion_coeff,   sigma=sigma)

# --- Local dataset ---
from datasets.ChestXRay import ChestXrayDataset

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


# ---------------- CLI ----------------

def parse_args():
    p = argparse.ArgumentParser("JAX SDE Chest X-ray trainer (overfit/tiny/full) with tqdm + wandb")

    # Data
    p.add_argument("--data_root", default="../datasets/cleaned")
    p.add_argument("--task", choices=["TB", "PNEUMONIA"], default="TB")
    p.add_argument("--split", choices=["train", "val", "test"], default="train")
    p.add_argument("--img_size", type=int, default=256)
    p.add_argument("--class_filter", type=int, default=1, help="Optional: keep a class index only (e.g., 1)")

    # Debug/overfit
    p.add_argument("--overfit_one", action="store_true")
    p.add_argument("--overfit_k", type=int, default=0)
    p.add_argument("--eval_mse_to_target", action="store_true")
    p.add_argument("--sample_every", type=int, default=1)

    # Model & training (good starting points)
    p.add_argument("--channels", type=str, default="64,128,256,512")
    p.add_argument("--embed_dim", type=int, default=256)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_per_device", type=int, default=4)
    p.add_argument("--sample_batch_size", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)

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

    return p.parse_args()


# ---------------- Training entry ----------------

def main():
    args = parse_args()

    # Derive mode string for clean run naming
    mode = "full"
    if args.overfit_one:
        mode = "of1"
    elif args.overfit_k > 0:
        mode = f"tiny{args.overfit_k}"

    # Channels tuple
    channels = tuple(int(c.strip()) for c in args.channels.split(",") if c.strip())

    # Construct run directory that is unique per job
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
        f"-lr{args.lr:g}-b{per_dev}x{ndev}"
    )
    if slurm_id:
        exp_slug += f"-slurm{slurm_id}"

    # If resuming, trust resume_dir; otherwise create a fresh run_dir
    if args.resume_dir:
        run_dir = args.resume_dir
        print(f"[info] Resuming into: {run_dir}")
    else:
        # Optional custom run name
        base_run_name = args.run_name or exp_slug
        run_dir = os.path.join(args.output_root, base_run_name, ts)

    ckpt_dir    = ensure_dir(os.path.join(run_dir, "ckpts"))
    samples_dir = ensure_dir(args.samples_dir or os.path.join(run_dir, "samples"))
    meta_path   = os.path.join(run_dir, "run_meta.json")

    # Effective checkpoint paths
    ckpt_latest = os.path.join(ckpt_dir, "last.flax")

    # Snapshot config to file for provenance
    cfg_dump = dict(vars(args))
    cfg_dump.update({
        "exp_slug": exp_slug,
        "run_dir": run_dir,
        "ckpt_latest": ckpt_latest,
        "samples_dir": samples_dir,  # override args.samples_dir (which may be None)
    })
    ensure_dir(run_dir)
    with open(meta_path, "w") as f:
        json.dump(cfg_dump, f, indent=2, sort_keys=True)

    # Make dirs early so wandb "dir" can point to run_dir
    os.makedirs(samples_dir, exist_ok=True)

    # RNG & shapes
    rng = jax.random.PRNGKey(args.seed)
    C = 1

    # --- Dataset & DataLoader ---
    ds = ChestXrayDataset(
        root_dir=args.data_root, task=args.task, split=args.split,
        img_size=args.img_size, class_filter=args.class_filter
    )

    # Small dataset summary -> wandb table later
    label_counts = Counter(ds.labels)
    ds_size = len(ds)

    # Overfit modes (dataset wrappers)
    if args.overfit_one:
        first_img, _ = ds[0]  # (1,H,W) in [-1,1]
        class RepeatOne(torch.utils.data.Dataset):
            def __init__(self, img, length=8192):
                self.img = img.clone()
                self.length = length
            def __len__(self): return self.length
            def __getitem__(self, idx): return self.img, 0
        train_ds = RepeatOne(first_img, length=max(8192, ds_size))
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

    # --- Model ---
    score_model = ScoreNet(
        marginal_prob_std_fn,
        channels=channels,
        embed_dim=args.embed_dim,
    )
    fake_x = jnp.ones((batch_size, H, W, C), dtype=jnp.float32)
    fake_t = jnp.ones((batch_size,), dtype=jnp.float32)
    params = score_model.init({'params': rng}, fake_x, fake_t)

    # Optimizer & TrainState
    tx = optax.adam(args.lr)
    host_state = TrainState.create(apply_fn=score_model.apply, params=params, tx=tx)

    # Resume from checkpoint if requested or present
    resume_ckpt = None
    if args.resume_dir and tf.io.gfile.exists(ckpt_latest):
        resume_ckpt = ckpt_latest
    elif args.ckpt_path and tf.io.gfile.exists(args.ckpt_path):
        resume_ckpt = args.ckpt_path

    if resume_ckpt:
        print(f"[info] Loading checkpoint from {resume_ckpt}")
        with tf.io.gfile.GFile(resume_ckpt, "rb") as f:
            host_state = from_bytes(host_state, f.read())
    else:
        print("[info] No checkpoint; fresh training.")

    # Replicate to devices & get pmapped step
    state = jax.device_put_replicated(host_state, jax.local_devices())
    train_step_fn = get_train_step_fn(score_model, marginal_prob_std_fn)

    # --- Weights & Biases init ---
    use_wandb = bool(args.wandb and _WANDB_AVAILABLE)
    if args.wandb and not _WANDB_AVAILABLE:
        print("[warn] wandb requested, but not installed. Proceeding without online logging.")
    run_name = args.run_name or exp_slug
    wandb_run = None
    if use_wandb:
        wandb_tags = [t.strip() for t in args.wandb_tags.split(",") if t.strip()]
        wandb_config = dict(**vars(args), exp_slug=exp_slug, run_dir=run_dir,
                            ds_size=ds_size, label_counts=dict(label_counts),
                            n_local_devices=ndev, effective_batch=batch_size, sigma=sigma)
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
        # Define metrics/axes
        wandb.define_metric("train/step")
        wandb.define_metric("train/*", step_metric="train/step")
        wandb.define_metric("epoch/*", step_metric="epoch/idx")
        # Dataset table
        table = wandb.Table(columns=["label", "count"])
        for k, v in sorted(label_counts.items()):
            table.add_data(str(k), int(v))
        wandb.log({"dataset/summary": table, "epoch/idx": 0})

    # --- Train ---
    global_step = 0
    running_loss = 0.0
    running_count = 0

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

            global_step += 1
            running_loss += loss_val
            running_count += 1
            if (global_step % max(1, args.postfix_every_steps)) == 0:
                inner.set_postfix(loss=f"{loss_val:.4f}")

            if use_wandb and (global_step % max(1, args.log_every_steps) == 0):
                mean_loss = running_loss / running_count
                wandb.log({
                    "train/loss": mean_loss,  # smoothed loss over last window
                    "train/step": global_step
                })
                # reset window
                running_loss = 0.0
                running_count = 0

        # Save unreplicated checkpoint(s)
        host_state = jax.device_get(jax.tree_map(lambda v: v[0], state))
        # epoch-specific file (keeps history) + rolling 'last'
        ep_path = os.path.join(ckpt_dir, f"ep{epoch+1:04d}.flax")
        with tf.io.gfile.GFile(ep_path, "wb") as f:
            f.write(to_bytes(host_state))
        with tf.io.gfile.GFile(ckpt_latest, "wb") as f:
            f.write(to_bytes(host_state))

        avg_loss = float(np.mean(losses)) if len(losses) else float("nan")
        print(f"[epoch {epoch+1}] avg loss: {avg_loss:.6f}")

        if use_wandb:
            wandb.log({"epoch/avg_loss": avg_loss, "epoch/idx": epoch+1,
                       "ckpt/last_path": ckpt_latest, "ckpt/epoch_path": ep_path})

        # Periodic sampling (also logs to wandb as images)
        if ((epoch + 1) % max(1, args.sample_every)) == 0:
            images, eval_dict = sample_and_log(
                rng_key=rng,
                score_model=score_model,
                params=host_state.params,
                H=H, W=W, img_size=args.img_size,
                batch_size=args.sample_batch_size,
                out_dir=samples_dir,
                epoch=epoch+1,
                target_np=target_np,
            )
            if use_wandb:
                # images: list of (caption, PIL or numpy)
                wandb_imgs = [wandb.Image(img, caption=cap) for cap, img in images]
                log_payload = {"samples/grid": wandb_imgs, "epoch/idx": epoch+1}
                log_payload.update({f"eval/{k}": v for k, v in eval_dict.items()})
                wandb.log(log_payload)

    print(f"[done] run dir: {run_dir}")
    if use_wandb:
        wandb.finish()


def sample_and_log(rng_key, score_model, params, H, W, img_size, batch_size, out_dir, epoch, target_np=None):
    """Returns (images_to_log, eval_metrics) where images_to_log is a list of (caption, image-array)."""
    import matplotlib.pyplot as plt
    from torchvision.utils import save_image

    rng_key, step_rng = jax.random.split(rng_key)
    samples = ode_sampler(
        rng=step_rng,
        score_model=score_model,
        params=params,
        marginal_prob_std=marginal_prob_std_fn,
        diffusion_coeff=diffusion_coeff_fn,
        batch_size=batch_size,
        img_size=img_size,
    )
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
    # Convert grid to displayable numpy
    grid_np = grid.permute(1, 2, 0).numpy()
    grid_np = np.clip(grid_np, 0.0, 1.0)

    # Save files
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

    # Return images for wandb
    return [("sample_grid", grid_np)], eval_metrics


if __name__ == "__main__":
    main()
