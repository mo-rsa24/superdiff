import argparse, os, json, hashlib, math, functools
import numpy as np
import jax
import jax.numpy as jnp
import tensorflow as tf
from diffusion.equations import marginal_prob_std as ve_marginal_prob_std, diffusion_coeff as ve_diffusion_coeff
from diffusion.sampling import pc_sampler, Euler_Maruyama_sampler, ode_sampler
from models.cxr_unet import ScoreNet
from flax.training.train_state import TrainState
from flax.serialization import from_bytes
import optax
import torch
import os
from typing import Tuple
from run.cxr import ensure_dir, sample_and_log

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
gpus = tf.config.list_physical_devices("GPU")
for g in gpus:
    tf.config.experimental.set_memory_growth(g, True)

def to_uint8_vis(x: np.ndarray) -> np.ndarray:
    """
    Map [0,1] float -> uint8 for duplicate hashing (sample_and_log already clips to [0,1]).
    """
    x = np.clip(x, 0.0, 1.0)
    return (x * 255.0).astype(np.uint8)

def duplicate_groups_from_batch(samples_bhwc01: np.ndarray):
    """
    Given samples as (B,H,W,1) in [0,1], return {hash: [indices]} for duplicates (len>1).
    """
    u8 = to_uint8_vis(samples_bhwc01)
    groups = {}
    for i in range(u8.shape[0]):
        h = hashlib.sha1(u8[i, :, :, 0].tobytes()).hexdigest()
        groups.setdefault(h, []).append(i)
    return {h: idxs for h, idxs in groups.items() if len(idxs) > 1}

def pprint_duplicate_groups(groups, label="samples"):
    if not groups:
        print(f"[{label}] ✅ No duplicate tiles detected.")
        return
    print(f"[{label}] ⚠️ Duplicate tiles:")
    for h, idxs in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        print(f"  - hash={h[:10]}… count={len(idxs)} -> indices {idxs}")

def load_run_meta(run_dir: str):
    meta_path = os.path.join(run_dir, "run_meta.json")
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            return json.load(f)
    return {}

def infer_ckpt_path(run_dir: str):
    ckpt_path = os.path.join(run_dir, "ckpts", "ep0003.flax")
    if not tf.io.gfile.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    return ckpt_path

# --------------------------
# Minimal model + ckpt load
# --------------------------

def build_model(img_size: int, channels: str, embed_dim: int, batch_size: int, sigma_max: float):
    """
    Mirror your trainer’s model init (ScoreNet(marginal_prob_std), fake_x/t) so
    restored params match shapes. :contentReference[oaicite:5]{index=5}
    """
    chans = tuple(int(c) for c in channels.split(",")) if isinstance(channels, str) else tuple(channels)
    mpstd = functools.partial(ve_marginal_prob_std, sigma=float(sigma_max))
    score_model = ScoreNet(mpstd, channels=chans, embed_dim=int(embed_dim))
    # score_model = ScoreNet(ve_marginal_prob_std, channels=chans, embed_dim=int(embed_dim))
    fake_x = jnp.ones((batch_size, img_size, img_size, 1), dtype=jnp.float32)
    fake_t = jnp.ones((batch_size,), dtype=jnp.float32)
    # params = score_model.init({'params': jax.random.PRNGKey(0)}, fake_x, fake_t)
    # create a minimal TrainState like in training (optimizer isn't used here) :contentReference[oaicite:6]{index=6}
    cpu = jax.devices("cpu")[0]
    with jax.default_device(cpu):
        params = score_model.init({'params': jax.random.PRNGKey(0)}, fake_x, fake_t)
    params = jax.device_put(params, jax.devices("gpu")[0])
    tx = optax.adamw(learning_rate=1e-3)
    host_state = TrainState.create(apply_fn=score_model.apply, params=params, tx=tx)
    return score_model, host_state

def restore_params_from_ckpt(host_state: TrainState, ckpt_path: str):
    """
    Your training code saves (TrainState, ema_params, ema_decay) in last.flax when possible;
    we mirror that restore path and prefer EMA for sampling. :contentReference[oaicite:7]{index=7}
    """
    with tf.io.gfile.GFile(ckpt_path, "rb") as f:
        blob = f.read()
    # Try the (TrainState, ema_params, ema_decay) tuple first
    try:
        restored_state, ema_params, ema_decay = from_bytes((host_state, host_state.params, 0.0), blob)
        print(f"[ckpt] Restored TrainState + EMA (decay={ema_decay}).")
        return restored_state, ema_params
    except Exception:
        # Fallback: some older runs may store just TrainState or just params
        try:
            restored_state = from_bytes(host_state, blob)
            print("[ckpt] Restored TrainState; sampling from its params.")
            return restored_state, restored_state.params
        except Exception:
            # Last resort: params only
            params_only = from_bytes(host_state.params, blob)
            host_state = host_state.replace(params=params_only)
            print("[ckpt] Restored raw params; sampling from them.")
            return host_state, params_only

# --------------------------
# A tiny sampler that mirrors cxr.sample_and_log’s internal logic
# (so we can get raw batch back for duplicate testing)
# --------------------------

def run_sampler_once(*, rng_key, score_model, params, img_size: int, batch_size: int,
                    sampler_name: str, num_steps: int, snr: float, eps: float,
                     epoch: int, sigma_max: float):
    """
    This mirrors cxr.sample_and_log: select sampler, fold_in(epoch), split(key),
    run sampler, clip to [0,1], and reshape to (B,H,W,1). :contentReference[oaicite:8]{index=8} :contentReference[oaicite:9]{index=9}
    """
    mpstd = functools.partial(ve_marginal_prob_std, sigma=float(sigma_max))
    dcoeff = functools.partial(ve_diffusion_coeff, sigma=float(sigma_max))
    sampler_name = (sampler_name or "pc").lower()
    if sampler_name in ("pc", "predictor-corrector"):
        def _run(rng):
            return pc_sampler(rng, score_model, params, mpstd, dcoeff,
                              batch_size=batch_size, img_size=img_size, num_steps=num_steps, snr=snr, eps=eps)
    elif sampler_name in ("em", "euler", "euler-maruyama"):
        def _run(rng):
            return Euler_Maruyama_sampler(rng, score_model, params, mpstd, dcoeff,
                                          batch_size=batch_size, num_steps=num_steps, eps=eps, img_size=img_size)
    elif sampler_name in ("ode", "pf-ode", "probability-flow-ode"):
        def _run(rng):
            return ode_sampler(rng, score_model, params, mpstd, dcoeff,
                               batch_size=batch_size, img_size=img_size, eps=eps)
    else:
        raise ValueError(f"Unknown sampler: {sampler_name}")

    # Important: same RNG handling as in cxr.sample_and_log (fold_in + split) :contentReference[oaicite:10]{index=10}
    rng_key = jax.random.fold_in(rng_key, int(epoch))
    rng_key, step_rng = jax.random.split(rng_key)

    samples = _run(step_rng)                          # (B,H,W,1) in [0,1]
    samples = jnp.clip(samples, 0.0, 1.0)
    samples = np.asarray(samples)                     # JAX -> NumPy
    return samples, rng_key

# --------------------------
# CLI
# --------------------------

def parse_args():
    ap = argparse.ArgumentParser("Investigate repeats in sampling/logging (standalone)")
    ap.add_argument("--run_dir", required=True, help="Path to a finished training run (folder with ckpts/ & run_meta.json)")
    ap.add_argument("--sigma_max", type=float, default=25.0,
                    help="VE sigma_max (used to wrap VE functions like in training)")
    ap.add_argument("--ckpt_path", default=None, help="Override checkpoint path (else uses run_dir/ckpts/last.flax)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epoch", type=int, default=1, help="Epoch number to fold into RNG (matches training logging).")
    # Fallbacks if run_meta.json is missing / incomplete:
    ap.add_argument("--img_size", type=int, default=256)
    ap.add_argument("--sample_batch_size", type=int, default=8)
    ap.add_argument("--channels", type=str, default="64,128,256,512")
    ap.add_argument("--embed_dim", type=int, default=256)
    ap.add_argument("--sampler", choices=["pc", "em", "ode"], default="pc")
    ap.add_argument("--num_steps", type=int, default=500)
    ap.add_argument("--snr", type=float, default=0.16)
    ap.add_argument("--eps", type=float, default=1e-3)
    ap.add_argument("--out_dir", default="smoke_samples")
    return ap.parse_args()

# --------------------------
# Main
# --------------------------

def main():
    args = parse_args()

    meta = load_run_meta(args.run_dir)
    # Prefer recorded meta; fall back to flags
    img_size  = int(meta.get("img_size", args.img_size))
    bsamp     = int(meta.get("sample_batch_size", args.sample_batch_size))
    channels  = meta.get("channels", args.channels)
    embed_dim = int(meta.get("embed_dim", args.embed_dim))
    sampler   = meta.get("sampler", args.sampler)
    num_steps = int(meta.get("num_steps", args.num_steps))
    snr       = float(meta.get("snr", args.snr))
    sigma_max = float(meta.get("sigma_max", args.sigma_max))
    eps       = float(meta.get("eps", args.eps))

    ckpt_path = args.ckpt_path or infer_ckpt_path(args.run_dir)

    print(f"[meta] img_size={img_size}  sample_batch_size={bsamp}  channels={channels}  "
          f"embed_dim={embed_dim}  sampler={sampler}  num_steps={num_steps}  snr={snr}  eps={eps}")
    print(f"[meta] sigma_max={sigma_max}  run_dir={args.run_dir}  ckpt_path={ckpt_path}")
    print(f"[ckpt] using {ckpt_path}")

    # Build model & restore params (EMA if available), exactly like the trainer :contentReference[oaicite:11]{index=11}
    # score_model, host_state = build_model(img_size, channels, embed_dim, bsamp)
    score_model, host_state = build_model(img_size, channels, embed_dim, bsamp, sigma_max)
    host_state, ema_or_params = restore_params_from_ckpt(host_state, ckpt_path)

    # 1) Reuse your cxr.sample_and_log to save the grid(s) exactly as training does. :contentReference[oaicite:12]{index=12}
    rng = jax.random.PRNGKey(args.seed)
    out_dir = ensure_dir(args.out_dir)

    images, eval_dict, rng = sample_and_log(
        rng_key=rng,
        score_model=score_model,
        params=ema_or_params,
        H=img_size, W=img_size, img_size=img_size,
        batch_size=bsamp,
        out_dir=out_dir,
        epoch=args.epoch,
        target_np=None,                     # set if you want MSE-to-target like in overfit_one path
        sampler_name=sampler,
        num_steps=num_steps,
        snr=snr,
        eps=eps,
        marginal_prob_std_fn=ve_marginal_prob_std,
        diffusion_coeff_fn=ve_diffusion_coeff,
    )
    print(f"[sample_and_log] eval: {eval_dict}")

    # 2) Run the sampler once (same internal logic as sample_and_log) so we can inspect per-tile data.
    samples_bhwc01, _ = run_sampler_once(
        rng_key=rng, score_model=score_model, params=ema_or_params,
        img_size=img_size, batch_size=bsamp,
        sampler_name=sampler, num_steps=num_steps, snr=snr, eps=eps, epoch=args.epoch
    )

    # Duplicate investigation
    means = samples_bhwc01[:, :8, :8, :].mean(axis=(1, 2, 3))
    print("[debug] first 8x8 per-tile means:", np.round(means, 6))
    groups = duplicate_groups_from_batch(samples_bhwc01)
    pprint_duplicate_groups(groups, label="sampler_batch")

    # Optional: also emit a duplicate-colored grid using torchvision (helpful when eyeballing)
    try:
        from torchvision.utils import make_grid, save_image
        # (B,H,W,1)->(B,1,H,W) tensor
        t = torch.tensor(samples_bhwc01.transpose(0,3,1,2))
        # Mark duplicates by normalizing unique groups (just a diagnostic)
        # Here we simply save the normal grid again; customize if you want colors.
        grid = make_grid(t, nrow=int(math.sqrt(max(1, t.shape[0]))))
        save_image(grid, os.path.join(out_dir, "grid_sampler_batch.png"))
        print(f"[saved] {os.path.join(out_dir, 'grid_sampler_batch.png')}")
    except Exception as e:
        print(f"[warn] torchvision save failed (ok to ignore): {e}")

    print("[done] Investigated duplicates. See printed groups and images in:", out_dir)

if __name__ == "__main__":
    main()
