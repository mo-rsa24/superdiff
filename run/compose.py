#!/usr/bin/env python3
# compose.py — compose two pretrained CXR score models and sample images
# Usage examples:
#   python compose.py --run_a tb256-of1-... --run_b pn256-of1-... --alpha 0.5 --mode fixed --sampler pc
#   python compose.py --run_a tb256-full-... --run_b pn256-full-... --sampler em --batch_size 16
#   python compose.py --run_a A --run_b B --estimate_ll 1 --batch_size 8
#   python compose.py --run_a tb256-tiny8-r2-tb-train-cxr256-of1-ch96x192x384x768-ve-pc-lr0.0002-b8x1-slurm155813 --run_b pn256-tiny8-r2-pneumonia-train-cxr256-of1-ch96x192x384x768-ve-pc-lr0.0002-b8x1-slurm155812 --estimate_ll 1 --batch_size 8
# Place this file at the repo root (same level as cxr_sde.slurm). It will add repo root to PYTHONPATH.

import os, sys, json, math, functools
from datetime import datetime
from typing import Tuple, Optional

# --- ensure repo root on sys.path (so 'diffusion', 'models' import cleanly) ---
_REPO_ROOT = os.path.abspath(os.path.dirname(__file__))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import argparse
import numpy as np
import jax
import jax.numpy as jnp
import optax
import tensorflow as tf

from flax.training.train_state import TrainState
from flax.serialization import to_bytes, from_bytes

# Project modules
from models.cxr_unet import ScoreNet
from diffusion.equations import marginal_prob_std, diffusion_coeff
# If you have VPSDE variants, you can extend build_sde_from_cfg below.

try:
    # Optional: LL composition (if you patched your sampling/equations as in earlier step)
    from diffusion.sampling import compose_and_estimate_log_likelihood_along_superposed_trajectory as compose_ll_images
    _HAS_LL = True
except Exception:
    _HAS_LL = False


# ------------------ utils ------------------

def n_local_devices() -> int:
    return jax.local_device_count()

def ensure_dir(p: str) -> str:
    os.makedirs(p, exist_ok=True)
    return p

def parse_channels(s: str):
    return tuple(int(c.strip()) for c in str(s).split(",") if c.strip())

def latest_subdir(path: str) -> str:
    """Return newest timestamped subdir if exists, else the dir itself."""
    if not os.path.isdir(path):
        raise FileNotFoundError(f"No such directory: {path}")
    subs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    if not subs:
        return path
    subs.sort()
    return os.path.join(path, subs[-1])

def resolve_run_dir(runs_root: str, run_name_or_path: str) -> str:
    """Accept either a name under runs_root or an absolute/relative path."""
    candidate = os.path.join(runs_root, run_name_or_path)
    if os.path.isdir(candidate):
        return latest_subdir(candidate)
    # else: user passed a full path already
    return latest_subdir(run_name_or_path)

def latest_ckpt(ckpt_dir: str) -> str:
    last = os.path.join(ckpt_dir, "last.flax")
    if tf.io.gfile.exists(last):
        return last
    eps = [p for p in tf.io.gfile.listdir(ckpt_dir) if p.startswith("ep") and p.endswith(".flax")]
    if not eps:
        # raise FileNotFoundError(f"No checkpoints in {ckpt_dir}")
        pass
    eps.sort()
    return os.path.join(ckpt_dir, eps[-1])

def load_config(run_dir: str) -> dict:
    meta = os.path.join(run_dir, "run_meta.json")
    if not tf.io.gfile.exists(meta):
        raise FileNotFoundError(f"Missing run_meta.json in {run_dir}")
    with tf.io.gfile.GFile(meta, "r") as f:
        return json.load(f)

def build_sde_from_cfg(cfg: dict):
    """Return (marginal_prob_std_fn, diffusion_coeff_fn, label). Currently VE(σmax)."""
    sigma_max = float(cfg.get("sigma_max", 25.0))
    mstd = functools.partial(marginal_prob_std, sigma=sigma_max)
    dcoeff = functools.partial(diffusion_coeff,   sigma=sigma_max)
    return mstd, dcoeff, f"VE(σmax={sigma_max:g})"

def instantiate_model_from_cfg(cfg: dict, marginal_prob_std_fn):
    ch = parse_channels(cfg.get("channels", "64,128,256,512"))
    emb = int(cfg.get("embed_dim", 256))
    model = ScoreNet(marginal_prob_std_fn, channels=ch, embed_dim=emb)
    return model, ch, emb

def load_params_tuple(path: str, state_template, ema_template, ema_decay_default=0.9995):
    """Supports two formats: (TrainState, ema_params, ema_decay) OR TrainState only."""
    with tf.io.gfile.GFile(path, "rb") as f:
        blob = f.read()
    # Try tuple first
    try:
        ts, ema, decay = from_bytes((state_template, ema_template, ema_decay_default), blob)
        return ts.params, ema, decay
    except Exception:
        ts = from_bytes(state_template, blob)
        return ts.params, ts.params, ema_decay_default

def load_model_from_run(runs_root: str, run_name_or_path: str):
    """Return (model, params_for_sampling, cfg, run_dir, (mstd,dcoeff,label))."""
    run_dir = resolve_run_dir(runs_root, run_name_or_path)
    cfg = load_config(run_dir)
    mstd, dcoeff, sde_label = build_sde_from_cfg(cfg)

    model, ch, emb = instantiate_model_from_cfg(cfg, mstd)
    H = int(cfg.get("img_size", 256))
    C = 1
    per_dev = max(1, int(cfg.get("batch_per_device", 4)))
    batch = per_dev * max(1, n_local_devices())

    fake_x = jnp.ones((batch, H, H, C), dtype=jnp.float32)
    fake_t = jnp.ones((batch,), dtype=jnp.float32)
    params = model.init({'params': jax.random.PRNGKey(0)}, fake_x, fake_t)
    state_tmpl = TrainState.create(apply_fn=model.apply, params=params, tx=optax.adam(1e-4))
    ema_tmpl = params

    ckpt_dir = os.path.join(run_dir, "ckpts")
    ckpt = latest_ckpt(ckpt_dir)
    params_model, ema_params, decay = load_params_tuple(ckpt, state_tmpl, ema_tmpl, cfg.get("ema_decay", 0.9995))
    # params_to_use = ema_params  # prefer EMA for sampling if present
    # return model, params_to_use, cfg, run_dir, (mstd, dcoeff, sde_label)
    return model, params,  cfg, run_dir, (mstd, dcoeff, sde_label)

def assert_compat(cfg_a: dict, cfg_b: dict):
    keys = ["img_size", "channels", "embed_dim", "sde", "sigma_max"]
    mismatches = []
    for k in keys:
        if str(cfg_a.get(k)) != str(cfg_b.get(k)):
            mismatches.append((k, cfg_a.get(k), cfg_b.get(k)))
    if mismatches:
        msg = "Incompatible runs for composition:\n" + "\n".join([f"  {k}: A={a} vs B={b}" for k,a,b in mismatches])
        raise ValueError(msg)

def make_composed_score_fn(score_model):
    """
    Returns a vmapped score(x,t) that blends two models' scores:
      fixed  : s = s_B + α (s_A - s_B)
      sum    : s = s_A + s_B
      normsum: s = (s_A + s_B) / √2
    We use vmap (batch) and optionally pmap (devices) via calling code.
    """
    def score(params_a, params_b, x, t, alpha: float, mode: str):
        s_a = score_model.apply(params_a, x, t)
        s_b = score_model.apply(params_b, x, t)
        if mode == "sum":
            s = s_a + s_b
        elif mode == "normsum":
            s = (s_a + s_b) / jnp.sqrt(2.0)
        else:  # fixed
            s = s_b + alpha * (s_a - s_b)
        return s
    return jax.vmap(score, in_axes=(None, None, 0, 0, None, None))

def to_device_batch(x_np, H, W, C):
    """(B,H,W,C) -> (n_dev, B//n_dev, H,W,C) for pmap, padding if needed."""
    devices = n_local_devices()
    B = x_np.shape[0]
    if B % devices != 0:
        pad = devices - (B % devices)
        x_np = np.concatenate([x_np, np.repeat(x_np[:1], pad, axis=0)], axis=0)
        B = x_np.shape[0]
    return x_np.reshape(devices, B // devices, H, W, C)

def from_device_batch(x):
    """Inverse of to_device_batch; drop any padded samples."""
    if isinstance(x, (jax.Array, jnp.ndarray)):
        x = np.asarray(x)
    n_dev, per, *rest = x.shape
    return x.reshape(n_dev * per, *rest)


# ------------------ samplers (PC/EM/ODE) ------------------

def pc_sampler(rng, composed_score, params_a, params_b, mstd, dcoeff,
               batch_size: int, img_size: int, snr: float = 0.16, steps: int = 500, eps: float = 1e-3,
               alpha: float = 0.5, mode: str = "fixed"):
    H = W = img_size
    C = 1
    devs = n_local_devices()
    per_dev = max(1, batch_size // devs)
    batch_size = per_dev * devs

    rng, sub = jax.random.split(rng)
    init = jax.random.normal(sub, (devs, per_dev, H, W, C)) * mstd(1.)
    x = init

    t_grid = jnp.linspace(1., eps, steps)
    step_size = float(t_grid[0] - t_grid[1])

    for t in t_grid:
        bt = jnp.ones((devs, per_dev)) * t

        # Corrector (Langevin)
        score = jax.pmap(composed_score)(params_a, params_b, x, bt, alpha, mode)
        # grad norm estimate per device (flatten per sample)
        grad_norm = jnp.sqrt(jnp.mean(jnp.sum(score**2, axis=(2,3,4))))
        noise_norm = np.sqrt(H * W * C)
        langevin_step = 2 * (snr * noise_norm / (grad_norm + 1e-8)) ** 2
        rng, sub = jax.random.split(rng)
        z = jax.random.normal(sub, x.shape)
        x = x + langevin_step * score + jnp.sqrt(2 * langevin_step) * z

        # Predictor (reverse SDE Euler step)
        g = dcoeff(t)
        score = jax.pmap(composed_score)(params_a, params_b, x, bt, alpha, mode)
        x_mean = x + (g ** 2) * score * step_size
        rng, sub = jax.random.split(rng)
        z = jax.random.normal(sub, x.shape)
        x = x_mean + jnp.sqrt(g ** 2 * step_size) * z

    return from_device_batch(x_mean)

def em_sampler(rng, composed_score, params_a, params_b, mstd, dcoeff,
               batch_size: int, img_size: int, steps: int = 500, eps: float = 1e-3,
               alpha: float = 0.5, mode: str = "fixed"):
    H = W = img_size
    C = 1
    devs = n_local_devices()
    per_dev = max(1, batch_size // devs)
    batch_size = per_dev * devs

    rng, sub = jax.random.split(rng)
    x = jax.random.normal(sub, (devs, per_dev, H, W, C)) * mstd(1.)
    t_grid = jnp.linspace(1., eps, steps)
    step_size = float(t_grid[0] - t_grid[1])

    for t in t_grid:
        bt = jnp.ones((devs, per_dev)) * t
        g = dcoeff(t)
        score = jax.pmap(composed_score)(params_a, params_b, x, bt, alpha, mode)
        mean_x = x + (g ** 2) * score * step_size
        rng, sub = jax.random.split(rng)
        x = mean_x + jnp.sqrt(step_size) * g * jax.random.normal(sub, x.shape)

    return from_device_batch(mean_x)

def ode_sampler(rng, composed_score, params_a, params_b, mstd, dcoeff,
                batch_size: int, img_size: int, steps: int = 500, eps: float = 1e-3,
                alpha: float = 0.5, mode: str = "fixed"):
    # A simple fixed-step RK2 for probability-flow ODE
    H = W = img_size
    C = 1
    devs = n_local_devices()
    per_dev = max(1, batch_size // devs)
    batch_size = per_dev * devs

    rng, sub = jax.random.split(rng)
    x = jax.random.normal(sub, (devs, per_dev, H, W, C)) * mstd(1.)
    t_grid = jnp.linspace(1., eps, steps)
    h = float(t_grid[0] - t_grid[1])

    def drift(x, t):
        bt = jnp.ones((devs, per_dev)) * t
        g = dcoeff(t)
        s = jax.pmap(composed_score)(params_a, params_b, x, bt, alpha, mode)
        return -0.5 * (g ** 2) * s

    for t in t_grid:
        k1 = drift(x, t)
        k2 = drift(x + 0.5 * h * k1, t - 0.5 * h)
        x = x + h * k2

    return from_device_batch(x)


# ------------------ CLI & main ------------------

def parse_args():
    p = argparse.ArgumentParser("Compose two pretrained CXR SDE models and sample images")
    p.add_argument("--runs_root", default="runs")
    p.add_argument("--run_a", required=True, help="Run folder name under runs/ or absolute path")
    p.add_argument("--run_b", required=True, help="Run folder name under runs/ or absolute path")

    p.add_argument("--alpha", type=float, default=0.5, help="Mixing for 'fixed' mode")
    p.add_argument("--mode", choices=["fixed","sum","normsum"], default="fixed")

    p.add_argument("--sampler", choices=["pc","em","ode"], default="pc")
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--snr", type=float, default=0.16)
    p.add_argument("--eps", type=float, default=1e-3)

    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--exp_name", default=None, help="Name for compose run dir; default auto from inputs")
    p.add_argument("--run_name", default=None, help="Optional sub-name for compose run dir")

    # Optional: estimate log-likelihood with superposed PF-ODE (needs patched diffusion.sampling/equations)
    p.add_argument("--estimate_ll", type=int, default=0)

    # --- Visualization (train batches) ---
    p.add_argument("--viz_train_grid", action="store_true",
                   help="Periodically save a grid of the current training batch (B,1,H,W).")
    p.add_argument("--viz_train_grid_every_steps", type=int, default=200,
                   help="How often (in global steps) to save a training grid.")
    p.add_argument("--viz_train_grid_max_images", type=int, default=64,
                   help="Cap the number of images shown in the grid.")
    p.add_argument("--viz_train_grid_nrow", type=int, default=0,
                   help="Grid columns; 0 means auto sqrt.")
    p.add_argument("--viz_train_grid_dir", default=None,
                   help="Output dir for training grids (defaults to run_dir/train_grids).")

    return p.parse_args()


def main():
    args = parse_args()

    # Load both models from their latest checkpoints
    model_a, params_a, cfg_a, dir_a, (mstd_a, dcoeff_a, sde_a) = load_model_from_run(args.runs_root, args.run_a)
    model_b, params_b, cfg_b, dir_b, (mstd_b, dcoeff_b, sde_b) = load_model_from_run(args.runs_root, args.run_b)

    # Sanity: same architecture/schedule
    assert_compat(cfg_a, cfg_b)

    # Use schedule from A (they match by assertion)
    mstd, dcoeff = mstd_a, dcoeff_a
    H = int(cfg_a.get("img_size", 256))
    img_size = H

    # Build composed score (vmap) and wrap with pmap at call sites
    composed_score = make_composed_score_fn(model_a)

    # Create output run dir
    base_name = args.exp_name or f"COMPOSE-{args.mode}-a[{os.path.basename(args.run_a)}]_b[{os.path.basename(args.run_b)}]_alpha{args.alpha:g}"
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = os.path.join(args.runs_root, base_name, ts)
    samples_dir = ensure_dir(os.path.join(out_dir, "samples"))

    # Optional: superposed PF-ODE with LL accumulation (image-space)
    if args.estimate_ll:
        if not _HAS_LL:
            print("[warn] --estimate_ll requested but diffusion.sampling.compose_and_estimate_log_likelihood_along_superposed_trajectory not found. Skipping.")
        else:
            # Make tiny TrainState wrappers that expose .apply_fn/.params
            state_a = TrainState.create(apply_fn=model_a.apply, params=params_a, tx=optax.sgd(0.0))
            state_b = TrainState.create(apply_fn=model_b.apply, params=params_b, tx=optax.sgd(0.0))

            B = max(1, args.batch_size)
            shape = (B, H, H, 1)
            rng = jax.random.PRNGKey(args.seed)
            traj, lla, llb = compose_ll_images(state_a, state_b, rng, dt=1e-3, t=1.0, shape=shape)

            # Final slice to image grid (assuming model domain in [0,1]; if [-1,1], map accordingly)
            x_final = traj[:, -1, ...]
            x_vis = jnp.clip(x_final, 0.0, 1.0)

            import torch
            from torchvision.utils import make_grid, save_image
            x_t = torch.tensor(np.asarray(jnp.transpose(x_vis, (0,3,1,2))))  # NCHW
            grid = make_grid(x_t, nrow=int(math.sqrt(max(1, x_t.shape[0]))))
            grid_np = grid.permute(1,2,0).numpy()

            import matplotlib.pyplot as plt
            plt.figure(figsize=(6,6)); plt.axis("off")
            plt.imshow(grid_np, vmin=0., vmax=1.); plt.tight_layout()
            out_png = os.path.join(samples_dir, f"composeLL_grid_{ts}.png")
            plt.savefig(out_png, bbox_inches="tight", pad_inches=0); plt.close()
            save_image(grid, os.path.join(samples_dir, f"composeLL_grid_{ts}_tv.png"))
            print(f"[compose-LL] saved {out_png}")
            print(f"[compose-LL] mean LL A: {lla[:,-1].mean():.4f}  |  mean LL B: {llb[:,-1].mean():.4f}")

    # Standard composed sampling (PC/EM/ODE)
    rng = jax.random.PRNGKey(args.seed)
    if args.sampler == "pc":
        imgs = pc_sampler(rng, composed_score, params_a, params_b, mstd, dcoeff,
                          batch_size=args.batch_size, img_size=img_size, snr=args.snr,
                          steps=args.steps, eps=args.eps, alpha=args.alpha, mode=args.mode)
    elif args.sampler == "em":
        imgs = em_sampler(rng, composed_score, params_a, params_b, mstd, dcoeff,
                          batch_size=args.batch_size, img_size=img_size,
                          steps=args.steps, eps=args.eps, alpha=args.alpha, mode=args.mode)
    else:
        imgs = ode_sampler(rng, composed_score, params_a, params_b, mstd, dcoeff,
                           batch_size=args.batch_size, img_size=img_size,
                           steps=args.steps, eps=args.eps, alpha=args.alpha, mode=args.mode)

    # NHWC [0,1] -> NCHW tensor grid
    imgs = jnp.clip(imgs, 0.0, 1.0)
    imgs = jnp.transpose(imgs.reshape((-1, img_size, img_size, 1)), (0, 3, 1, 2))
    import torch
    from torchvision.utils import make_grid, save_image
    imgs_t = torch.tensor(np.asarray(imgs))
    grid = make_grid(imgs_t, nrow=int(math.sqrt(max(1, imgs_t.shape[0]))))

    # Save
    import matplotlib.pyplot as plt
    grid_np = grid.permute(1,2,0).numpy()
    plt.figure(figsize=(6,6)); plt.axis("off")
    plt.imshow(grid_np, vmin=0., vmax=1.); plt.tight_layout()
    out_png = os.path.join(samples_dir, f"compose_{args.mode}_alpha{args.alpha:g}_{ts}.png")
    plt.savefig(out_png, bbox_inches="tight", pad_inches=0); plt.close()
    save_image(grid, os.path.join(samples_dir, f"compose_{args.mode}_alpha{args.alpha:g}_{ts}_tv.png"))
    print(f"[compose] saved {out_png}")

    # Also save a minimal meta for provenance
    with open(os.path.join(out_dir, "compose_meta.json"), "w") as f:
        json.dump({
            "run_a": dir_a, "run_b": dir_b,
            "mode": args.mode, "alpha": args.alpha,
            "sampler": args.sampler, "steps": args.steps,
            "batch_size": args.batch_size, "eps": args.eps,
            "seed": args.seed
        }, f, indent=2)

if __name__ == "__main__":
    main()
