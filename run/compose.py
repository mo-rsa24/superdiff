import argparse
import os
import math
import json
import functools
from datetime import datetime
from torchvision.utils import save_image, make_grid
import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf
import tqdm
import torch
from flax.serialization import from_bytes
from flax.training.train_state import TrainState
import optax  # Still needed for TrainState template

# --- Visualization Imports ---
import matplotlib.pyplot as plt
from torchvision.utils import save_image

# --- Project modules (only model architecture is needed) ---
from models.cxr_unet import ScoreNet

# =================================================================================
# SUPERPOSITION SAMPLER LOGIC
# Copied from your equations.py and sampling.py for a self-contained script
# =================================================================================

# Note: These SDE parameters are placeholders; the actual ones will be loaded from model configs.
beta_0 = 0.1
beta_1 = 20.0
log_alpha = lambda t: -0.5 * t * beta_0 - 0.25 * t ** 2 * (beta_1 - beta_0)
dlog_alphadt = jax.grad(lambda t: log_alpha(t).sum())
log_sigma = lambda t: jnp.log(t)


def score_function_hutchinson_estimator(key, model, params, t, x):
    """Gets the score and its divergence using Hutchinson's estimator."""
    eps = jax.random.randint(key, x.shape, 0, 2).astype(float) * 2 - 1.0
    _score_fn_x = lambda _x: model.apply(params, _x, t)
    score_val, jvp_val = jax.jvp(_score_fn_x, (x,), (eps,))
    divergence = (jvp_val * eps).sum(axis=tuple(range(1, x.ndim)))
    return score_val, divergence


score_function_hutchinson_estimator_jit = jax.jit(score_function_hutchinson_estimator, static_argnums=(1,))


# ---------- replace get_kappa + solver in compose.py ----------

def _vel_and_dlog_rng(rng, model, params, t_batch, x, diffusion_coeff_fn):
    """
    Returns:
      v(x,t)   : VE velocity field  = -(1/2) g(t)^2 * sθ(x,t)
      dlog(x,t): Hutchinson estimate of  -div v(x,t)  (this matches the notebook's 'dlog_*')
    """
    # score: sθ(x,t)
    score = model.apply(params, x, t_batch)  # ScoreNet signature: (x, t)

    # VE velocity: v = -(1/2) g(t)^2 * sθ
    g = diffusion_coeff_fn(t_batch)                   # (B,)
    g2 = (g ** 2).reshape((-1,) + (1,) * (x.ndim - 1))  # broadcast to x
    v = -0.5 * g2 * score

    # Hutchinson on the velocity map (NOT on the score!)
    eps = jax.random.randint(rng, x.shape, 0, 2).astype(x.dtype) * 2 - 1
    v_fn = lambda _x: -0.5 * ((diffusion_coeff_fn(t_batch) ** 2)
                              .reshape((-1,) + (1,) * (_x.ndim - 1))) * model.apply(params, _x, t_batch)
    jvp_out = jax.jvp(v_fn, (x,), (eps,))[1]
    # notebook computes: div = -(eps * jvp).sum(...)  → we return that as dlog
    red_axes = tuple(range(1, x.ndim))
    dlog = -(eps * jvp_out).sum(axis=red_axes)  # shape (B,)
    return v, dlog


@jax.jit
def get_kappa_velocity(sigma_t, dlog_a, dlog_b, v_a, v_b, eps=1e-6):
    red_axes = tuple(range(1, v_a.ndim))
    num = sigma_t * (dlog_a - dlog_b) + ((v_a - v_b) * (v_a + v_b)).sum(axis=red_axes)
    den = ((v_a - v_b) ** 2).sum(axis=red_axes) + eps
    return num / den


def ito_dynamic_estimator_solver(
    key, model_a, params_a, model_b, params_b,
    marginal_prob_std_fn, diffusion_coeff_fn,
    shape, num_steps=500, eps=1e-3
):
    """
    VE reverse-ODE sampler with AND superposition in velocity space.
    """
    B = shape[0]

    # init from the correct VE prior
    key, sub = jax.random.split(key)
    x = jax.random.normal(sub, shape) * marginal_prob_std_fn(1.0)

    # time grid (1 → eps)
    t_grid = jnp.linspace(1.0, eps, num_steps, dtype=x.dtype)
    dt = t_grid[0] - t_grid[1]  # positive scalar

    for t in t_grid:
        t_batch = jnp.full((B,), t, dtype=x.dtype)

        # velocities + dlogs for both models (identical schedules/functions)
        key, k1, k2 = jax.random.split(key, 3)
        v_a, dlog_a = _vel_and_dlog_rng(k1, model_a, params_a, t_batch, x, diffusion_coeff_fn)
        v_b, dlog_b = _vel_and_dlog_rng(k2, model_b, params_b, t_batch, x, diffusion_coeff_fn)

        # κ(t,x)
        sigma_t = marginal_prob_std_fn(t_batch)  # (B,)
        kappa = get_kappa_velocity(sigma_t, dlog_a, dlog_b, v_a, v_b)  # (B,)
        kappa = kappa.reshape((-1,) + (1,) * (x.ndim - 1))  # broadcast

        # mixed velocity and ODE step (VE: dx/dt = v_mix)
        v_mix = v_b + kappa * (v_a - v_b)
        x = x + v_mix * dt

    return x


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p


def _resolve_run_dir(output_root: str, run_name: str):
    """Finds the latest timestamped subdirectory for a given run name."""
    base = os.path.join(output_root, run_name)
    if not os.path.isdir(base):
        raise FileNotFoundError(f"Base run directory not found: {base}")
    subs = [d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d))]
    if not subs: return base  # Allows for runs without a timestamped subdir
    return os.path.join(base, sorted(subs)[-1])


def _latest_ckpt(ckpt_dir: str):
    """Finds the path to the latest checkpoint file."""
    last = os.path.join(ckpt_dir, "last.flax")
    if tf.io.gfile.exists(last): return last
    eps = sorted([p for p in tf.io.gfile.listdir(ckpt_dir) if p.startswith("ep") and p.endswith(".flax")])
    if not eps: raise FileNotFoundError(f"No checkpoints in {ckpt_dir}")
    return os.path.join(ckpt_dir, eps[-1])


def _load_config(run_dir: str):
    """Loads the run_meta.json config from a run directory."""
    meta_path = os.path.join(run_dir, "run_meta.json")
    if not tf.io.gfile.exists(meta_path): raise FileNotFoundError(f"Missing run_meta.json in {run_dir}")
    with tf.io.gfile.GFile(meta_path, "r") as f: return json.load(f)


def _build_sde_from_cfg(cfg):
    """Instantiates SDE functions based on a loaded config."""
    sde_type = cfg.get("sde", "VE")
    if sde_type == "VE":
        from diffusion.equations import marginal_prob_std, diffusion_coeff
        sigma_max = float(cfg.get("sigma_max",  12.0))
        mstd_fn = functools.partial(marginal_prob_std, sigma=sigma_max)
        dcoeff_fn = functools.partial(diffusion_coeff, sigma=sigma_max)
        return mstd_fn, dcoeff_fn
    else:
        # Assuming VPSDE for simplicity. Add more SDE types if needed.
        from diffusion.equations import vpsde_marginal_prob_std, vpsde_diffusion_coeff
        return vpsde_marginal_prob_std, vpsde_diffusion_coeff


def _load_model_from_run(output_root, run_name):
    """Loads a complete model (architecture, EMA params, config) from a training run directory."""
    print(f"[*] Loading model from run: {run_name}")
    run_dir = _resolve_run_dir(output_root, run_name)
    cfg = _load_config(run_dir)
    mstd_fn, dcoeff_fn = _build_sde_from_cfg(cfg)

    # Instantiate model architecture
    channels = tuple(int(c.strip()) for c in cfg.get("channels", "64,128,256,512").split(","))
    embed_dim = int(cfg.get("embed_dim", 256))
    model = ScoreNet(marginal_prob_std=mstd_fn, channels=channels, embed_dim=embed_dim)

    # Create dummy templates to deserialize the checkpoint
    img_size = int(cfg.get("img_size", 256))
    dummy_x = jnp.ones((1, img_size, img_size, 1), dtype=jnp.float32)
    dummy_t = jnp.ones((1,), dtype=jnp.float32)
    # Correct: Creates a nested template that matches the saved file
    params_template = model.init({'params': jax.random.PRNGKey(0)}, dummy_x, dummy_t)

    # The TrainState template needs an optimizer, but it won't be used.
    state_template = TrainState.create(apply_fn=model.apply, params=params_template, tx=optax.adam(1e-4))

    # Load the checkpoint
    ckpt_path = _latest_ckpt(os.path.join(run_dir, "ckpts"))
    print(f"    - Loading checkpoint: {ckpt_path}")
    with tf.io.gfile.GFile(ckpt_path, "rb") as f:
        blob = f.read()

    # The new format saved (TrainState, ema_params, ema_decay). We only need ema_params.
    try:
        target_template = (None, params_template, None)
        restored_tuple = from_bytes(target_template, blob)
        restored_ema_params = restored_tuple[1]
        print("    - Successfully loaded EMA parameters.")
        return model, restored_ema_params, cfg, (mstd_fn, dcoeff_fn)
    except Exception as e:
        print(f"    - FAILED to load EMA params. The model structure in your script might not match the"
              f" checkpoint. \n      Original error: {type(e).__name__}: {e}")
        raise

def _assert_compat(cfg_a, cfg_b):
    """Ensures two model configs are compatible for superposition."""
    keys = ["img_size", "channels", "embed_dim", "sde", "sigma_max"]
    for k in keys:
        if str(cfg_a.get(k)) != str(cfg_b.get(k)):
            raise ValueError(
                f"Incompatible runs: Mismatch on '{k}'. "
                f"Model A has '{cfg_a.get(k)}', Model B has '{cfg_b.get(k)}'"
            )
    print("[+] Models appear compatible.")


def main(args):
    # 1. Load both models and their configs
    model_a, params_a, cfg_a, (mstd_a, dcoeff_a) = _load_model_from_run(args.runs_root, args.run_a)
    model_b, params_b, cfg_b, (mstd_b, dcoeff_b) = _load_model_from_run(args.runs_root, args.run_b)

    # 2. Check for compatibility
    _assert_compat(cfg_a, cfg_b)

    # Create the output directory
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = ensure_dir(os.path.join(args.output_dir, f"compose_{args.run_a}_vs_{args.run_b}_{ts}"))
    print(f"[*] Saving samples to: {out_dir}")

    # 3. Prepare for sampling
    rng = jax.random.PRNGKey(args.seed)
    img_size = int(cfg_a.get("img_size", 256))
    sample_shape = (args.batch_size, img_size, img_size, 1)

    # The SDE functions and model apply_fn should be identical, so we can just use set 'a'
    marginal_prob_std_fn = mstd_a
    diffusion_coeff_fn = dcoeff_a

    # 4. Run the superposition sampler
    samples = ito_dynamic_estimator_solver(
        key=rng,
        model_a=model_a, params_a=params_a,
        model_b=model_b, params_b=params_b,
        marginal_prob_std_fn=marginal_prob_std_fn,
        diffusion_coeff_fn=diffusion_coeff_fn,
        shape=sample_shape,
        num_steps=args.num_steps,
        eps=args.eps
    )

    # 5. Process and save the results
    # In main()
    # 5. Process and save the results
    samples = jnp.clip(samples, 0.0, 1.0)
    samples = jnp.transpose(samples.reshape((-1, img_size, img_size, 1)), (0, 3, 1, 2))
    samples_t = torch.tensor(np.asarray(samples))

    # First, create the 3D grid from the 4D batch of samples
    grid_t = make_grid(samples_t, nrow=int(math.sqrt(args.batch_size)))

    # Save the grid using save_image
    save_image(grid_t, os.path.join(out_dir, "composed_samples_grid.png"))

    # Now, permute the 3D grid for matplotlib visualization
    grid_np = grid_t.permute(1, 2, 0).numpy()
    plt.figure(figsize=(10, 10))
    plt.imshow(grid_np, cmap='gray')
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "composed_samples_grid_mpl.png"), bbox_inches="tight", pad_inches=0)
    plt.close()

    print(f"[+] Done. Samples saved in {out_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Superposition Inference Script for Chest X-Ray Models")
    parser.add_argument("--runs_root", default="runs", help="Root directory where training runs are stored.")
    parser.add_argument("--run_a", required=True, help="Name of the first model's run directory.")
    parser.add_argument("--run_b", required=True, help="Name of the second model's run directory.")
    parser.add_argument("--output_dir", default="composition_results", help="Directory to save generated images.")
    parser.add_argument("--batch_size", type=int, default=8, help="Number of images to generate.")
    parser.add_argument("--num_steps", type=int, default=1000, help="Number of steps for the ODE sampler.")
    parser.add_argument("--eps", type=float, default=1e-5, help="Final time for the ODE sampler.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for generation.")

    cli_args = parser.parse_args()
    main(cli_args)