# compose.py
# Corrected script for two-model SUPERDIFF 'AND' composition.

import argparse
import os
import math
import json
import functools
from datetime import datetime

import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf
import torch
import tqdm
from flax.serialization import from_bytes
from flax.training.train_state import TrainState
import optax

from torchvision.utils import save_image, make_grid

from models.unet import ScoreNet


# Assuming your unconditional ScoreNet and helpers are in this module



# --- Utilities ---

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p


def _load_config(run_dir: str):
    meta_path = os.path.join(run_dir, "run_meta.json")
    if tf.io.gfile.exists(meta_path):
        with tf.io.gfile.GFile(meta_path, "r") as f:
            return json.load(f)
    eps = sorted([p for p in tf.io.gfile.listdir(run_dir)])
    if not eps:
        raise FileNotFoundError(f"Missing run_meta.json in {run_dir}")
    with tf.io.gfile.GFile(os.path.join(run_dir, eps[-1], "run_meta.json"), "r") as f:
        return json.load(f)

def _latest_ckpt(ckpt_dir: str):
    path = os.path.join(ckpt_dir, "last.flax")
    if tf.io.gfile.exists(path):
        return path
    eps = sorted([p for p in tf.io.gfile.listdir(ckpt_dir) if p.startswith("ep") and p.endswith(".flax")])
    if not eps:
        raise FileNotFoundError(f"No checkpoints in {ckpt_dir}")
    return os.path.join(ckpt_dir, eps[-1])


def _load_model_from_run(run_dir):
    print(f"[*] Loading model from run: {run_dir}")
    cfg = _load_config(run_dir)
    from diffusion.equations import marginal_prob_std, diffusion_coeff
    mstd_fn = functools.partial(marginal_prob_std, sigma=float(cfg.get("sigma_max", 25.0)))
    dcoeff_fn = functools.partial(diffusion_coeff, sigma=float(cfg.get("sigma_max", 25.0)))

    model = ScoreNet(
        marginal_prob_std=mstd_fn,
        channels=tuple(int(c) for c in cfg.get("channels").split(",")),
        embed_dim=cfg.get("embed_dim"),
    )

    img_size = cfg.get("img_size")
    dummy_x = jnp.ones((1, img_size, img_size, 1))
    dummy_t = jnp.ones((1,))
    params_template = model.init({'params': jax.random.PRNGKey(0)}, dummy_x, dummy_t)['params']

    ckpt_path = _latest_ckpt(os.path.join(run_dir, "ckpts"))
    print(f"    - Loading checkpoint: {ckpt_path}")
    with tf.io.gfile.GFile(ckpt_path, "rb") as f:
        blob = f.read()

    # The trainer saves a tuple of (TrainState, ema_params, ema_decay)
    _, ema_params, _ = from_bytes((None, params_template, 0.999), blob)
    print("    - Successfully loaded EMA parameters.")
    return model, ema_params, cfg, mstd_fn, dcoeff_fn


def make_time_grid(eps, num_steps, dtype, reverse=True):
    tg = jnp.linspace(1.0, eps, num_steps, dtype=dtype)
    return tg if reverse else tg[::-1]


def score_fn(model, params, x, t_batch):
    # This function should be defined once.
    # It correctly calls the unconditional model.
    return model.apply({'params': params}, x, t_batch)


# --- SUPERDIFF 'AND' Sampler for Two Unconditional Models ---

def sampler_superdiff_and_ode(rng, model_a, pa, model_b, pb,
                              marginal_prob_std_fn, diffusion_coeff_fn,
                              shape, num_steps=1000, eps=1e-5):
    B = shape[0]
    rng, sub = jax.random.split(rng)
    x = jax.random.normal(sub, shape) * marginal_prob_std_fn(1.0)

    t_grid = make_time_grid(eps, num_steps, x.dtype, reverse=True)
    dt = t_grid[0] - t_grid[1]

    @jax.jit
    def get_vel_and_div(model, p, _x, _t):
        _s_fn = lambda _x_in: score_fn(model, p, _x_in, _t)
        eps_noise = jax.random.rademacher(jax.random.PRNGKey(0), _x.shape, dtype=_x.dtype)
        s, jvp_val = jax.jvp(_s_fn, (_x,), (eps_noise,))
        div_s = (eps_noise * jvp_val).sum(axis=(1, 2, 3))
        g2 = (diffusion_coeff_fn(_t) ** 2)[:, None, None, None]
        vel = -0.5 * g2 * s
        div_v = -0.5 * g2.squeeze() * div_s
        return vel, div_v

    for t in tqdm.tqdm(t_grid, desc="SUPERDIFF AND Sampling"):
        t_batch = jnp.full((B,), t, dtype=x.dtype)
        v_a, div_a = get_vel_and_div(model_a, pa, x, t_batch)
        v_b, div_b = get_vel_and_div(model_b, pb, x, t_batch)

        v_diff = v_b - v_a
        v_sum = v_b + v_a
        kappa_num = div_b - div_a - (v_diff * v_sum).sum(axis=(1, 2, 3))
        kappa_den = (v_diff ** 2).sum(axis=(1, 2, 3)) + 1e-6
        kappa = kappa_num / kappa_den

        v_final = v_a + kappa[:, None, None, None] * v_diff
        x = x + v_final * dt

    return jnp.clip(x, 0.0, 1.0)


# --- Main Execution ---

def main(args):
    # Load the two expert models
    run_a_dir = os.path.join("runs", args.run_a_dir)
    run_b_dir = os.path.join("runs", args.run_a_dir)
    model_a, params_a, cfg_a, mstd_fn_a, dcoeff_fn_a = _load_model_from_run(run_a_dir)
    model_b, params_b, cfg_b, _, _ = _load_model_from_run(run_b_dir)

    # Basic compatibility check
    assert cfg_a['sde'] == cfg_b['sde'] and cfg_a['sigma_max'] == cfg_b['sigma_max'], "SDE schedules must match!"
    assert cfg_a['img_size'] == cfg_b['img_size'], "Image sizes must match!"

    # Use the SDE functions from model A (since they are identical)
    marginal_prob_std_fn = mstd_fn_a
    diffusion_coeff_fn = dcoeff_fn_a

    # Create output directory
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_a_name = os.path.basename(os.path.normpath(args.run_a_dir))
    run_b_name = os.path.basename(os.path.normpath(args.run_b_dir))
    out_dir = ensure_dir(os.path.join("compositions", f"compose_{run_a_name}_AND_{run_b_name}_{ts}"))
    print(f"[+] Saving results to: {out_dir}")

    # Run sampler
    rng = jax.random.PRNGKey(args.seed)
    rng, sub_rng = jax.random.split(rng)
    sample_shape = (args.batch_size, cfg_a['img_size'], cfg_a['img_size'], 1)

    # --- THIS IS THE CORRECTED SAMPLER CALL ---
    # It now correctly uses the two loaded models and their parameters.
    samples = sampler_superdiff_and_ode(
        sub_rng, model_a, params_a, model_b, params_b,
        marginal_prob_std_fn, diffusion_coeff_fn,
        sample_shape, num_steps=args.num_steps, eps=args.eps
    )

    # Save results (assuming model output is in [0, 1] range)
    samples_pt = torch.tensor(np.asarray(samples)).permute(0, 3, 1, 2)  # NHWC -> NCHW
    save_image(samples_pt, os.path.join(out_dir, "composition_grid.png"), nrow=int(math.sqrt(args.batch_size)))

    print("[+] Composition finished successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Two-Model Superposition (AND operator)")
    parser.add_argument("--run_a_dir", type=str, required=True,
                        help="Path to the first model's run directory (e.g., Normal_Expert).")
    parser.add_argument("--run_b_dir", type=str, required=True,
                        help="Path to the second model's run directory (e.g., TB_Expert).")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_steps", type=int, default=1000)
    parser.add_argument("--eps", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=0)
    main(parser.parse_args())