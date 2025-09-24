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
from flax.serialization import from_bytes
from flax.training.train_state import TrainState
import optax
import tqdm

from torchvision.utils import save_image

# --- Import your new conditional model ---
from models.cxr_unet import ScoreNet


# --- Utilities to load model and config ---
def ensure_dir(p): os.makedirs(p, exist_ok=True); return p


def _load_config(run_dir: str):
    meta_path = os.path.join(run_dir, "config.json")
    if not tf.io.gfile.exists(meta_path):
        raise FileNotFoundError(f"Missing config.json in {run_dir}")
    with tf.io.gfile.GFile(meta_path, "r") as f: return json.load(f)


def _latest_ckpt(ckpt_dir: str):
    eps = sorted([p for p in tf.io.gfile.listdir(ckpt_dir) if p.startswith("epoch_") and p.endswith(".flax")])
    if not eps: raise FileNotFoundError(f"No checkpoints in {ckpt_dir}")
    return os.path.join(ckpt_dir, eps[-1])


def load_model_from_run(run_dir):
    print(f"[*] Loading model from run: {run_dir}")
    cfg = _load_config(run_dir)
    from diffusion.equations import marginal_prob_std, diffusion_coeff
    if cfg.get("sde") == "VE":
        mstd_fn = functools.partial(marginal_prob_std, sigma=float(cfg.get("sigma_max", 25.0)))
        dcoeff_fn = functools.partial(diffusion_coeff, sigma=float(cfg.get("sigma_max", 25.0)))
    else:  # Add VPSDE support if needed
        raise NotImplementedError("Only VE SDE is supported in this compose script.")

    model = ScoreNet(
        marginal_prob_std=mstd_fn,
        channels=tuple(int(c) for c in cfg.get("channels").split(",")),
        embed_dim=cfg.get("embed_dim"),
        num_classes=cfg.get("num_classes"),
    )

    img_size = cfg.get("img_size")
    dummy_x = jnp.ones((1, img_size, img_size, 1))
    dummy_t = jnp.ones((1,))
    dummy_y = jnp.zeros((1,), dtype=jnp.int32)

    params_template = model.init({'params': jax.random.PRNGKey(0)}, dummy_x, dummy_t, dummy_y)['params']
    state_template = TrainState.create(apply_fn=model.apply, params=params_template, tx=optax.adam(1e-4))

    ckpt_path = _latest_ckpt(os.path.join(run_dir, "ckpts"))
    print(f"    - Loading checkpoint: {ckpt_path}")
    with tf.io.gfile.GFile(ckpt_path, "rb") as f:
        blob = f.read()

    _, ema_params, _ = from_bytes((state_template, params_template, 0.999), blob)
    print("    - Successfully loaded EMA parameters.")
    return model, ema_params, cfg, mstd_fn, dcoeff_fn


# --- Core Superposition Sampler ---
def ito_superposition_sampler(
        rng, model, params, mstd_fn, dcoeff_fn, shape,
        label_base, label_a, label_b,
        num_steps=1000, eps=1e-5, guidance_scale=1.0
):
    B = shape[0]
    rng, sub = jax.random.split(rng)
    x = jax.random.normal(sub, shape) * mstd_fn(1.0)

    time_steps = jnp.linspace(1., eps, num_steps)
    dt = time_steps[0] - time_steps[1]

    # JIT the score function for performance
    @jax.jit
    def get_score(p, _x, _t, _y):
        return model.apply({'params': p}, _x, _t, _y)

    y_base = jnp.full((B,), label_base, dtype=jnp.int32)
    y_a = jnp.full((B,), label_a, dtype=jnp.int32)
    y_b = jnp.full((B,), label_b, dtype=jnp.int32)

    # Hutchinson's estimator for divergence
    @jax.jit
    def get_vel_and_div(p, _x, _t, _y):
        v_fn = lambda _x_in: -0.5 * (dcoeff_fn(_t) ** 2)[:, None, None, None] * get_score(p, _x_in, _t, _y)
        eps_noise = jax.random.rademacher(jax.random.PRNGKey(0), _x.shape, dtype=_x.dtype)
        vel, jvp_val = jax.jvp(v_fn, (_x,), (eps_noise,))
        div = (eps_noise * jvp_val).sum(axis=(1, 2, 3))
        return vel, div

    for t in tqdm.tqdm(time_steps, desc="Superposition Sampling"):
        t_batch = jnp.full((B,), t)

        # Get velocities and divergences for the two target classes
        v_a, div_a = get_vel_and_div(params, x, t_batch, y_a)
        v_b, div_b = get_vel_and_div(params, x, t_batch, y_b)

        # Get baseline velocity
        v_base = -0.5 * (dcoeff_fn(t_batch) ** 2)[:, None, None, None] * get_score(params, x, t_batch, y_base)

        # Calculate kappa (mixing coefficient for AND)
        # Simplified from paper for clarity: kappa ≈ [div(v_b) - div(v_a) - <v_b-v_a, v_b+v_a>] / ||v_b-v_a||^2
        v_diff = v_b - v_a
        v_sum = v_b + v_a

        kappa_num = div_b - div_a - (v_diff * v_sum).sum(axis=(1, 2, 3))
        kappa_den = (v_diff ** 2).sum(axis=(1, 2, 3)) + 1e-6
        kappa = kappa_num / kappa_den

        # Combine velocities using the full formula
        v_final = v_base + guidance_scale * ((v_a - v_base) + kappa[:, None, None, None] * (v_b - v_a))

        # Update step (Euler-Maruyama for the ODE)
        x = x - v_final * dt

    return jnp.clip(x, -1.0, 1.0)


def main(args):
    # Load the single conditional model
    run_dir = args.run_dir
    model, params, cfg, mstd_fn, dcoeff_fn = load_model_from_run(run_dir)

    # Create output directory
    ts = datetime.now().strftime("%Ym%d-%H%M%S")
    comp_name = f"compose_{args.label_a}_AND_{args.label_b}_base_{args.label_base}"
    out_dir = ensure_dir(os.path.join(run_dir, "compositions", f"{comp_name}_{ts}"))
    print(f"[+] Saving results to: {out_dir}")

    # Load class map from training dataset to resolve string labels to integers
    # This assumes your new dataset saves its class_map. For now, let's hardcode it based on `new_ChestXray.py`
    class_map = {'NORMAL': 0, 'PNEUMONIA': 1, 'TB': 2}  # Adjust if your order differs
    print(f"[*] Using class map: {class_map}")

    try:
        label_base_idx = class_map[args.label_base]
        label_a_idx = class_map[args.label_a]
        label_b_idx = class_map[args.label_b]
    except KeyError as e:
        print(f"[Error] Label '{e.args[0]}' not found in class map. Available: {list(class_map.keys())}")
        return

    # Run sampler
    rng = jax.random.PRNGKey(args.seed)
    shape = (args.batch_size, cfg['img_size'], cfg['img_size'], 1)

    samples = ito_superposition_sampler(
        rng, model, params, mstd_fn, dcoeff_fn, shape,
        label_base=label_base_idx, label_a=label_a_idx, label_b=label_b_idx,
        num_steps=args.num_steps, eps=args.eps, guidance_scale=args.guidance_scale
    )

    # Save results
    samples_0_1 = (np.asarray(samples) + 1.0) * 0.5
    samples_pt = torch.tensor(samples_0_1).permute(0, 3, 1, 2)
    save_image(samples_pt, os.path.join(out_dir, "composition_grid.png"), nrow=int(math.sqrt(args.batch_size)))

    print("[+] Composition finished successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Conditional Model Superposition (AND operator)")
    parser.add_argument("--run_dir", type=str, required=True,
                        help="Path to the trained conditional model's run directory.")
    parser.add_argument("--label_base", type=str, default="NORMAL", help="Name of the base class for CFG.")
    parser.add_argument("--label_a", type=str, default="PNEUMONIA", help="Name of the first class to compose.")
    parser.add_argument("--label_b", type=str, default="TB", help="Name of the second class to compose.")
    parser.add_argument("--guidance_scale", type=float, default=1.0, help="Scale for the conditional guidance.")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_steps", type=int, default=1000)
    parser.add_argument("--eps", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=0)
    main(parser.parse_args())