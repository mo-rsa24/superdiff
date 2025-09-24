"""
Main training script for a class-conditional Score-SDE model on Chest X-rays.

This script handles:
1. Parsing command-line arguments for configuration.
2. Setting up the appropriate SDE (VE or VPSDE).
3. Loading the ChestXrayDataset.
4. Initializing the conditional ScoreNet model, optimizer, and training state.
5. Running the main training loop with periodic sampling and checkpointing.
"""
import argparse
import functools
import json
import os
from datetime import datetime

import jax
import jax.numpy as jnp
import numpy as np
import optax
import torch
import tqdm
from flax.serialization import  to_bytes
from flax.training.train_state import TrainState
from torch.utils.data import DataLoader

# --- Local Project Modules ---
# Note: Ensure these paths are correct relative to your project structure.
from datasets.ChestXRay import ChestXrayDataset
from diffusion.equations import (diffusion_coeff, marginal_prob_std,
                                 vpsde_diffusion_coeff,
                                 vpsde_marginal_prob_std)
from diffusion.sampling import Euler_Maruyama_sampler, ode_sampler, pc_sampler
from models.cxr_unet import ScoreNet
from train.train_score_sde import get_train_step_fn


def parse_args():
    """Parses command-line arguments for the training run."""
    p = argparse.ArgumentParser("JAX Conditional SDE Chest X-ray Trainer")
    # Data
    p.add_argument("--data_root", default="../datasets/cleaned_2/TB", help="Path to the dataset.")
    p.add_argument("--split", choices=["train", "val", "test"], default="train")
    p.add_argument("--img_size", type=int, default=128)
    p.add_argument("--num_classes", type=int, default=3, help="Number of classes for conditioning.")
    # Model
    p.add_argument("--channels", type=str, default="64,128,256,512", help="U-Net channel progression.")
    p.add_argument("--embed_dim", type=int, default=256)
    # SDE/Sampler
    p.add_argument("--sde", choices=["VE", "VPSDE"], default="VE", help="Type of SDE to use.")
    p.add_argument("--sigma_max", type=float, default=25.0, help="Maximum sigma for VE SDE.")
    p.add_argument("--sampler", choices=["pc", "em", "ode"], default="pc", help="Sampler to use for generation.")
    p.add_argument("--num_steps", type=int, default=500, help="Number of steps for the sampler.")
    # Training
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch_per_device", type=int, default=4)
    p.add_argument("--sample_batch_size", type=int, default=18, help="Must be a multiple of num_classes.")
    p.add_argument("--ema_decay", type=float, default=0.999)
    # Logistics
    p.add_argument("--output_root", default="runs", help="Root directory for experiment outputs.")
    p.add_argument("--exp_name", default="cxr_conditional_sde")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sample_every", type=int, default=5, help="Frequency of sampling (in epochs).")
    p.add_argument("--ckpt_every", type=int, default=10, help="Frequency of checkpointing (in epochs).")
    return p.parse_args()

# --- Helper Functions ---
def ensure_dir(p): os.makedirs(p, exist_ok=True); return p
def tree_ema_update(ema, new, decay): return jax.tree_map(lambda e, p: e * decay + (1.0 - decay) * p, ema, new)

def main():
    args = parse_args()

    # --- 1. Setup & Configuration ---
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = ensure_dir(os.path.join(args.output_root, f"{args.exp_name}_{ts}"))
    ckpt_dir = ensure_dir(os.path.join(run_dir, "ckpts"))
    samples_dir = ensure_dir(os.path.join(run_dir, "samples"))
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    # Set up SDE functions based on user choice.
    if args.sde == "VE":
        marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=args.sigma_max)
        diffusion_coeff_fn = functools.partial(diffusion_coeff, sigma=args.sigma_max)
    else:
        marginal_prob_std_fn = vpsde_marginal_prob_std
        diffusion_coeff_fn = vpsde_diffusion_coeff

    # --- 2. Dataset and DataLoader ---
    ds = ChestXrayDataset(root_dir=args.data_root, split=args.split, img_size=args.img_size)
    if args.num_classes != len(ds.class_map):
        raise ValueError(f"--num_classes={args.num_classes} but found {len(ds.class_map)} classes in data.")

    batch_size = args.batch_per_device * jax.local_device_count()
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True, pin_memory=True)

    # --- 3. Model, Optimizer, and State Initialization ---
    rng = jax.random.PRNGKey(args.seed)
    H, W, C = args.img_size, args.img_size, 1
    channels = tuple(int(c.strip()) for c in args.channels.split(","))

    score_model = ScoreNet(
        marginal_prob_std=marginal_prob_std_fn,
        channels=channels,
        embed_dim=args.embed_dim,
        num_classes=args.num_classes,  # This enables conditioning
    )

    # Initialize model parameters with dummy data.
    rng, init_rng = jax.random.split(rng)
    fake_x = jnp.ones((1, H, W, C))
    fake_t = jnp.ones((1,))
    fake_y = jnp.zeros((1,), dtype=jnp.int32)
    params = score_model.init({'params': init_rng}, fake_x, fake_t, fake_y)['params']

    # Create TrainState and EMA parameters.
    tx = optax.adamw(learning_rate=args.lr)
    state = TrainState.create(apply_fn=score_model.apply, params=params, tx=tx)
    ema_params = state.params

    # Replicate state across all devices for pmap.
    replicated_state = jax.device_put_replicated(state, jax.local_devices())
    train_step_fn = get_train_step_fn(score_model, marginal_prob_std_fn)

    # --- 4. Training Loop ---
    global_step = 0
    for epoch in tqdm.trange(args.epochs, desc="Epochs"):
        inner_loop = tqdm.tqdm(loader, desc=f"Training Epoch {epoch + 1}", leave=False)
        for x_pt, y_pt in inner_loop:
            # Prepare batch for JAX: Move to NumPy and reshape for sharding.
            x_np = x_pt.permute(0, 2, 3, 1).numpy()
            y_np = y_pt.numpy()
            sharded_x = x_np.reshape(jax.local_device_count(), -1, H, W, C)
            sharded_y = y_np.reshape(jax.local_device_count(), -1)

            # Generate a unique random key for each device.
            rng, *step_rngs = jax.random.split(rng, jax.local_device_count() + 1)
            step_rngs = jnp.asarray(step_rngs)

            # Execute one training step.
            loss, replicated_state = train_step_fn(step_rngs, sharded_x, sharded_y, replicated_state)
            inner_loop.set_postfix(loss=f"{loss[0]:.4f}")

            # Update Exponential Moving Average of parameters.
            params_host = jax.device_get(jax.tree_map(lambda x: x[0], replicated_state.params))
            ema_params = tree_ema_update(ema_params, params_host, args.ema_decay)
            global_step += 1

        # --- 5. Periodic Sampling and Checkpointing ---
        if (epoch + 1) % args.sample_every == 0:
            sample_and_log_conditional(
                rng, score_model, ema_params, args, epoch + 1, samples_dir, ds.class_map, diffusion_coeff_fn
            )

        if (epoch + 1) % args.ckpt_every == 0:
            state_host = jax.device_get(jax.tree_map(lambda x: x[0], replicated_state))
            payload = to_bytes((state_host, ema_params))
            path = os.path.join(ckpt_dir, f"epoch_{epoch + 1}.flax")
            with open(path, "wb") as f: f.write(payload)
            print(f"\nSaved checkpoint: {path}")


def sample_and_log_conditional(rng, model, params, args, epoch, out_dir, class_map, diffusion_coeff_fn):
    from torchvision.utils import save_image
    print(f"\n--- Starting sampling for epoch {epoch} ---")

    H, W = args.img_size, args.img_size
    devices = jax.local_device_count()
    if args.sample_batch_size % args.num_classes != 0:
        print(f"[Warning] sample_batch_size {args.sample_batch_size} not easily divisible by num_classes {args.num_classes}.")
    samples_per_class = args.sample_batch_size // args.num_classes

    all_samples = []
    class_names = sorted(class_map, key=class_map.get)

    sampler_fn = {'pc': pc_sampler, 'em': Euler_Maruyama_sampler, 'ode': ode_sampler}[args.sampler]

    for i, class_name in enumerate(class_names):
        print(f"Sampling for class '{class_name}' (label {i})...")
        rng, sample_rng = jax.random.split(rng)

        # Create conditional labels, sharded for pmap.
        y_cond_flat = jnp.full((samples_per_class * devices,), i, dtype=jnp.int32)
        y_cond = y_cond_flat.reshape(devices, -1)

        samples = sampler_fn(
            sample_rng, model, params,
            marginal_prob_std=model.marginal_prob_std,
            diffusion_coeff=diffusion_coeff_fn, # BUG FIX: Use the correct diffusion_coeff
            batch_size=samples_per_class * devices,
            img_size=args.img_size,
            num_steps=args.num_steps,
            y_cond=y_cond
        )
        all_samples.append(samples)

    # Combine samples from all classes and save a grid.
    final_samples = jnp.concatenate(all_samples, axis=1)  # Combine along batch axis
    final_samples = jnp.clip(final_samples.reshape(-1, H, W, 1), 0.0, 1.0)
    final_samples_pt = torch.tensor(np.asarray(final_samples)).permute(0, 3, 1, 2)

    grid_path = os.path.join(out_dir, f"grid_epoch_{epoch:04d}.png")
    save_image(final_samples_pt, grid_path, nrow=samples_per_class)
    print(f"Saved conditional sample grid to {grid_path}")


if __name__ == "__main__":
    # To prevent TensorFlow from occupying GPU memory.
    # This is important if you're using TF for data loading but JAX for training.
    import tensorflow as tf
    tf.config.experimental.set_visible_devices([], "GPU")
    main()
