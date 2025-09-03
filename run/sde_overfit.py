#!/usr/bin/env python3
"""
Single-example / tiny-dataset overfitting driver for your SDE diffusion pipeline on grayscale shapes.

Goals
-----
• Prove the pipeline can learn a trivial task fast (debugging aid)
• Options to overfit ONE exemplar or a tiny K-sample dataset
• Pmap-safe batching (tiles to multiple of local_device_count)
• Periodic sampling + optional MSE-to-target for numeric convergence

Assumptions
-----------
• Your project defines:
    - `diffusion.equations`: marginal_prob_std_fn, diffusion_coeff_fn
    - `diffusion.sampling`: ode_sampler
    - `unet.ScoreNet` (UNet score model)
    - `Shapes.GrayscaleShapesDataset` (returns (tensor[-1,1], label))
• `train_score_sde.py` exposes `get_train_step_fn(model, marginal_prob_std_fn)`

Usage examples
--------------
# fastest single-example overfit (deterministic exemplar)
python sde_overfit.py --sanity_checks \
  --overfit_one --fixed_shape circle --disable_variations \
  --eval_mse_to_target --sample_every 1 --auto_per_device 8

# tiny dataset (first K synthetic samples)
python sde_overfit.py --sanity_checks \
  --overfit_k 10 --disable_variations \
  --sample_every 1 --auto_per_device 8
"""

import argparse, os, math
from datetime import datetime

import jax
import jax.numpy as jnp
import numpy as np
import optax
import tensorflow as tf
import tqdm
import torch

from flax.training.train_state import TrainState
from flax.serialization import to_bytes, from_bytes
from torch.utils.data import DataLoader, Dataset

# ------------------ Project imports (robust) ------------------
try:
    from diffusion.equations import marginal_prob_std_fn, diffusion_coeff_fn
    from diffusion.sampling import ode_sampler
except ImportError as e:
    raise ImportError(
        "Could not import diffusion.* modules. Ensure PYTHONPATH includes your project's root "
        "so that 'diffusion' is importable (e.g., export PYTHONPATH=\"$PWD:$PYTHONPATH\").\n"
        f"Original error: {e}"
    )

try:
    from models.unet import ScoreNet
except Exception:
    try:
        from models.unet import ScoreNet  # alternative layout
    except Exception as e:
        raise ImportError("Could not import ScoreNet from unet/models.unet") from e

try:
    from datasets.Shapes import GrayscaleShapesDataset
except Exception:
    try:
        from datasets.Shapes import GrayscaleShapesDataset  # alternative layout
    except Exception as e:
        raise ImportError("Could not import GrayscaleShapesDataset from Shapes/datasets.Shapes") from e

try:
    from train.train_score_sde import get_train_step_fn
except Exception as e:
    raise ImportError("train_score_sde.py must define get_train_step_fn(model, marginal_prob_std_fn)") from e

# ------------------ Utilities ------------------

def n_local_devices():
    return jax.local_device_count()


def round_up_to_multiple(n, k):
    return ((n + k - 1) // k) * k


def tile_to_multiple(x_np, multiple):
    """Tile along batch dimension so len%multiple==0, then return new array."""
    B = x_np.shape[0]
    if B % multiple == 0:
        return x_np
    pad = multiple - (B % multiple)
    reps = (pad + B - 1) // B + 1  # enough to cover "pad" with tiling
    tiled = np.concatenate([x_np] * reps, axis=0)
    return tiled[:B + pad]


def to_pmap_batch(x_np, H, W, C):
    """Ensure batch is a multiple of device_count, then reshape to (D, perD, H, W, C)."""
    devices = n_local_devices()
    x_np = tile_to_multiple(x_np, devices)
    return x_np.reshape(devices, -1, H, W, C)


def make_grid_torch(imgs_tensor, nrow=None):
    """imgs_tensor: (N,1,H,W) or (N,3,H,W) in [0,1]"""
    from torchvision.utils import make_grid
    N = imgs_tensor.shape[0]
    if nrow is None:
        nrow = int(math.sqrt(N))
    return make_grid(imgs_tensor, nrow=nrow)


# ------------------ Debug datasets ------------------

class RepeatSingleExample(Dataset):
    """Wrap a single image tensor and repeat it to create an arbitrarily large dataset.
    Expects tensor scaled to [-1,1] with shape (1,H,W)."""

    def __init__(self, image_tensor, length):
        self.image = image_tensor.detach().clone()
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # return (image, dummy_label) compatible with the rest of the pipeline
        return self.image, torch.tensor(0)


# ------------------ CLI & Config ------------------

def build_config(args):
    # Base config toggled by --sanity_checks
    if args.sanity_checks:
        cfg = dict(
            img_size=28,
            shapes=["circle", "square", "triangle"],
            dataset_size=3000,
            location_variation=True,
            size_variation=True,
            size_range=(0.85, 1.15),
            channels=(16, 32, 64, 128),
            embed_dim=256,
            n_epochs=20,
            batch_size=32,
            lr=2e-3,
            num_workers=4,
            sample_batch_size=36,
            seed=0,
            ckpt_path="ckpt.flax",
            samples_dir="samples",
        )
    else:
        cfg = dict(
            img_size=64,
            shapes=["circle", "square", "triangle"],
            dataset_size=50_000,
            location_variation=True,
            size_variation=True,
            size_range=(0.8, 1.2),
            channels=(32, 64, 128, 256),
            embed_dim=256,
            n_epochs=200,
            batch_size=128,
            lr=1e-4,
            num_workers=8,
            sample_batch_size=64,
            seed=0,
            ckpt_path="ckpt.flax",
            samples_dir="samples",
        )

    # Debug/overfit knobs
    cfg["overfit_one"] = args.overfit_one
    cfg["overfit_k"] = args.overfit_k
    cfg["fixed_shape"] = args.fixed_shape
    cfg["disable_variations"] = args.disable_variations
    cfg["auto_per_device"] = args.auto_per_device
    cfg["sample_every"] = args.sample_every
    cfg["eval_mse_to_target"] = args.eval_mse_to_target

    return cfg


def get_dataloader(cfg):
    """Return (loader, target_np_or_None). If overfitting one example, also return that target image for eval."""
    if cfg["overfit_one"]:
        # Deterministic single exemplar (no jitter) from a fixed shape
        base_ds = GrayscaleShapesDataset(
            shapes=[cfg["fixed_shape"]],
            size=1,
            img_size=cfg["img_size"],
            location_variation=not cfg["disable_variations"] and False,
            size_variation=not cfg["disable_variations"] and False,
            size_range=cfg["size_range"],
        )
        base_img, _ = base_ds[0]  # (1,H,W) in [-1,1]
        # Repeat so we have many batches per epoch
        desired_len = max(8 * cfg["batch_size"], cfg["dataset_size"])
        ds = RepeatSingleExample(base_img, length=desired_len)
        target = base_img  # keep for MSE/PSNR eval
    else:
        # Tiny-dataset option (first K items) for quick checks
        if cfg["overfit_k"] > 0:
            small_size = cfg["overfit_k"]
        else:
            small_size = cfg["dataset_size"]

        ds = GrayscaleShapesDataset(
            shapes=cfg["shapes"],
            size=small_size,
            img_size=cfg["img_size"],
            location_variation=cfg["location_variation"] and not cfg["disable_variations"],
            size_variation=cfg["size_variation"] and not cfg["disable_variations"],
            size_range=cfg["size_range"],
        )
        target = None

    loader = DataLoader(
        ds,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg["num_workers"],
        drop_last=True,
        pin_memory=True,
    )
    return loader, (target.numpy() if target is not None else None)


# ------------------ Main ------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sanity_checks", action="store_true", help="Use fast hyperparams for quick convergence.")
    # Overfit/tiny-dataset debug knobs
    parser.add_argument("--overfit_one", action="store_true", help="Repeat a single exemplar and learn it perfectly.")
    parser.add_argument("--fixed_shape", type=str, default="circle", choices=["circle", "square", "triangle"],
                        help="Which shape to use for --overfit_one.")
    parser.add_argument("--overfit_k", type=int, default=0, help="If >0, train only on the first K samples (tiny dataset).")
    parser.add_argument("--disable_variations", action="store_true", help="Turn off size/location jitter for determinism.")
    parser.add_argument("--auto_per_device", type=int, default=None,
                        help="If set, batch_size = this * jax.local_device_count() (useful under pmap).")
    parser.add_argument("--sample_every", type=int, default=1, help="Epoch cadence for sampling grids during training.")
    parser.add_argument("--eval_mse_to_target", action="store_true",
                        help="When overfitting one sample, compute MSE of samples to target.")
    args = parser.parse_args()

    cfg = build_config(args)
    os.makedirs(cfg["samples_dir"], exist_ok=True)
    rng = jax.random.PRNGKey(cfg["seed"])

    # Auto batch size aligned to devices
    if cfg["auto_per_device"] is not None:
        cfg["batch_size"] = max(1, cfg["auto_per_device"]) * n_local_devices()

    # ------------------ Model init ------------------
    score_model = ScoreNet(
        marginal_prob_std_fn,
        channels=cfg["channels"],
        embed_dim=cfg["embed_dim"],
    )

    fake_input = jnp.ones((cfg["batch_size"], cfg["img_size"], cfg["img_size"], 1), dtype=jnp.float32)
    fake_time = jnp.ones((cfg["batch_size"],), dtype=jnp.float32)
    params = score_model.init({'params': rng}, fake_input, fake_time)

    tx = optax.adam(cfg["lr"])
    host_state = TrainState.create(apply_fn=score_model.apply, params=params, tx=tx)

    ckpt_path = cfg["ckpt_path"]

    # --------- Resume from checkpoint if it exists ---------
    if tf.io.gfile.exists(ckpt_path):
        print(f"[info] Loading checkpoint from {ckpt_path}")
        with tf.io.gfile.GFile(ckpt_path, 'rb') as fin:
            host_state = from_bytes(host_state, fin.read())
    else:
        print("[info] No checkpoint found. Starting fresh.")

    # Replicate to devices
    state = jax.device_put_replicated(host_state, jax.local_devices())

    # Train step fn from your training utilities
    train_step_fn = get_train_step_fn(score_model, marginal_prob_std_fn)

    # ------------------ Data ------------------
    loader, target_np = get_dataloader(cfg)
    H = cfg["img_size"]
    W = cfg["img_size"]
    C = 1

    # sanity: batch must map evenly to devices for pmap
    if cfg["batch_size"] % n_local_devices() != 0:
        print(f"[warn] batch_size ({cfg['batch_size']}) not divisible by local devices ({n_local_devices()}); "
              f"consider --auto_per_device. We'll tile few extra items when needed.")

    # ------------------ Training ------------------
    num_batches_seen = 0
    tqdm_epoch = tqdm.trange(cfg["n_epochs"])
    for epoch in tqdm_epoch:
        epoch_losses = []
        for batch in loader:
            if isinstance(batch, (list, tuple)):
                x = batch[0]  # (B,1,H,W) in [-1,1]
            else:
                x = batch

            # NHWC for JAX, to [0,1]
            x = x.permute(0, 2, 3, 1).contiguous()  # (B,H,W,1)
            x = (x + 1.0) * 0.5
            x_np = x.numpy()
            x_np = to_pmap_batch(x_np, H, W, C)     # (D, perD, H, W, 1)

            rng, *step_rng = jax.random.split(rng, n_local_devices() + 1)
            step_rng = jnp.asarray(step_rng)
            loss, state = train_step_fn(step_rng, x_np, state)
            loss_host = float(jax.device_get(loss)[0])
            epoch_losses.append(loss_host)
            num_batches_seen += 1

        # Host (unreplicated) checkpoint each epoch
        host_state = jax.device_get(jax.tree_map(lambda v: v[0], state))
        with tf.io.gfile.GFile(ckpt_path, 'wb') as fout:
            fout.write(to_bytes(host_state))

        avg_loss = float(np.mean(epoch_losses)) if len(epoch_losses) else float('nan')
        tqdm_epoch.set_description(f'AvgLoss {avg_loss:.6f} | Batches {num_batches_seen}')

        # Periodic sampling & eval
        if ((epoch + 1) % max(1, cfg["sample_every"])) == 0:
            # Use the just-saved/unreplicated params
            sample_params = host_state.params
            eval_and_save_samples(
                rng_key=rng,
                score_model=score_model,
                params=sample_params,
                cfg=cfg,
                epoch=epoch + 1,
                target_np=target_np
            )

    print(f"[info] Checkpoint saved to: {cfg['ckpt_path']}")


def eval_and_save_samples(rng_key, score_model, params, cfg, epoch, target_np=None):
    from torchvision.utils import save_image
    import matplotlib.pyplot as plt

    sample_batch_size = cfg["sample_batch_size"]
    rng_key, step_rng = jax.random.split(rng_key)

    samples = ode_sampler(
        rng=step_rng,
        score_model=score_model,
        params=params,
        marginal_prob_std=marginal_prob_std_fn,
        diffusion_coeff=diffusion_coeff_fn,
        batch_size=sample_batch_size,
    )
    samples = jnp.clip(samples, 0.0, 1.0)
    samples = jnp.transpose(samples.reshape((-1, cfg["img_size"], cfg["img_size"], 1)), (0, 3, 1, 2))
    samples_t = torch.tensor(np.asarray(samples))  # float32 in [0,1]

    # optional eval to the training exemplar
    if cfg["eval_mse_to_target"] and (target_np is not None):
        target01 = (target_np + 1.0) * 0.5  # [-1,1] -> [0,1]
        target01 = np.expand_dims(target01, axis=0)  # (1,1,H,W)
        # compute per-sample MSE to target
        diff = samples_t.numpy() - target01
        mse = np.mean(diff * diff, axis=(1, 2, 3))
        best, mean = float(np.min(mse)), float(np.mean(mse))
        print(f"[eval] epoch {epoch} MSE-to-target: best={best:.6f}, mean={mean:.6f}")

    grid = make_grid_torch(samples_t)
    # Show and save
    plt.figure(figsize=(6, 6))
    plt.axis('off')
    plt.imshow(grid.permute(1, 2, 0), vmin=0., vmax=1.)
    plt.tight_layout()
    plt.show()

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    os.makedirs(cfg["samples_dir"], exist_ok=True)
    out_matplotlib = os.path.join(cfg["samples_dir"], f"grid_ep{epoch:03d}_{timestamp}.png")
    out_torchvision = os.path.join(cfg["samples_dir"], f"grid_ep{epoch:03d}_{timestamp}_tv.png")
    plt.savefig(out_matplotlib, bbox_inches="tight", pad_inches=0)
    save_image(grid, out_torchvision)

    print(f"[info] Saved images to:\n  {out_matplotlib}\n  {out_torchvision}")


if __name__ == "__main__":
    main()
