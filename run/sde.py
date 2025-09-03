import argparse, os
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
from torch.utils.data import DataLoader
from diffusion.equations import marginal_prob_std_fn, diffusion_coeff_fn
from diffusion.sampling import ode_sampler
from models.unet import ScoreNet
from datasets.Shapes import GrayscaleShapesDataset                # :contentReference[oaicite:7]{index=7}

# ------------------ CLI & Config ------------------
def build_config(sanity_checks: bool):
    if sanity_checks:
        return dict(
            # data
            img_size=28,
            shapes=["circle", "square", "triangle"],
            dataset_size=3_000,
            location_variation=True,
            size_variation=True,
            size_range=(0.85, 1.15),

            # model
            channels=(16, 32, 64, 128),
            embed_dim=256,

            # train
            n_epochs=20,
            batch_size=32,
            lr=2e-3,
            num_workers=4,

            # sampling
            sample_batch_size=36,   # 6x6 grid
            seed=0,

            # io
            ckpt_path="ckpt.flax",
            samples_dir="samples",
        )
    else:
        return dict(
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

def get_dataloader(cfg):
    ds = GrayscaleShapesDataset(
        shapes=cfg["shapes"],
        size=cfg["dataset_size"],
        img_size=cfg["img_size"],
        location_variation=cfg["location_variation"],
        size_variation=cfg["size_variation"],
        size_range=cfg["size_range"],
    )  # returns tensors in [-1, 1] with shape [1, H, W]  :contentReference[oaicite:8]{index=8}

    # NOTE: To match ScoreNet (NHWC), we will permute to (B,H,W,1) in the loop.
    return DataLoader(
        ds,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg["num_workers"],
        drop_last=True,
        pin_memory=True,
    )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sanity_checks", action="store_true",
                        help="Use fast hyperparams for quick convergence.")
    args = parser.parse_args()
    cfg = build_config(args.sanity_checks)

    os.makedirs(cfg["samples_dir"], exist_ok=True)
    rng = jax.random.PRNGKey(cfg["seed"])

    # ------------------ Model init ------------------
    score_model = ScoreNet(
        marginal_prob_std_fn,
        channels=cfg["channels"],
        embed_dim=cfg["embed_dim"],
    )

    fake_input = jnp.ones((cfg["batch_size"], cfg["img_size"], cfg["img_size"], 1), dtype=jnp.float32)
    fake_time  = jnp.ones((cfg["batch_size"],), dtype=jnp.float32)
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

    # Train step fn from your training utilities (unchanged API)  :contentReference[oaicite:10]{index=10}
    from train.train_score_sde import get_train_step_fn
    train_step_fn = get_train_step_fn(score_model, marginal_prob_std_fn)

    # ------------------ Data ------------------
    loader = get_dataloader(cfg)
    assert cfg["batch_size"] % jax.local_device_count() == 0, \
        f"batch_size ({cfg['batch_size']}) must be divisible by num devices ({jax.local_device_count()})"
    data_shape = (jax.local_device_count(), -1, cfg["img_size"], cfg["img_size"], 1)

    # ------------------ Training ------------------
    tqdm_epoch = tqdm.trange(cfg["n_epochs"])
    for epoch in tqdm_epoch:
        avg_loss = 0.0
        num_items = 0

        for batch in loader:
            # batch: (B, 1, H, W) in [-1,1]
            if isinstance(batch, (list, tuple)):
                x = batch[0]
            else:
                x = batch
            # NHWC for JAX
            x = x.permute(0, 2, 3, 1).contiguous()               # (B,H,W,1)
            x = (x + 1.0) * 0.5                                  # to [0,1] if your loss expects it; keep if consistent with your pipeline
            x = x.numpy().reshape(data_shape)

            rng, *step_rng = jax.random.split(rng, jax.local_device_count() + 1)
            step_rng = jnp.asarray(step_rng)
            loss, state = train_step_fn(step_rng, x, state)

            loss_host = jax.device_get(loss)[0]
            avg_loss += float(loss_host) * x.shape[0]
            num_items += x.shape[0]

        tqdm_epoch.set_description('Average Loss: {:5f}'.format(avg_loss / num_items))

        # Save host (unreplicated) checkpoint each epoch
        host_state = jax.device_get(jax.tree_map(lambda v: v[0], state))
        with tf.io.gfile.GFile(ckpt_path, 'wb') as fout:
            fout.write(to_bytes(host_state))

    # ------------------ Sampling ------------------
    from torchvision.utils import make_grid, save_image
    import matplotlib.pyplot as plt

    sample_batch_size = cfg["sample_batch_size"]
    fake_input = jnp.ones((sample_batch_size, cfg["img_size"], cfg["img_size"], 1), dtype=jnp.float32)
    fake_time  = jnp.ones((sample_batch_size,), dtype=jnp.float32)

    rng = jax.random.PRNGKey(cfg["seed"] + 1)
    params = score_model.init({'params': rng}, fake_input, fake_time)
    sample_state = TrainState.create(apply_fn=score_model.apply, params=params, tx=optax.adam(cfg["lr"]))

    if tf.io.gfile.exists(ckpt_path):
        print(f"[info] Loading checkpoint for sampling from {ckpt_path}")
        with tf.io.gfile.GFile(ckpt_path, 'rb') as fin:
            sample_state = from_bytes(sample_state, fin.read())
    else:
        print("[warn] No checkpoint found for sampling; using freshly-initialized params.")

    rng, step_rng = jax.random.split(rng)
    samples = ode_sampler(
        rng=step_rng,
        score_model=score_model,
        params=sample_state.params,
        marginal_prob_std=marginal_prob_std_fn,
        diffusion_coeff=diffusion_coeff_fn,
        batch_size=sample_batch_size,
    )

    samples = jnp.clip(samples, 0.0, 1.0)
    samples = jnp.transpose(samples.reshape((-1, cfg["img_size"], cfg["img_size"], 1)), (0, 3, 1, 2))
    samples_t = torch.tensor(np.asarray(samples))  # float32 in [0,1]

    # Make grid and save
    nrow = int(np.sqrt(sample_batch_size))
    grid = make_grid(samples_t, nrow=nrow)

    plt.figure(figsize=(6, 6))
    plt.axis('off')
    plt.imshow(grid.permute(1, 2, 0).cpu(), vmin=0., vmax=1.)
    plt.show()

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_matplotlib = os.path.join(cfg["samples_dir"], f"grid_matplotlib_{timestamp}.png")
    out_torchvision = os.path.join(cfg["samples_dir"], f"grid_torchvision_{timestamp}.png")
    plt.savefig(out_matplotlib, bbox_inches="tight", pad_inches=0)
    save_image(grid, out_torchvision)

    print(f"[info] Saved images to:\n  {out_matplotlib}\n  {out_torchvision}")
    print(f"[info] Checkpoint saved to: {cfg['ckpt_path']}")

if __name__ == "__main__":
    main()
