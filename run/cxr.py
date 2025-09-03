# run.py  (standalone JAX entrypoint for Chest X-ray SDE training / overfit)
# - Loads chest X-rays via ChestXrayDataset (torch DataLoader)
# - Runs your JAX SDE training loop and samplers (imports equations, sampling, unet, train_score_sde)
# - Includes overfit-one / tiny-K diagnostics like your grayscale setup
# - Applies 256x256 hyperparams by default (img_size=256, channels=64,128,256,512, lr=2e-4, etc.)

import argparse, os, math
from datetime import datetime

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
from diffusion.equations import marginal_prob_std
from diffusion.equations import diffusion_coeff
from diffusion.sampling import ode_sampler
from models.cxr_unet import ScoreNet
from train.train_score_sde import get_train_step_fn
import functools
sigma = 25.0
marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=sigma)
diffusion_coeff_fn   = functools.partial(diffusion_coeff,   sigma=sigma)
# --- Local dataset ---
from datasets.ChestXRay import ChestXrayDataset

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
    p.add_argument("--eval_mse_to_target", action="store_true")
    p.add_argument("--sample_every", type=int, default=1)

    # Model & training (D hyperparams)
    p.add_argument("--channels", type=str, default="64,128,256,512")
    p.add_argument("--embed_dim", type=int, default=256)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_per_device", type=int, default=4)
    p.add_argument("--sample_batch_size", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ckpt_path", default="ckpt_cxr.flax")
    p.add_argument("--samples_dir", default="samples_cxr")

    # Sampler steps (you can keep defaults used in equations/sampling)
    p.add_argument("--ode_steps", type=int, default=500)

    return p.parse_args()

# ---------------- Training entry ----------------

def main():
    args = parse_args()
    os.makedirs(args.samples_dir, exist_ok=True)
    rng = jax.random.PRNGKey(args.seed)

    channels = tuple(int(c.strip()) for c in args.channels.split(",") if c.strip())
    H = W = int(args.img_size)
    C = 1

    # --- Dataset & DataLoader ---
    ds = ChestXrayDataset(root_dir=args.data_root, task=args.task, split=args.split,
                          img_size=args.img_size, class_filter=args.class_filter)

    # Overfit modes:
    if args.overfit_one:
        # Repeat the first exemplar to produce many batches
        first_img, _ = ds[0]  # (1,H,W) in [-1,1]
        class RepeatOne(torch.utils.data.Dataset):
            def __init__(self, img, length=8192):
                self.img = img.clone()
                self.length = length
            def __len__(self): return self.length
            def __getitem__(self, idx): return self.img, 0
        train_ds = RepeatOne(first_img, length=max(8192, len(ds)))
        target_np = first_img.numpy()  # keep for MSE eval
    elif args.overfit_k > 0:
        # Subsample first K images
        class FirstK(torch.utils.data.Dataset):
            def __init__(self, base, k): self.base, self.k = base, int(k)
            def __len__(self): return min(self.k, len(self.base))
            def __getitem__(self, i): return self.base[i]
        train_ds = FirstK(ds, args.overfit_k)
        target_np = None
    else:
        train_ds = ds
        target_np = None

    per_dev = max(1, args.batch_per_device)
    batch_size = per_dev * n_local_devices()

    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=8, drop_last=True, pin_memory=True)

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

    # Resume
    if tf.io.gfile.exists(args.ckpt_path):
        print(f"[info] Loading checkpoint from {args.ckpt_path}")
        with tf.io.gfile.GFile(args.ckpt_path, "rb") as f:
            host_state = from_bytes(host_state, f.read())
    else:
        print("[info] No checkpoint; fresh training.")

    # Replicate to devices
    state = jax.device_put_replicated(host_state, jax.local_devices())
    train_step_fn = get_train_step_fn(score_model, marginal_prob_std_fn)

    # --- Train ---
    num_batches = 0
    for epoch in tqdm.trange(args.epochs, desc="epochs"):
        losses = []
        for x, _ in loader:
            # Torch (B,1,H,W) in [-1,1] -> JAX NHWC [0,1]
            x = x.permute(0, 2, 3, 1).contiguous()
            x = (x + 1.0) * 0.5
            x_np = x.numpy()
            x_np = to_pmap_batch(x_np, H, W, C)

            rng, *step_rng = jax.random.split(rng, n_local_devices() + 1)
            step_rng = jnp.asarray(step_rng)
            loss, state = train_step_fn(step_rng, x_np, state)
            losses.append(float(jax.device_get(loss)[0]))
            num_batches += 1

        # Save unreplicated checkpoint
        host_state = jax.device_get(jax.tree_map(lambda v: v[0], state))
        with tf.io.gfile.GFile(args.ckpt_path, "wb") as f:
            f.write(to_bytes(host_state))

        print(f"[epoch {epoch+1}] avg loss: {np.mean(losses):.6f}")

        # Periodic sampling
        if ((epoch + 1) % max(1, args.sample_every)) == 0:
            sample_and_log(
                rng_key=rng,
                score_model=score_model,
                params=host_state.params,
                H=H, W=W, img_size=args.img_size,
                batch_size=args.sample_batch_size,
                out_dir=args.samples_dir,
                epoch=epoch+1,
                target_np=target_np,
            )

    print(f"[done] ckpt: {args.ckpt_path}")

def sample_and_log(rng_key, score_model, params, H, W, img_size, batch_size, out_dir, epoch, target_np=None):
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

    if (target_np is not None):
        target01 = (target_np + 1.0) * 0.5  # [-1,1] -> [0,1]
        target01 = np.expand_dims(target01, axis=0)     # (1,1,H,W)
        diff = samples_t.numpy() - target01
        mse = np.mean(diff * diff, axis=(1, 2, 3))
        print(f"[eval] epoch {epoch} MSE-to-target: best={np.min(mse):.6f}, mean={np.mean(mse):.6f}")

    grid = make_grid_torch(samples_t)
    plt.figure(figsize=(6, 6)); plt.axis("off")
    plt.imshow(grid.permute(1, 2, 0), vmin=0., vmax=1.); plt.tight_layout(); plt.show()

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    os.makedirs(out_dir, exist_ok=True)
    out_matplotlib = os.path.join(out_dir, f"grid_ep{epoch:03d}_{ts}.png")
    out_torchvision = os.path.join(out_dir, f"grid_ep{epoch:03d}_{ts}_tv.png")
    plt.savefig(out_matplotlib, bbox_inches="tight", pad_inches=0); save_image(grid, out_torchvision)
    print(f"[saved] {out_matplotlib}\n[saved] {out_torchvision}")

if __name__ == "__main__":
    main()
