import jax
import jax.numpy as jnp
import os
from datetime import datetime

import optax
from flax.training.train_state import TrainState
from flax.serialization import to_bytes, from_bytes
import tensorflow as tf
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
import tqdm
import torch
import numpy as np
from diffusion.equations import marginal_prob_std_fn, diffusion_coeff_fn
from diffusion.sampling import ode_sampler
from models.unet import ScoreNet
from train.train_score_sde import get_train_step_fn
import argparse, json, hashlib, time

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--resume", choices=["auto", "never", "require"], default="auto")
    p.add_argument("--out_root", type=str, default="runs")
    p.add_argument("--run_name", type=str, default=None)
    p.add_argument("--ckpt_name", type=str, default="ckpt.flax")
    return p.parse_args()

# ------------------ Config ------------------

args = parse_args()
run_name = args.run_name or time.strftime("exp-%Y%m%d-%H%M%S")
out_dir = os.path.join(args.out_root, run_name)
os.makedirs(out_dir, exist_ok=True)
ckpt_path = os.path.join(out_dir, args.ckpt_name)
samples_dir = os.path.join(out_dir, "samples")
os.makedirs(samples_dir, exist_ok=True)

manifest = {
    "run_name": run_name,
    "resume": args.resume,
    "n_epochs": 50,
    "batch_size": 256,
    "lr": 1e-4,
}
with open(os.path.join(out_dir, "manifest.json"), "w") as f:
    json.dump(manifest, f, indent=2)


def maybe_load_ckpt(host_state, ckpt_path, policy):
    if policy == "never":
        print("[info] Resume policy=never; starting fresh.")
        return host_state
    exists = tf.io.gfile.exists(ckpt_path)
    if policy == "require" and not exists:
        raise FileNotFoundError(f"Resume policy=require but missing checkpoint: {ckpt_path}")
    if exists:
        print(f"[info] Resume policy={policy}; loading checkpoint from {ckpt_path}")
        with tf.io.gfile.GFile(ckpt_path, 'rb') as fin:
            return from_bytes(host_state, fin.read())
    print("[info] No checkpoint found. Starting fresh.")
    return host_state


n_epochs   = 50
batch_size = 256
lr         = 1e-4
rng = jax.random.PRNGKey(0)
fake_input = jnp.ones((batch_size, 28, 28, 1))
fake_time = jnp.ones(batch_size)
score_model = ScoreNet(marginal_prob_std_fn)
params = score_model.init({'params': rng}, fake_input, fake_time)

dataset = MNIST('.', train=True, transform=transforms.ToTensor(), download=True)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
tx = optax.adam(lr)
host_state = TrainState.create(apply_fn=score_model.apply, params=params, tx=tx)
# --------- Resume from checkpoint if it exists (HOST state) ---------
if tf.io.gfile.exists(ckpt_path):   # <- NEW
    print(f"[info] Loading checkpoint from {ckpt_path}")
    with tf.io.gfile.GFile(ckpt_path, 'rb') as fin:
        host_state = from_bytes(host_state, fin.read())
else:
    print("[info] No checkpoint found. Starting fresh.")
state = jax.device_put_replicated(host_state, jax.local_devices())
train_step_fn = get_train_step_fn(score_model, marginal_prob_std_fn)
tqdm_epoch = tqdm.trange(n_epochs)

assert batch_size % jax.local_device_count() == 0
data_shape = (jax.local_device_count(), -1, 28, 28, 1)

# optimizer = flax.jax_utils.replicate(optimizer)
for epoch in tqdm_epoch:
  avg_loss = 0.
  num_items = 0
  for x, y in data_loader:
    x = x.permute(0, 2, 3, 1).numpy().reshape(data_shape)
    rng, *step_rng = jax.random.split(rng, jax.local_device_count() + 1)
    step_rng = jnp.asarray(step_rng)
    loss, state = train_step_fn(step_rng, x, state)
    loss_host = jax.device_get(loss)[0]
    avg_loss += float(loss_host) * x.shape[0]
    num_items += x.shape[0]
  # Print the averaged training loss so far.
  tqdm_epoch.set_description('Average Loss: {:5f}'.format(avg_loss / num_items))
  host_state = jax.device_get(jax.tree_map(lambda x: x[0], state))
  with tf.io.gfile.GFile(ckpt_path, 'wb') as fout:
      fout.write(to_bytes(host_state))

# ------- Sampling -------
from torchvision.utils import make_grid, save_image
sample_batch_size = 64
sampler = ode_sampler

# Rebuild a state to load params
fake_input = jnp.ones((sample_batch_size, 28, 28, 1))
fake_time  = jnp.ones((sample_batch_size,))
rng = jax.random.PRNGKey(0)
params = score_model.init({'params': rng}, fake_input, fake_time)

sample_state = TrainState.create(apply_fn=score_model.apply, params=params, tx=optax.adam(lr))

# Load if checkpoint exists (it should after training)
if tf.io.gfile.exists(ckpt_path):
    print(f"[info] Loading checkpoint for sampling from {ckpt_path}")
    with tf.io.gfile.GFile(ckpt_path, 'rb') as fin:
        sample_state = from_bytes(sample_state, fin.read())
else:
    print("[warn] No checkpoint found for sampling; using freshly-initialized params.")
    
    
rng, step_rng = jax.random.split(rng)
samples = sampler(
    rng=step_rng,
    score_model=score_model,
    params=sample_state.params,
    marginal_prob_std=marginal_prob_std_fn,
    diffusion_coeff=diffusion_coeff_fn,
    batch_size=sample_batch_size
)

samples = jnp.clip(samples, 0.0, 1.0)
samples = jnp.transpose(samples.reshape((-1, 28, 28, 1)), (0, 3, 1, 2))
samples_t = torch.tensor(np.asarray(samples))  # float32 in [0,1]

# Make grid for display
nrow = int(np.sqrt(sample_batch_size))
sample_grid = make_grid(samples_t, nrow=nrow)

# ---------- Show and SAVE the image ----------
import matplotlib.pyplot as plt
plt.figure(figsize=(6, 6))
plt.axis('off')
plt.imshow(sample_grid.permute(1, 2, 0).cpu(), vmin=0., vmax=1.)
plt.show()

# Save via matplotlib (after show)   # <- NEW
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
matplotlib_out = os.path.join(samples_dir, f"grid_matplotlib_{timestamp}.png")
plt.savefig(matplotlib_out, bbox_inches="tight", pad_inches=0)

# Also save via torchvision (robust, no GUI dependency)  # <- NEW
torchvision_out = os.path.join(samples_dir, f"grid_torchvision_{timestamp}.png")
save_image(sample_grid, torchvision_out)

print(f"[info] Saved images to:\n  {matplotlib_out}\n  {torchvision_out}")
print(f"[info] Checkpoint saved to: {ckpt_path}")
