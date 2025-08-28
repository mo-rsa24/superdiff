import jax
import jax.numpy as jnp
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

n_epochs =   50
batch_size =  256
lr=1e-4

rng = jax.random.PRNGKey(0)
fake_input = jnp.ones((batch_size, 28, 28, 1))
fake_time = jnp.ones(batch_size)
score_model = ScoreNet(marginal_prob_std_fn)
params = score_model.init({'params': rng}, fake_input, fake_time)

dataset = MNIST('.', train=True, transform=transforms.ToTensor(), download=True)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
tx = optax.adam(lr)
state = TrainState.create(apply_fn=score_model.apply, params=params, tx=tx)
state = jax.device_put_replicated(state, jax.local_devices())
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
  with tf.io.gfile.GFile('ckpt.flax', 'wb') as fout:
      fout.write(to_bytes(host_state))

# ------- Sampling -------
from torchvision.utils import make_grid
sample_batch_size = 64
sampler = ode_sampler

# Rebuild a state to load params
fake_input = jnp.ones((sample_batch_size, 28, 28, 1))
fake_time  = jnp.ones((sample_batch_size,))
rng = jax.random.PRNGKey(0)
params = score_model.init({'params': rng}, fake_input, fake_time)

sample_state = TrainState.create(apply_fn=score_model.apply, params=params, tx=optax.adam(lr))

with tf.io.gfile.GFile('ckpt.flax', 'rb') as fin:
    sample_state = from_bytes(sample_state, fin.read())

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

import matplotlib.pyplot as plt
sample_grid = make_grid(torch.tensor(np.asarray(samples)), nrow=int(np.sqrt(sample_batch_size)))

plt.figure(figsize=(6, 6))
plt.axis('off')
plt.imshow(sample_grid.permute(1, 2, 0).cpu(), vmin=0., vmax=1.)
plt.show()