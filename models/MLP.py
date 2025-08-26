import flax.linen as nn
import jax.numpy as jnp

class GaussianMLP(nn.Module):
  num_hid : int
  num_out : int

  @nn.compact
  def __call__(self, t, x):
    h = jnp.hstack([t,x])
    h = nn.Dense(features=self.num_hid)(h)
    h = nn.relu(h)
    h = nn.Dense(features=self.num_hid)(h)
    h = nn.swish(h)
    h = nn.Dense(features=self.num_hid)(h)
    h = nn.swish(h)
    h = nn.Dense(features=self.num_out)(h)
    return h

class GrayscaleLatentMLP(nn.Module):
  num_hid: int
  num_out: int

  @nn.compact
  def __call__(self, t, x):
    # The model expects t to have a trailing dimension.
    if t.ndim == 1:
      t = t[:, None]
    h = jnp.concatenate([t, x], axis=1)
    h = nn.Dense(features=self.num_hid)(h)
    h = nn.swish(h)  # swish is the same as silu
    h = nn.Dense(features=self.num_hid)(h)
    h = nn.swish(h)
    h = nn.Dense(features=self.num_hid)(h)
    h = nn.swish(h)
    h = nn.Dense(features=self.num_hid)(h)
    h = nn.swish(h)
    h = nn.Dense(features=self.num_out)(h)
    return h