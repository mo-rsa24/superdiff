# unet.py  (robust 256×256 UNet for SDE score model)
import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Any, Tuple

class GaussianFourierProjection(nn.Module):
    embed_dim: int
    scale: float = 30.

    @nn.compact
    def __call__(self, x):
        W = self.param('W', jax.nn.initializers.normal(stddev=self.scale),
                       (self.embed_dim // 2,))
        W = jax.lax.stop_gradient(W)
        x_proj = x[:, None] * W[None, :] * 2 * jnp.pi
        return jnp.concatenate([jnp.sin(x_proj), jnp.cos(x_proj)], axis=-1)

class DenseToMap(nn.Module):
    features: int
    @nn.compact
    def __call__(self, x):
        return nn.Dense(self.features)(x)[:, None, None, :]

def _pick_gn_groups(C: int) -> int:
    g = min(32, C)
    while g > 1 and (C % g) != 0:
        g //= 2
    return max(1, g)

def _block(x, c, t_embed, act=nn.swish):
    x = nn.Conv(c, (3,3), strides=(1,1), padding='SAME', use_bias=False)(x)
    x = x + DenseToMap(c)(t_embed)
    x = nn.GroupNorm(num_groups=_pick_gn_groups(c))(x)
    return act(x)

class ScoreNet(nn.Module):
    marginal_prob_std: Any
    channels: Tuple[int, ...] = (64, 128, 256, 512)
    embed_dim: int = 256

    @nn.compact
    def __call__(self, x, t):
        act = nn.swish
        temb = act(nn.Dense(self.embed_dim)(
            GaussianFourierProjection(self.embed_dim)(t)))

        # Encoder (SAME padding, stride-2 downsamples)
        h1 = _block(x,              self.channels[0], temb, act)
        d1 = nn.Conv(self.channels[0], (3,3), strides=(2,2), padding='SAME', use_bias=False)(h1)

        h2 = _block(d1,             self.channels[1], temb, act)
        d2 = nn.Conv(self.channels[1], (3,3), strides=(2,2), padding='SAME', use_bias=False)(h2)

        h3 = _block(d2,             self.channels[2], temb, act)
        d3 = nn.Conv(self.channels[2], (3,3), strides=(2,2), padding='SAME', use_bias=False)(h3)

        h4 = _block(d3,             self.channels[3], temb, act)   # bottleneck

        # Decoder (transpose-conv upsample + skip concat; all SAME)
        u3 = nn.ConvTranspose(self.channels[2], (4,4), strides=(2,2), padding='SAME', use_bias=False)(h4)
        u3 = jnp.concatenate([u3, h3], axis=-1)
        u3 = _block(u3,            self.channels[2], temb, act)

        u2 = nn.ConvTranspose(self.channels[1], (4,4), strides=(2,2), padding='SAME', use_bias=False)(u3)
        u2 = jnp.concatenate([u2, h2], axis=-1)
        u2 = _block(u2,            self.channels[1], temb, act)

        u1 = nn.ConvTranspose(self.channels[0], (4,4), strides=(2,2), padding='SAME', use_bias=False)(u2)
        u1 = jnp.concatenate([u1, h1], axis=-1)
        u1 = _block(u1,            self.channels[0], temb, act)

        out = nn.Conv(1, (3,3), strides=(1,1), padding='SAME')(u1)

        # Score normalization
        out = out / self.marginal_prob_std(t)[:, None, None, None]
        return out
