# models/cxr_unet.py
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

class ResBlock(nn.Module):
    c: int
    embed_dim: int
    scale_skip: bool = True

    @nn.compact
    def __call__(self, x, t_embed):
        act = nn.swish
        in_ch = x.shape[-1]

        # main branch: conv → t-bias → GN → act → conv → GN
        h = nn.Conv(self.c, (3, 3), padding='SAME', use_bias=False)(x)
        h = h + DenseToMap(self.c)(t_embed)
        h = nn.GroupNorm(num_groups=_pick_gn_groups(self.c))(h)
        h = act(h)
        h = nn.Conv(self.c, (3, 3), padding='SAME', use_bias=False)(h)
        h = nn.GroupNorm(num_groups=_pick_gn_groups(self.c))(h)

        # skip branch: project if channels don't match
        if in_ch != self.c:
            x = nn.Conv(self.c, (1, 1), padding='SAME', use_bias=False, name='skip_proj')(x)

        if self.scale_skip:
            x = x * (1.0 / jnp.sqrt(2.0))

        return act(h + x)


class SelfAttention2D(nn.Module):
    num_heads: int = 4
    @nn.compact
    def __call__(self, x):
        B,H,W,C = x.shape
        h = nn.LayerNorm()(x)
        h = h.reshape((B, H*W, C))
        h = nn.SelfAttention(num_heads=self.num_heads)(h)
        h = h.reshape((B,H,W,C))
        return x + h  # residual

class ScoreNet(nn.Module):
    marginal_prob_std: Any
    channels: Tuple[int, ...] = (64, 128, 256, 512)
    embed_dim: int = 256
    attn_bottleneck: bool = True
    num_heads: int = 4

    @nn.compact
    def __call__(self, x, t):
        act = nn.swish
        temb = act(nn.Dense(self.embed_dim)(
            GaussianFourierProjection(self.embed_dim)(t)))

        # Encoder
        h1 = ResBlock(self.channels[0], self.embed_dim)(x, temb)
        d1 = nn.Conv(self.channels[0], (3,3), strides=(2,2), padding='SAME', use_bias=False)(h1)

        h2 = ResBlock(self.channels[1], self.embed_dim)(d1, temb)
        d2 = nn.Conv(self.channels[1], (3,3), strides=(2,2), padding='SAME', use_bias=False)(h2)

        h3 = ResBlock(self.channels[2], self.embed_dim)(d2, temb)
        d3 = nn.Conv(self.channels[2], (3,3), strides=(2,2), padding='SAME', use_bias=False)(h3)

        h4 = ResBlock(self.channels[3], self.embed_dim)(d3, temb)  # 32×32 bottleneck
        if self.attn_bottleneck:
            h4 = SelfAttention2D(num_heads=self.num_heads)(h4)

        # Decoder
        u3 = nn.ConvTranspose(self.channels[2], (4,4), strides=(2,2), padding='SAME', use_bias=False)(h4)
        u3 = jnp.concatenate([u3, h3], axis=-1)
        u3 = ResBlock(self.channels[2], self.embed_dim)(u3, temb)

        u2 = nn.ConvTranspose(self.channels[1], (4,4), strides=(2,2), padding='SAME', use_bias=False)(u3)
        u2 = jnp.concatenate([u2, h2], axis=-1)
        u2 = ResBlock(self.channels[1], self.embed_dim)(u2, temb)

        u1 = nn.ConvTranspose(self.channels[0], (4,4), strides=(2,2), padding='SAME', use_bias=False)(u2)
        u1 = jnp.concatenate([u1, h1], axis=-1)
        u1 = ResBlock(self.channels[0], self.embed_dim)(u1, temb)

        out = nn.Conv(1, (3,3), strides=(1,1), padding='SAME')(u1)
        # score normalization must use the *same* marginal_prob_std as loss
        out = out / self.marginal_prob_std(t)[:, None, None, None]
        return out
