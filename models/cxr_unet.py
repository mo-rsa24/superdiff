import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Any, Tuple, Optional

# --- Helper Modules ---

class GaussianFourierProjectionV2(nn.Module):
    """
    Fourier features with a stable, explicit parameter shape for time embedding.

    This module converts a scalar time `t` into a high-dimensional feature vector.
    It uses a fixed, non-trainable random projection matrix `W`, ensuring that the
    time embedding is consistent across training.

    Attributes:
        embed_dim: The dimensionality of the output embedding.
        scale: The standard deviation of the random Gaussian projection matrix.
    """
    embed_dim: int
    scale: float = 30.0

    # In your new models/cxr_unet.py

    @nn.compact
    def __call__(self, t: jnp.ndarray) -> jnp.ndarray:
        """
        Transforms scalar time `t` into a high-dimensional feature vector using
        the original, robust broadcasting method.

        Args:
            t: A JAX array of shape `(B,)` representing timesteps.

        Returns:
            A JAX array of shape `(B, embed_dim)` for the time embedding.
        """
        D = self.embed_dim // 2
        def _normal_init(key, shape, dtype=jnp.float32):
            return jax.random.normal(key, shape, dtype) * self.scale
        W = self.param('W_v2', _normal_init, (D,))
        W = jax.lax.stop_gradient(W)
        t_proj = t[:, None] * W[None, :] * (2.0 * jnp.pi)
        return jnp.concatenate([jnp.sin(t_proj), jnp.cos(t_proj)], axis=-1)

class DenseToMap(nn.Module):
    """Lifts a dense vector to a 4D feature map for broadcasting."""
    features: int
    @nn.compact
    def __call__(self, x):
        return nn.Dense(self.features)(x)[:, None, None, :]

def _pick_gn_groups(C: int, g: int = 32) -> int:
    """Helper to select a valid number of groups for GroupNorm."""
    while g > 1 and (C % g) != 0:
        g //= 2
    return max(1, g)

class ResBlock(nn.Module):
    """A residual block with time/class embedding injection."""
    c: int
    embed_dim: int
    scale_skip: bool = True

    @nn.compact
    def __call__(self, x, t_embed):
        act = nn.swish
        in_ch = x.shape[-1]

        h = nn.GroupNorm(num_groups=_pick_gn_groups(in_ch))(x)
        h = act(h)
        h = nn.Conv(self.c, (3, 3), padding='SAME', use_bias=False)(h)

        # Inject the combined time/label embedding.
        h += DenseToMap(self.c)(t_embed)
        h = nn.GroupNorm(num_groups=_pick_gn_groups(self.c))(h)
        h = act(h)
        h = nn.Conv(self.c, (3, 3), padding='SAME', use_bias=False)(h)

        # Skip connection.
        if in_ch != self.c:
            x = nn.Conv(self.c, (1, 1), padding='SAME', use_bias=False, name='skip_proj')(x)

        return (x + h) * (1.0 / jnp.sqrt(2.0) if self.scale_skip else 1.0)

class SelfAttention2D(nn.Module):
    """Self-attention block for 2D feature maps."""
    num_heads: int = 4
    @nn.compact
    def __call__(self, x):
        B, H, W, C = x.shape
        h = nn.GroupNorm(num_groups=_pick_gn_groups(C))(x)
        h = h.reshape((B, H * W, C))
        h = nn.SelfAttention(num_heads=self.num_heads, qkv_features=C)(h)
        h = h.reshape((B, H, W, C))
        return x + h

# --- Main Conditional ScoreNet Model ---

class ScoreNet(nn.Module):
    """
    A class-conditional U-Net model for score-based generation.

    This model predicts the score (gradient of the log-probability density)
    of a noisy image, conditioned on both the noise level (time `t`) and a
    class label (`y`).

    Attributes:
        marginal_prob_std: A function that returns the standard deviation of the
                           perturbation kernel for a given time.
        channels: A tuple of channel counts for each resolution level.
        embed_dim: The dimensionality of the time and class embeddings.
        num_classes: The number of classes for conditioning. If 0, the model is
                     unconditional.
        attn_bottleneck: If True, apply self-attention at the bottleneck.
        num_heads: Number of heads for the self-attention layer.
    """
    marginal_prob_std: Any
    channels: Tuple[int, ...] = (64, 128, 256, 512)
    embed_dim: int = 256
    num_classes: int = 0
    attn_bottleneck: bool = True
    num_heads: int = 4

    @nn.compact
    def __call__(self, x: jnp.ndarray, t: jnp.ndarray, y: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        """
        Forward pass for the ScoreNet model.

        Args:
            x: Input noisy images, shape `(B, H, W, C)`.
            t: Timesteps, shape `(B,)`.
            y: Class labels, shape `(B,)`. Required if `num_classes > 0`.

        Returns:
            The predicted score, shape `(B, H, W, C)`.
        """
        act = nn.swish

        # 1. Create the combined conditioning vector (time + class).
        # Time embedding
        temb = GaussianFourierProjectionV2(self.embed_dim)(t)
        temb = act(nn.Dense(self.embed_dim)(temb))

        # Conditional label embedding
        if self.num_classes > 0:
            if y is None:
                raise ValueError("Class label `y` must be provided for a conditional model.")
            y_embed = nn.Embed(num_embeddings=self.num_classes, features=self.embed_dim)(y)
            # Combine by adding the two embeddings.
            temb += y_embed

        # 2. U-Net Body
        # Encoder
        h1 = ResBlock(self.channels[0], self.embed_dim)(x, temb)
        d1 = nn.Conv(self.channels[0], (3,3), strides=(2,2), padding='SAME', use_bias=False)(h1)

        h2 = ResBlock(self.channels[1], self.embed_dim)(d1, temb)
        d2 = nn.Conv(self.channels[1], (3,3), strides=(2,2), padding='SAME', use_bias=False)(h2)

        h3 = ResBlock(self.channels[2], self.embed_dim)(d2, temb)
        d3 = nn.Conv(self.channels[2], (3,3), strides=(2,2), padding='SAME', use_bias=False)(h3)

        # Bottleneck
        h4 = ResBlock(self.channels[3], self.embed_dim)(d3, temb)
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

        # 3. Project to output channels and scale by marginal prob std.
        out = nn.Conv(x.shape[-1], (3,3), strides=(1,1), padding='SAME')(u1)
        out = out / self.marginal_prob_std(t)[:, None, None, None]
        return out

