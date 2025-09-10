from typing import Any, Callable, Sequence, Tuple
import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.struct import dataclass

# ---------------- Utils ----------------

@dataclass
class DiagonalGaussian:
    mean: jnp.ndarray
    logvar: jnp.ndarray  # log(σ^2)

    def sample(self, key):
        eps = jax.random.normal(key, self.mean.shape)
        return self.mean + jnp.exp(0.5 * self.logvar) * eps

    def mode(self):
        return self.mean

    def kl(self):
        # KL(q||N(0,1)) = 0.5 * sum(μ^2 + σ^2 - 1 - logσ^2) per-element
        return 0.5 * jnp.sum(
            jnp.square(self.mean) + jnp.exp(self.logvar) - 1.0 - self.logvar,
            axis=tuple(range(1, self.mean.ndim))
        )

# --------------- Building blocks ---------------

class ResBlock(nn.Module):
    ch: int
    dropout: float = 0.0

    @nn.compact
    def __call__(self, x, train=True):
        h = nn.GroupNorm(num_groups=32)(x)
        h = nn.swish(h)
        h = nn.Conv(self.ch, (3,3), padding="SAME")(h)
        h = nn.GroupNorm(num_groups=32)(h)
        h = nn.swish(h)
        if self.dropout > 0 and train:
            h = nn.Dropout(self.dropout)(h, deterministic=not train)
        h = nn.Conv(self.ch, (3,3), padding="SAME")(h)
        if x.shape[-1] != self.ch:
            x = nn.Conv(self.ch, (1,1))(x)
        return x + h

class Down(nn.Module):
    ch: int
    @nn.compact
    def __call__(self, x):
        return nn.Conv(self.ch, (3,3), strides=(2,2), padding="SAME")(x)

class Up(nn.Module):
    ch: int
    @nn.compact
    def __call__(self, x):
        B,H,W,C = x.shape
        x = jax.image.resize(x, (B, H*2, W*2, C), method="nearest")
        return nn.Conv(self.ch, (3,3), padding="SAME")(x)

# --------------- Encoder/Decoder ---------------

class Encoder(nn.Module):
    ch_mults: Sequence[int] = (64,128,256,256)
    in_ch: int = 1
    z_ch: int = 4
    num_res_blocks: int = 2
    dropout: float = 0.0
    double_z: bool = True  # like LDM: produce 2*z_ch for μ,logσ²

    @nn.compact
    def __call__(self, x, train=True):
        h = nn.Conv(self.ch_mults[0], (3,3), padding="SAME")(x)
        for i, ch in enumerate(self.ch_mults):
            for _ in range(self.num_res_blocks):
                h = ResBlock(ch, self.dropout)(h, train=train)
            if i < len(self.ch_mults)-1:
                h = Down(self.ch_mults[i+1])(h)
        h = ResBlock(self.ch_mults[-1], self.dropout)(h, train=train)
        out_ch = (2*self.z_ch) if self.double_z else self.z_ch
        h = nn.Conv(out_ch, (3,3), padding="SAME")(h)
        return h

class Decoder(nn.Module):
    ch_mults: Sequence[int] = (64,128,256,256)
    out_ch: int = 1
    z_ch: int = 4
    num_res_blocks: int = 2
    dropout: float = 0.0

    @nn.compact
    def __call__(self, z, train=True):
        h = nn.Conv(self.ch_mults[-1], (3,3), padding="SAME")(z)
        h = ResBlock(self.ch_mults[-1], self.dropout)(h, train=train)
        for i in reversed(range(len(self.ch_mults))):
            if i > 0:
                h = Up(self.ch_mults[i-1])(h)
            for _ in range(self.num_res_blocks):
                h = ResBlock(self.ch_mults[i], self.dropout)(h, train=train)
        h = nn.GroupNorm(num_groups=32)(h)
        h = nn.swish(h)
        h = nn.Conv(self.out_ch, (3,3), padding="SAME")(h)
        return h

class AutoencoderKL(nn.Module):
    enc_cfg: dict
    dec_cfg: dict
    embed_dim: int
    @nn.compact
    def __call__(self, x, *, rng=None, sample_posterior=True, train=True):
        enc_z = self.enc_cfg.get("z_ch", self.embed_dim)
        dec_z = self.dec_cfg.get("z_ch", self.embed_dim)
        enc_kwargs = {k: v for k, v in self.enc_cfg.items() if k != "z_ch"}
        dec_kwargs = {k: v for k, v in self.dec_cfg.items() if k != "z_ch"}
        enc = Encoder(z_ch=enc_z, **enc_kwargs, name="encoder")
        dec = Decoder(z_ch=dec_z, **dec_kwargs, name="decoder")
        moments = enc(x, train=train)
        mu, logvar = jnp.split(moments, 2, axis=-1)
        q = DiagonalGaussian(mu, logvar)
        if sample_posterior:
            assert rng is not None, "rng required when sampling posterior."
            z = q.sample(rng)
        else:
            z = q.mode()
        xrec = dec(z, train=train)
        return xrec, q

    # helpers for separate calls
    def encode(self, x, *, train=True):
        enc_z = self.enc_cfg.get("z_ch", self.embed_dim)
        enc_kwargs = {k: v for k, v in self.enc_cfg.items() if k != "z_ch"}
        enc = Encoder(z_ch=enc_z, **enc_kwargs, name="encoder")
        moments = enc(x, train=train)
        mu, logvar = jnp.split(moments, 2, axis=-1)
        return DiagonalGaussian(mu, logvar)

    def decode(self, z, *, train=True):
        dec_z = self.dec_cfg.get("z_ch", self.embed_dim)
        dec_kwargs = {k: v for k, v in self.dec_cfg.items() if k != "z_ch"}
        dec = Decoder(z_ch=dec_z, **dec_kwargs, name="decoder")
        return dec(z, train=train)
