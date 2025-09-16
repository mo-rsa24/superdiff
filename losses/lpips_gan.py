from typing import Optional, Dict, Tuple
import jax
import jax.numpy as jnp
from flax import linen as nn
from dataclasses import dataclass


class NLayerDiscriminator(nn.Module):
    in_channels: int = 1
    n_layers: int = 3
    use_actnorm: bool = False

    @nn.compact
    def __call__(self, x, train=True):
        ch = 64
        h = nn.Conv(ch, (4,4), strides=(2,2), padding="SAME")(x)
        h = nn.leaky_relu(h, 0.2)
        for i in range(1, self.n_layers):
            mult = min(2**i, 8)
            h = nn.Conv(ch*mult, (4,4), strides=(2,2), padding="SAME")(h)
            h = nn.GroupNorm(num_groups=32)(h)
            h = nn.leaky_relu(h, 0.2)
        h = nn.Conv(ch*8, (4,4), padding="SAME")(h)
        h = nn.GroupNorm(num_groups=32)(h)
        h = nn.leaky_relu(h, 0.2)
        h = nn.Conv(1, (4,4), padding="SAME")(h)
        return h  # (N,H',W',1) logits

# --------- GAN losses ---------

def hinge_d_loss(real_logits, fake_logits):
    return jnp.mean(jnp.maximum(0., 1. - real_logits)) + jnp.mean(jnp.maximum(0., 1. + fake_logits))

def hinge_g_loss(fake_logits):
    return -jnp.mean(fake_logits)

def vanilla_d_loss(real_logits, fake_logits):
    real = jnp.mean(nn.sigmoid_cross_entropy_with_logits(logits=real_logits, labels=jnp.ones_like(real_logits)))
    fake = jnp.mean(nn.sigmoid_cross_entropy_with_logits(logits=fake_logits, labels=jnp.zeros_like(fake_logits)))
    return real + fake

def vanilla_g_loss(fake_logits):
    return jnp.mean(nn.sigmoid_cross_entropy_with_logits(logits=fake_logits, labels=jnp.ones_like(fake_logits)))

# --------- LPIPS-like hook (optional) ---------

@dataclass
class PerceptualHook:
    """Pluggable perceptual feature distance. If fn is None, contributes 0."""
    fn: Optional[callable] = None
    weight: float = 0.0

    def __call__(self, x, y):
        if self.fn is None or self.weight <= 0.0:
            return 0.0
        return self.weight * self.fn(x, y)

# --------- Full loss (LDM-style interface) ---------

@dataclass
class LPIPSGANConfig:
    disc_start: int = 0
    kl_weight: float = 1.0
    pixel_weight: float = 1.0
    disc_factor: float = 1.0
    disc_weight: float = 1.0
    disc_num_layers: int = 3
    disc_in_channels: int = 1
    disc_loss: str = "hinge"  # or "vanilla"
    perceptual_weight: float = 0.0  # set >0 if you wire PerceptualHook

class LPIPSWithDiscriminatorJAX(nn.Module):
    cfg: LPIPSGANConfig
    perc: PerceptualHook = PerceptualHook()  # defaults to zero

    def setup(self):
        self.logvar = self.param("logvar", lambda k: jnp.array(0.0))
        self.discriminator = NLayerDiscriminator(
            in_channels=self.cfg.disc_in_channels,
            n_layers=self.cfg.disc_num_layers,
        )
        self.disc_is_hinge = (self.cfg.disc_loss == "hinge")

    def _adopt_weight(self, step):
        return jnp.where(step >= self.cfg.disc_start, self.cfg.disc_factor, 0.0)

    def _pixel_loss(self, x, y):
        # L1 + (optional) perceptual
        rec_l1 = jnp.mean(jnp.abs(x - y), axis=(1,2,3))
        p = self.perc(x, y)
        return self.cfg.pixel_weight * rec_l1 + p

    def __call__(self, *, x_in, x_rec, posterior, step, last_layer=None, train=True):
        """
        Returns:
          gen_loss, gen_logs, disc_loss, disc_logs
        Interface mirrors the PyTorch version’s split of optimizer_idx==0/1. :contentReference[oaicite:6]{index=6}
        """
        # NLL term with heteroscedastic logvar (scalar)
        rec = self._pixel_loss(x_in, x_rec)  # (N,)
        nll_per = rec / jnp.exp(self.logvar) + self.logvar
        nll = jnp.mean(nll_per)            # scalar
        kl = jnp.array(0.0) if (posterior is None) else jnp.mean(posterior.kl())
        # kl = jnp.mean(posterior.kl())      # scalar

        # generator loss (GAN)
        d_logits_fake = self.discriminator(x_rec, train=train)
        d_logits_fake = jnp.reshape(d_logits_fake, (d_logits_fake.shape[0], -1))
        d_logits_fake = jnp.mean(d_logits_fake, axis=1)  # Patch avg

        if self.disc_is_hinge:
            g_loss = hinge_g_loss(d_logits_fake)
        else:
            g_loss = vanilla_g_loss(d_logits_fake)

        d_weight = self.cfg.disc_weight  # adaptive weight can be added if you want grad-based matching

        disc_factor = self._adopt_weight(step)
        gen_loss = nll + self.cfg.kl_weight * kl + d_weight * disc_factor * g_loss

        logs_g = {
            "train/total": gen_loss,
            "train/nll": nll,
            "train/kl": kl,
            "train/g_loss": g_loss,
            "train/logvar": self.logvar,
            "train/disc_factor": disc_factor,
        }

        # discriminator pass
        d_logits_real = self.discriminator(x_in, train=train)
        d_logits_real = jnp.mean(jnp.reshape(d_logits_real, (d_logits_real.shape[0], -1)), axis=1)

        d_logits_fake_det = self.discriminator(jax.lax.stop_gradient(x_rec), train=train)
        d_logits_fake_det = jnp.mean(jnp.reshape(d_logits_fake_det, (d_logits_fake_det.shape[0], -1)), axis=1)

        if self.disc_is_hinge:
            d_loss = hinge_d_loss(d_logits_real, d_logits_fake_det)
        else:
            d_loss = vanilla_d_loss(d_logits_real, d_logits_fake_det)

        d_loss = disc_factor * d_loss
        logs_d = {"train/disc_loss": d_loss}

        return gen_loss, logs_g, d_loss, logs_d
