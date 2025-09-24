from functools import partial
import jax
import jax.numpy as jnp
from flax.training.train_state import TrainState
from typing import Callable, Tuple
from flax import linen as nn

# This import assumes your SDE definitions are in a 'diffusion' directory.
from diffusion.equations import marginal_prob_std_fn

def loss_fn(
    rng: jax.Array,
    model: nn.Module,
    params: dict,
    x: jnp.ndarray,
    y: jnp.ndarray,
    marginal_prob_std: marginal_prob_std_fn,
    eps: float = 1e-5
) -> jnp.ndarray:
    """
    The loss function for training class-conditional score-based models.

    Args:
        rng: A JAX random key.
        model: The score-based model (e.g., ScoreNet).
        params: The parameters of the model.
        x: A batch of clean images, shape `(B, H, W, C)`.
        y: A batch of corresponding class labels, shape `(B,)`.
        marginal_prob_std: A function that gives the standard deviation of the
                           perturbation kernel.
        eps: A small value to prevent sampling `t=0`.

    Returns:
        The mean loss over the batch.
    """
    # 1. Sample a random time t for each image in the batch.
    rng, step_rng = jax.random.split(rng)
    random_t = jax.random.uniform(step_rng, (x.shape[0],), minval=eps, maxval=1.)

    # 2. Sample noise and create the perturbed image x_t.
    rng, step_rng = jax.random.split(rng)
    z = jax.random.normal(step_rng, x.shape)
    std = marginal_prob_std(random_t)
    perturbed_x = x + z * std[:, None, None, None]

    # 3. Get the model's score prediction, conditioned on x_t, t, and y.
    score = model.apply({'params': params}, perturbed_x, random_t, y)

    # 4. Calculate the score-matching loss.
    loss = jnp.mean(jnp.sum((score * std[:, None, None, None] + z) ** 2,
                            axis=(1, 2, 3)))
    return loss

def get_train_step_fn(
    model: nn.Module,
    marginal_prob_std: marginal_prob_std_fn
) -> Callable[[jax.random.KeyArray, jnp.ndarray, jnp.ndarray, TrainState], Tuple[jnp.ndarray, TrainState]]:
    """
    Creates a pmapped, one-step training function for the conditional model.

    Args:
        model: The score-based model (e.g., ScoreNet).
        marginal_prob_std: A function that gives the standard deviation of the
                           perturbation kernel.

    Returns:
        A function that takes a PRNG key, a sharded batch of images and labels,
        and the training state, and returns the loss and updated state.
    """
    # Create a function that computes both the loss and its gradient.
    val_and_grad_fn = jax.value_and_grad(loss_fn, argnums=2)

    def step_fn(
        rng: jax.random.KeyArray,
        x: jnp.ndarray,
        y: jnp.ndarray,
        state: TrainState
    ) -> Tuple[jnp.ndarray, TrainState]:
        """A single training step."""
        params = state.params
        loss, grad = val_and_grad_fn(rng, model, params, x, y, marginal_prob_std)

        # Average gradients and loss across all devices.
        mean_grad = jax.lax.pmean(grad, axis_name='device')
        mean_loss = jax.lax.pmean(loss, axis_name='device')

        # Update model parameters.
        new_state = state.apply_gradients(grads=mean_grad)

        return mean_loss, new_state

    # pmap the step function for parallel execution on multiple devices.
    # Note the in_axes: rng, x, and y are sharded; state is replicated.
    return jax.pmap(step_fn, in_axes=(0, 0, 0, None), axis_name='device')
