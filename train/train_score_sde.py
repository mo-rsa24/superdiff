from diffusion.equations import *

def loss_fn(rng, model, params, x, marginal_prob_std, eps=1e-5):
    """The loss function for training score-based generative models.

    Args:
      model: A `flax.linen.Module` object that represents the structure of
        the score-based model.
      params: A dictionary that contains all trainable parameters.
      x: A mini-batch of training data.
      marginal_prob_std: A function that gives the standard deviation of
        the perturbation kernel.
      eps: A tolerance value for numerical stability.
    """
    rng, step_rng = jax.random.split(rng)
    random_t = jax.random.uniform(step_rng, (x.shape[0],), minval=eps, maxval=1.)
    rng, step_rng = jax.random.split(rng)
    z = jax.random.normal(step_rng, x.shape)
    std = marginal_prob_std(random_t)
    perturbed_x = x + z * std[:, None, None, None]
    score = model.apply(params, perturbed_x, random_t)
    loss = jnp.mean(jnp.sum((score * std[:, None, None, None] + z) ** 2,
                            axis=(1, 2, 3)))
    return loss


def get_train_step_fn(model, marginal_prob_std):
    """Create a one-step training function.

    Args:
      model: A `flax.linen.Module` object that represents the structure of
        the score-based model.
      marginal_prob_std: A function that gives the standard deviation of
        the perturbation kernel.
    Returns:
      A function that runs one step of training.
    """

    val_and_grad_fn = jax.value_and_grad(loss_fn, argnums=2)

    def step_fn(rng, x, state):
        params = state.params
        loss, grad = val_and_grad_fn(rng, model, params, x, marginal_prob_std)
        mean_grad = jax.lax.pmean(grad, axis_name='device')
        mean_loss = jax.lax.pmean(loss, axis_name='device')
        new_state = state.apply_gradients(grads=mean_grad)

        return mean_loss, new_state

    return jax.pmap(step_fn, axis_name='device')