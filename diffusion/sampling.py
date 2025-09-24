# Samplers updated for conditional generation.
import jax
import jax.numpy as jnp
import numpy as np
import tqdm
from scipy import integrate
from typing import Optional, Callable
import flax.linen as nn


def make_pmap_score_fn(score_model: nn.Module, conditional: bool) -> Callable:
    """
    Creates a pmapped score function that handles optional conditioning.

    Args:
        score_model: The Flax model used to predict the score.
        conditional: A boolean indicating whether to expect a class label `y`.

    Returns:
        A pmapped function for score evaluation.
    """
    if conditional:
        def score_fn(params, x, t, y):
            # Assumes the model's apply method handles {'params': ...} dictionary
            return score_model.apply({'params': params}, x, t, y)
        # Pmap over the batch dimensions of x, t, and y.
        return jax.pmap(score_fn, in_axes=(None, 0, 0, 0))
    else:
        def score_fn(params, x, t):
            return score_model.apply({'params': params}, x, t)
        # Pmap over the batch dimensions of x and t.
        return jax.pmap(score_fn, in_axes=(None, 0, 0))

def pc_sampler(
    rng: jax.Array,
    score_model: nn.Module,
    params: dict,
    marginal_prob_std: Callable,
    diffusion_coeff: Callable, # <-- MODIFIED: Replaced sde_fn_type with Callable
    batch_size: int,
    img_size: int,
    num_steps: int = 500,
    snr: float = 0.16,
    eps: float = 1e-3,
    y_cond: Optional[jnp.ndarray] = None
) -> jnp.ndarray:
    """
    Predictor-Corrector sampler for conditional generation.

    Args:
        y_cond: Sharded class labels of shape `(num_devices, B // num_devices)`.
                If provided, enables conditional sampling.

    Returns:
        Generated samples.
    """
    devices = jax.local_device_count()
    is_conditional = y_cond is not None
    pmap_score_fn = make_pmap_score_fn(score_model, conditional=is_conditional)
    time_shape = (devices, batch_size // devices)
    sample_shape = time_shape + (img_size, img_size, 1)

    rng, step_rng = jax.random.split(rng)
    init_x = jax.random.normal(step_rng, sample_shape) * marginal_prob_std(1.)
    time_steps = jnp.linspace(1., eps, num_steps)
    step_size = time_steps[0] - time_steps[1]
    x = init_x

    for time_step in tqdm.tqdm(time_steps, desc="PC Sampler"):
        batch_time_step = jnp.ones(time_shape) * time_step
        args = (params, x, batch_time_step, y_cond) if is_conditional else (params, x, batch_time_step)

        # Corrector step (Langevin MCMC)
        grad = pmap_score_fn(*args)
        grad_norm = jnp.linalg.norm(grad.reshape(grad.shape[0], -1), axis=-1).mean()
        noise_norm = np.sqrt(np.prod(x.shape[2:])) # H*W*C
        langevin_step_size = 2 * (snr * noise_norm / (grad_norm + 1e-6)) ** 2
        rng, step_rng = jax.random.split(rng)
        z = jax.random.normal(step_rng, x.shape)
        x = x + langevin_step_size * grad + jnp.sqrt(2 * langevin_step_size) * z

        # Predictor step (Reverse SDE)
        g = diffusion_coeff(time_step)
        score = pmap_score_fn(*args)
        x_mean = x + (g ** 2) * score * step_size
        rng, step_rng = jax.random.split(rng)
        z = jax.random.normal(step_rng, x.shape)
        x = x_mean + jnp.sqrt(g ** 2 * step_size) * z

    return x_mean

def Euler_Maruyama_sampler(
    rng: jax.Array,
    score_model: nn.Module,
    params: dict,
    marginal_prob_std: Callable,
    diffusion_coeff: Callable, # <-- MODIFIED: Replaced sde_fn_type with Callable
    batch_size: int,
    img_size: int,
    num_steps: int = 500,
    eps: float = 1e-3,
    y_cond: Optional[jnp.ndarray] = None
) -> jnp.ndarray:
    """Euler-Maruyama sampler for conditional generation."""
    devices = jax.local_device_count()
    is_conditional = y_cond is not None
    pmap_score_fn = make_pmap_score_fn(score_model, conditional=is_conditional)
    time_shape = (devices, batch_size // devices)
    sample_shape = time_shape + (img_size, img_size, 1)

    rng, step_rng = jax.random.split(rng)
    init_x = jax.random.normal(step_rng, sample_shape) * marginal_prob_std(1.)
    time_steps = jnp.linspace(1., eps, num_steps)
    step_size = time_steps[0] - time_steps[1]
    x = init_x

    for time_step in tqdm.tqdm(time_steps, desc="EM Sampler"):
        batch_time_step = jnp.ones(time_shape) * time_step
        g = diffusion_coeff(time_step)
        args = (params, x, batch_time_step, y_cond) if is_conditional else (params, x, batch_time_step)
        score = pmap_score_fn(*args)
        mean_x = x + (g ** 2) * score * step_size
        rng, step_rng = jax.random.split(rng)
        noise = jax.random.normal(step_rng, x.shape)
        x = mean_x + jnp.sqrt(step_size) * g * noise
    return mean_x

def ode_sampler(
    rng: jax.Array,
    score_model: nn.Module,
    params: dict,
    marginal_prob_std: Callable,
    diffusion_coeff: Callable, # <-- MODIFIED: Replaced sde_fn_type with Callable
    batch_size: int,
    img_size: int,
    eps: float = 1e-3,
    y_cond: Optional[jnp.ndarray] = None,
    atol: float = 1e-5,
    rtol: float = 1e-5
) -> jnp.ndarray:
    """ODE sampler for conditional generation using SciPy's ODE solver."""
    devices = jax.local_device_count()
    is_conditional = y_cond is not None
    pmap_score_fn = make_pmap_score_fn(score_model, conditional=is_conditional)
    time_shape = (devices, batch_size // devices)
    sample_shape = time_shape + (img_size, img_size, 1)

    rng, step_rng = jax.random.split(rng)
    init_x = jax.random.normal(step_rng, sample_shape) * marginal_prob_std(1.)

    # Wrapper to interface JAX score function with SciPy's NumPy-based solver.
    def score_eval_wrapper(sample, time_steps):
        # Convert NumPy inputs from solver back to JAX arrays.
        sample = jnp.asarray(sample, dtype=jnp.float32).reshape(sample_shape)
        time_steps = jnp.asarray(time_steps).reshape(time_shape)
        args = (params, sample, time_steps, y_cond) if is_conditional else (params, sample, time_steps)
        score = pmap_score_fn(*args)
        # Convert JAX output back to NumPy for the solver.
        return np.asarray(score).reshape((-1,)).astype(np.float64)

    def ode_func(t, x):
        """The ODE function for the probability flow ODE."""
        time_steps = np.ones((batch_size,)) * t
        g = diffusion_coeff(t)
        return -0.5 * (g ** 2) * score_eval_wrapper(x, time_steps)

    # Solve the ODE.
    res = integrate.solve_ivp(
        ode_func, (1., eps), np.asarray(init_x).reshape(-1),
        rtol=rtol, atol=atol, method='RK45'
    )
    # Reshape final result back to image dimensions.
    x = jnp.asarray(res.y[:, -1]).reshape(sample_shape)
    return x

