from typing import Tuple
import jax.numpy as jnp
import numpy as np
from jax import *
from tqdm import trange
import tqdm
import math
from diffusion.equations import diffusion, drift, score_function_hutchinson_estimator, get_kappa, dlog_alphadt, beta, \
    dlogqdt

def _sum_except_batch(x):
    axes = tuple(range(1, x.ndim))
    return jnp.sum(x, axis=axes, keepdims=True)


def _broadcast_time(t_scalar, x):
    """Make t broadcast like x: (N,1[,1,1...])"""
    N = x.shape[0]
    extra_ones = (1,) * (x.ndim - 1)
    return jnp.ones((N,) + extra_ones, dtype=x.dtype) * t_scalar

def reverse_sde(state, sample_data, key, dt:float = 1e-2, xi: float = 1.0, t: float = 1.0):
  """
   The trajectory is progressive. Meaning that starting at t=0, we sample noise once.
   Perform operations using the previous values.

   You evolve the same 512 particles across time, recording their states along the pat
   xi is pronounced "Kai"
  """
  datapoints, coordinates = sample_data.shape
  num_timesteps = int(t/dt)+1 # time index i = 0, 1, .... n (include start and 100 steps)
  t = t * jnp.ones((datapoints, 1)) # Broadcasted time vector per particle (starts at 1)
  key, subkey = random.split(key, num=2)
  trajectory_field = jnp.zeros((datapoints, num_timesteps, coordinates)) # Storage for whole trajectory
  pure_noise = random.normal(subkey, shape=(datapoints, coordinates))
  trajectory_field = trajectory_field.at[:, 0, :].set(pure_noise)
  trajectory = euler_murayama(key, t, state, trajectory_field, xi=xi, dt=dt, num_timesteps=num_timesteps, shape=sample_data.shape)
  return trajectory


def euler_murayama(key, t, state, trajectory, xi: float = 1.0, dt: float = 1e-2, num_timesteps: int=100,  shape: Tuple[int,int] = (512, 2)):
  # diffusion at timestep i
  for timestep in trange(num_timesteps-1):
    key, subkey = random.split(key, 2)
    epsilon = random.normal(subkey, shape)
    diffusion_term = diffusion(dt, state, t, trajectory[:, timestep, :], xi)
    drift_term  = drift(t, epsilon, dt)
    dx = diffusion_term + drift_term
    trajectory = trajectory.at[:, timestep+1, :].set(trajectory[:, timestep, :] + dx) # Take a step
    t += -dt
  return trajectory

def compose_and_estimate_log_likelihood_along_superposed_trajectory(state_a, state_b, key, dt:float = 1e-2, t: float = 1.0, shape: Tuple[int, int] = (512, 2)):
    assert len(shape) >= 2, "shape must be (N, ...)"
    datapoints = shape[0]
    data_dims = shape[1:]
    num_coords = int(np.prod(data_dims))  # pass to dlogqdt
    num_timesteps = int(t / dt) + 1
    t_array = _broadcast_time(jnp.asarray(t, dtype=jnp.float32), jnp.zeros(shape, dtype=jnp.float32))

    key, subkey = random.split(key, num=2)
    trajectory = jnp.zeros((datapoints, num_timesteps) + data_dims, dtype=jnp.float32)
    pure_noise = random.normal(subkey, shape=(datapoints,) + data_dims)
    trajectory = trajectory.at[:, 0, ...].set(pure_noise)
    trajectory, lla, llb = ito_dynamic_estimator_solver(
        key, t_array, state_a, state_b, trajectory,
        dt=dt, num_timesteps=num_timesteps, data_dims=data_dims, num_coords=num_coords
    )
    return trajectory, lla, llb

def ito_dynamic_estimator_solver(
        key,
        t,
        state_a,
        state_b,
        trajectory,
        dt: float = 1e-2,
        num_timesteps: int = 100,
        data_dims: Tuple[int, ...] = (2,),
        num_coords: int = 2,
    ):
    """
    Integrate the reverse probability-flow ODE with the **superposed** drift:
      x' = dlog_alphadt(t) * x - beta(t) * ( s_b + κ (s_a - s_b) )
    and accumulate log-likelihoods for model A and B via instantaneous CoV.

    Args:
      t            : (N, 1[,1,1...]) broadcastable time tensor
      trajectory   : (N, T, *data_dims)
      data_dims    : tuple of sample dimensions (e.g., (H,W,1))
      num_coords   : H*W*C (or 2 for vectors)

    Returns:
      trajectory   : updated in-place
      loglik_a/b   : (N, T) arrays
    """
    N = trajectory.shape[0]
    log_likelihood_model_a = np.zeros((N, num_timesteps), dtype=np.float32)
    log_likelihood_model_b = np.zeros((N, num_timesteps), dtype=np.float32)

    for timestep in trange(num_timesteps - 1, desc="compose-ito"):
        x_t = trajectory[:, timestep, ...]
        key, subkey = random.split(key, 2)

        # score + divergence for both models at (t, x_t)
        score_a, div_a = score_function_hutchinson_estimator(subkey, t, state_a, x_t)
        key, subkey = random.split(key, 2)
        score_b, div_b = score_function_hutchinson_estimator(subkey, t, state_b, x_t)

        # κ(t, x_t) (shape (N,1[,1,1...]) broadcastable over x)
        kappa = get_kappa(t, (div_a, div_b), (score_a, score_b))

        # reverse PF ODE drift with superposition (no noise)
        reverse_drift_ode = dlog_alphadt(t) * x_t - beta(t) * (score_b + kappa * (score_a - score_b))

        # one deterministic step backward in time
        x_next = x_t - dt * reverse_drift_ode
        trajectory = trajectory.at[:, timestep + 1, ...].set(x_next)

        # per-model log-likelihood increments (Eq.10; pass num_coords)
        lla = dlogqdt(t, x_t, score_a, div_a, reverse_drift_ode, ndim=num_coords).reshape(N)
        llb = dlogqdt(t, x_t, score_b, div_b, reverse_drift_ode, ndim=num_coords).reshape(N)
        log_likelihood_model_a[:, timestep + 1] = log_likelihood_model_a[:, timestep] - float(dt) * np.asarray(lla)
        log_likelihood_model_b[:, timestep + 1] = log_likelihood_model_b[:, timestep] - float(dt) * np.asarray(llb)

        # step time backward
        t = t - dt

    return trajectory, log_likelihood_model_a, log_likelihood_model_b



num_steps = 500
#
# def score_fn(score_model, params, x, t):
#     return score_model.apply(params, x, t)


# pmap_score_fn = jax.pmap(score_fn, in_axes=(None, None, 0, 0))

def make_pmap_score_fn(score_model):
    def score_fn(params, x, t):
        return score_model.apply(params, x, t)
    return jax.pmap(score_fn, in_axes=(None, 0, 0))

def Euler_Maruyama_sampler(rng, score_model, params, marginal_prob_std, diffusion_coeff,
                           batch_size=64, num_steps=num_steps, eps=1e-3, img_size=28):
    devices = jax.local_device_count()
    if batch_size % devices != 0:
        raise ValueError(
            f"sample_batch_size ({batch_size}) must be divisible by local_device_count ({devices}). "
            "Choose a multiple to avoid degenerate sampling.")
    pmap_score_fn = make_pmap_score_fn(score_model)
    time_shape   = (devices, batch_size // devices)
    sample_shape = time_shape + (img_size, img_size, 1)
    rng, step_rng = jax.random.split(rng)
    init_x = jax.random.normal(step_rng, sample_shape) * marginal_prob_std(1.)
    time_steps = jnp.linspace(1., eps, num_steps)
    step_size = time_steps[0] - time_steps[1]
    x = init_x
    for time_step in tqdm.tqdm(time_steps):
        batch_time_step = jnp.ones(time_shape) * time_step
        g = diffusion_coeff(time_step)
        mean_x = x + (g ** 2) * pmap_score_fn(params, x, batch_time_step) * step_size
        rng, step_rng = jax.random.split(rng)
        x = mean_x + jnp.sqrt(step_size) * g * jax.random.normal(step_rng, x.shape)
    return mean_x



# @title Define the Predictor-Corrector sampler (double click to expand or collapse)

signal_to_noise_ratio = 0.16  # @param {'type':'number'}

## The number of sampling steps.
num_steps = 500  # @param {'type':'integer'}


def pc_sampler(rng,
               score_model,
               params,
               marginal_prob_std,
               diffusion_coeff,
               batch_size=64,
               img_size=28,
               num_steps=num_steps,
               snr=signal_to_noise_ratio,
               eps=1e-3):
    """Generate samples from score-based models with Predictor-Corrector method.

    Args:
      rng: A JAX random state.
      score_model: A `flax.linen.Module` that represents the
        architecture of the score-based model.
      params: A dictionary that contains the parameters of the score-based model.
      marginal_prob_std: A function that gives the standard deviation
        of the perturbation kernel.
      diffusion_coeff: A function that gives the diffusion coefficient
        of the SDE.
      batch_size: The number of samplers to generate by calling this function once.
      num_steps: The number of sampling steps.
        Equivalent to the number of discretized time steps.
      eps: The smallest time step for numerical stability.

    Returns:
      Samples.
    """
    devices = jax.local_device_count()
    pmap_score_fn = make_pmap_score_fn(score_model)
    time_shape = (devices, batch_size // devices)
    if batch_size % devices != 0:
        raise ValueError(
                    f"sample_batch_size ({batch_size}) must be divisible by local_device_count ({devices}). "
            "Choose a multiple to avoid degenerate sampling.")

    sample_shape = time_shape + (img_size, img_size, 1)
    rng, step_rng = jax.random.split(rng)
    init_x = jax.random.normal(step_rng, sample_shape) * marginal_prob_std(1.)
    time_steps = jnp.linspace(1., eps, num_steps)
    step_size = time_steps[0] - time_steps[1]
    x = init_x
    for time_step in tqdm.tqdm(time_steps):
        batch_time_step = jnp.ones(time_shape) * time_step
        # Corrector step (Langevin MCMC)
        grad = pmap_score_fn(params, x, batch_time_step)
        grad_norm = jnp.linalg.norm(grad.reshape(sample_shape[0], sample_shape[1], -1),
                                    axis=-1).mean()
        noise_norm = np.sqrt(np.prod(x.shape[1:]))
        langevin_step_size = 2 * (snr * noise_norm / grad_norm) ** 2
        rng, step_rng = jax.random.split(rng)
        z = jax.random.normal(step_rng, x.shape)
        x = x + langevin_step_size * grad + jnp.sqrt(2 * langevin_step_size) * z

        # Predictor step (Euler-Maruyama)
        g = diffusion_coeff(time_step)
        score = pmap_score_fn(params, x, batch_time_step)
        x_mean = x + (g ** 2) * score * step_size
        rng, step_rng = jax.random.split(rng)
        z = jax.random.normal(step_rng, x.shape)
        x = x_mean + jnp.sqrt(g ** 2 * step_size) * z

        # The last step does not include any noise
    return x_mean


# @title Define the ODE sampler (double click to expand or collapse)

from scipy import integrate

## The error tolerance for the black-box ODE solver
error_tolerance = 1e-5  # @param {'type': 'number'}


def ode_sampler(rng,
                score_model,
                params,
                marginal_prob_std,
                diffusion_coeff,
                batch_size=64,
                atol=error_tolerance,
                rtol=error_tolerance,
                z=None,
                img_size=28,
                eps=1e-3):
    """Generate samples from score-based models with black-box ODE solvers.

    Args:
      rng: A JAX random state.
      score_model: A `flax.linen.Module` object  that represents architecture
        of the score-based model.
      params: A dictionary that contains model parameters.
      marginal_prob_std: A function that returns the standard deviation
        of the perturbation kernel.
      diffusion_coeff: A function that returns the diffusion coefficient of the SDE.
      batch_size: The number of samplers to generate by calling this function once.
      atol: Tolerance of absolute errors.
      rtol: Tolerance of relative errors.
      z: The latent code that governs the final sample. If None, we start from p_1;
        otherwise, we start from the given z.
      eps: The smallest time step for numerical stability.
    """
    devices = jax.local_device_count()
    if batch_size % devices != 0:
        raise ValueError(f"sample_batch_size ({batch_size}) must be divisible by local_device_count ({devices}). "
            + "Choose a multiple to avoid degenerate sampling."
            )
    pmap_score_fn = make_pmap_score_fn(score_model)
    time_shape = (devices, batch_size // devices)
    sample_shape = time_shape + (img_size, img_size, 1)
    # Create the latent code
    if z is None:
        rng, step_rng = jax.random.split(rng)
        z = jax.random.normal(step_rng, sample_shape)
        init_x = z * marginal_prob_std(1.)
    else:
        init_x = z

    shape = init_x.shape

    def score_eval_wrapper(sample, time_steps):
        sample = jnp.asarray(sample, dtype=jnp.float32).reshape(sample_shape)
        time_steps = jnp.asarray(time_steps).reshape(time_shape)
        score = pmap_score_fn(params, sample, time_steps)
        return np.asarray(score).reshape((-1,)).astype(np.float64)

    def ode_func(t, x):
        """The ODE function for use by the ODE solver."""
        time_steps = np.ones(time_shape) * t
        g = diffusion_coeff(t)
        return -0.5 * (g ** 2) * score_eval_wrapper(x, time_steps)

    # Run the black-box ODE solver.
    res = integrate.solve_ivp(ode_func, (1., eps), np.asarray(init_x).reshape(-1),
                              rtol=rtol, atol=atol, method='RK45')
    print(f"Number of function evaluations: {res.nfev}")
    x = jnp.asarray(res.y[:, -1]).reshape(shape)

    return x

def select_sampler(name: str):
    name = name.lower()
    if name in ("pc", "predictor-corrector"):
        return pc_sampler
    if name in ("em", "euler", "euler-maruyama"):
        return Euler_Maruyama_sampler
    if name in ("ode", "pf-ode", "probability-flow-ode"):
        return ode_sampler
    raise ValueError(f"Unknown sampler: {name}")