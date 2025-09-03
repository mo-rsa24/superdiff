from typing import Tuple
import jax.numpy as jnp
import numpy as np
from jax import *
from tqdm import trange
import tqdm

from diffusion.equations import diffusion, drift, score_function_hutchinson_estimator, get_kappa, dlog_alphadt, beta, \
    dlogqdt


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
    datapoints, coordinates = shape
    num_timesteps = int(t / dt) + 1  # time index i = 0, 1, .... n (include start and 100 steps)
    t = t * jnp.ones((datapoints, 1))  # Broadcasted time vector per particle (starts at 1)
    key, subkey = random.split(key, num=2)
    trajectory_field = jnp.zeros((datapoints, num_timesteps, coordinates))  # Storage for whole trajectory
    pure_noise = random.normal(subkey, shape=(datapoints, coordinates))
    trajectory_field = trajectory_field.at[:, 0, :].set(pure_noise)
    trajectory, log_likelihood_model_a, log_likelihood_model_b = ito_dynamic_estimator_solver(key, t, state_a, state_b, trajectory_field, dt=dt, num_timesteps=num_timesteps,  shape=shape)
    return trajectory, log_likelihood_model_a, log_likelihood_model_b

def ito_dynamic_estimator_solver(key, t, state_a, state_b, trajectory, dt: float = 1e-2, num_timesteps: int=100,  shape: Tuple[int,int] = (512, 2)):
    datapoints, coordinates = shape
    log_likelihood_model_a = np.zeros((datapoints, num_timesteps))
    log_likelihood_model_b = np.zeros((datapoints, num_timesteps))
    for timestep in trange(num_timesteps-1):
        x_t = trajectory[:, timestep, :]
        key, subkey = random.split(key, 2)
        score_model_a, score_divergence_model_a = score_function_hutchinson_estimator(key, t, state_a, x_t)
        score_model_b, score_divergence_model_b = score_function_hutchinson_estimator(key, t, state_b, x_t)
        kappa = get_kappa(t, (score_divergence_model_a, score_divergence_model_b), (score_model_a, score_model_b))
        reverse_drift_ode = dlog_alphadt(t)*x_t - beta(t)*(score_model_b + kappa*(score_model_a-score_model_b))
        trajectory = trajectory.at[:, timestep + 1, :].set(x_t -  dt*reverse_drift_ode)  # Take a step
        log_likelihood_model_a[:,timestep+1] = log_likelihood_model_a[:,timestep] - dt*dlogqdt(t, x_t, score_model_a, score_divergence_model_a, reverse_drift_ode).squeeze()
        log_likelihood_model_b[:,timestep+1] = log_likelihood_model_b[:,timestep] - dt*dlogqdt(t, x_t, score_model_b, score_divergence_model_b, reverse_drift_ode).squeeze()
        t += -dt
    return trajectory, log_likelihood_model_a, log_likelihood_model_b



num_steps = 500

def _make_pmap_score_fn(score_model):
    """Bind the Flax Module in a closure so pmap sees only JAX types."""
    def _score_fn(params, x, t):
        # If your TrainState.params is the full variables dict, this is already fine.
        # If it's only the 'params' subtree, change to: score_model.apply({'params': params}, x, t)
        return score_model.apply(params, x, t)
    # params is broadcast (None); x and t are mapped on the leading device axis.
    return jax.pmap(_score_fn, in_axes=(None, 0, 0))


def Euler_Maruyama_sampler(rng, score_model, params, marginal_prob_std, diffusion_coeff,
                           batch_size=64, num_steps=num_steps, eps=1e-3):
    ndev = jax.local_device_count()
    assert batch_size % ndev == 0, "batch_size must be divisible by #devices"
    time_shape   = (ndev, batch_size // ndev)
    sample_shape = time_shape + (28, 28, 1)
    rng, step_rng = jax.random.split(rng)
    init_x = jax.random.normal(step_rng, sample_shape) * marginal_prob_std(1.)
    time_steps = jnp.linspace(1., eps, num_steps)
    step_size  = time_steps[0] - time_steps[1]
    x = init_x

    pmap_score = _make_pmap_score_fn(score_model)  # <-- bind once

    for time_step in tqdm.tqdm(time_steps):
        batch_t = jnp.ones(time_shape) * time_step
        g = diffusion_coeff(time_step)
        mean_x = x + (g ** 2) * pmap_score(params, x, batch_t) * step_size
        rng, step_rng = jax.random.split(rng)
        x = mean_x + jnp.sqrt(step_size) * g * jax.random.normal(step_rng, x.shape)
    return mean_x


# @title Define the Predictor-Corrector sampler (double click to expand or collapse)

signal_to_noise_ratio = 0.16  # @param {'type':'number'}

## The number of sampling steps.
num_steps = 500  # @param {'type':'integer'}


def pc_sampler(rng, score_model, params, marginal_prob_std, diffusion_coeff,
               batch_size=64, num_steps=num_steps, snr=signal_to_noise_ratio, eps=1e-3):
    ndev = jax.local_device_count()
    assert batch_size % ndev == 0, "batch_size must be divisible by #devices"
    time_shape   = (ndev, batch_size // ndev)
    sample_shape = time_shape + (28, 28, 1)
    rng, step_rng = jax.random.split(rng)
    init_x = jax.random.normal(step_rng, sample_shape) * marginal_prob_std(1.)
    time_steps = jnp.linspace(1., eps, num_steps)
    step_size  = time_steps[0] - time_steps[1]
    x = init_x

    pmap_score = _make_pmap_score_fn(score_model)  # <-- bind once

    for time_step in tqdm.tqdm(time_steps):
        batch_t = jnp.ones(time_shape) * time_step
        # Corrector (Langevin)
        grad = pmap_score(params, x, batch_t)
        grad_norm  = jnp.linalg.norm(grad.reshape(sample_shape[0], sample_shape[1], -1), axis=-1).mean()
        noise_norm = np.sqrt(np.prod(x.shape[1:]))
        langevin_step = 2 * (snr * noise_norm / grad_norm) ** 2
        rng, step_rng = jax.random.split(rng)
        z = jax.random.normal(step_rng, x.shape)
        x = x + langevin_step * grad + jnp.sqrt(2 * langevin_step) * z
        # Predictor (EM)
        g = diffusion_coeff(time_step)
        score = pmap_score(params, x, batch_t)
        x_mean = x + (g ** 2) * score * step_size
        rng, step_rng = jax.random.split(rng)
        z = jax.random.normal(step_rng, x.shape)
        x = x_mean + jnp.sqrt(g ** 2 * step_size) * z
    return x_mean

from scipy import integrate

## The error tolerance for the black-box ODE solver
error_tolerance = 1e-5  # @param {'type': 'number'}


def ode_sampler(rng, score_model, params, marginal_prob_std, diffusion_coeff,
                batch_size=64, atol=error_tolerance, rtol=error_tolerance, z=None, eps=1e-3):
    ndev = jax.local_device_count()
    assert batch_size % ndev == 0, "batch_size must be divisible by #devices"
    time_shape   = (ndev, batch_size // ndev)
    sample_shape = time_shape + (28, 28, 1)

    if z is None:
        rng, step_rng = jax.random.split(rng)
        z = jax.random.normal(step_rng, sample_shape)
        init_x = z * marginal_prob_std(1.)
    else:
        init_x = z

    pmap_score = _make_pmap_score_fn(score_model)  # <-- bind once

    def score_eval_wrapper(sample, time_steps):
        sample = jnp.asarray(sample, dtype=jnp.float32).reshape(sample_shape)
        time_steps = jnp.asarray(time_steps).reshape(time_shape)
        score = pmap_score(params, sample, time_steps)
        return np.asarray(score).reshape((-1,)).astype(np.float64)

    def ode_func(t, x):
        time_steps = np.ones(time_shape) * t
        g = diffusion_coeff(t)
        return -0.5 * (g ** 2) * score_eval_wrapper(x, time_steps)

    res = integrate.solve_ivp(ode_func, (1., eps), np.asarray(init_x).reshape(-1),
                              rtol=rtol, atol=atol, method='RK45')
    x = jnp.asarray(res.y[:, -1]).reshape(init_x.shape)
    return x
