from typing import Tuple
import jax.numpy as jnp
import numpy as np
from jax import *
from tqdm import trange

from diffusion.equations import diffusion, drift, score_function_hutchinson_estimator, get_kappa, dlog_alphadt, beta, \
    dlogqdt, SELECT_KAPPA


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

def compose_and_estimate_log_likelihood_along_superposed_trajectory(state_a, state_b, key, dt:float = 1e-2, t: float = 1.0, shape: Tuple[int, int] = (512, 2), kappa: SELECT_KAPPA = "iso_surface"):
    datapoints, coordinates = shape
    num_timesteps = int(t / dt) + 1  # time index i = 0, 1, .... n (include start and 100 steps)
    t = t * jnp.ones((datapoints, 1))  # Broadcasted time vector per particle (starts at 1)
    key, subkey = random.split(key, num=2)
    trajectory_field = jnp.zeros((datapoints, num_timesteps, coordinates))  # Storage for whole trajectory
    pure_noise = random.normal(subkey, shape=(datapoints, coordinates))
    trajectory_field = trajectory_field.at[:, 0, :].set(pure_noise)
    trajectory, log_likelihood_model_a, log_likelihood_model_b = ito_dynamic_estimator_solver(key, t, state_a, state_b, trajectory_field, dt=dt, num_timesteps=num_timesteps,  shape=shape, kappa=kappa)
    return trajectory, log_likelihood_model_a, log_likelihood_model_b

def ito_dynamic_estimator_solver(key, t, state_a, state_b, trajectory, dt: float = 1e-2, num_timesteps: int=100,  shape: Tuple[int,int] = (512, 2), select_kappa: SELECT_KAPPA = "iso_surface"):
    datapoints, coordinates = shape
    log_likelihood_model_a = np.zeros((datapoints, num_timesteps))
    log_likelihood_model_b = np.zeros((datapoints, num_timesteps))
    for timestep in trange(num_timesteps-1):
        x_t = trajectory[:, timestep, :]
        key, subkey = random.split(key, 2)
        score_model_a, score_divergence_model_a = score_function_hutchinson_estimator(key, t, state_a, x_t)
        score_model_b, score_divergence_model_b = score_function_hutchinson_estimator(key, t, state_b, x_t)
        kappa = get_kappa(t, (score_divergence_model_a, score_divergence_model_b), (score_model_a, score_model_b), kappa=select_kappa)
        reverse_drift_ode = dlog_alphadt(t)*x_t - beta(t)*(score_model_b + kappa*(score_model_a-score_model_b))
        trajectory = trajectory.at[:, timestep + 1, :].set(x_t -  dt*reverse_drift_ode)  # Take a step
        log_likelihood_model_a[:,timestep+1] = log_likelihood_model_a[:,timestep] - dt*dlogqdt(t, x_t, score_model_a, score_divergence_model_a, reverse_drift_ode).squeeze()
        log_likelihood_model_b[:,timestep+1] = log_likelihood_model_b[:,timestep] - dt*dlogqdt(t, x_t, score_model_b, score_divergence_model_b, reverse_drift_ode).squeeze()
        t += -dt
    return trajectory, log_likelihood_model_a, log_likelihood_model_b