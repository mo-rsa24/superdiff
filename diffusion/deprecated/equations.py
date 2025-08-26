from typing import Tuple

import jax.numpy as jnp
import numpy as np
from jax import *
from numpy import ndarray
from tqdm import trange

beta_0 = 0.1
beta_1 = 20.0

log_alpha = lambda t: -0.5*t*beta_0-0.25*t**2*(beta_1-beta_0)
log_sigma = lambda t: jnp.log(t)

dlog_alphadt = jax.grad(lambda t: log_alpha(t).sum())
dlog_sigmadt = jax.grad(lambda t: log_sigma(t).sum())

beta = lambda t: (1 + 0.5*t*beta_0 + 0.5*t**2*(beta_1-beta_0)) # Controls how fast noise is added to data
diffusion = lambda dt, state, t, trajectory_at_time_t, xi: -dt * vector_field(state, t, trajectory_at_time_t, xi)
drift = lambda xi, t, noise, dt:jnp.sqrt(2 * xi * beta(t) * dt) * noise

def q_t(data, t, standard_noise): # Forward Diffusion
  x_t = jnp.exp(log_alpha(t))*data + jnp.exp(log_sigma(t))*standard_noise
  return x_t

def forward_step(key, data, timesteps: ndarray = np.linspace(0.0, 1.0, 6)):
  return forward_steps(key, data, timesteps)[-1]

def forward_steps(key, data, timesteps: ndarray = np.linspace(0.0, 1.0, 6)):
  x_ts = []
  for t in timesteps:
    key, subkey = random.split(key)
    epsilon = random.normal(subkey, data.shape)
    x_ts.append(q_t(data, t, epsilon))
  return x_ts

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

@jax.jit
def vector_field(state, t,x,xi: float=0.0, stochastic_sampling: bool = False):
  sdlogqdx = lambda _t, _x: state.apply_fn(state.params, _t, _x) # Score Function
  dxdt = dlog_alphadt(t) * x - beta(t) * sdlogqdx(t, x) - xi * beta(t) / jnp.exp(log_sigma(t)) * sdlogqdx(t, x)
  return dxdt

def euler_murayama(key, t, state, trajectory, xi: float = 1.0, dt: float = 1e-2, num_timesteps: int=100,  shape: Tuple[int,int] = (512, 2)):

  # diffusion at timestep i
  for timestep in trange(num_timesteps):
    key, subkey = random.split(key, 2)
    epsilon = random.normal(subkey, shape)
    diffusion_term = diffusion(dt, state, t, trajectory[:, timestep, :], xi)
    drift_term  = drift(xi, t, epsilon, dt)
    dx = diffusion_term + drift_term
    trajectory = trajectory.at[:, timestep+1, :].set(trajectory[:, timestep, :] + dx) # Take a step
    t += -dt
  return trajectory
