import jax.numpy as jnp
import numpy as np
from jax import *
from numpy import ndarray
from typing import Literal

SELECT_KAPPA = Literal["iso_surface", "simple_average", "proportional_average"]

beta_0 = 0.1
beta_1 = 20.0

log_alpha = lambda t: -0.5*t*beta_0-0.25*t**2*(beta_1-beta_0)
log_sigma = lambda t: jnp.log(t)

dlog_alphadt = jax.grad(lambda t: log_alpha(t).sum())
dlog_sigmadt = jax.grad(lambda t: log_sigma(t).sum())

beta = lambda t: (1 + 0.5*t*beta_0 + 0.5*t**2*(beta_1-beta_0)) # Controls how fast noise is added to data
diffusion = lambda dt, state, t, trajectory_at_time_t, xi: -dt * vector_field(state, t, trajectory_at_time_t, xi)
drift = lambda t, noise, dt:jnp.sqrt(2*jnp.exp(log_sigma(t))*beta(t)*dt) * noise

def q_t(data, t, standard_noise): # Forward Diffusion
  x_t = jnp.exp(log_alpha(t))*data + jnp.exp(log_sigma(t))*standard_noise
  return x_t

def last_step_in_forward_diffusion(key, data, timesteps: ndarray = np.linspace(0.0, 1.0, 6)):
  return forward_diffusion_over_time(key, data, timesteps)[-1]

def forward_diffusion_over_time(key, data, timesteps: ndarray = np.linspace(0.0, 1.0, 6)):
  """
  :return: A list of samples that have been diffused over time
  Example:  [first_batch_of_samples_at_time_0, first_batch_of_samples_at_time_1, ..., first_batch_of_samples_at_time_6] each of size (512,2)
  """
  x_ts = []
  for t in timesteps:
    key, subkey = random.split(key)
    epsilon = random.normal(subkey, data.shape)
    x_ts.append(q_t(data, t, epsilon))
  return x_ts

@jax.jit
def vector_field(state, t,x,xi: float=0.0, stochastic_sampling: bool = True):
  sdlogqdx = lambda _t, _x: state.apply_fn(state.params, _t, _x) # Score Function
  if stochastic_sampling:
    """
    Type: Reverse SDE Sampler (DDIM)
    SDE: If you want stochastic sampling (reverse diffusion with noise)
    Note: 👉 So, likelihood evaluation is not feasible directly with the SDE sampler. 
    - The dynamics include Brownian noise (dWt) 
    - If you try to compute log-likelihood of a datapoint, 
      you’d need to integrate over all random noise paths that could reach that point.
    - That expectation is intractable without special tricks (e.g. Monte Carlo path sampling).
    """
    dxdt = dlog_alphadt(t) * x - 2 * beta(t) * sdlogqdx(t, x) # Default in DDPM
  else:
    """
    Type: Probability-Flow ODE Sampler (Deterministic)
    ODE: If you want deterministic trajectories
    Note: The ODE is noise-free. Each datapoint has a unique trajectory back to Gaussian noise.
    - You can use the instantaneous change of variables formula:
    Benefit: Allows likelihood evaluation
    💖 Trick: This is tractable with Hutchinson’s trace trick
    """
    dxdt = dlog_alphadt(t) * x - beta(t) * sdlogqdx(t, x) - xi * beta(t) / jnp.exp(log_sigma(t)) * sdlogqdx(t, x)
  return dxdt


@jax.jit
def score_function_hutchinson_estimator(key, t, state, x): # Get score & divergence
  eps = jax.random.randint(key, x.shape, 0, 2).astype(float) * 2 - 1.0
  sdlogqdx = lambda _x: state.apply_fn(state.params, t, _x)
  sdlogdx_val, jvp_val = jax.jvp(sdlogqdx, (x,), (eps,)) # Gets divergence
  return sdlogdx_val, (jvp_val * eps).sum(1, keepdims=True)

def dlogqdt(t, x, score, score_divergence, true_drift, ndim: int =2): # Proposition 5(Equation 10): -divergence(v) - <score, v- u>
  """
  Used in likelihood estimation: evaluates the log-probability of generated samples without retraining.
  """
  model_vector_field = dlog_alphadt(t) * x - beta(t) * score # v(x,t)
  divergence_of_v = -dlog_alphadt(t) * ndim + beta(t) * score_divergence # -🔻. v(x,t)
  normalized_score = score / jnp.exp(log_sigma(t))
  correction = -(normalized_score * (model_vector_field - true_drift)).sum(1, keepdims=True)
  smooth_density_estimator = divergence_of_v + correction
  return smooth_density_estimator

@jax.jit
def get_sscore(state,t,x):
  return state.apply_fn(state.params, t, x)

@jax.jit
def get_stoch_dll(t,dt,x,dx,sscore, ndim: int = 2):
  output = ndim*dt*dlog_alphadt(t) - dt*beta(t)*(sscore**2)/jnp.exp(log_sigma(t))
  output += ((dx + dt*dlog_alphadt(t)*x)*sscore/jnp.exp(log_sigma(t)))
  return output.sum(1)

@jax.jit
def select_kappa(ikey,t,dt,x,sdlogdx_1,sdlogdx_2, bs: int = 512):
  noise = jnp.sqrt(2*jnp.exp(log_sigma(t))*beta(t)*dt)*random.normal(ikey, shape=(bs,2))
  dx_ind = -dt*(dlog_alphadt(t)*x - 2*beta(t)*sdlogdx_2) + noise
  kappa = -dt*beta(t)*(sdlogdx_1-sdlogdx_2)*(sdlogdx_1+sdlogdx_2)/jnp.exp(log_sigma(t))
  kappa += ((dx_ind + dt*dlog_alphadt(t)*x)*(sdlogdx_1-sdlogdx_2)/jnp.exp(log_sigma(t)))
  kappa = -kappa.sum(1)/(dt*2*beta(t)*(sdlogdx_1-sdlogdx_2)**2/jnp.exp(log_sigma(t))).sum(1)
  return kappa


@jax.jit
def get_kappa(t, divlogs, sdlogdxs, ll, kappa: SELECT_KAPPA = "iso_surface"):
    """
    Used in model superposition: balances two pretrained diffusion models by computing the optimal mixing coefficient at each timestep
    """
    divlog_1, divlog_2 = divlogs
    sdlogdx_1, sdlogdx_2 = sdlogdxs
    ll_1, ll_2 = ll
    if kappa == "iso_surface":
      kappa = jnp.exp(log_sigma(t))*(divlog_1-divlog_2) + (sdlogdx_1*(sdlogdx_1-sdlogdx_2)).sum(1, keepdims=True)
      kappa /= ((sdlogdx_1-sdlogdx_2)**2).sum(1, keepdims=True)
    elif kappa == "simple_average":
      kappa = 0.5
    elif kappa == "proportional_average":
      max_ll = jnp.maximum(ll_1, ll_2)
      kappa = jnp.exp(ll_1 - max_ll) / (jnp.exp(ll_1 - max_ll) + jnp.exp(ll_2 - max_ll))
      kappa = kappa[:, None]
    return kappa

