import jax.numpy as jnp
import numpy as np
from jax import *
from numpy import ndarray
import functools

beta_0 = 0.1
beta_1 = 20.0

log_alpha = lambda t: -0.5*t*beta_0-0.25*t**2*(beta_1-beta_0)
log_sigma = lambda t: jnp.log(t)

dlog_alphadt = jax.grad(lambda t: log_alpha(t).sum())
dlog_sigmadt = jax.grad(lambda t: log_sigma(t).sum())

beta = lambda t: (1 + 0.5*t*beta_0 + 0.5*t**2*(beta_1-beta_0)) # Controls how fast noise is added to data
diffusion = lambda dt, state, t, trajectory_at_time_t, xi: -dt * vector_field(state, t, trajectory_at_time_t, xi)
drift = lambda t, noise, dt:jnp.sqrt(2*jnp.exp(log_sigma(t))*beta(t)*dt) * noise


def _sum_except_batch(x):
  """Sum over all non-batch axes, keep batch axis."""
  assert x.ndim >= 2, "Expected (N, ...)"
  axes = tuple(range(1, x.ndim))
  return jnp.sum(x, axis=axes, keepdims=True)


def q_t(data, t, standard_noise): # Forward Diffusion
  x_t = jnp.exp(log_alpha(t))*data + jnp.exp(log_sigma(t))*standard_noise
  return x_t

def _log_alpha_bar_cosine(t, s=0.008):
  # Nichol & Dhariwal cosine schedule in continuous time
  c = (t + s) / (1.0 + s)
  # add tiny epsilon for numerical stability
  return 2.0 * jnp.log(jnp.cos(jnp.pi * c / 2.0 + 1e-12))


# d/dt log alpha_bar
_dlogab_dt = jax.grad(lambda _t: _log_alpha_bar_cosine(_t).sum())


def vpsde_marginal_prob_std(t):
  # std_t = sqrt(1 - alpha_bar(t))
  return jnp.sqrt(1.0 - jnp.exp(_log_alpha_bar_cosine(t)))


def vpsde_diffusion_coeff(t):
  # β(t) = -d/dt log alpha_bar(t)
  beta_t = -_dlogab_dt(t)
  # g(t) = sqrt(β(t))
  return jnp.sqrt(jnp.clip(beta_t, 1e-12, 1e12))

def get_sdlogqdx_fn(model, params, train: bool = False):
  """
  Equivalent to `sdlogqdx = lambda _t, _x: state.apply_fn(params, _t, _x)`  # 🔻qt(x)
  Returns a function that evaluates ∇ log q_t(x) (score function).

  Args:
      model: Flax Module
      params: parameters to apply
      train: whether to run in training mode (affects dropout/BN if present)

  Returns:
      sdlogqdx(t, x, rng=None) -> model output
  """

  def sdlogqdx(t, x, labels=None, rng=None):
    variables = {"params": params}
    kwargs = dict(train=train, mutable=False)
    if rng is not None:
      kwargs["rngs"] = {"dropout": rng}
    # If your model is conditional, it likely has signature (t, x, labels, ...)
    if labels is None:
      return model.apply(variables, t, x, **kwargs)
    else:
      return model.apply(variables, t, x, labels, **kwargs)
  return sdlogqdx

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
def vector_field(model, state, t, data, labels, xi: float=0.0, stochastic_sampling: int = 0):
  x, _ = data
  t = t*jnp.ones((x.shape[0],1,1,1))
  sdlogqdx = get_sdlogqdx_fn(model, state.model_params, train=False)
  if stochastic_sampling == 0:
    """
    Type: Reverse SDE Sampler (DDIM)
    SDE: If you want stochastic sampling (reverse diffusion with noise)
    Note: 👉 So, likelihood evaluation is not feasible directly with the SDE sampler. 
    - The dynamics include Brownian noise (dWt) 
    - If you try to compute log-likelihood of a datapoint, 
      you’d need to integrate over all random noise paths that could reach that point.
    - That expectation is intractable without special tricks (e.g. Monte Carlo path sampling).
    """
    dxdt = dlog_alphadt(t) * x - 2 * beta(t) * sdlogqdx(t, x, labels) # Default in DDPM
  elif stochastic_sampling == 1:
    dxdt = dlog_alphadt(t) * x - beta(t) * sdlogqdx(t, x, labels)  # Match super-diffusion dataset
  else:
    """
    Type: Probability-Flow ODE Sampler (Deterministic)
    ODE: If you want deterministic trajectories
    Note: The ODE is noise-free. Each datapoint has a unique trajectory back to Gaussian noise.
    - You can use the instantaneous change of variables formula:
    Benefit: Allows likelihood evaluation
    💖 Trick: This is tractable with Hutchinson’s trace trick
    """
    dxdt = dlog_alphadt(t) * x - beta(t) * sdlogqdx(t, x, labels) - xi * beta(t) / jnp.exp(log_sigma(t)) * sdlogqdx(t, x, labels)
  return dxdt


@jax.jit
def score_function_hutchinson_estimator(key, t, state, x):
  """
  Returns:
    score: same shape as x  (N, ...), ∇_x log q_t(x)
    div  : shape (N, 1[,1,1...]) (summed over all non-batch dims)
  """
  # Rademacher noise with same shape as x
  eps = jax.random.randint(key, x.shape, 0, 2).astype(x.dtype) * 2.0 - 1.0

  # Model must map x -> score with t broadcasted per-sample.
  # state.apply_fn(state.params, t, x) matches your current usage.
  def f(_x):
    return state.apply_fn(state.params, t, _x)

  score_val, jvp_val = jax.jvp(f, (x,), (eps,))
  div = _sum_except_batch(jvp_val * eps)  # Hutchinson trace trick
  return score_val, div

def dlogqdt(t, x, score, score_divergence, true_drift, ndim: int =2): # Proposition 5(Equation 10): -divergence(v) - <score, v- u>
  """
  Used in likelihood estimation: evaluates the log-probability of generated samples without retraining.
  """
  model_vector_field = dlog_alphadt(t) * x - beta(t) * score # v(x,t)
  divergence_of_v = -dlog_alphadt(t) * ndim + beta(t) * score_divergence # -🔻. v(x,t)
  normalized_score = score / jnp.exp(log_sigma(t))
  # correction = -(normalized_score * (model_vector_field - true_drift)).sum(1, keepdims=True)
  correction = -_sum_except_batch(normalized_score * (model_vector_field - true_drift))
  smooth_density_estimator = divergence_of_v + correction
  return smooth_density_estimator

@jax.jit
def get_kappa(t, divlogs, sdlogdxs):
  """
  Optimal mixing coefficient κ(t) per Eq. (superposition).
  Accepts high-dim tensors; sums are taken over non-batch dims.
  Shapes:
    divlog_* : (N, 1[,1,1...])
    sdlogdx_*: (N, ... same as x)
  Returns:
    kappa: (N, 1[,1,1...])  (broadcastable over x)
  """
  div1, div2 = divlogs
  s1, s2 = sdlogdxs
  # numerator: exp(log_sigma(t)) * (div1-div2) + <s1, s1 - s2>
  num = jnp.exp(log_sigma(t)) * (div1 - div2) + _sum_except_batch(s1 * (s1 - s2))
  den = _sum_except_batch((s1 - s2) ** 2) + 1e-12  # stabilize
  return num / den


# @title Set up the SDE

def marginal_prob_std(t, sigma):
  """Compute the mean and standard deviation of $p_{0t}(x(t) | x(0))$.

  Args:
    t: A vector of time steps.
    sigma: The $\sigma$ in our SDE.

  Returns:
    The standard deviation.
  """
  return jnp.sqrt((sigma ** (2 * t) - 1.) / 2. / jnp.log(sigma))


def diffusion_coeff(t, sigma):
  """Compute the diffusion coefficient of our SDE.

  Args:
    t: A vector of time steps.
    sigma: The $\sigma$ in our SDE.

  Returns:
    The vector of diffusion coefficients.
  """
  return sigma ** t


sigma = 25.0  # @param {'type':'number'}
marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=sigma)
diffusion_coeff_fn = functools.partial(diffusion_coeff, sigma=sigma)