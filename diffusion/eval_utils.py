from typing import Any
from functools import partial

import math

import jax
import jax.numpy as jnp
import diffrax

from models import utils as mutils


def get_bpd_estimator(model, vector_field, use_ema=True):

  def get_bpd(key, state, batch):
    x_0 = batch['image']
    net = mutils.get_model_fn(model, 
                              state.params_ema if use_ema else state.model_params, 
                              train=False)
    key, eps_key = jax.random.split(key)
    eps = jax.random.randint(eps_key, x_0.shape, 0, 2).astype(float)*2 - 1.0
    
    def vf_jac(t, data, args):
      x, log_p = data
      dxdt = lambda _x: vector_field(t, _x, net)
      dxdt_val, jvp_val = jax.jvp(dxdt, (x,), (eps,))
      return (dxdt_val, (jvp_val*eps).sum((1,2,3)))
      
    solve = partial(diffrax.diffeqsolve, 
                    terms=diffrax.ODETerm(vf_jac), 
                    solver=diffrax.Dopri5(), 
                    t0=0.0, t1=1.0, dt0=1e-2, 
                    saveat=diffrax.SaveAt(ts=[1.0]),
                    stepsize_controller=diffrax.ConstantStepSize())

    solution = solve(y0=(x_0, jnp.zeros(x_0.shape[0])))
    x_1, delta_log_p = solution.ys[0][-1], solution.ys[1][-1]
    D = jnp.array(x_1.shape[1:]).prod()
    log_p_1 = -0.5*(x_1**2).sum((1,2,3)) - 0.5*D*math.log(2*math.pi)
    log_p_0 = log_p_1 + delta_log_p
    bpd = -log_p_0 / math.log(2) / D + 7.0
    return jax.lax.pmean(bpd.mean(), axis_name='batch'), solution.stats['num_steps']

  return get_bpd

def get_generator(models, config, vector_field, train=False):
  shape = (config.eval.batch_size//jax.local_device_count(),
           config.data.image_size, 
           config.data.image_size, 
           config.data.num_channels)

  def train_generator(key, labels, state):
    keys = jax.random.split(key, num=2)
    
    dt = 1e-2
    t = 1.0
    n = int(t/dt)
    logq = jnp.zeros((shape[0], len(models)))
    x = jax.random.normal(keys[0], shape)
    for _ in range(n):
      data = (x,logq)
      dx, dlogq = vector_field(t, data, args={'key': keys[1], 'labels': labels, 'dt': dt, 'state': state})
      x += dx
      logq += dlogq
      t += -dt
    return x, n
  
  if train:
    return train_generator

  def artifact_generator(key, labels):
    keys = jax.random.split(key, num=2)
    
    dt = 5e-3
    t = 1.0
    n = int(t/dt)
    logq = jnp.zeros((shape[0], len(models)))
    x = jax.random.normal(keys[0], shape)
    for _ in range(n):
      data = (x,logq)
      dx, dlogq = vector_field(t, data, args={'key': keys[1], 'labels': labels, 'dt': dt})
      x += dx
      logq += dlogq
      t += -dt
    return x, n
    
  return artifact_generator
