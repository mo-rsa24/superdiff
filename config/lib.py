import os
import orbax
import flax
from jax import random
from flax.training import checkpoints
import optax
import numpy as np

from models import utils as mutils

def get_optimizer(config):
  schedule = optax.join_schedules([optax.linear_schedule(0.0, config.train.lr, config.train.warmup),
                                   optax.constant_schedule(config.train.lr)],
                                   boundaries=[config.train.warmup])
  optimizer = optax.adam(learning_rate=schedule, b1=config.train.beta1, eps=config.train.eps)
  optimizer = optax.chain(
    optax.clip(config.train.grad_clip),
    optimizer
  )
  return optimizer

def init_model(key, config, workdir):
  key, init_key = random.split(key)
  model, initial_params = mutils.init_model(init_key, config)
  optimizer = get_optimizer(config)
  opt_state = optimizer.init(initial_params)
  state = mutils.State(step=1, opt_state=opt_state,
                       model_params=initial_params,
                       ema_rate=config.model.ema_rate,
                       params_ema=initial_params,
                       sampler_state=0.5,
                       key=key, wandbid=np.random.randint(int(1e7),int(1e8)))

  mgr_options = orbax.checkpoint.CheckpointManagerOptions(
    create=True, max_to_keep=50, step_prefix='chkpt')
  ckpt_mgr = orbax.checkpoint.CheckpointManager(
    workdir, orbax.checkpoint.Checkpointer(orbax.checkpoint.PyTreeCheckpointHandler()), mgr_options)
  ckpt_mgr.reload()
  # read preemptied run
  if ckpt_mgr.latest_step() is not None:
    restore_args = flax.training.orbax_utils.restore_args_from_target(state, mesh=None)
    state = ckpt_mgr.restore(ckpt_mgr.latest_step(), items=state, restore_kwargs={'restore_args': restore_args})
  return state, ckpt_mgr, optimizer, model