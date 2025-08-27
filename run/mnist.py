import tensorflow as tf
import jax.numpy as jnp
from jax import random
import flax.jax_utils as flax_utils
import diffusion.eval_utils as eutils
from config.lib import init_model
from config.mnist_config import get_config
from diffusion.equations import vector_field, q_t
from models.ddpm import *
from train.train_score_net import score_loss, get_step_fn
from utils.viz import plot_grid
import jax
tf.config.set_visible_devices([], 'GPU')
import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
from datasets.MNIST import get_mnist_dataset, get_image_scaler, get_image_inverse_scaler

key = random.PRNGKey(0)
key, init_key = random.split(key)

workdir = "./checkpoints/"
config = get_config()
state, ckpt_mgr, optimizer, model = init_model(key, config, workdir)
initial_step = int(state.step)
key = state.key
qt, loss_fn, v_field = q_t, score_loss, vector_field
step_fn = get_step_fn(model, optimizer, loss_fn)
step_fn = jax.pmap(step_fn, axis_name='batch')
artifact_generator = eutils.get_generator([model], config, jax.jit(vector_field), train=True)
artifact_generator = jax.vmap(artifact_generator, axis_name='batch')

train = get_mnist_dataset()
plot_grid(train)

train_iter = iter(train)
scaler = get_image_scaler()
inverse_scalar = get_image_inverse_scaler()

pstate = flax_utils.replicate(state)
key = jax.random.fold_in(key, jax.process_index())
for step in range(initial_step, config.train.n_iters+1):
    batch = jax.tree_map(lambda x: x._numpy(), next(train_iter))
    batch['image'] = scaler(batch['image'])
    key, *next_key = random.split(key, num=jax.local_device_count() + 1)
    next_key = jnp.asarray(next_key)
    (_, pstate), ploss = step_fn((next_key, pstate), batch)
    loss = ploss.mean()