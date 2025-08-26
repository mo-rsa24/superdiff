import optax
from flax.training import train_state
from jax import random
import jax
from tqdm import trange

from config import Config
from diffusion.equations import q_t
import numpy as np

from models.MLP import GaussianMLP


def score_loss(state, key, params, sample_data):
    datapoints, dimension = sample_data.shape
    keys = random.split(key ,)
    t = random.uniform(keys[0], [datapoints, 1])
    epsilon = random.normal(keys[1], (datapoints, dimension))
    x_t = q_t(sample_data, t, epsilon) # xt = qt(x)
    sdlogqdx = lambda _t, _x: state.apply_fn(params, _t, _x) # 🔻qt(x)
    loss = ((epsilon + sdlogqdx(t, x_t))**2).sum(1)
    return loss.mean()

@jax.jit
def train_step(state, sample_data, key):
    grad_fn = jax.value_and_grad(score_loss, argnums=2)
    loss, grads = grad_fn(state,key, state.params, sample_data)
    state = state.apply_gradients(grads=grads)
    return state, loss

def train_mlp(key, state, sample_data):
    key, loc_key = random.split(key)

    num_iterations = 20_000
    loss_plot = np.zeros(num_iterations)
    key, loop_key = random.split(key)

    for iter in trange(num_iterations):
        iter_key = random.fold_in(loop_key, iter)
        state, loss = train_step(state, sample_data, iter_key)
        loss_plot[iter] = loss
    return loss_plot, state

def train_group(model, sample_data, x_t):
    key = random.PRNGKey(Config.SEED)
    key, init_key = random.split(key)
    optimizer = optax.adam(learning_rate=2e-4)
    state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=model.init(init_key, np.ones((512, 1)), x_t),
        tx=optimizer
    )
    loss, state = train_mlp(key, state, sample_data)
    return loss, state