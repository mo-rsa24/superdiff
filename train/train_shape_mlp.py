from flax.jax_utils import prefetch_to_device
from tqdm import trange

import optax
from flax.training import train_state
from diffusion.equations import *
from config import Config
from jax import random


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
    def sdlogqdx(t, x, rng=None):
        variables = {"params": params}
        if not train:
            return model.apply(variables, t, x, train=False, mutable=False)
        else:
            rngs = {"dropout": rng} if rng is not None else None
            return model.apply(variables, t, x, train=True, mutable=False, rngs=rngs)

    return sdlogqdx


def score_loss(state, key, params, batch):
    datapoints, dimension  = batch.shape
    keys = random.split(key, )
    t = random.uniform(keys[0], [datapoints, 1])
    epsilon = random.normal(keys[1], (datapoints, dimension))
    x_t = q_t(batch, t, epsilon)  # xt = qt(x)
    sdlogqdx = lambda _t, _x: state.apply_fn(params, _t, _x)  # 🔻qt(x)
    loss = ((epsilon + sdlogqdx(t, x_t)) ** 2).sum(1)
    return loss.mean()

@jax.jit
def train_step(state, batch, key):
    grad_fn = jax.value_and_grad(score_loss, argnums=2)
    loss, grads = grad_fn(state, key, state.params, batch)
    state = state.apply_gradients(grads=grads)
    return state, loss

def to_device(iterable):
    for b in iterable:
        yield jax.device_put(jnp.asarray(b))

def train_latent_shape_model(key, state, dataset, epochs: int = 900, model_name: str = "Circle Model"):
    key, loop_key = random.split(key)
    prefetched = to_device(dataset)
    train_loss = []
    for epoch in (pbar := trange(epochs, desc=f"Training {model_name}")):
        epoch_loss = []
        for batch in prefetched:
            key, step_key = random.split(key)
            state, loss = train_step(state, batch, step_key)
            epoch_loss.append(loss)
        avg_loss = np.mean(epoch_loss)
        train_loss.append(avg_loss)
        pbar.set_postfix_str(f"Loss: {avg_loss:.4f}")
    print(f"Training complete for {model_name}.")
    return state, train_loss

def train_latent_shape_group(model, dataset, epochs: int = 900, model_name: str = "Circle Model"):
    key = random.PRNGKey(Config.SEED)
    key, init_key = random.split(key)
    optimizer = optax.adam(learning_rate=2e-4)
    state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=model.init(init_key, jnp.ones((1, 1)), jnp.ones((1,2))),
        tx=optimizer
    )
    state, loss = train_latent_shape_model(key, state, dataset, epochs=epochs, model_name=model_name)
    return state, loss


def train_expert(key, model, dataset, batch_size: int = 512, epochs: int = 900, model_name: str = "Circle Model"):
    """Main training loop for a single expert model."""
    print(f"--- Training expert for {model_name} ---")
    init_key, train_key = random.split(key)
    optimizer = optax.adam(learning_rate=2e-4)
    state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=model.init(init_key, jnp.ones((1, 1)), jnp.ones((1, 2))),
        tx=optimizer
    )

    num_batches = len(dataset) // batch_size
    train_loss = []
    for epoch in (pbar := trange(epochs, desc=f"Training {model_name}")):
        key, perm_key = random.split(key)
        perms = random.permutation(perm_key, len(dataset))
        perms = perms[:num_batches * batch_size]
        perms = perms.reshape((num_batches, batch_size))

        epoch_loss = []
        for perm in perms:
            batch = dataset[perm, :]
            key, step_key = random.split(key)
            state, loss = train_step(state, batch, step_key)
            epoch_loss.append(loss)

        avg_loss = np.mean(epoch_loss)
        train_loss.append(avg_loss)
        pbar.set_postfix_str(f"Loss: {avg_loss:.4f}")

    print(f"Training complete for {model_name}.")
    return state, train_loss  # Return the full state for generation

