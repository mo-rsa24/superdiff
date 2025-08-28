import math
import optax
from diffusion.equations import *
from jax import random

def sample_time(sample_size, u0, t_0, t_1):
    u = (u0 + math.sqrt(2)*jnp.arange(sample_size*jax.device_count())) % 1
    u0=u[-1]
    t = (t_1-t_0)*u[jax.process_index()*sample_size:(jax.process_index()+1)*sample_size] + t_0
    return t, u0


def score_loss(key, model, params, state, batch, t_0: float = 0.0, t_1: float = 1.0, train: bool = False):
    k_t, k_eps, k_model = random.split(key, 3)

    data = batch["image"]  # [B,H,W,C]
    labels = batch.get("label", None)  # Optional: [B] or one-hot

    sample_size = data.shape[0]
    t, next_sampler_state = sample_time(sample_size, state, t_0, t_1)
    t = jnp.expand_dims(t, (1, 2, 3))

    epsilon = random.normal(k_eps[0], data.shape)
    x_t = q_t(batch, t, epsilon)  # xt = qt(x)

    sdlogqdx = get_sdlogqdx_fn(model, params, train=train)
    score = sdlogqdx(t, x_t, labels=labels, rng=k_model)

    loss = jnp.mean(jnp.sum((epsilon + score) ** 2, axis=(1, 2, 3)))
    return loss.mean(), next_sampler_state

def get_step_fn(model, optimizer, loss_fn):

  def step_fn(carry_state, batch):
    (key, state) = carry_state
    key, iter_key = jax.random.split(key)
    grad_fn = jax.value_and_grad(loss_fn, argnums=1, has_aux=True)
    (loss, new_sampler_state), grad = grad_fn(iter_key, model,  state.model_params, state.sampler_state, batch)
    grad = jax.lax.pmean(grad, axis_name='batch')
    updates, opt_state = optimizer.update(grad, state.opt_state, state.model_params)
    new_params = optax.apply_updates(state.model_params, updates)
    new_params_ema = jax.tree_map(
      lambda p_ema, p: p_ema * state.ema_rate + p * (1. - state.ema_rate),
      state.params_ema, new_params
    )
    new_state = state.replace(
      step=state.step+1,
      opt_state=opt_state,
      sampler_state=new_sampler_state,
      model_params=new_params,
      params_ema=new_params_ema
    )

    loss = jax.lax.pmean(loss, axis_name='batch')
    new_carry_state = (key, new_state)
    return new_carry_state, loss

  return step_fn


def stack_imgs(x, n=8, m=8):
    im_size = x.shape[2]
    big_img = np.zeros((n*im_size,m*im_size,x.shape[-1]),dtype=np.uint8)
    for i in range(n):
        for j in range(m):
            p = x[i*m+j] * 255
            p = p.clip(0, 255).astype(np.uint8)
            big_img[i*im_size:(i+1)*im_size, j*im_size:(j+1)*im_size, :] = p
    return big_img


# -------------------------
# 5) Batch helpers
# -------------------------
def iterate_array_batches(images, labels, batch_size, key):
    N = images.shape[0]
    assert labels is None or labels.shape[0] == N
    perm_key, = random.split(key, 1)
    idx = random.permutation(perm_key, N)
    for start in range(0, N, batch_size):
        sel = idx[start:start+batch_size]
        batch = {"image": images[sel]}
        if labels is not None:
            batch["label"] = labels[sel]
        yield batch

def to_device(iterable):
    for b in iterable:
        yield jax.device_put(jnp.asarray(b))



# -------------------------
# 6) The main training loop
# -------------------------
# def train_model(key, model, dataset, *, batch_size=64, epochs=100, lr=2e-4,
#                 t_0=0.0, t_1=1.0, model_name="Score-UNet"):
#     """
#     dataset can be (i) an iterator yielding {'image', 'label'} dicts, or
#     (ii) a tuple (images, labels) arrays.
#     """
#     print(f"--- Training {model_name} ---")
#     # Peek a batch to infer shapes for init
#     if isinstance(dataset, tuple):
#         images, labels = dataset
#         H, W, C = images.shape[1], images.shape[2], images.shape[3]
#         init_batch = {"image": images[:1], "label": None if labels is None else labels[:1]}
#         batch_iter_factory = lambda k: iterate_array_batches(images, labels, batch_size, k)
#     else:
#         # Assume iterable of dicts; grab one then rebuild iterator if needed
#         first = next(iter(dataset))
#         H, W, C = first["image"].shape[1], first["image"].shape[2], first["image"].shape[3]
#         init_batch = first
#         def batch_iter_factory(k):
#             # If dataset is a re-iterable (e.g., DataLoader), just return iter(dataset)
#             return iter(dataset)
#
#     # Init optimizer/state
#     init_key, key = random.split(key)
#     optimizer = optax.adam(lr)
#     # Model expects (t, x, [labels]) like the wrapper above
#     params = model.init(init_key,
#                         jnp.ones((1, 1, 1, 1), dtype=jnp.float32),   # t
#                         jnp.ones((1, H, W, C), dtype=jnp.float32),    # x
#                         init_batch.get("label", None)                 # labels if conditional
#                         )
#     state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=optimizer)
#
#     losses = []
#     for epoch in (pbar := trange(epochs, desc=f"{model_name}")):
#         epoch_losses = []
#         # fresh iterator each epoch (and fresh permutation for array-backed dataset)
#         key, epoch_key = random.split(key)
#         batch_iter = batch_iter_factory(epoch_key)
#
#         for batch in batch_iter:
#             key, step_key = random.split(key)
#             state, loss = train_step(state, batch, step_key, model, t_0=t_0, t_1=t_1)
#             epoch_losses.append(loss)
#
#         avg = float(np.mean(jax.device_get(jnp.array(epoch_losses))))
#         losses.append(avg)
#         pbar.set_postfix_str(f"loss={avg:.4f}")
#
#     print(f"Training complete for {model_name}.")
#     return state, losses