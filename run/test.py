from diffusion.sampling import reverse_sde
from train.train_mlp import train_mlp
from models.MLP import GaussianMLP
import optax
from flax.training import train_state
from diffusion.equations import *
from config import Config
from jax import random

from utils.viz import visualize_forward_and_reverse_diffusion, plot_loss

key = random.PRNGKey(Config.SEED)
key, init_key = random.split(key)

from datasets.Gaussians import GaussianDataset
gaussian_dataset = GaussianDataset()
sample_data = gaussian_dataset.sample_data()
datapoints, dimension = sample_data.shape

timesteps = np.linspace(0.0, 1.0, 6)
x_ts = forward_diffusion_over_time(gaussian_dataset.key, sample_data, timesteps)
x_t = last_step_in_forward_diffusion(gaussian_dataset.key, sample_data, timesteps)

model = GaussianMLP(num_hid=512, num_out=2)
optimizer = optax.adam(learning_rate=2e-4)
state = train_state.TrainState.create(
    apply_fn=model.apply,
    params=model.init(init_key, np.ones((512, 1)), x_t),
    tx=optimizer
)

loss, state = train_mlp(key, state, sample_data)
plot_loss(loss)
trajectory = reverse_sde(state, sample_data, key)
visualize_forward_and_reverse_diffusion(x_ts, trajectory)
print('Done.')