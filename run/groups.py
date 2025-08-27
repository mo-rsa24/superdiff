from diffusion.sampling import reverse_sde, compose_and_estimate_log_likelihood_along_superposed_trajectory
from train.train_mlp import train_group
from models.MLP import GaussianMLP
from diffusion.equations import *
from config import Config
from jax import random

from utils.viz import visualize_forward_and_reverse_diffusion, \
    visualize_log_likelihood_along_superposed_trajectory, \
    visualize_forward_diffusion_process_of_samples_over_time, visualize_composition

key = random.PRNGKey(Config.SEED)
key, init_key = random.split(key)
timesteps = np.linspace(0.0, 1.0, 6)

from datasets.Gaussians import GaussianDataset


gaussian_dataset = GaussianDataset()

sample_data_up, sample_data_down = gaussian_dataset.get_groups(subsample=False)
visualize_forward_diffusion_process_of_samples_over_time(key, sample_data_up, sample_data_up.shape)
visualize_forward_diffusion_process_of_samples_over_time(key, sample_data_down, sample_data_down.shape)

up_steps = forward_diffusion_over_time(gaussian_dataset.key, sample_data_up)
up_forward = forward_diffusion_over_time(gaussian_dataset.key, sample_data_up)[-1]

down_steps = forward_diffusion_over_time(gaussian_dataset.key, sample_data_down)
down_forward = forward_diffusion_over_time(gaussian_dataset.key, sample_data_down)[-1]


model_upper_left = GaussianMLP(num_hid=512, num_out=2)
model_upper_right = GaussianMLP(num_hid=512, num_out=2)
model_lower_left = GaussianMLP(num_hid=512, num_out=2)
model_lower_right = GaussianMLP(num_hid=512, num_out=2)

up_loss, up_state = train_group(model_up, sample_data_up, up_forward)
up_trajectory = reverse_sde(up_state, sample_data_up, key)
visualize_forward_and_reverse_diffusion(up_steps, up_trajectory)

down_loss,down_state = train_group(model_down, sample_data_down, down_forward)
down_trajectory = reverse_sde(down_state, sample_data_down, key)
visualize_forward_and_reverse_diffusion(down_steps, down_trajectory)

trajectory, log_likelihood_model_a, log_likelihood_model_b = compose_and_estimate_log_likelihood_along_superposed_trajectory(up_state, down_state, key)
visualize_log_likelihood_along_superposed_trajectory(log_likelihood_model_a, log_likelihood_model_b)

visualize_composition(trajectory, up_steps, down_steps)