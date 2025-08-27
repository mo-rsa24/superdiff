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

ul, ur, lr, ll = gaussian_dataset.get_quadrants(subsample=True)
visualize_forward_diffusion_process_of_samples_over_time(key, ul, ul.shape)
visualize_forward_diffusion_process_of_samples_over_time(key, ur, ur.shape)
visualize_forward_diffusion_process_of_samples_over_time(key, ll, ll.shape)
visualize_forward_diffusion_process_of_samples_over_time(key, lr, lr.shape)

ul_steps = forward_diffusion_over_time(gaussian_dataset.key, ul)
ul_forward = forward_diffusion_over_time(gaussian_dataset.key, ul)[-1]

ur_steps = forward_diffusion_over_time(gaussian_dataset.key, ur)
ur_forward = forward_diffusion_over_time(gaussian_dataset.key, ur)[-1]

ll_steps = forward_diffusion_over_time(gaussian_dataset.key, ll)
ll_forward = forward_diffusion_over_time(gaussian_dataset.key, ll)[-1]

lr_steps = forward_diffusion_over_time(gaussian_dataset.key, lr)
lr_forward = forward_diffusion_over_time(gaussian_dataset.key, lr)[-1]



model_ul = GaussianMLP(num_hid=512, num_out=2)
model_ur = GaussianMLP(num_hid=512, num_out=2)
model_ll = GaussianMLP(num_hid=512, num_out=2)
model_lr = GaussianMLP(num_hid=512, num_out=2)

ul_loss, ul_state = train_group(model_ul, ul, ur_forward)
ul_trajectory = reverse_sde(ul_state, ul, key)
visualize_forward_and_reverse_diffusion(ul_steps, ul_trajectory)

ur_loss, ur_state = train_group(model_ur, ur, ur_forward)
ur_trajectory = reverse_sde(ur_state, ur, key)
visualize_forward_and_reverse_diffusion(ur_steps, ur_trajectory)

lr_loss, lr_state = train_group(model_lr, lr, ur_forward)
lr_trajectory = reverse_sde(lr_state, lr, key)
visualize_forward_and_reverse_diffusion(lr_steps, lr_trajectory)

ll_loss, ll_state = train_group(model_ll, ll, ur_forward)
ll_trajectory = reverse_sde(ll_state, ll, key)
visualize_forward_and_reverse_diffusion(ll_steps, ll_trajectory)


trajectory, log_likelihood_model_a, log_likelihood_model_b = compose_and_estimate_log_likelihood_along_superposed_trajectory(ul_state, ur_state, key)
visualize_log_likelihood_along_superposed_trajectory(log_likelihood_model_a, log_likelihood_model_b)
visualize_composition(trajectory, ul_steps, ur_steps)


trajectory, log_likelihood_model_a, log_likelihood_model_b = compose_and_estimate_log_likelihood_along_superposed_trajectory(ll_state, lr_state, key)
visualize_log_likelihood_along_superposed_trajectory(log_likelihood_model_a, log_likelihood_model_b)
visualize_composition(trajectory, ll_state, lr_state)


trajectory, log_likelihood_model_a, log_likelihood_model_b = compose_and_estimate_log_likelihood_along_superposed_trajectory(ul_state, ll_state, key)
visualize_log_likelihood_along_superposed_trajectory(log_likelihood_model_a, log_likelihood_model_b)
visualize_composition(trajectory, ul_steps, ll_steps)


trajectory, log_likelihood_model_a, log_likelihood_model_b = compose_and_estimate_log_likelihood_along_superposed_trajectory(ur_state, lr_state, key)
visualize_log_likelihood_along_superposed_trajectory(log_likelihood_model_a, log_likelihood_model_b)
visualize_composition(trajectory, ur_steps, lr_steps)