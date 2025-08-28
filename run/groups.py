from diffusion.sampling import reverse_sde, compose_and_estimate_log_likelihood_along_superposed_trajectory
from train.train_mlp import train_group
from models.MLP import GaussianMLP
from diffusion.equations import *
from config import Config
from jax import random

from utils.viz import visualize_forward_and_reverse_diffusion, \
    visualize_log_likelihood_along_superposed_trajectory, \
    visualize_forward_diffusion_process_of_samples_over_time, visualize_composition, \
    visualize_forward_diffusion_process_of_all_groups_over_time, visualize_forward_and_reverse_diffusion_on_all_latents, \
    visualize_compositions

key = random.PRNGKey(Config.SEED)
key, init_key = random.split(key)
timesteps = np.linspace(0.0, 1.0, 6)

from datasets.Gaussians import GaussianDataset


gaussian_dataset = GaussianDataset()

ul, ur, lr, ll = gaussian_dataset.get_quadrants(subsample=True)
datapoints, coordinates = ul.shape
t_shape = (datapoints, 1)

ul_steps = forward_diffusion_over_time(gaussian_dataset.key, ul)
ul_forward = forward_diffusion_over_time(gaussian_dataset.key, ul)[-1]

ur_steps = forward_diffusion_over_time(gaussian_dataset.key, ur)
ur_forward = forward_diffusion_over_time(gaussian_dataset.key, ur)[-1]

ll_steps = forward_diffusion_over_time(gaussian_dataset.key, ll)
ll_forward = forward_diffusion_over_time(gaussian_dataset.key, ll)[-1]

lr_steps = forward_diffusion_over_time(gaussian_dataset.key, lr)
lr_forward = forward_diffusion_over_time(gaussian_dataset.key, lr)[-1]
labels = ["Upper Left", "Upper Right", "Lower Right", "Lower Left"]
samples = [ul_steps, ur_steps, lr_steps, ll_steps]
visualize_forward_diffusion_process_of_all_groups_over_time(samples, labels)



model_ul = GaussianMLP(num_hid=datapoints, num_out=coordinates)
model_ur = GaussianMLP(num_hid=datapoints, num_out=coordinates)
model_ll = GaussianMLP(num_hid=datapoints, num_out=coordinates)
model_lr = GaussianMLP(num_hid=datapoints, num_out=coordinates)

ul_loss, ul_state = train_group(model_ul, ul, ul_forward, t_shape=t_shape)
ul_trajectory = reverse_sde(ul_state, ul, key)

ur_loss, ur_state = train_group(model_ur, ur, ur_forward, t_shape=t_shape)
ur_trajectory = reverse_sde(ur_state, ur, key)

lr_loss, lr_state = train_group(model_lr, lr, lr_forward, t_shape=t_shape)
lr_trajectory = reverse_sde(lr_state, lr, key)

ll_loss, ll_state = train_group(model_ll, ll, ll_forward, t_shape=t_shape)
ll_trajectory = reverse_sde(ll_state, ll, key)

trajectories = [ul_trajectory, ur_trajectory, lr_trajectory, ll_trajectory]
visualize_forward_and_reverse_diffusion_on_all_latents(samples, trajectories, labels, title="Forward & Reverse Diffusion Over Samples")

# Simple Average
trajectory, log_likelihood_model_a, log_likelihood_model_b = compose_and_estimate_log_likelihood_along_superposed_trajectory(ul_state, ur_state, key, kappa='simple_average')
visualize_log_likelihood_along_superposed_trajectory(log_likelihood_model_a, log_likelihood_model_b)
visualize_compositions(trajectory, samples, labels, title="Sampling From An Iso-Surface Of Upper Left & Upper Right")

trajectory, log_likelihood_model_a, log_likelihood_model_b = compose_and_estimate_log_likelihood_along_superposed_trajectory(ll_state, lr_state, key, kappa='simple_average')
visualize_log_likelihood_along_superposed_trajectory(log_likelihood_model_a, log_likelihood_model_b)
visualize_compositions(trajectory, samples, labels, title="Sampling From An Iso-Surface Of Lower Left & Lower Right")

trajectory, log_likelihood_model_a, log_likelihood_model_b = compose_and_estimate_log_likelihood_along_superposed_trajectory(ul_state, ll_state, key, kappa='simple_average')
visualize_log_likelihood_along_superposed_trajectory(log_likelihood_model_a, log_likelihood_model_b)
visualize_compositions(trajectory, samples, labels, title="Sampling From An Iso-Surface Of Upper Left & Lower Left")

trajectory, log_likelihood_model_a, log_likelihood_model_b = compose_and_estimate_log_likelihood_along_superposed_trajectory(ur_state, lr_state, key, kappa='simple_average')
visualize_log_likelihood_along_superposed_trajectory(log_likelihood_model_a, log_likelihood_model_b)
visualize_compositions(trajectory, samples, labels, title="Sampling From An Iso-Surface Of Upper Right & Lower Right")

# Proportional Average
trajectory, log_likelihood_model_a, log_likelihood_model_b = compose_and_estimate_log_likelihood_along_superposed_trajectory(ul_state, ur_state, key, kappa='proportional_average')
visualize_log_likelihood_along_superposed_trajectory(log_likelihood_model_a, log_likelihood_model_b)
visualize_compositions(trajectory, samples, labels, title="Sampling From An Iso-Surface Of Upper Left & Upper Right")

trajectory, log_likelihood_model_a, log_likelihood_model_b = compose_and_estimate_log_likelihood_along_superposed_trajectory(ll_state, lr_state, key, kappa='proportional_average')
visualize_log_likelihood_along_superposed_trajectory(log_likelihood_model_a, log_likelihood_model_b)
visualize_compositions(trajectory, samples, labels, title="Sampling From An Iso-Surface Of Lower Left & Lower Right")

trajectory, log_likelihood_model_a, log_likelihood_model_b = compose_and_estimate_log_likelihood_along_superposed_trajectory(ul_state, ll_state, key, kappa='proportional_average')
visualize_log_likelihood_along_superposed_trajectory(log_likelihood_model_a, log_likelihood_model_b)
visualize_compositions(trajectory, samples, labels, title="Sampling From An Iso-Surface Of Upper Left & Lower Left")

trajectory, log_likelihood_model_a, log_likelihood_model_b = compose_and_estimate_log_likelihood_along_superposed_trajectory(ur_state, lr_state, key, kappa='proportional_average')
visualize_log_likelihood_along_superposed_trajectory(log_likelihood_model_a, log_likelihood_model_b)
visualize_compositions(trajectory, samples, labels, title="Sampling From An Iso-Surface Of Upper Right & Lower Right")

# Iso-Surface
trajectory, log_likelihood_model_a, log_likelihood_model_b = compose_and_estimate_log_likelihood_along_superposed_trajectory(ul_state, ur_state, key)
visualize_log_likelihood_along_superposed_trajectory(log_likelihood_model_a, log_likelihood_model_b)
visualize_compositions(trajectory, samples, labels, title="Sampling From An Iso-Surface Of Upper Left & Upper Right")

trajectory, log_likelihood_model_a, log_likelihood_model_b = compose_and_estimate_log_likelihood_along_superposed_trajectory(ll_state, lr_state, key)
visualize_log_likelihood_along_superposed_trajectory(log_likelihood_model_a, log_likelihood_model_b)
visualize_compositions(trajectory, samples, labels, title="Sampling From An Iso-Surface Of Lower Left & Lower Right")

trajectory, log_likelihood_model_a, log_likelihood_model_b = compose_and_estimate_log_likelihood_along_superposed_trajectory(ul_state, ll_state, key)
visualize_log_likelihood_along_superposed_trajectory(log_likelihood_model_a, log_likelihood_model_b)
visualize_compositions(trajectory, samples, labels, title="Sampling From An Iso-Surface Of Upper Left & Lower Left")

trajectory, log_likelihood_model_a, log_likelihood_model_b = compose_and_estimate_log_likelihood_along_superposed_trajectory(ur_state, lr_state, key)
visualize_log_likelihood_along_superposed_trajectory(log_likelihood_model_a, log_likelihood_model_b)
visualize_compositions(trajectory, samples, labels, title="Sampling From An Iso-Surface Of Upper Right & Lower Right")