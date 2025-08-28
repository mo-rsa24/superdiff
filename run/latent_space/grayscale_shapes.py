from torch.utils.data import DataLoader
from datasets.Shapes import GrayscaleShapesDataset, get_batches, get_samples
from diffusion.sampling import reverse_sde, compose_and_estimate_log_likelihood_along_superposed_trajectory
from models.Transform import GrayscalePCA, get_latent_codes
from models.MLP import GrayscaleLatentMLP
from diffusion.equations import *
from config import Config
from jax import random

from train.train_shape_mlp import train_latent_shape_model, train_latent_shape_group, train_expert
from utils.viz import visualize_forward_and_reverse_diffusion, \
    visualize_log_likelihood_along_superposed_trajectory, \
    visualize_forward_diffusion_process_of_samples_over_time, visualize_composition, \
    visualize_forward_diffusion_process_groups_over_time, plot_loss

import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')

shapes = ("circle", "square")
shape_to_idx = {s: i for i, s in enumerate(shapes)}
idx_to_shape = {i: s for i, s in shape_to_idx.items()}

key = random.PRNGKey(Config.SEED)
key, circle_key, square_key = random.split(key, 3)

dataset = GrayscaleShapesDataset(shapes=shapes, location_variation=False, size_variation=False)
batch_size = 512
epochs = 500
dataloader = DataLoader(dataset, batch_size=len(dataset))
method = "umap"
vae_training_params = {'epochs': 25, 'beta': 4.0}
latent_codes_np, labels_np = get_latent_codes(
    method=method,
    dataset=dataset,
    n_components=2,
    vae_params=vae_training_params if method == 'vae' else None
)

latent_codes = jnp.array(latent_codes_np)
labels = labels_np

circles = latent_codes[labels == shape_to_idx["circle"]]
circle_samples = get_samples(circle_key, circles, 512)

squares = latent_codes[labels == shape_to_idx["square"]]
square_samples = get_samples(square_key, squares, 512)

timesteps = np.linspace(0.0, 1.0, 6)

circle_forward_steps = forward_diffusion_over_time(key, circle_samples, timesteps)
square_forward_steps = forward_diffusion_over_time(key, square_samples, timesteps)

visualize_forward_diffusion_process_groups_over_time(circle_forward_steps, square_forward_steps, lim=(-20, 20))

circle_dataset = get_batches(circles, batch_size, Config.SEED, dataset_size=len(circles))
square_dataset = get_batches(squares, batch_size, Config.SEED, dataset_size=len(squares))

circle_model = GrayscaleLatentMLP(num_hid = 512, num_out = 2)
circle_state, circle_loss = train_expert(circle_key, circle_model, circles, epochs=epochs, model_name="Circle Model")
circle_trajectory = reverse_sde(circle_state, circle_samples, key)
plot_loss(circle_loss)
visualize_forward_and_reverse_diffusion(circle_forward_steps, circle_trajectory, lim=(-20, 20))

square_model = GrayscaleLatentMLP(num_hid = 512, num_out = 2)
square_state, square_loss  = train_expert(square_key, square_model, squares, epochs=epochs, model_name="Square Model")
square_trajectory = reverse_sde(square_state, square_samples, key)
visualize_forward_and_reverse_diffusion(square_forward_steps, square_trajectory, lim=(-20, 20))

trajectory, log_likelihood_model_a, log_likelihood_model_b = compose_and_estimate_log_likelihood_along_superposed_trajectory(circle_state, square_state, key, dt=1e-3)
visualize_log_likelihood_along_superposed_trajectory(log_likelihood_model_a, log_likelihood_model_b)
visualize_composition(trajectory, circle_forward_steps, square_forward_steps)