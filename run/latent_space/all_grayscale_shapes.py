from torch.utils.data import DataLoader
from datasets.Shapes import GrayscaleShapesDataset, get_batches, get_samples
from diffusion.sampling import reverse_sde, compose_and_estimate_log_likelihood_along_superposed_trajectory
from models.Transform import GrayscalePCA, get_latent_codes
from models.MLP import GrayscaleLatentMLP
from diffusion.equations import *
from config import Config
from jax import random
from train.train_shape_mlp import train_expert
from utils.viz import visualize_forward_and_reverse_diffusion, \
    visualize_log_likelihood_along_superposed_trajectory, \
    visualize_compositions, \
    plot_loss, \
    visualize_forward_diffusion_process_of_all_groups_over_time, visualize_forward_and_reverse_diffusion_on_all_latents

import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')

shapes = ("circle", "square", "triangle")
shape_to_idx = {s: i for i, s in enumerate(shapes)}
idx_to_shape = {i: s for i, s in shape_to_idx.items()}

key = random.PRNGKey(Config.SEED)
key, circle_key, square_key, triangle_key = random.split(key, 4)

dataset = GrayscaleShapesDataset(shapes=shapes)
batch_size = 512
epochs = 600
dataloader = DataLoader(dataset, batch_size=len(dataset))
method = "vae"
vae_training_params = {'epochs': 600, 'beta': 4.0}
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

triangles = latent_codes[labels == shape_to_idx["triangle"]]
triangle_samples = get_samples(triangle_key, triangles, 512)

timesteps = np.linspace(0.0, 1.0, 6)

circle_forward_steps = forward_diffusion_over_time(key, circle_samples, timesteps)
square_forward_steps = forward_diffusion_over_time(key, square_samples, timesteps)
triangle_forward_steps = forward_diffusion_over_time(key, triangle_samples, timesteps)
steps = [circle_forward_steps, square_forward_steps, triangle_forward_steps]

visualize_forward_diffusion_process_of_all_groups_over_time(steps, shapes, lim=(-5,5), title="Forward Diffusion Over All Shapes")

circle_model = GrayscaleLatentMLP(num_hid = 512, num_out = 2)
circle_state, circle_loss = train_expert(circle_key, circle_model, circles, epochs=epochs, model_name="Circle Model")
circle_trajectory = reverse_sde(circle_state, circle_samples, circle_key, dt=1e-4)

square_model = GrayscaleLatentMLP(num_hid = 512, num_out = 2)
square_state, square_loss  = train_expert(square_key, square_model, squares, epochs=epochs, model_name="Square Model")
square_trajectory = reverse_sde(square_state, square_samples, square_key, dt=1e-4)

triangle_model = GrayscaleLatentMLP(num_hid = 512, num_out = 2)
triangle_state, triangle_loss  = train_expert(triangle_key, triangle_model, triangles, epochs=epochs, model_name="Triangle Model")
triangle_trajectory = reverse_sde(triangle_state, triangle_samples, triangle_key, dt=1e-4)

trajectories = [circle_trajectory, square_trajectory, triangle_trajectory]
visualize_forward_and_reverse_diffusion_on_all_latents(steps, trajectories, shapes, lim=(-5,5), title="Forward & Reverse Diffusion Over All Shapes")

# Between Squares And Circles
trajectory, log_likelihood_model_a, log_likelihood_model_b = compose_and_estimate_log_likelihood_along_superposed_trajectory(circle_state, square_state, key, dt=1e-4)
visualize_log_likelihood_along_superposed_trajectory(log_likelihood_model_a, log_likelihood_model_b)
visualize_compositions(trajectory, steps, shapes, lim=(-5,5), title="Sampling From An Iso-Surface Of Squares & Circles")

# Between Triangles And Circles
trajectory, log_likelihood_model_a, log_likelihood_model_b = compose_and_estimate_log_likelihood_along_superposed_trajectory(triangle_state, circle_state, key, dt=1e-4)
visualize_log_likelihood_along_superposed_trajectory(log_likelihood_model_a, log_likelihood_model_b)
visualize_compositions(trajectory, steps, shapes, lim=(-5,5), title="Sampling From An Iso-Surface Of Triangles & Circles")


# Between Triangles And Squares
trajectory, log_likelihood_model_a, log_likelihood_model_b = compose_and_estimate_log_likelihood_along_superposed_trajectory(triangle_state, square_state, key, dt=1e-4)
visualize_log_likelihood_along_superposed_trajectory(log_likelihood_model_a, log_likelihood_model_b)
visualize_compositions(trajectory, steps, shapes, lim=(-5,5), title="Sampling From An Iso-Surface Of Triangles & Squares")