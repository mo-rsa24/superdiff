import numpy as np
from typing import Tuple, Optional
from jax import random
import matplotlib.pyplot as plt
from tensorflow.python.data.ops.prefetch_op import _PrefetchDataset
import tensorflow as tf
from datasets.Gaussians import GaussianDataset
from diffusion.equations import q_t
from utils.image_manipulation import rescale_for_visualization


def scatter_plot(data, figsize: Tuple[int,int] =(23,5)):
    plt.figure(figsize=figsize)
    plt.scatter(data[:, 0], data[:, 1], alpha=0.3)
    plt.xlim(-3, 3)
    plt.ylim(-3, 3)
    plt.grid()
    plt.show()

def plot_loss(loss, figsize: Tuple[int,int] =(23,5)):
    plt.figure(figsize=figsize)
    plt.plot(loss)
    plt.grid()
    plt.show()


def visualize_forward_diffusion_process_of_samples_over_time(key, sample_data, shape: Tuple[int, int], figsize: Tuple[int,int] =(23, 5), num_axis: int = 6):
    t_axis = np.linspace(0.0, 1.0, 6)
    plt.figure(figsize=figsize)
    for i in range(num_axis):
        plt.subplot(1, num_axis, i+1) # 1 row, 6 columns
        _, *subkeys = random.split(key, 3)
        standard_noise = random.normal(key, shape=shape)
        x_t = q_t(sample_data, t_axis[i], standard_noise)
        plt.scatter(x_t[:, 0], x_t[:, 1], alpha=0.3)
        plt.title(f't={t_axis[i]}')
        plt.xlim(-3, 3)
        plt.ylim(-3, 3)
        plt.grid()
    plt.show()

def visualize_forward_diffusion_process_groups_over_time(sample_1, sample_2, figsize: Tuple[int,int] =(23, 5), lim=(-3, 3)):
    """
    sample_1, sample_2: A list of samples that have been diffused over time
    Example:  [first_batch_of_samples_at_time_0, first_batch_of_samples_at_time_1, ..., first_batch_of_samples_at_time_6] each of size (512,2)
    Prerequisite: Call forward_diffusion_over_time()
    """
    timesteps: np.ndarray = np.linspace(0.0, 1.0, np.array(len(sample_1)))
    plt.figure(figsize=figsize)
    for (i, timestep) in enumerate(timesteps):
        plt.subplot(1, len(timesteps), i+1)
        plt.scatter(sample_1[i][:, 0], sample_1[i][:, 1], label='Sample 1', s=10, )
        plt.scatter(sample_2[i][:, 0], sample_2[i][:, 1], label='Sample 2', s=10, )
        plt.title(f't={timestep}')
        plt.xlim(lim)
        plt.ylim(lim)
        plt.grid()
        if i == 0:
            plt.legend(fontsize=15)
    plt.show()

def visualize_forward_diffusion_process_of_all_groups_over_time(samples, labels, figsize: Tuple[int,int] =(23, 5), lim=(-3, 3), title: str = "Forward Diffusion Over All Latents"):
    """
    sample_1, sample_2: A list of samples that have been diffused over time
    Example:  [first_batch_of_samples_at_time_0, first_batch_of_samples_at_time_1, ..., first_batch_of_samples_at_time_6] each of size (512,2)
    Prerequisite: Call forward_diffusion_over_time()
    """
    timesteps: np.ndarray = np.linspace(0.0, 1.0, np.array(len(samples[0])))
    plt.figure(figsize=figsize)
    for (i, timestep) in enumerate(timesteps):
        plt.subplot(1, len(timesteps), i+1)
        for j, (shape_samples, label) in enumerate(zip(samples, labels)):
            plt.scatter(shape_samples[i][:, 0], shape_samples[i][:, 1], label=f'{label}', s=10, )
        plt.title(f't={timestep}')
        plt.xlim(lim)
        plt.ylim(lim)
        plt.grid()
        if i == 0:
            plt.legend(fontsize=15)
    plt.suptitle(title, fontsize=20)
    plt.show()

def visualize_forward_and_reverse_diffusion(sample_data, generated_data, figsize: Tuple[int,int] =(23, 5), lim=(-3, 3)):
    timesteps: np.ndarray = np.linspace(0.0, 1.0, np.array(len(sample_data)))
    gen_num_timesteps = generated_data.shape[1]-1
    plt.figure(figsize=figsize)
    for (i, timestep) in enumerate(timesteps):
        plt.subplot(1, len(timesteps), i+1)
        reverse_index = len(timesteps) - 1 - i
        t = timesteps[reverse_index]
        plt.scatter(sample_data[reverse_index][:, 0], sample_data[reverse_index][:, 1], label='Noise Data', s=10, )
        plt.scatter(generated_data[:, int(gen_num_timesteps*(timesteps[i])),0], generated_data[:, int(gen_num_timesteps*(timesteps[i])),1], label='Generated Data',s=10, alpha=0.5)
        plt.title(f't={t}')
        plt.xlim(lim)
        plt.ylim(lim)
        plt.grid()
        if i == 0:
            plt.legend(fontsize=15)
    plt.show()

def visualize_forward_and_reverse_diffusion_on_all_latents(samples, generated_data, labels, figsize: Tuple[int,int] =(23, 5), lim=(-3, 3), title: str = "Forward & Reverse Diffusion Over All Latents"):
    timesteps: np.ndarray = np.linspace(0.0, 1.0, np.array(len(samples[0])))
    gen_num_timesteps = generated_data[0].shape[1]-1
    plt.figure(figsize=figsize)
    for (i, timestep) in enumerate(timesteps):
        plt.subplot(1, len(timesteps), i+1)
        reverse_index = len(timesteps) - 1 - i
        t = timesteps[reverse_index]
        for j, (label, sub_samples, gen_data) in enumerate(zip(labels, samples, generated_data)):
            plt.scatter(sub_samples[reverse_index][:, 0], sub_samples[reverse_index][:, 1], label=label, s=10, )
            plt.scatter(gen_data[:, int(gen_num_timesteps*(timesteps[i])),0], gen_data[:, int(gen_num_timesteps*(timesteps[i])),1], label=f'{label} gen data',s=10, alpha=0.5)
        plt.title(f't={t}')
        plt.xlim(lim)
        plt.ylim(lim)
        plt.grid()
        if i == 0:
            plt.legend(fontsize=15)
    plt.suptitle(title, fontsize=20)
    plt.show()


def forward_schedule_(dataset: Optional[GaussianDataset],  figsize: Tuple[int,int] =(23,5), num_axis: int = 6):
    t_axis = np.linspace(0.0, 1.0, 6)
    key = dataset.key
    sample_data = dataset.sample_data()
    plt.figure(figsize=figsize)
    for i in range(num_axis):
        plt.subplot(1, num_axis, i+1) # 1 row, 6 columns
        _, *subkeys = random.split(key, 3)
        standard_noise = random.normal(key, shape=dataset.shape)
        x_t = q_t(sample_data, t_axis[i], standard_noise)
        plt.scatter(x_t[:, 0], x_t[:, 1], alpha=0.3)
        plt.title(f't={t_axis[i]}')
        plt.xlim(-3, 3)
        plt.ylim(-3, 3)
        plt.grid()
    plt.show()

def visualize_log_likelihood_along_superposed_trajectory(log_likelihood_a, log_likelihood_b):
    plt.plot((log_likelihood_a - log_likelihood_b)[:20, :].T)
    plt.grid()
    plt.show()

def visualize_composition(superposed_trajectory, sample_data_steps_1, sample_data_steps_2, sample_data_steps_1_label: str = "Up Noise", sample_data_steps_2_label: str = "Down Noise", figsize: Tuple[int,int] =(23,5), num_axis: int = 6):
    timesteps: np.ndarray = np.linspace(0.0, 1.0, np.array(len(sample_data_steps_1)))
    gen_num_timesteps = superposed_trajectory.shape[1] - 1
    plt.figure(figsize=figsize)
    for i in range(num_axis):
        plt.subplot(1, num_axis, i + 1)  # 1 row, 6 columns
        reverse_index = len(timesteps) - 1 - i
        t = timesteps[reverse_index]
        plt.scatter(sample_data_steps_1[reverse_index][:, 0], sample_data_steps_1[reverse_index][:, 1], label=sample_data_steps_1_label, s=10, )
        plt.scatter(sample_data_steps_2[reverse_index][:, 0], sample_data_steps_2[reverse_index][:, 1], label=sample_data_steps_2_label, s=10, )
        plt.scatter(superposed_trajectory[:, int(gen_num_timesteps * (timesteps[i])), 0], superposed_trajectory[:, int(gen_num_timesteps* (timesteps[i])), 1], label='Composition')
        plt.title(f't={t}')
        plt.xlim(-3, 3)
        plt.ylim(-3, 3)
        plt.grid()
        if i == 0:
            plt.legend(fontsize=15)
    plt.show()

def visualize_compositions(superposed_trajectory, samples, labels, figsize: Tuple[int,int] =(23,5), num_axis: int = 6, lim=(-3,3), title: str = "Sampling From An Iso-Surface Of All Latents"):
    timesteps: np.ndarray = np.linspace(0.0, 1.0, np.array(len(samples[0])))
    gen_num_timesteps = superposed_trajectory.shape[1] - 1
    plt.figure(figsize=figsize)
    for i in range(num_axis):
        plt.subplot(1, num_axis, i + 1)  # 1 row, 6 columns
        reverse_index = len(timesteps) - 1 - i
        t = timesteps[reverse_index]
        for j, (shape_samples, label) in enumerate(zip(samples, labels)):
            plt.scatter(shape_samples[reverse_index][:, 0], shape_samples[reverse_index][:, 1], label=label, s=10, )
        plt.scatter(superposed_trajectory[:, int(gen_num_timesteps * (timesteps[i])), 0], superposed_trajectory[:, int(gen_num_timesteps* (timesteps[i])), 1], label='Composition')
        plt.title(f't={t}')
        plt.xlim(lim)
        plt.ylim(lim)
        plt.grid()
        if i == 0:
            plt.legend(fontsize=15)
    plt.suptitle(title, fontsize=20)
    plt.show()

def plot_grid(train: _PrefetchDataset):
     data = next(iter(train))
     imgs = data['image']  # shape: (1, 128, 32, 32, 3)
     labels = data['label']  # shape: (1, 128) if present
     imgs = rescale_for_visualization(imgs)
     grid = imgs[:64]
     rows, cols = 8, 8
     fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.6, rows * 1.6))
     axes = axes.ravel()

     for i in range(rows * cols):
         axes[i].imshow(grid[i])
         axes[i].axis('off')
         if labels is not None:
            axes[i].set_title(int(tf.squeeze(labels)[i]), fontsize=8)
     plt.tight_layout()
     plt.show()