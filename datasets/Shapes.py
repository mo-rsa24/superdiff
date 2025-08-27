from typing import Tuple

import torch
from jax import random
from torch.utils.data import Dataset
from torchvision.transforms import Compose, ToTensor, Lambda, Grayscale
from PIL import Image, ImageDraw, ImageFilter
import numpy as np, tensorflow as tf, tensorflow_datasets as tfds
import random as pyrandom

class GrayscaleShapesDataset(Dataset):
    """
        Generates grayscale images of simple shapes.
    """
    def __init__(self, shapes, size=10000, img_size=64, location_variation: bool = False):
        self.size = size
        self.img_size = img_size
        self.shapes = shapes
        self.shape_to_idx = {s: i for i, s in enumerate(self.shapes)}
        self.location_variation = location_variation
        self.transform = Compose([
            ToTensor(),
            Lambda(lambda t: (t * 2) - 1)  # Scale to [-1, 1]
        ])

    def _draw_shape(self, shape, draw):
        margin = self.img_size // 4
        top_left = (margin, margin)
        bottom_right = (self.img_size - margin, self.img_size - margin)
        if shape == "circle":
            draw.ellipse([top_left, bottom_right], fill="white")
        elif shape == "square":
            draw.rectangle([top_left, bottom_right], fill="white")
        elif shape == "triangle":
            p1 = (self.img_size // 2, margin)
            p2 = (margin, self.img_size - margin)
            p3 = (self.img_size - margin, self.img_size - margin)
            draw.polygon([p1, p2, p3], fill="white")

    def _draw_shape_with_location_variation(self, shape, draw):
        margin = self.img_size // 4
        dx = pyrandom.randint(-margin // 2, margin // 2)
        dy = pyrandom.randint(-margin // 2, margin // 2)

        top_left = (margin + dx, margin + dy)
        bottom_right = (self.img_size - margin + dx, self.img_size - margin + dy)

        if shape == "circle":
            draw.ellipse([top_left, bottom_right], fill="white")
        elif shape == "square":
            draw.rectangle([top_left, bottom_right], fill="white")
        elif shape == "triangle":
            p1 = (self.img_size // 2 + dx, margin + dy)
            p2 = (margin + dx, self.img_size - margin + dy)
            p3 = (self.img_size - margin + dx, self.img_size - margin + dy)
            draw.polygon([p1, p2, p3], fill="white")

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        shape_name = self.shapes[idx % len(self.shapes)]
        shape_label = torch.tensor(self.shape_to_idx[shape_name])

        # Create a grayscale ('L') image
        image = Image.new("L", (self.img_size, self.img_size), "black")
        draw = ImageDraw.Draw(image)
        if self.location_variation:
            self._draw_shape_with_location_variation(shape_name, draw)
        else:
            self._draw_shape(shape_name, draw)

        return self.transform(image), shape_label

class ShapesDataset(Dataset):
    """Generates images of simple shapes with specified colors on the fly."""

    def __init__(self, size=5000, img_size=64, mode='rgb'):
        self.size = size
        self.img_size = img_size
        self.mode = mode  # 'rgb', 'shape', or 'color'

        self.shapes = ["circle", "square", "triangle"]
        self.colors = ["red", "green", "blue"]

        self.all_combinations = [(s, c) for s in self.shapes for c in self.colors]

        self.transform_rgb = Compose([
            ToTensor(),
            Lambda(lambda t: (t * 2) - 1)  # Scale to [-1, 1]
        ])

        self.transform_shape = Compose([
            Grayscale(num_output_channels=1),
            ToTensor(),
            Lambda(lambda t: (t * 2) - 1)
        ])

    def __len__(self):
        return self.size

    def _draw_shape(self, shape, color_name, draw):
        margin = self.img_size // 4
        top_left, bottom_right = (margin, margin), (self.img_size - margin, self.img_size - margin)
        if shape == "circle":
            draw.ellipse([top_left, bottom_right], fill=color_name)
        elif shape == "square":
            draw.rectangle([top_left, bottom_right], fill=color_name)
        elif shape == "triangle":
            p1 = (self.img_size // 2, margin)
            p2 = (margin, self.img_size - margin)
            p3 = (self.img_size - margin, self.img_size - margin)
            draw.polygon([p1, p2, p3], fill=color_name)

    def __getitem__(self, idx):
        shape_name, color_name = self.all_combinations[idx % len(self.all_combinations)]

        # Create the base RGB image
        image = Image.new("RGB", (self.img_size, self.img_size), "black")
        draw = ImageDraw.Draw(image)
        self._draw_shape(shape_name, color_name, draw)

        if self.mode == 'shape':
            # Return grayscale version for shape-only training
            return self.transform_shape(image)

        elif self.mode == 'color':
            # For color training, we destroy shape info by blurring heavily
            # This creates a "color blob"
            blob_image = image.filter(ImageFilter.GaussianBlur(radius=self.img_size / 4))
            return self.transform_rgb(blob_image)

        else:  # self.mode == 'rgb'
            # Return the original colored shape
            return self.transform_rgb(image)

def get_batches(array, batch_size, seed, dataset_size: int = 10_000):
    dataset = tf.data.Dataset.from_tensor_slices(array.astype(np.float32))
    dataset = dataset.shuffle(buffer_size=min(len(array), dataset_size), seed=seed, reshuffle_each_iteration=True)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return tfds.as_numpy(dataset)

def get_samples(key, shape_latent_codes, datapoints):
    indices = random.choice(key, shape_latent_codes.shape[0], (datapoints,), replace=False)
    return shape_latent_codes[indices]
