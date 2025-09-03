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
    def __init__(self, shapes, size=10000, img_size=64, location_variation: bool = False,
                 size_variation: bool = False,
                 size_range: Tuple[float, float] = (0.8, 1.2),
                 ):
        self.size = size
        self.img_size = img_size
        self.shapes = shapes
        self.shape_to_idx = {s: i for i, s in enumerate(self.shapes)}
        self.location_variation = location_variation
        self.size_variation = size_variation
        self.size_range = size_range
        self.transform = Compose([
            ToTensor(),
            Lambda(lambda t: (t * 2) - 1)  # Scale to [-1, 1]
        ])

    def _sample_bbox(self):
        """Return (left, top, right, bottom) given size/location toggles."""
        margin_base = self.img_size // 4
        side_base = self.img_size - 2 * margin_base

        # size jitter
        if self.size_variation:
            s = pyrandom.uniform(*self.size_range)
        else:
            s = 1.0
        side = max(4, int(round(side_base * s)))

        # location jitter (choose center so bbox stays inside bounds)
        cx_nom, cy_nom = self.img_size // 2, self.img_size // 2
        min_c = margin_base + side // 2
        max_c = self.img_size - margin_base - side // 2

        if self.location_variation and max_c >= min_c:
            cx = pyrandom.randint(min_c, max_c)
            cy = pyrandom.randint(min_c, max_c)
        else:
            cx, cy = cx_nom, cy_nom

        left = cx - side // 2
        top = cy - side // 2
        right = left + side
        bottom = top + side
        return (left, top, right, bottom)

    def _draw_shape(self, shape, draw):
        # unified drawer that uses bbox (supports both toggles)
        bbox = self._sample_bbox()
        if shape == "circle":
            draw.ellipse(bbox, fill="white")
        elif shape == "square":
            draw.rectangle(bbox, fill="white")
        elif shape == "triangle":
            # inscribe triangle in bbox
            l, t, r, b = bbox
            cx = (l + r) // 2
            p1 = (cx, t)       # top mid
            p2 = (l, b)        # bottom-left
            p3 = (r, b)        # bottom-right
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

        image = Image.new("L", (self.img_size, self.img_size), "black")
        draw = ImageDraw.Draw(image)
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
