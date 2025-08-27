import os
import jax
import numpy as np

from jax import grad, jit, vmap, random, jvp

from flax.training import train_state
import optax
from sklearn.decomposition import PCA
import joblib
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import os
from tqdm.auto import trange
from functools import partial

class Config:
    batch_size = 128
    prefetch_size = -1
    shuffle_buffer_size = 10000