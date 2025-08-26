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
    # Data
    SEED = 42
    IMG_SIZE = 64
    N_SAMPLES_DATASET = 10000
    SHAPES = ["circle", "square", "triangles"]
    SHAPE_TO_IDX = {s: i for i, s in enumerate(SHAPES)}
    IDX_TO_SHAPE = {i: s for s, i in SHAPE_TO_IDX.items()}

    # PCA
    N_COMPONENTS = 2
    CHECKPOINT_DIR = "checkpoints"
    PCA_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "pca.joblib")
    # Scale the latent space for better visualization, similar to the example image.
    LATENT_SPACE_SCALE = 2.5

    # Model
    NUM_HIDDEN_MLP = 512
    NUM_OUT_MLP = N_COMPONENTS  # Matches ndim from superposition_edu.py

    # Training
    SEED = 42
    LEARNING_RATE = 2e-4
    EPOCHS = 900  # A reasonable number of epochs
    BATCH_SIZE = 512

    # Generation
    T_START = 1.0
    DT = 1e-3  # Corresponds to dt = 1e-3 in superposition_edu.py
    N_STEPS = int(T_START / DT)