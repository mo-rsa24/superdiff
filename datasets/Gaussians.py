from jax import random
from jax import numpy as jnp
from typing import Literal, Tuple

Quadrant = Literal["upper_left", "upper_right", "lower_right", "lower_left"]

class GaussianDataset:
    def __init__(self, num_keys: int = 3, num_clusters: int = 4, datapoints: int = 512, seed: int = 42, dimensions: int = 2, stretch: float = 0.5, scaling_factor: float = 3.0, normal: bool = True):
        self.num_keys = num_keys
        self.datapoints = datapoints
        self.dimensions = dimensions
        self.shape = (self.datapoints, self.dimensions)
        self.stretch = stretch
        self.num_clusters = jnp.log2(num_clusters)
        self.scaling_factor = scaling_factor
        self.key, *self.subkeys = random.split(random.PRNGKey(seed), self.num_keys)
        self.normal = normal
        """
        Number of clusters is controlled by minval and maxval of randint.
        because you can only get values between [0, 2) which is [0,1] in integers.
        So, 2^{n} generates the number of clusters. 2^{maxval} will be number of clusters
        
        A single key in JAX is a pair of integers that allow deterministic random generation
        - Each key generates different set of random numbers
        A single key in JAX can split into multiple subkeys: 
        - A subkey allows us to deterministically generate multiple random sequences
        -- Example: n1 = [4,52,1,5,2 ] {key=0}
                    n2 = [4,5,6,7,1,2] {subkey=1} - this is true all the time
        - Why: If we run the code with the same seed, the key and its subkeys generate same set of random numbers
        """

    def sample_data(self):
        x_1 = random.randint(self.key, minval=0, maxval=self.num_clusters, shape=self.shape)
        x_1 = self.scaling_factor * (x_1.astype(jnp.float32) - self.stretch)
        if self.normal: # If false, then you only get 4 circles because the datapoints are repeated on the same coordinates
            x_1 += 4e-1 * random.normal(self.subkeys[0], shape=self.shape)
        return x_1

    def sample_group(self, up: bool = True, subsample: bool = True):
        keys = random.split(self.key, 3)
        if subsample:
            datapoints, coordinates = self.shape[0] // 2, self.shape[1]
        else:
            datapoints, coordinates = self.shape
        if up:
            x_1 = random.randint(keys[0], minval=jnp.array([0, 1]), maxval=jnp.array([2, 2]), shape=(datapoints, coordinates))
        else:
            x_1 = random.randint(keys[0], minval=jnp.array([0, 0]), maxval=jnp.array([2, 1]), shape=(datapoints, coordinates))
        x_1 = 3 * (x_1.astype(jnp.float32) - 0.5)
        x_1 += 4e-1 * random.normal(keys[1], shape=(datapoints, coordinates))
        return x_1

    def sample_quadrant(self, quadrant: Quadrant, subsample: bool = True):
        """
        Sample a SINGLE Gaussian cluster at one of:
        'upper_left', 'upper_right', 'lower_right', 'lower_left'.

        If subsample=True, we return ~1/4 of the total datapoints (so
        calling this 4 times yields ~self.datapoints total).
        """
        # fresh keys each call
        keys = self._next_keys(3)

        if subsample:
            datapoints, coordinates = self.shape[0] // 4, self.shape[1]
        else:
            datapoints, coordinates = self.shape

        # Choose per-dimension integer ranges for the target cell
        if quadrant == "upper_left":
            minval = jnp.array([0, 1])
            maxval = jnp.array([1, 2])
        elif quadrant == "upper_right":
            minval = jnp.array([1, 1])
            maxval = jnp.array([2, 2])
        elif quadrant == "lower_right":
            minval = jnp.array([1, 0])
            maxval = jnp.array([2, 1])
        elif quadrant == "lower_left":
            minval = jnp.array([0, 0])
            maxval = jnp.array([1, 1])
        else:
            raise ValueError(f"Unknown quadrant: {quadrant}")

        x_1 = random.randint(keys[0], minval=minval, maxval=maxval, shape=(datapoints, coordinates))
        # Keep your existing scaling and jitter so distributions match your current groups API
        x_1 = 3 * (x_1.astype(jnp.float32) - 0.5)
        x_1 += 4e-1 * random.normal(keys[1], shape=(datapoints, coordinates))
        return x_1

    def get_quadrants(self, subsample: bool = True) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Convenience wrapper to fetch all four clusters, ordered as:
        (upper_left, upper_right, lower_right, lower_left)

        With subsample=True each cluster is ~1/4 of self.datapoints,
        so concatenating all four recovers ~self.datapoints total samples.
        """
        ul = self.sample_quadrant("upper_left", subsample=subsample)
        ur = self.sample_quadrant("upper_right", subsample=subsample)
        lr = self.sample_quadrant("lower_right", subsample=subsample)
        ll = self.sample_quadrant("lower_left", subsample=subsample)
        return ul, ur, lr, ll

    def _next_keys(self, n: int = 3):
        """
        Optional helper to advance PRNG for fresh randomness across calls.
        Feel free to remove and revert to random.split(self.key, n) if you prefer fixed sequences.
        """
        self.key, *ks = random.split(self.key, n + 1)
        return ks

    def get_groups(self, subsample: bool = True):
        up = self.sample_group(up=True, subsample=subsample)
        down = self.sample_group(up=False, subsample=subsample)
        return up, down