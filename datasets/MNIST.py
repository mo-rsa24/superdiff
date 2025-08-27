import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')
import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
import tensorflow_datasets as tfds
from config.Config import Config
from typing import List, Tuple
from tensorflow.python.data.ops.prefetch_op import _PrefetchDataset
from tensorflow_datasets.core import DatasetBuilder

def resize(img, image_size: int = 32):
    img = tf.image.convert_image_dtype(img, tf.float32)
    return tf.image.resize(img, [image_size, image_size], antialias=True)

def preprocess_fn(dataset):
    """Basic preprocessing function scales data to [0, 1) and randomly flips."""
    img = resize(dataset['image'])
    img = tf.tile(img, [1, 1, 3]) # Step 2: Tile from (32, 32, 1) to (32, 32, 3)
    img = (tf.random.uniform(img.shape, dtype=tf.float32) + img * 255.) / 256.
    return dict(image=img, label=dataset['label'])

def get_mnist_dataset():
     mnist_builder: DatasetBuilder = tfds.builder('mnist')
     mnist_builder.download_and_prepare()
     dataset_options = get_options()
     read_config = tfds.ReadConfig(options=dataset_options)
     dataset: _PrefetchDataset = mnist_builder.as_dataset(split='train', shuffle_files=True, read_config=read_config)
     train = create_dataset_for_split(dataset)
     return train

def get_image_scaler():
    def scaler(x):
        return (x - 0.5) / 0.5
    return scaler

def get_image_inverse_scaler():
  def inv_scaler(x):
    return x*0.5 + 0.5
  return inv_scaler

def get_options():
    dataset_options = tf.data.Options()
    dataset_options.experimental_optimization.map_parallelization = True
    dataset_options.threading.private_threadpool_size = 48
    dataset_options.threading.max_intra_op_parallelism = 1
    return dataset_options


def create_dataset_for_split(dataset: _PrefetchDataset, prefetch_size: int = -1,
                             batch_dims: Tuple[int, int] = (1, 128)):
    dataset = dataset.repeat(count=None)
    dataset = dataset.shuffle(buffer_size=Config.shuffle_buffer_size)
    dataset = dataset.map(preprocess_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    for batch_size in reversed(batch_dims):
        dataset = dataset.batch(batch_size, drop_remainder=True)
    return dataset.prefetch(prefetch_size)
