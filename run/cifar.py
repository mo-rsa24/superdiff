import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')
import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'