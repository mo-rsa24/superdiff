import numpy as np
import tensorflow as tf

def rescale_for_visualization(imgs): # mapping from [-1, 1] to [0, 1]
    imgs = tf.squeeze(imgs, axis=0)
    imgs = imgs.numpy()
    if imgs.max() <= 1.5 and imgs.min() >= -1.5:
        imgs = (imgs + 1.0) / 2.0
    else:
        imgs = imgs / 255.0
    imgs = np.clip(imgs, 0, 1)
    return imgs