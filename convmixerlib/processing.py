from typing import Tuple
import tensorflow as tf
from typing import Tuple

def train_val_split(images: tf.Tensor, labels: tf.Tensor, val_perc: float) -> Tuple[Tuple[tf.Tensor, tf.Tensor], Tuple[tf.Tensor, tf.Tensor]]:
    """
    split data into training and validation

    Args:
        images (tf.Tensor): images data
        labels (tf.Tensor): labes for images data
        val_perc (float): percentage of observation to reserve for validation data

    Returns:
        Tuple[Tuple[tf.Tensor, tf.Tensor], Tuple[tf.Tensor, tf.Tensor]]: (images train, labels train), (images val, labels val)
    """

    # shuffle data before creating train validation split
    indexes = tf.range(start=0, limit=labels.shape[0], dtype=tf.int32)
    indexes = tf.random.shuffle(indexes)
    images = tf.gather(images, indexes)
    labels = tf.gather(labels, indexes)

    # validation size
    val_size = int(val_perc * len(labels))

    # train validation split
    images_train = images[:-val_size]
    images_val = images[-val_size:]    
    labels_train = labels[:-val_size]
    labels_val = labels[-val_size:]

    return (images_train, labels_train), (images_val, labels_val)

def data_augmentation(images_batch: tf.Tensor, labels_batch: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    apply random transformations to augment data

    Args:
        images_batch (tf.Tensor): batch of images data
        labels_batch (tf.Tensor): batch of labels

    Returns:
        Tuple[tf.Tensor, tf.Tensor]: batch of images data, batch of labels
    """
    # random horizontal flip
    images_batch = tf.image.random_flip_left_right(images_batch)

    # random vertical flip
    images_batch = tf.image.random_flip_up_down(images_batch)

    # small hue change
    images_batch = tf.image.random_hue(images_batch, max_delta=0.05)

    # small saturation change
    images_batch = tf.image.random_saturation(images_batch, lower=0.95, upper=1.05) 

    # small contrast change
    images_batch = tf.image.random_contrast(images_batch, lower=0.90, upper=1.10)

    # small brightness change
    images_batch = tf.image.random_brightness(images_batch, max_delta=0.10)

    # clip values out of range
    images_batch = tf.clip_by_value(images_batch, clip_value_min=0.0, clip_value_max=255.0)

    return images_batch, labels_batch
