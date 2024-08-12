import tensorflow as tf
import keras
from typing import Tuple

def read_cfar10_data(val_perc: float) -> Tuple[Tuple[tf.Tensor, tf.Tensor], Tuple[tf.Tensor, tf.Tensor], Tuple[tf.Tensor, tf.Tensor]]:
    """
    load cfar10 data

    Args:
        val_perc (float): percentage of observation to reserve for validation data

    Returns:
        Tuple[Tuple[tf.Tensor, tf.Tensor], Tuple[tf.Tensor, tf.Tensor], Tuple[tf.Tensor, tf.Tensor]]: _description_
    """
    # number of classes in cifar10
    num_classes = 10

    # load cifar10 data
    (images, labels), (images_test, labels_test) = keras.datasets.cifar10.load_data()

    # convert images data to float
    images = tf.cast(images, dtype=tf.float32)
    images_test = tf.cast(images_test, dtype=tf.float32)

    # convert labels to one hot encoding
    labels = tf.one_hot(labels, depth=num_classes, dtype=tf.float32)
    labels = tf.reshape(labels, shape=(labels.shape[0], num_classes))
    labels_test = tf.one_hot(labels_test, depth=num_classes, dtype=tf.float32)
    labels_test = tf.reshape(labels_test, shape=(labels_test.shape[0], num_classes))

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

    return (images_train, labels_train), (images_val, labels_val), (images_test, labels_test)