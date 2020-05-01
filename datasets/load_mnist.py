import logging
from typing import Tuple

import tensorflow as tf
from sklearn.model_selection import train_test_split

logger = tf.get_logger()
logger.setLevel(logging.DEBUG)

SHUFFLE_BUFFER_SIZE = 1000


def load_dataset(batch_sizes: Tuple[int, int, int] = None,
                 with_log: bool = False):
    """load dataset extended method
    Args:
        batch_sizes: [train_batch_size, valid_batch_size, test_batch_size]
    Returns:
        train_dataset, valid_dataset, test_dataset
    """
    if batch_sizes is None:
        batch_sizes = [120, 240, 240]
    (train_x, train_y), (test_x, test_y) = tf.keras.datasets.mnist.load_data()
    train_x, valid_x, train_y, valid_y = train_test_split(train_x, train_y,
                                                          test_size=0.2,
                                                          random_state=42)

    assert train_x.shape[0] == train_y.shape[0]
    assert valid_x.shape[0] == valid_y.shape[0]
    assert test_x.shape[0] == test_y.shape[0]

    logger.info("train data size: {}".format(train_x.shape[0]))
    logger.info("valid data size: {}".format(valid_x.shape[0]))
    logger.info("test data size: {}".format(test_x.shape[0]))

    @tf.function
    def _parse_function(img, label):
        feature = {}
        img = tf.cast(img, dtype=tf.float32)
        img = img[..., tf.newaxis]
        img = tf.image.resize(img, (24, 24))
        img = img / 255.0
        feature["img"] = img
        feature["label"] = label
        return feature

    train_dataset_raw = tf.data.Dataset.from_tensor_slices(
        (train_x, train_y)).map(_parse_function)
    valid_dataset_raw = tf.data.Dataset.from_tensor_slices(
        (valid_x, valid_y)).map(_parse_function)
    test_dataset_raw = tf.data.Dataset.from_tensor_slices(
        (test_x, test_y)).map(_parse_function)

    train_dataset = train_dataset_raw.shuffle(
        SHUFFLE_BUFFER_SIZE).batch(batch_sizes[0])
    valid_dataset = valid_dataset_raw.shuffle(
        SHUFFLE_BUFFER_SIZE).batch(batch_sizes[1])
    test_dataset = test_dataset_raw.shuffle(
        SHUFFLE_BUFFER_SIZE).batch(batch_sizes[2])

    sample = next(train_dataset.as_numpy_iterator())
    logger.info("image size: {}".format(sample["img"].shape[1:]))

    return (train_dataset, valid_dataset, test_dataset)


def check():
    train, valid, test = load_dataset(with_log=True)
