#!/usr/bin/env python

import logging
from typing import Tuple
import tensorflow_datasets as tfds
import tensorflow as tf

AUTOTUNE = tf.data.experimental.AUTOTUNE

logger = tf.get_logger()
logger.setLevel(logging.DEBUG)

SHUFFLE_BUFFER_SIZE = 10000

@tf.function
def convert(sample):
    img = sample["image"]
    label = sample["label"]
    img = tf.cast(img, tf.float32)
    img = img / 255.0
    if img.get_shape().is_fully_defined():
        h,w, c = img.get_shape().as_list()
    else:
       static_shape = img.get_shape().with_rank(rank=3).as_list()
       dynamic_shape = tf.unstack(tf.shape(img), 3)
       h, w, c = [s if s is not None else d for s, d in zip(static_shape, dynamic_shape)]
    small = tf.minimum(h, w)
    img = tf.image.resize_with_crop_or_pad(img, small, small)
    img = tf.image.resize(img, [48, 48], method="bilinear", preserve_aspect_ratio=True)
    img = tf.clip_by_value(img, 0.0, 1.0)
    data = {}
    data['img'] = img
    data['label'] = label
    return data

@tf.function
def augument(sample):
    img = sample["image"]
    label = sample["label"]
    img = tf.cast(img, tf.float32)
    img = img / 255.0
    if img.get_shape().is_fully_defined():
        h,w, c = img.get_shape().as_list()
    else:
       static_shape = img.get_shape().with_rank(rank=3).as_list()
       dynamic_shape = tf.unstack(tf.shape(img), 3)
       h, w, c = [s if s is not None else d for s, d in zip(static_shape, dynamic_shape)]
    small = tf.minimum(h, w)
    img = tf.image.resize_with_crop_or_pad(img, small, small)
    
    img = tf.image.resize(img, [64, 64], method="bilinear", preserve_aspect_ratio=True)
    img = tf.image.random_crop(img, [48, 48, 3])
    img = tf.image.random_brightness(img, max_delta=0.1)
    img = tf.clip_by_value(img, 0.0, 1.0)
    data = {}
    data['img'] = img
    data['label'] = label
    return data


def load_dataset(batch_sizes: Tuple[int, int, int] = None, with_log: bool = False):
    """load dataset extended method
    Args:
        batch_sizes: [train_batch_size, valid_batch_size, test_batch_size]
    Returns:
        train_datasets, valid_datasets, test_datasets
    """
    if batch_sizes is None:
        batch_sizes = [120, 240, 240]
    ds = tfds.load("oxford_flowers102", shuffle_files=True)
    # assert isinstance(ds, tf.data.Dataset)
    if with_log:
        logger.info(ds)
    train_datasets = (
        ds["train"]
        .cache()
        .shuffle(SHUFFLE_BUFFER_SIZE)
        .map(augument, num_parallel_calls=AUTOTUNE)
        .batch(batch_sizes[0])
        .prefetch(AUTOTUNE)
    )
    valid_datasets = (
        ds["validation"].map(convert, num_parallel_calls=AUTOTUNE).batch(batch_sizes[1])
    )
    test_datasets = (
        ds["test"].map(convert, num_parallel_calls=AUTOTUNE).batch(batch_sizes[2])
    )
    if with_log:
        sample = next(train_datasets.as_numpy_iterator())
        logger.info("image size: {}".format(sample['img'].shape[1:]))
        print(tf.reduce_max(sample['img']))
        print(tf.reduce_min(sample['img']))
        print(sample['img'].shape)
        import matplotlib.pyplot as plt

        for i in range(25):
            plt.subplot(5, 5, i + 1)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            print(sample['img'][i].shape)
            plt.imshow(sample['img'][i])
            plt.xlabel(sample['label'][i])
        plt.show()
    return train_datasets, valid_datasets, test_datasets


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--view", action="store_true", help="log image with view")
    args = parser.parse_args()

    load_dataset(with_log=args.view)
