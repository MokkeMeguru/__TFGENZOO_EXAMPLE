#!/usr/bin/env python3

import logging
from typing import Tuple
import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np
from skimage import io, color

AUTOTUNE = tf.data.experimental.AUTOTUNE

logger = tf.get_logger()
logger.setLevel(logging.DEBUG)

SHUFFLE_BUFFER_SIZE = 10000


def rgb_to_lab(rgb: np.array):

    return color.rgb2lab(rgb)


@tf.function
def convert(sample):
    img = sample["image"]
    label = sample["label"]
    img = tf.py_function(func=rgb_to_lab, inp=[img], Tout=tf.float32)
    print(img.shape)
    return {"image": img, "label": label}


ds = tfds.load("oxford_flowers102", shuffle_files=True)

import matplotlib.pyplot as plt

for i in ds["train"].take(1).as_numpy_iterator():
    arr = i["image"]
    arr = arr / 255.0
    break

train_ds = ds["train"].map(convert)
print(train_ds)
for i in train_ds.take(1).as_numpy_iterator():
    tarr = i["image"]
    break
plt.figure()
plt.imshow(color.lab2rgb(tarr))

print(arr.shape)
plt.figure()
plt.imshow(arr)
arr2 = color.rgb2lab(arr)
arr_rec = color.lab2rgb(arr2)
plt.figure()
plt.imshow(arr_rec)
print(np.abs(arr - arr_rec).sum())
plt.show()
