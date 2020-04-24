import numpy as np
import tensorflow as tf


def bits_x(log_likelihood: tf.Tensor,
           log_det_jacobian: tf.Tensor, pixels: int, n_bits: int = 8):
    """bits/dims
    Args:
        log_likelihood: shape is [batch_size,]
        log_det_jacobian: shape is [batch_size,]
        pixels: e.g. HWC image => H * W * C
        n_bits: e.g [0 255] image => 8 = log(256)

    Returns:
        bits_x: shape is [batch_size,]

    formula:
        (log_likelihood + log_det_jacobian)
          / (log 2 * h * w * c) + log(2^n_bits) / log(2.)
    """
    nobj = - 1.0 * (log_likelihood + log_det_jacobian)
    _bits_x = nobj / (np.log(2.) * pixels) + n_bits
    return _bits_x



class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """for warmup learning rate
    ref:
    https://www.tensorflow.org/tutorials/text/transformer
    """

    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()
        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps**-1.5)
        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)
