"""Example code for training Glow
"""
import logging
from typing import Dict

import hydra
import numpy as np
import tensorflow as tf
from omegaconf import DictConfig

from future import CustomSchedule, bits_x
from models.glow import Glow

logger = tf.get_logger()
logger.setLevel(logging.DEBUG)



class Task:
    def __init__(self, hparams: Dict):
        self.hparams = hparams

        self.input_shape = [
            self.hparams["images"]["width"],
            self.hparams["images"]["height"],
            self.hparams["images"]["channel"]]
        self.pixels = np.prod(self.input_shape)

        self.glow = Glow(self.hparams)

        if self.hparams["check_model"]:
            self.check_model()

    def check_model(self):
        x = tf.keras.Input(self.input_shape)
        # self.glow.build(x.shape)
        z, ldj, zaux, ll = self.glow(x)
        self.z_shape = list(z.shape)
        self.zaux_shape = list(zaux.shape)
        self.z_dims = np.prod(z.shape[1:])
        self.zaux_dims = np.prod(zaux.shape[1:])

        # summarize
        logger.info("z_f's shape             : {}".format(self.z_shape))
        logger.info("log_det_jacobian's shape: {}".format(ldj.shape))
        logger.info("z_aux's shape           : {}".format(self.zaux_shape))
        self.glow.summary()


@hydra.main(config_path="./conf/config.yaml")
def main(cfg: DictConfig):
    task = Task(cfg)



if __name__ == '__main__':
    cfg = {
        "check_model": True,
        "images": {
            "channel": 1,
            "height": 32,
            "width": 32},
        "model": {
            "hidden_width": 512,
            "conditional": False,
            "K": 16,
            "L": 3},
        "conditional": True,
        "hidden_width": 512}
    main()
