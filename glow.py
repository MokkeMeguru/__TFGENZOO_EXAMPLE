"""Example code for training Glow
"""
import logging
from pathlib import Path
from typing import Dict

import hydra
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from omegaconf import DictConfig
from tensorflow.keras import metrics, optimizers
from TFGENZOO.optimizers import transformer_schedule
from tqdm import tqdm

from datasets import load_mnist
from future import bits_x
from models.glow import Glow

Mean = metrics.Mean
Adam = optimizers.Adam
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
        self.check_model()
        self.load_dataset()
        self.setup_target_distribution()
        self.setup_optimizer()
        self.setup_metrics()
        self.setup_checkpoint(
            Path(self.hparams.get("checkpoint_path",
                                  "checkpoints/glow")))
        self.setup_writer()

    def check_model(self):
        x = tf.keras.Input(self.input_shape)
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

    def load_dataset(self):
        self.train_dataset, self.valid_dataset, self.test_dataset = load_mnist.load_dataset(
            batch_sizes=self.hparams["batch_sizes"], with_log=True)

    def setup_target_distribution(self):
        z_distribution = tfp.distributions.MultivariateNormalDiag(
            tf.zeros([self.z_dims]), tf.ones([self.z_dims]))
        zaux_distribution = tfp.distributions.MultivariateNormalDiag(
            tf.zeros([self.zaux_dims]), tf.ones([self.zaux_dims]))
        self.target_distribution = (z_distribution, zaux_distribution)

    def setup_optimizer(self):
        self.learning_rate_schedule = transformer_schedule.CustomSchedule(
            self.pixels)
        self.optimizer = tf.keras.optimizers.Adam(
            self.learning_rate_schedule)

    def setup_metrics(self):
        self.train_nll = Mean(name='nll', dtype=tf.float32)
        self.valid_nll = Mean(name='nll', dtype=tf.float32)
        self.train_ldj = Mean(name='ldj', dtype=tf.float32)
        self.valid_ldj = Mean(name='ldj', dtype=tf.float32)

    def setup_checkpoint(self, checkpoint_path: Path):
        logger.info("checkpoint will be saved at {}".format(checkpoint_path))
        ckpt = tf.train.Checkpoint(step=tf.Variable(0),
                                   model=self.glow,
                                   optimizer=self.optimizer)
        ckpt_manager = tf.train.CheckpointManager(
            ckpt, checkpoint_path, max_to_keep=7)
        if ckpt_manager.latest_checkpoint:
            ckpt.restore(ckpt_manager.latest_checkpoint)
            logger.info("latest checkpoint was restored !!!")
            for sample in self.train_dataset.take(1):
                logger.info("restored loss (log_prob bits/dims) {}".format(
                    tf.reduce_mean(self.test_step(sample['img']))))
        self.ckpt = ckpt
        self.ckpt_manager = ckpt_manager

    def setup_writer(self):
        self.writer = tf.summary.create_file_writer(
            logdir="glow_log")

    @tf.function
    def train_step(self, img):
        x = img
        with tf.GradientTape() as tape:
            z, ldj, zaux, ll = self.glow(x, training=True)
            z = tf.reshape(z, [-1, self.z_dims])
            zaux = tf.reshape(zaux, [-1, self.zaux_dims])
            lp = self.target_distribution[0].log_prob(z)
            loss = bits_x(lp + ll, ldj, self.pixels)
        variables = tape.watched_variables()
        grads = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(
            zip(grads, variables))
        self.train_nll(loss)
        self.train_ldj(ldj)

    @tf.function
    def valid_step(self, img):
        x = img
        z, ldj, zaux, ll = self.glow(x, training=False)
        z = tf.reshape(z, [-1, self.z_dims])
        zaux = tf.reshape(zaux, [-1, self.zaux_dims])
        lp = self.target_distribution[0].log_prob(z)
        loss = bits_x(lp + ll, ldj, self.pixels)
        self.valid_nll(loss)
        self.valid_ldj(ldj)

    @tf.function
    def eval_step(self, img):
        x = img
        z, ldj, zaux, ll = self.glow(x, training=False)
        z = tf.reshape(z, [-1, self.z_dims])
        zaux = tf.reshape(zaux, [-1, self.zaux_dims])
        lp = self.target_distribution[0].log_prob(z)
        loss = bits_x(lp + ll, ldj, self.pixels)
        return loss

    def sample_image(self,
                     beta_z: float = 0.75, beta_zaux: float = 0.75):
        z_distribution = tfp.distributions.MultivariateNormalDiag(
            tf.zeros([self.z_dims]), tf.broadcast_to(beta_z, [self.z_dims]))
        z = z_distribution.sample(4)
        z = tf.reshape(z, [-1] + self.z_shape)
        x, ildj = self.glow.inverse(
            z, None, training=False, temparature=beta_zaux)
        tf.summary.image(
            "generated image",
            x, step=self.optimizer.iterations, max_outputs=4)
        for x in self.test_dataset.take(1):
            tf.summary.image("reference image", x['img'][:4],
                             max_outputs=4,
                             step=self.optimizer.iterations)
            z, ldj, zaux, ll = self.glow(x["img"][:4],
                                         training=False)
            x, ildj = self.glow.inverse(z, zaux, training=False)
            tf.summary.image("reversed image", x,
                             max_outputs=4,
                             step=self.optimizer.iterations)

    def train(self):
        for epoch in range(self.hparams['epochs']):
            for x in tqdm(self.train_dataset):
                self.train_step(x['img'])
            for x in tqdm(self.valid_dataset):
                self.valid_step(x['img'])
            ckpt_save_path = self.ckpt_manager.save()
            with self.writer.as_default():
                self.sample_image(self.hparams['inference']['beta_z'],
                                  self.hparams['inference']['beta_zaux'])
                tf.summary.scalar('train/nll', self.train_nll.result(),
                                  step=self.optimizer.iterations)
                tf.summary.scalar('valid/nll', self.valid_nll.result(),
                                  step=self.optimizer.iterations)
                tf.summary.scalar('train/ldj', self.train_ldj.result(),
                                  step=self.optimizer.iterations)
                tf.summary.scalar('valid/ldj', self.valid_ldj.result(),
                                  step=self.optimizer.iterations)
                logger.info("epoch {}: train_loss = {}, valid_loss = {}, saved_at = {}".format(
                    epoch,
                    self.train_nll.result().numpy(),
                    self.valid_nll.result().numpy(),
                    ckpt_save_path))
                self.train_nll.reset_states()
                self.train_ldj.reset_states()
                self.valid_nll.reset_states()
                self.valid_ldj.reset_states()

    def eval(self):
        losses = []
        for x in tqdm(self.train_dataset):
            losses.append(tf.reduce_mean(self.eval_step(x['img'])))
        logger.info("eval: nll {}".format(tf.reduce_mean(losses)))


@hydra.main(config_path="./conf/config.yaml")
def main(cfg: DictConfig):
    task = Task(cfg)
    task.train()
    task.eval()

if __name__ == '__main__':
    main()
