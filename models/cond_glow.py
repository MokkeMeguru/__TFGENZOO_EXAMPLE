#!/usr/bin/env python3

import tensorflow as tf
from typing import Dict, List

from tensorflow.keras import Model

from TFGENZOO.flows import (
    Actnorm,
    AffineCoupling,
    AffineCouplingMask,
    FlowModule,
    Inv1x1Conv,
)
from TFGENZOO.flows.cond_affine_coupling import ConditionalAffineCoupling
from TFGENZOO.flows.flowbase import ConditionalFlowModule

from TFGENZOO.flows.factor_out import FactorOut, FactorOutBase
from TFGENZOO.flows.quantize import LogitifyImage
from TFGENZOO.flows.squeeze import Squeezing
from TFGENZOO.layers.resnet import ShallowResNet


def CondNet(input_shape):
    inputs = tf.keras.layers.Input(shape=input_shape)

    resolution_levels = [
        tf.keras.Sequential(
            [
                tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding="same"),
                tf.keras.layers.LeakyReLU(),
                tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding="same"),
            ]
        ),
        tf.keras.Sequential(
            [
                tf.keras.layers.Conv2D(filters=128, kernel_size=3, padding="same"),
                tf.keras.layers.LeakyReLU(),
                tf.keras.layers.Conv2D(
                    filters=128, kernel_size=3, padding="same", strides=2
                ),
            ]
        ),
        tf.keras.Sequential(
            [
                tf.keras.layers.Conv2D(filters=128, kernel_size=3, padding="same"),
                tf.keras.layers.LeakyReLU(),
                tf.keras.layers.Conv2D(
                    filters=128, kernel_size=3, padding="same", strides=2
                ),
            ]
        ),
        tf.keras.Sequential(
            [
                tf.keras.layers.Conv2D(filters=128, kernel_size=3, padding="same"),
                tf.keras.layers.LeakyReLU(),
                tf.keras.layers.Conv2D(
                    filters=128, kernel_size=3, padding="same", strides=2
                ),
            ]
        ),
    ]

    x = inputs
    outputs = []
    for resolution in resolution_levels:
        x = resolution(x)
        outputs.append(x)
    return tf.keras.Model(inputs, outputs[1:])

# CondNet([48, 48, 1]).summary()

class CGlow(Model):
    def __init__(self, hparams: Dict):
        super().__init__()
        self.model_params = hparams["model"]
        K = self.model_params["K"]
        L = self.model_params["L"]
        conditional = self.model_params["conditional"]
        flows = []
        flows.append(LogitifyImage())
        for layer in range(L):

            # Squeezing Layer
            if layer == 0:
                flows.append(Squeezing(with_zaux=False))
            else:
                flows.append(Squeezing(with_zaux=True))
            fml = []

            # build flow module layer
            for k in range(K):
                fml.append(Actnorm())
                fml.append(Inv1x1Conv())
                scale_shift_net = ShallowResNet(width=self.model_params["hidden_width"])
                fml.append(
                    ConditionalAffineCoupling(
                        mask_type=AffineCouplingMask.ChannelWise,
                        scale_shift_net=scale_shift_net,
                    )
                )
            flows.append(ConditionalFlowModule(fml))

            # Factor Out Layer
            if layer == 0:
                flows.append(FactorOut(conditional=conditional))
            elif layer != L - 1:
                flows.append(FactorOut(with_zaux=True, conditional=conditional))

        self.flows = flows
        self.cond_conf = hparams["images"]["cond"]
        self.cond_net = CondNet(
            [
                self.cond_conf["height"],
                self.cond_conf["width"],
                self.cond_conf["channel"],
            ]
        )
        self.cond_net.summary()

    def call(
        self,
        x: tf.Tensor,
        zaux: tf.Tensor = None,
        inverse: bool = False,
        training: bool = True,
        temparature: float = 1.0,
    ):
        # setup conditional input
<<<<<<< HEAD
        conds = self.cond_net(x[..., 0])
=======
        conds = self.cond_net(x[..., 0:1])
>>>>>>> origin/master
        x = x[..., 1:]

        assert (
            len(conds) == self.model_params["L"]
        ), "conditional inputs' length is as same as L"
        if inverse:
            return self.inverse(x, zaux, conds, training, temparature)
        else:
            return self.forward(x, conds, training)

    def inverse(
        self,
        x: tf.Tensor,
        zaux: tf.Tensor,
        conds: List[tf.Tensor],
        training: bool,
        temparature: float = 1.0,
    ):
        """inverse
        latent -> object
        """
        inverse_log_det_jacobian = tf.zeros(tf.shape(x)[0:1])
        idx = 0
        for flow in enumerate(reversed(self.flows)):
            if isinstance(flow, Squeezing):
                if flow.with_zaux:
                    if zaux is not None:
                        x, zaux = flow(x, zaux=zaux, inverse=True)
                    else:
                        x = flow(x, inverse=True)
                else:
                    x = flow(x, inverse=True)
            elif isinstance(flow, FactorOutBase):
                if flow.with_zaux:
                    x, zaux = flow(x, zaux=zaux, inverse=True, temparature=temparature)
                else:
                    x = flow(x, inverse=True, temparature=temparature)

            elif isinstance(flow, ConditionalFlowModule):
                x, ldj = flow(
                    x,
                    inverse=True,
                    cond=conds[self.model_params["L"] - idx - 1],
                    training=training,
                )
                inverse_log_det_jacobian += ldj
<<<<<<< HEAD
                idx += 1
=======
<<<<<<< HEAD
            raise Exception()
=======
>>>>>>> 89f04e8713221a05f858b611fd1fffbba237af2b
            else:
                x, ldj = flow(
                    x,
                    inverse=True,
                    training=training,
                )
                inverse_log_det_jacobian += ldj 
>>>>>>> origin/master
        return x, inverse_log_det_jacobian

    def forward(self, x: tf.Tensor, conds: List[tf.Tensor], training: bool):
        """forward
        object -> latent
        """
        print(x.shape)
        zaux = None
        log_det_jacobian = tf.zeros(tf.shape(x)[0:1])
        log_likelihood = tf.zeros(tf.shape(x)[0:1])
        idx = 0
        for flow in self.flows:
            if isinstance(flow, Squeezing):
                if flow.with_zaux:
                    x, zaux = flow(x, zaux=zaux)
                else:
                    x = flow(x)
            elif isinstance(flow, FactorOutBase):
                x, zaux, ll = flow(x, zaux=zaux)
                log_likelihood += ll
            elif isinstance(flow, ConditionalFlowModule):
               x, ldj = flow(x, cond=conds[idx], training=training)
                idx += 1
            else:
<<<<<<< HEAD
                x, ldj = flow(x, training=training)
=======
<<<<<<< HEAD
                raise Exception()
=======
                x, ldj = flow(x, cond=conds[idx], training=training)
>>>>>>> origin/master
>>>>>>> 89f04e8713221a05f858b611fd1fffbba237af2b
        return x, log_det_jacobian, zaux, log_likelihood


cglow = CGlow(
    hparams={
        "batch_sizes": [128, 256, 256],
        "epochs": 512,
        "model": {"K": 4, "L": 3, "conditional": True, "hidden_width": 512},
        "inference": {"beta_z": 0.75, "beta_zaux": 0.75},
        "image_name": "flower",
        "images": {
            "cond": {"width": 48, "height": 48, "channel": 1},
            "base": {"width": 48, "height": 48, "channel": 2},
        },
    }
)

sample_input = tf.math.sin(tf.random.normal([16, 48, 48, 3]))
cglow(sample_input)
