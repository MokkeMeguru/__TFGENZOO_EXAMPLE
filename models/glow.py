from typing import Dict

import tensorflow as tf
from tensorflow.keras import Model
from TFGENZOO.flows import (Actnorm, AffineCoupling, AffineCouplingMask,
                            FlowModule, Inv1x1Conv)
from TFGENZOO.flows.factor_out import FactorOut, FactorOutBase
from TFGENZOO.flows.quantize import LogitifyImage
from TFGENZOO.flows.squeeze import Squeezing
from TFGENZOO.flows.utils.util import ShallowResNet


class Glow(Model):
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
                scale_shift_net = ShallowResNet(
                    width=self.model_params["hidden_width"])
                fml.append(AffineCoupling(
                    mask_type=AffineCouplingMask.ChannelWise,
                    scale_shift_net=scale_shift_net))
            flows.append(FlowModule(fml))

            # Factor Out Layer
            if layer == 0:
                flows.append(FactorOut(conditional=conditional))
            elif layer != L - 1:
                flows.append(
                    FactorOut(with_zaux=True,
                              conditional=conditional))

        self.flows = flows

    def call(self,
             x: tf.Tensor,
             zaux: tf.Tensor = None,
             inverse: bool = False,
             training: bool = True,
             temparature: float = 1.0):
        if inverse:
            return self.inverse(x, zaux, training, temparature)
        else:
            return self.forward(x, training)

    def inverse(self,
                x: tf.Tensor,
                zaux: tf.Tensor,
                training: bool,
                temparature: float):
        """inverse
        latent -> object
        """
        inverse_log_det_jacobian = tf.zeros(tf.shape(x)[0:1])

        for flow in reversed(self.flows):
            if isinstance(flow, Squeezing):
                if flow.with_zaux and zaux is not None:
                    x, zaux = flow(x, zaux=zaux, inverse=True)
                else:
                    x = flow(x, inverse=True)
            elif isinstance(flow, FactorOutBase):
                if flow.with_zaux and zaux is not None:
                    x, zaux = flow(x, zaux=zaux, inverse=True,
                                   temparature=temparature)
                else:
                    x = flow(x, inverse=True, temparature=temparature)
            else:
                x, ldj = flow(x, inverse=True, training=training)
                inverse_log_det_jacobian += ldj
        return x, inverse_log_det_jacobian

    def forward(self,
                x: tf.Tensor,
                training: bool):
        """forward
        object -> latent
        """
        zaux = None
        log_det_jacobian = tf.zeros(tf.shape(x)[0:1])
        log_likelihood = tf.zeros(tf.shape(x)[0:1])
        for flow in self.flows:
            if isinstance(flow, Squeezing):
                if flow.with_zaux:
                    x, zaux = flow(x, zaux=zaux)
                else:
                    x = flow(x)
            elif isinstance(flow, FactorOutBase):
                x, zaux, ll = flow(x, zaux=zaux)
                log_likelihood += ll
            else:
                x, ldj = flow(x, training=training)
                log_det_jacobian += ldj
        return x, log_det_jacobian, zaux, log_likelihood
