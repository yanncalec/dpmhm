"""TODO
Yu, H., Wang, K., Li, Y., Zhao, W., 2019. Representation Learning With Class Level Autoencoder for Intelligent Fault Diagnosis. IEEE Signal Processing Letters 26, 1476–1480. https://doi.org/10.1109/LSP.2019.2936310

https://github.com/KWflyer/CLAE

CNN on the raw wavform.
Input: fixed-length trunks of wavform

The original method can also be implemented in a self-supervised fashion by considering each wavform record a class.
"""


# import sys
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras, linalg
from tensorflow.keras import models, layers, regularizers, callbacks, losses
from tensorflow.keras.applications import resnet

from dataclasses import dataclass, field

# from tensorflow.keras.losses import cosine_similarity

# import numpy as np


@dataclass
class Config:
    # AUTO = tf.data.AUTOTUNE
    batch_size:int = 512
    epochs:int = 5
    # n_batch:int = ?  # data_size/batch_size
    training_steps:int = 10**4  # epochs * n_batch

    # encoder:dict = field(default_factory=lambda: dict(include_top=False, weights='imagenet', pooling='avg'))
    train_encoder:bool = True

    projector_units:int = 2048
    predictor_units:int = 512
    use_bias:bool = False
    weight_decay:float = 0.0005

    # projector:dict = field(default_factory=lambda: dict(units=2048, use_bias=False))

    @classmethod
    def from_dict(cls, obj: dict):
        return cls(**obj)

    # def optimizer(self, learning_rate=0.05, weight_decay=1e-4, momentum=0.9):
    #     lr = keras.optimizers.schedules.CosineDecay(
    #         learning_rate * self.batch_size / 256,
    #         self.training_steps
    #     )
    #     return tfa.optimizers.SGDW(weight_decay=weight_decay, learning_rate=lr, momentum=momentum)


class CLAE(models.Model):
    def __init__(self, input_shape, c:Config):
        # input_shape, train_encoder:bool=True, tau:float=1e-1, **kwargs):
        super().__init__()
        self._config = c
        self.loss_tracker = keras.metrics.Mean(name='loss')
        # self.mae_metric = keras.metrics.MeanAbsoluteError(name="mae")

        # config for the network
        self._encoder = resnet.ResNet50(input_shape=input_shape, include_top=False, weights='imagenet', pooling='avg')
        self._encoder.trainable = c.train_encoder

        self._projector = models.Sequential([
            layers.Flatten(name='flatten'),
            layers.Dense(c.projector_units, use_bias=c.use_bias, activation='relu', kernel_regularizer=regularizers.l2(c.weight_decay), name='proj_fc1'),
            layers.BatchNormalization(),
            # layers.Dense(c.projector_units, use_bias=c.use_bias, activation='relu', kernel_regularizer=regularizers.l2(c.weight_decay), name='proj_fc2'),
            # layers.BatchNormalization(),
            layers.Dense(c.projector_units, use_bias=c.use_bias, activation=None, kernel_regularizer=regularizers.l2(c.weight_decay), name='proj_fc2'),
            ], name='projector')

        # self._encoder_projector = models.Sequential([
        #     self._encoder,
        #     self._projector
        # ], name='encoder_projector')

        self._decoder = models.Sequential([
            # layers.Flatten(name='flatten'),
            layers.Dense(c.predictor_units, use_bias=c.use_bias, activation='relu', kernel_regularizer=regularizers.l2(c.weight_decay), name='pred_fc1'),
            layers.BatchNormalization(),
            layers.Dense(c.projector_units, activation=None, name='pred_fc2'),
        ], name='predictor')

        # self.trainable_variables = self._encoder.trainable_variables + self._projector.trainable_variables + self.predictor.trainable_variables

    # @property
    # def metrics(self):
    #     return [self.loss_tracker]


    def _loss_func(self, p, z):
        return tf.reduce_mean(losses.cosine_similarity(p, tf.stop_gradient(z)), axis=0)

    def train_step(self, X, Y):
        # x1, x2 = inputs
        with tf.GradientTape() as tape:
            # z1, z2 = *self.call(inputs)
            Z = self._decoder(self._encoder(X))
             mZ = tf.reduce_mean(Z)  # per class mean

            loss_ae = losses.mse(x, z)
            loss_cr = losses.mse(X, )
            loss = loss_ae + self.beta * loss_cr
            loss = (self._loss_func(p1, z2) + self._loss_func(p2, z1)) / 2

        # Compute gradients
        gradients = tape.gradient(loss, self.trainable_variables)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        # Update metrics (includes the metric that tracks the loss)
        # self.compiled_metrics.update_state(y, y_pred)
        # Return a dict mapping metric names to current value
        self.loss_tracker.update_state(loss)
        # return {m.name: m.result() for m in self.metrics}
        return {'loss': self.loss_tracker.result()}
