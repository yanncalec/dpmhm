"""
MoCo v1:
- He, K., Fan, H., Wu, Y., Xie, S., Girshick, R., 2020. Momentum Contrast for Unsupervised Visual Representation Learning. Presented at the Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 9729–9738.

Code:
https://github.com/facebookresearch/moco

ToDo:
- Running failed with TF and PyTorch backend
- Constant loss function value with Jax
"""

# import sys
import tensorflow as tf

from keras import ops, models, layers, regularizers, callbacks, losses
# from keras.applications import resnet
# from dataclasses import dataclass

# from queue import Queue
from collections import deque

# import numpy as np

from ..losses import InfoNCE
from ..ul import autoencoder
from ..pretrained import get_base_encoder
from ..config import AbstractConfig

import logging
logger = logging.getLogger(__name__)


# For EMA:
# https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage

# @dataclass
# class Config(AbstractConfig):
#     # batch_size:int = 256
#     tau:float = 0.01  # temperature
#     maxlen:int = 1000  # size of the memory bank
#     name:str = 'VGG16'  # name of the base encoder 'ResNet50', 'CNN'
#     encoder_kwargs:dict = {}  # keyword parameters

#     def optimizer(self):
#         # https://www.tensorflow.org/addons/api_docs/python/tfa/optimizers/SGDW
#         import tensorflow_addons as tfa
#         return tfa.optimizers.SGDW(weight_decay=1e-4, learning_rate=0.03, momentum=0.9)


class MoCo_Callback(callbacks.Callback):
    """
    https://keras.io/api/callbacks/base_callback/
    """
    def on_train_begin(self, logs=None):
        # Convert Keras variables to Tensorflow variables, needed by EMA
        self._tf_variables = [tf.Variable(v) for v in self.model._online.weights]  # in Keras `.weights` is identical to `.variables`
        self.model._ema.apply(self._tf_variables)  # create a shadow copy

    def on_epoch_end(self, epoch, logs=None):
    # def on_train_batch_end(self, batch, logs=None):
        # Update the value of variables
        for v, w in zip(self._tf_variables, self.model._online.weights):
            v.assign(w)
        # EMA update of the target network
        self.model._target.set_weights(
            [self.model._ema.average(v) for v in self._tf_variables]
        )


class MoCo(models.Model):
    def __init__(self, input_shape:tuple, *, tau:float=0.1, momentum:float=0.999, maxlen:int=100, name:str='VGG16', encoder_kwargs:dict={}):
        super().__init__()
        self._input_shape = input_shape
        self._tau = tau

        self._memory = deque(maxlen=maxlen)  # queue of preceding encoded mini-batches
        # self.loss_tracker = keras.metrics.Mean(name='loss')
        # self.mae_metric = keras.metrics.MeanAbsoluteError(name="mae")

        try:
            self._encoder = get_base_encoder(input_shape, name, **encoder_kwargs)
        except:
            self._encoder = autoencoder.CAES(input_shape, **encoder_kwargs).encoder
        # self._encoder.trainable = train_encoder

        self._projector = models.Sequential([
            layers.Flatten(name='flatten'),
            layers.Dense(1024, activation='relu', name='fc1'),
            layers.BatchNormalization(),
            layers.Dense(256, activation='relu', name='fc2'),
            # layers.BatchNormalization(),
            # layers.Dense(64, activation=None, name='fc3'),
        ], name='projector')

        self._online = models.Sequential([
            self._encoder,
            self._projector,
        ], name='online')  # online network

        self._target = models.clone_model(self._online)  # momentum encoder network
        self._target.trainable = False

        # self._ema_rate = momentum
        self._ema = tf.train.ExponentialMovingAverage(momentum)

    # @tf.function
    def call(self, inputs, training=True):
        xq, xk = inputs  # treated as an iterator, not allowed in graph mode
        yq = self._online(xq, training=training)
        yk = self._target(xk, training=training)
        # print(ops.mean(yq), ops.mean(yk))

        try:
            # tensor of memory, has shape `(memlen, batch, feature)`
            mk = ops.stack(list(self._memory), axis=0)  # must convert to list first
            # assert ops.size(mk) > 0
        except Exception as msg:  # memory is empty
            logger.error(msg)
            mk = ops.expand_dims(yk, 0)

        self.add_loss(
            InfoNCE(yq, yk, mk, self._tau)
        )
        # Note: by default a deque object (with `append()` and `pop`) is LIFO. Use `appendleft` to make it FIFO.
        self._memory.append(yk)
        # no need to explicitly apply `pop()`, this is automatically handled by `deque`.
        # print('Length of the memory:', len(self._memory))

        return yq, yk

    # def train_step(self, inputs):
    #     # print(f"Eager execution mode: {tf.executing_eagerly()}")
    #     # https://keras.io/guides/customizing_what_happens_in_fit
    #     with tf.GradientTape() as tape:
    #         yq, yk = self.call(inputs)
    #         try:
    #             mk = tf.concat(list(self._memory), axis=0)  # must convert to list first
    #             # # However, the following doesn't work properly
    #             # mk = tf.stack(self._memory)[:,-1]
    #             assert tf.size(mk) > 0
    #         except:  # memory is empty
    #             mk = ops.copy(yk)
    #         loss = self._loss_func(yq, yk, mk)
    #         # loss += self._loss_func(yk, yk, mk)
    #         # loss = tf.reduce_sum(InfoNCE(yq, yq, mk, self._temperature))

    #     # Note: by default a deque object (with `append()` and `pop`) is LIFO. Use `appendleft` to make it FIFO.
    #     self._memory.append(yk)
    #     # no need to explicitly apply `pop()`, this is automatically handled by `deque`.
    #     # print('Length of the memory:', len(self._memory))

    #     # Compute gradients
    #     gradients = tape.gradient(loss, self.trainable_variables)
    #     # Update weights
    #     self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

    #     # # Update metrics (includes the metric that tracks the loss)
    #     # self.compiled_metrics.update_state(y, y_pred)
    #     # # Return a dict mapping metric names to current value
    #     # return {m.name: m.result() for m in self.metrics}

    #     # Compute our own metrics
    #     self.loss_tracker.update_state(loss)
    #     # self.mae_metric.update_state(y, y_pred)
    #     # return {'loss': self.loss_tracker.result(), "mae": self.mae_metric.result()}
    #     return {'loss': self.loss_tracker.result()}
