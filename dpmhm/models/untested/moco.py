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

import keras
from keras import ops, models, layers, callbacks
from collections import deque

from ..losses import InfoNCE
from ..ul import autoencoder
from ..pretrained import get_base_encoder

import logging
logger = logging.getLogger(__name__)


# For EMA:
# https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage


class MoCo_Callback(callbacks.Callback):
    """
    https://keras.io/api/callbacks/base_callback/
    """
    def on_train_begin(self, logs=None):
        batch_size = ops.shape(self.model._online.input_shape)
        logger.info(logs)  # contain only information on the loss

        # Creation of variables here is before the model being built.
        # # Convert Keras variables to Tensorflow variables, needed by EMA
        # self._tf_variables = [tf.Variable(v) for v in self.model._online.weights]  # in Keras `.weights` is identical to `.variables`
        # logger.warn(len(self._tf_variables))
        # logger.warn(len(self.model._online.weights))
        # self.model._ema.apply(self._tf_variables)  # create a shadow copy

    def on_train_batch_end(self, batch, logs=None):
        logger.info(f"Callback: batch={batch}")
        # logger.info(f"batch_size={batch_size}")

        try:
            # Update the value of variables
            for v, w in zip(self._variables, self.model._online.weights):
                v.assign(w)
            # EMA update of the target network
            self.model._target.set_weights(
                [self.model._ema.average(v) for v in self._variables]
            )
        except:
            logger.info("Initializing variables...")
            self._variables = [tf.Variable(v) for v in self.model._online.weights]  # in Keras `.weights` is identical to `.variables`
            logger.info(len(self._variables))
            logger.info(len(self.model._online.weights))
            self.model._ema.apply(self._variables)  # create a shadow copy

        # Deque doesn't play well in the graphical mode.
        self.model._memory.append(self.model._yk.numpy())
        # Conversion results in strange error in InfoNCE loss
        self.model._memory_tensor = ops.stack(self.model._memory, axis=0)
        # self.model._memory_tensor = ops.stack(list(self.model._memory))
        logger.info(ops.shape(self.model._memory_tensor))

        # logger.info(self.model._memory_tensor)
        # try:
        #     self.model._memory = ops.take(
        #         ops.vstack([
        #             ops.expand_dims(self.model._yk, 0),
        #             # ops.reshape(self._memory, (-1, *ops.shape(yk)))
        #             self.model._memory,
        #         ]),
        #         range(self.model._maxlen), axis=0
        #     )
        # except:
        #     self.model._memory = ops.expand_dims(self.model._yk, 0),


class MoCo(models.Model):
    def __init__(self, input_shape:tuple, output_dim:int=256, *, tau:float=0.1, momentum:float=0.999, maxlen:int=100, name:str='VGG16', encoder_kwargs:dict={}):
        super().__init__()
        self._input_shape = input_shape
        self._output_dim = output_dim
        self._tau = tau
        self._maxlen = maxlen

        self._memory = deque(maxlen=maxlen)  # queue of preceding
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
            layers.Dense(output_dim, activation='relu', name='fc2'),
        ], name='projector')

        self._online = models.Sequential([
            self._encoder,
            self._projector,
        ], name='online')  # online network

        self._target = models.clone_model(self._online)  # momentum encoder network
        self._target.trainable = False

        # self._output_shape = keras.Variable(
        #     ops.cast(ops.zeros((2,)), int), dtype=int
        # )

        # self._ema_rate = momentum
        self._ema = tf.train.ExponentialMovingAverage(momentum)

    def build(self, input_shape):
        logger.info(input_shape)
        batch_size = input_shape[0][0]
        # output_dim = self._online.output_shape[-1]  # not available yet
        self._yk = keras.Variable(
            ops.zeros((batch_size, self._output_dim)),
            trainable=False
        )

    #     self(keras.random.normal(input_shape))
    #     # # self.model._memory = ops.array((1, ))
    #     # # Convert Keras variables to Tensorflow variables, needed by EMA
    #     # self._tf_variables = [tf.Variable(v) for v in self._online.weights]  # in Keras `.weights` is identical to `.variables`
    #     # logger.warn(len(self._tf_variables))
    #     # logger.warn(len(self._online.weights))
    #     # self._ema.apply(self._tf_variables)  # create a shadow copy

    # @tf.function
    def call(self, inputs, training=True):
        xq, xk = inputs  # treated as an iterator, not allowed in graph mode
        yq = self._online(xq, training=training)
        yk = self._target(xk, training=training)
        logger.info(f"call: {ops.shape(yq)}, {ops.shape(yk)}")

        try:
            loss = InfoNCE(
                # yq, yk, self._memory, self._tau
                yq, yk, self._memory_tensor, self._tau
            )
        except Exception as msg:
            logger.error(f"call: {msg}")
            loss = InfoNCE(
                yq, yk, ops.expand_dims(yk, 0), self._tau
            )

        self.add_loss(loss)
        self._yk.assign(yk)

        # try:
        #     self._yk.assign(yk)
        # except:
        #     self._output_shape.assign(ops.array(ops.shape(yk)))

        # self.update_memory(yk)

        # # tensor of memory, has shape `(memlen, batch, feature)`
        # mk = ops.stack(list(self._memory), axis=0)  # must convert to list first

        # if ops.equal(ops.size(mk), 0):
        # # if len(self._memory) == 0:
        #     mk = ops.expand_dims(yk, 0)

        # # try:
        # #     # tensor of memory, has shape `(memlen, batch, feature)`
        # #     mk = ops.stack(list(self._memory), axis=0)  # must convert to list first
        # #     ops.asser ops.any(tf.greater(tf.size(mk), 0))
        # # except Exception as msg:  # memory is empty
        # #     logger.error(f"DPMHM error message: {msg}")
        # #     mk = ops.expand_dims(yk, 0)

        # self.add_loss(
        #     InfoNCE(yq, yk, mk, self._tau)
        # )
        # Note: by default a deque object (with `append()` and `pop`) is LIFO. Use `appendleft` to make it FIFO.
        # self._memory.append(yk)
        # logger.info("DPMHM: Memory updated.")
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
