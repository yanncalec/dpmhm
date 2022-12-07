"""
MoCo v1:
- He, K., Fan, H., Wu, Y., Xie, S., Girshick, R., 2020. Momentum Contrast for Unsupervised Visual Representation Learning. Presented at the Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 9729–9738.

Code:
https://github.com/facebookresearch/moco

See the original description of the algorithm.
"""

# import sys
import tensorflow as tf
from tensorflow import keras, linalg
from tensorflow.keras import models, layers, regularizers, callbacks, losses
from tensorflow.keras.applications import resnet
import tensorflow_addons as tfa

# from queue import Queue
from collections import deque

# import numpy as np

Config = {
    'batch_size': 256,
    'optimizer': tfa.optimizers.SGDW(weight_decay=1e-4, learning_rate=0.03, momentum=0.9),  # https://www.tensorflow.org/addons/api_docs/python/tfa/optimizers/SGDW
}


def InfoNCE(X, Y, K, tau:float):
    """
    X: query sample
    Y: positive sample
    K: key samples
    tau: temperature
    """
    # assert tf.shape(X)[0] == tf.shape(Y)[0] and tf.shape(X)[1] == tf.shape(K)[1]
    B = tf.shape(X)[0]  # batch size
    S = -losses.cosine_similarity(X, Y) / tau  # has size B
    S = S[:,None]
    # S = tf.reshape(S, [-1,1])
    # print(X.shape, K.shape)
    N = -losses.cosine_similarity(tf.expand_dims(X,1), K) / tau  # has shape B x ?
    # `K` alone has the same result as `tf.expand_dims(K,0)`
    return losses.sparse_categorical_crossentropy(
        tf.zeros(B),
        tf.concat([S, N], axis=-1),  # has shape B x (?+1)
        from_logits=True)
    # equivalent to:
    # return -(S - tf.reduce_logsumexp(tf.concat([S, N], axis=-1), axis=-1)[:, None])

"""
tf.reduce_logsumexp(?, axis=-1) is equivalent to
tf.math.log(tf.reduce_sum(tf.math.exp(?), axis=-1))
"""

class MoCo_Callback(callbacks.Callback):
    """
    https://keras.io/api/callbacks/base_callback/
    """
    def on_train_batch_end(self, batch, logs=None):
        self.model._target.set_weights([self.model._ema.average(v)
                                        for v in self.model._online.variables])


class MoCo(models.Model):
    def __init__(self, input_shape, train_encoder:bool=True, tau:float=0.07, msize:int=100, momentum:float=0.999, **kwargs):
        super().__init__()
        self._temperature = tau
        self._msize = msize
        self._loss_func = lambda X,Y,K: tf.reduce_sum(InfoNCE(X,Y,K,tau))

        self._memory = deque(maxlen=self._msize)  # queue of preceding encoded mini-batches
        self.loss_tracker = keras.metrics.Mean(name='loss')
        # self.mae_metric = keras.metrics.MeanAbsoluteError(name="mae")

        # config for the network
        # weights can also be loaded from a path, which takes the same amount of time ~1s
        self._encoder = resnet.ResNet50(include_top=False, weights='imagenet', input_shape=input_shape, pooling='avg')
        self._encoder.trainable = train_encoder

        self._projector = models.Sequential([
            layers.Flatten(name='flatten'),
            # layers.BatchNormalization(),
            layers.Dense(128, activation='relu', name='fc'),
        ], name='projector')

        self._online = models.Sequential([
            self._encoder,
            self._projector,
        ], name='online')  # encoder network

        self._target = models.clone_model(self._online)  # momentum encoder network
        self._target.trainable = False

        self._ema_rate = momentum
        self._ema = tf.train.ExponentialMovingAverage(self._ema_rate)
        self._ema.apply(self._online.variables)  # create shadow copy

    # @tf.function
    def call(self, inputs):
        xq, xk = inputs  # treated as an iterator, not allowed in graph mode
        yq, yk = self._online(xq), self._target(xk)
        return yq, yk

    def train_step(self, inputs):
        # print(f"Eager execution mode: {tf.executing_eagerly()}")
        # https://keras.io/guides/customizing_what_happens_in_fit
        with tf.GradientTape() as tape:
            yq, yk = self.call(inputs)
            try:
                mk = tf.concat(list(self._memory), axis=0)  # must convert to list first
                # # However, the following doesn't work properly
                # mk = tf.stack(self._memory)[:,-1]
                assert tf.size(mk) > 0
            except:  # memory is empty
                mk = tf.ones_like(yk)
            # mk = tf.ones_like(yk)
            loss = self._loss_func(yq, yk, mk)
            # loss += self._loss_func(yk, yk, mk)
            # loss = tf.reduce_sum(InfoNCE(yq, yq, mk, self._temperature))

        # Note: by default a deque object (with `append()` and `pop`) is LIFO. Use `appendleft` to make it FIFO.
        self._memory.append(yk)
        # no need to explicitly apply `pop()`, this is automatically handled by `deque`.
        # print('Length of the memory:', len(self._memory))

        # Compute gradients
        gradients = tape.gradient(loss, self.trainable_variables)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        # # Update metrics (includes the metric that tracks the loss)
        # self.compiled_metrics.update_state(y, y_pred)
        # # Return a dict mapping metric names to current value
        # return {m.name: m.result() for m in self.metrics}

        # Compute our own metrics
        self.loss_tracker.update_state(loss)
        # self.mae_metric.update_state(y, y_pred)
        # return {'loss': self.loss_tracker.result(), "mae": self.mae_metric.result()}
        return {'loss': self.loss_tracker.result()}
