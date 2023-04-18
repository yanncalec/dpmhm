"""Swapping Assignments between multiple Views of the same image (SwAV).

Reference
---------
Caron, M., Misra, I., Mairal, J., Goyal, P., Bojanowski, P., Joulin, A., 2020. Unsupervised Learning of Visual Features by Contrasting Cluster Assignments, in: Advances in Neural Information Processing Systems. Curran Associates, Inc., pp. 9912–9924.

Code:
https://github.com/facebookresearch/swav
https://github.com/ayulockin/SwAV-TF

How the prototype C is updated (via the soft Q?) is not clear.
"""

# import sys
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras, linalg, nn
from tensorflow.keras import models, layers, regularizers, callbacks, losses
from tensorflow.keras.applications import resnet

from dataclasses import dataclass, field


@dataclass
class Config:
    batch_size:int = 512
    epochs:int = 5
    # n_batch:int = ?  # data_size/batch_size
    training_steps:int = 10**4  # epochs * n_batch

    # encoder:dict = field(default_factory=lambda: dict(include_top=False, weights='imagenet', pooling='avg'))
    train_encoder:bool = True

    prototype_units:int = 2048
    use_bias:bool = False
    temperature:float = 0.0005

    @classmethod
    def from_dict(cls, obj: dict):
        return cls(**obj)

    def optimizer(self, learning_rate=0.05, weight_decay=1e-4, momentum=0.9):
        lr = keras.optimizers.schedules.CosineDecay(
            learning_rate * self.batch_size / 256,
            self.training_steps
        )
        return tfa.optimizers.SGDW(weight_decay=weight_decay, learning_rate=lr, momentum=momentum)


class Prototype(layers.Layer):
    # https://keras.io/guides/making_new_layers_and_models_via_subclassing/#making-new-layers-and-models-via-subclassing
    def __init__(self, units):
        super().__init__()
        self.units = units

    def build(self, input_shape):  # Create the state of the layer (weights)
        self.w = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer="random_normal",
            dtype=tf.float32,
            trainable=True,
        )
        # w_init = tf.random_normal_initializer()
        # self.w = tf.Variable(
        #     initial_value=w_init(shape=(input_shape[-1], self.units),
        #                         dtype=tf.float32),
        #     trainable=True)

    def call(self, inputs):  # Defines the computation from inputs to outputs
        wn, _ = linalg.normalize(self.w, axis=0)
        return tf.matmul(inputs, wn)  # tf.stop_gradient(wn) ??


class SwAV(models.Model):
    def __init__(self, input_shape, c:Config):
        # input_shape, train_encoder:bool=True, tau:float=1e-1, **kwargs):
        super().__init__()
        self._config = c
        self.loss_tracker = keras.metrics.Mean(name='loss')
        # self.mae_metric = keras.metrics.MeanAbsoluteError(name="mae")

        # config for the network
        self._encoder = resnet.ResNet50(input_shape=input_shape, include_top=False, weights='imagenet', pooling='avg')
        self._encoder.trainable = c.train_encoder

        self._prototype = Prototype(c.prototype_units)

        # self._projector = models.Sequential([
        #     layers.Flatten(name='flatten'),
        #     layers.Dense(c.prototype_units, use_bias=c.use_bias, activation=None, name='proto'),
        #     ], name='projector')

        # self.trainable_variables = self._encoder.trainable_variables + self._projector.trainable_variables + self.predictor.trainable_variables

    @staticmethod
    def sinkhorn(S, niter:int=3):
        """
        S: ~ (B,K)
        """
        Q = tf.transpose(tf.exp(S))
        Q /= tf.reduce_sum(Q)
        K, B = tf.cast(tf.shape(Q), tf.float32)

        for _ in range(niter):
            u = tf.reduce_sum(Q, axis=1, keepdims=True)
            Q /= K*u
            v = tf.reduce_sum(Q, axis=0, keepdims=True)
            Q /= B*v

        return tf.transpose(Q / tf.reduce_sum(Q, axis=0, keepdims=True))

    # @property
    # def metrics(self):
    #     return [self.loss_tracker]

    # @tf.function
    # def call(self, inputs):
    #     x1, x2 = inputs  # treated as an iterator, not allowed in graph mode
    #     y1, y2 = self._projector(self._encoder(x1)), self._projector(self._encoder(x2))
    #     return y1, y2

    def _loss_func(self, q, y):
        # tf.stop_gradient(q) is not necessary here since q was not recorded on the gradient tape.
        return tf.reduce_mean(nn.softmax_cross_entropy_with_logits(labels=q, logits=y, axis=-1))

    def train_step(self, inputs):
        x1, x2 = inputs
        tau = self._config.temperature
        with tf.GradientTape() as tape:
            # z1, z2 = *self.call(inputs)
            z1 = tf.linalg.normalize(self._encoder(x1), axis=-1)
            z2 = tf.linalg.normalize(self._encoder(x2), axis=-1)
            # z1, z2 = self._encoder(x1), self._encoder(x2)
            # z1 /= tf.linalg.norm(z1, axis=-1)
            # z2 /= tf.linalg.norm(z2, axis=-1)
            y1, y2 = self._prototype(z1)/tau, self._prototype(z2)/tau # z^T c/tau of (eq.2)
            # p1, p2 = nn.softmax(y1, axis=-1), tf.nn.softmax(y2, axis=-1)
            with tape.stop_recording():
                q1, q2 = self.sinkhorn(y1), self.sinkhorn(y2)

            loss = (self._loss_func(q1, y2) + self._loss_func(q2, y1)) / 2

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
