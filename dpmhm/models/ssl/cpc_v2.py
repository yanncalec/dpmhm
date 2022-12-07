"""Contrastive Predictive Coding. TODO.

References
----------
Oord, A. van den, Li, Y., Vinyals, O., 2018. Representation learning with contrastive predictive coding. arXiv preprint arXiv:1807.03748.

Code
----
https://github.com/Spijkervet/contrastive-predictive-coding
https://github.com/giovannimaffei/contrastive_predictive_coding

"""

# import sys
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras, linalg
from tensorflow.keras import models, layers, regularizers, callbacks, losses
from tensorflow.keras.applications import resnet

from dataclasses import dataclass, field

"""
X = tf.random.normal([4, 3])
Y = tf.random.normal([4, 3])

C = losses.cosine_similarity(X[:,None,:], Y[None,:,:])  #
assert C[i,j] == losses.cosine_similarity(X[i,:], Y[j,:])

C = losses.cosine_similarity(X[None,:,:],Y[:,None,:])  # ~ 4x4
# C = losses.cosine_similarity(X[:,:], Y[:,None,:])  # same same
assert C[i,j] == losses.cosine_similarity(X[j,:], Y[i,:])

C = losses.cosine_similarity(X, Y)  # no broadcast, ~ 4
assert C[i] == == losses.cosine_similarity(X[i,:], Y[i,:])
"""


@dataclass
class Config:
    # AUTO = tf.data.AUTOTUNE
    time_steps = 10
    encoder_units = 512

    batch_size:int = 512
    epochs:int = 5
    # n_batch:int = ?  # data_size/batch_size
    training_steps:int = 10**4  # epochs * n_batch

    # encoder:dict = field(default_factory=lambda: dict(include_top=False, weights='imagenet', pooling='avg'))
    # train_encoder:bool = True

    # projector_units:int = 2048
    # predictor_units:int = 512
    use_bias:bool = False
    # weight_decay:float = 0.0005

    # projector:dict = field(default_factory=lambda: dict(units=2048, use_bias=False))

    @classmethod
    def from_dict(cls, obj: dict):
        return cls(**obj)

    def optimizer(self, learning_rate=0.05, weight_decay=1e-4, momentum=0.9):
        lr = keras.optimizers.schedules.CosineDecay(
            learning_rate * self.batch_size / 256,
            self.training_steps
        )
        return tfa.optimizers.SGDW(weight_decay=weight_decay, learning_rate=lr, momentum=momentum)


class CPC(models.Model):
    """CPC raw wavform.
    """
    def __init__(self, input_shape, c:Config):
        # input has shape (batch, time, channel)
        # if input has (batch, feature, time, channel), convert first to (batch, time, feature*channel)

        # input_shape, train_encoder:bool=True, tau:float=1e-1, **kwargs):
        super().__init__()
        self._config = c
        self.loss_tracker = keras.metrics.Mean(name='loss')
        # self.mae_metric = keras.metrics.MeanAbsoluteError(name="mae")

        # config for the network
        self._encoder = models.Sequential([
            layers.Input(input_shape),
            layers.Conv1D(512, kernel_size=10, strides=5, activation='relu', padding='valid', use_bias=True),
            # layers.BatchNormalization(),
            layers.Conv1D(512, kernel_size=8, strides=4, activation='relu', padding='valid', use_bias=True),
            layers.Conv1D(512, kernel_size=4, strides=2, activation='relu', padding='valid', use_bias=True),
            layers.Conv1D(512, kernel_size=4, strides=2, activation='relu', padding='valid', use_bias=True),
            layers.Conv1D(512, kernel_size=4, strides=2, activation='relu', padding='valid', use_bias=True),
            # ResNet1D(input_shape=input_shape, units=512, pooling='avg'),
        ], name='encoder')  # (batch, time, feature)
        # self._encoder.trainable = c.train_encoder

        self._autoregressor = models.Sequential([
            layers.GRU(256, return_sequences=True),
        ], name='autoregressor')  # (batch, time, state)

        self._projector = models.Sequential([
            layers.Dense(c.time_steps * 512, use_bias=False, activation=None, name='projector'),  # (batch, time, K*feature)
            # layers.Reshape()
        ])

        # self.trainable_variables = self._encoder.trainable_variables + self._projector.trainable_variables + self.predictor.trainable_variables

    # @property
    # def metrics(self):
    #     return [self.loss_tracker]

    # @tf.function
    # def call(self, inputs):
    #     x1, x2 = inputs  # treated as an iterator, not allowed in graph mode
    #     y1, y2 = self._projector(self._encoder(x1)), self._projector(self._encoder(x2))
    #     return y1, y2

    def _loss_func(self, y, z):
        nb, nt, nf = tf.shape(y)
        nk = self._config.time_steps
        y = tf.reshape(y, (-1, nt, nk, nf//nk))

        # z = tf.expand_dims()
        loss = 0
        for k in range(nk):
            zt = z
            yt = y[:,:(nt-k),k,:]  # Wk * ct of (eq. 3)
            lt = tf.range(nt-k)  # for zt = z

            # zt = tf.roll(z, k, axis=1)
            # yt = y[:,:(nt-k),k,:]  # Wk * ct of (eq. 3)
            # lt = tf.zeros(nt-k)

            # zt = z[:,k:,:]
            # yt = y[:, :(nt-k), k, :]
            # lt = tf.zeros(nt-k)

            # F = tf.reduce_mean(
            #     losses.cosine_similarity(yt[:,:,None,:], zt[:,None,:,:])  # ~ (batch, time y, time z)
            #     , axis=0)

            F = tf.tensordot(yt, zt, axes=[[0,2], [0,2]]) / tf.cast(nb, tf.float32)  # ~ (time y, time z)
            # tf.print(k, nt, F.shape)

            loss += tf.reduce_mean(
                # tf.nn.softmax(F)
                losses.sparse_categorical_crossentropy(lt,F,from_logits=True)  # ~ nt-k
            )

        return loss

    def train_step(self, x):
        with tf.GradientTape() as tape:
            # z1, z2 = *self.call(inputs)
            z = self._encoder(x)
            c = self._autoregressor(z)
            y = self._projector(c)
            loss = self._loss_func(y, z)

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
