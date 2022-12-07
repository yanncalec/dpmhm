"""Barlow Twins.

References
----------
- Zbontar, J., Jing, L., Misra, I., LeCun, Y., Deny, S., 2021. Barlow Twins: Self-Supervised Learning via Redundancy Reduction, in: Proceedings of the 38th International Conference on Machine Learning. Presented at the International Conference on Machine Learning, PMLR, pp. 12310–12320.

Code
----
https://github.com/facebookresearch/barlowtwins

Notes
-----
Barlow Twins benefits very high-dimensional embeddings.
"""

from dataclasses import dataclass, field

import tensorflow as tf
# import tensorflow_addons as tfa
import tensorflow_models as tfm
# from tensorflow_models import optimization  # cannot import tensorflows_models.optimization directly
import tensorflow_probability as tfp

from tensorflow import keras, linalg
from tensorflow.keras import models, layers #, regularizers, callbacks, losses
from tensorflow.keras.applications import resnet
# from tensorflow_probability import stats
# from tensorflow_models.optimization.lars_optimizer import LARS

from dpmhm.models.lr_scheduler import WarmupCosineDecay

import logging
logger = logging.getLogger(__name__)


@dataclass
class Config:
    # AUTO = tf.data.AUTOTUNE
    # input_shape:tuple = (224, 224)
    batch_size:int = 2048  # can work well with 256 too
    epochs:int = 1000
    # n_batch:int = ?  # data_size/batch_size
    training_steps:int = 10**4  # epochs * n_batch

    train_encoder:bool = True  # train the encoder?

    projector_units:int = 8192  # dimension of the projector's layers
    use_bias:bool = False  # use bias in the projector layers
    # weight_decay:float = 1.5e-6  # l2 regularization

    loss_lambda:float = 5e-3  # lambda for the off-diagonal terms

    @classmethod
    def from_dict(cls, obj: dict):
        return cls(**obj)

    def optimizer(self, learning_rate=0.2, weight_decay_rate=1.5e-6, momentum=0.9):
        # Note: The original implementation uses a learning rate of 0.2 for the weights and 0.0048 for the biases and batch normalization parameters.

        try:
            # with warmup
            lr = WarmupCosineDecay(
                learning_rate * self.batch_size / 256,
                0,
                self.epochs // 10,
                self.training_steps
            )
        except Exception as msg:
            # without warmup
            logger.exception(msg)
            lr = keras.optimizers.schedules.CosineDecay(
                learning_rate * self.batch_size / 256,
                self.training_steps
            )

        # return keras.optimizers.SGD(learning_rate=lr, momentum=momentum)
        return tfm.optimization.lars_optimizer.LARS(weight_decay_rate=weight_decay_rate, learning_rate=lr, momentum=momentum)


class BarlowTwins(models.Model):
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
            layers.Dense(c.projector_units, use_bias=c.use_bias, activation='relu', name='proj_fc1'),
            layers.BatchNormalization(),
            # layers.Dense(c.projector_units, use_bias=c.use_bias, activation='relu', name='proj_fc2'),
            # layers.BatchNormalization(),
            layers.Dense(c.projector_units, use_bias=c.use_bias, activation=None, name='proj_fc3'),
            ], name='projector')

        # self._encoder_projector = models.Sequential([
        #     self._encoder,
        #     self._projector
        # ], name='encoder_projector')

        # self.trainable_variables = self._encoder.trainable_variables + self._projector.trainable_variables

    # @property
    # def metrics(self):
    #     return [self.loss_tracker]

    @tf.function
    def call(self, inputs):
        x1, x2 = inputs  # treated as an iterator, not allowed in graph mode
        y1, y2 = self._projector(self._encoder(x1)), self._projector(self._encoder(x2))
        return y1, y2

    # def _loss_func(self, p, z):
    #     return tf.reduce_mean(losses.cosine_similarity(p, tf.stop_gradient(z)), axis=0)

    def train_step(self, inputs):
        x1, x2 = inputs
        with tf.GradientTape() as tape:
            # z1, z2 = *self.call(inputs)
            z1, z2 = self._projector(self._encoder(x1)), self._projector(self._encoder(x2))
            corr = tfp.stats.correlation(z1, z2, sample_axis=0, event_axis=-1)
            loss = tf.reduce_sum((1-linalg.diag_part(corr))**2 + self._config.loss_lambda * (corr-linalg.diag(corr))**2)

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


__all__ = ['Config', 'BarlowTwins']