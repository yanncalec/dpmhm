""" SimCLR v1.

References
----------
- Chen, T., Kornblith, S., Norouzi, M., Hinton, G., 2020. A Simple Framework for Contrastive Learning of Visual Representations, in: International Conference on Machine Learning. Presented at the International Conference on Machine Learning, PMLR, pp. 1597–1607.
- Supplementary materials

Code
----
- [SimCLR Link](https://github.com/google-research/simclr)

Notes
-----
SimCLR benefits from
- larger batch size and more training steps,
- deeper and wider networks.
"""

from dataclasses import dataclass, field

import tensorflow as tf
# import tensorflow_addons as tfa
import tensorflow_models as tfm
# import tensorflow_probability as tfp

from tensorflow import keras, linalg
from tensorflow.keras import models, layers, regularizers, callbacks, losses
from tensorflow.keras.applications import resnet

from dpmhm.models.lr_scheduler import WarmupCosineDecay
from dpmhm.models.losses import NT_Xent

import logging
logger = logging.getLogger(__name__)


# """
# Iteration over a tensor is not allowed in graph mode:
# [losses.cosine_similarity(x, Y) for x in X], X has shape `(batch, dim)`

# use instead:
# losses.cosine_similarity(tf.expand_dims(X,1), Y)

# Test:
# V1 = tf.stack([losses.cosine_similarity(x, Y) for x in X])
# V2 = losses.cosine_similarity(tf.expand_dims(X,1), Y)
# V1-V2 is all zero
# """

@dataclass
class Config:
    input_shape:tuple = (224, 224)
    batch_size:int = 256  # 256 to 8192
    epochs:int = 100
    training_steps:int = 1000

    train_encoder:bool = True  # train the encoder?
    projector_units:int = 8192  # dimension of the projector's layers
    use_bias:bool = False  # use bias in the projector layers
    # weight_decay:float = 1.5e-6  # l2 regularization

    temperature:float = 1.0

    @classmethod
    def from_dict(cls, obj: dict):
        return cls(**obj)

    def optimizer(self, learning_rate=0.3, weight_decay_rate=1e-6, momentum=0.9):
        try:
            # with warmup
            lr = WarmupCosineDecay(
                learning_rate * self.batch_size / 256,
                # 0.075 * np.sqrt(self.batch_size),
                0,
                10,
                self.training_steps
            )
        except Exception as msg:
            # without warmup
            logger.exception(msg)
            lr = keras.optimizers.schedules.CosineDecay(
                learning_rate * self.batch_size / 256,
                self.training_steps
            )

        return tfm.optimization.lars_optimizer.LARS(weight_decay_rate=weight_decay_rate, learning_rate=lr, momentum=momentum)

        # return keras.optimizers.SGD(learning_rate=lr, momentum=momentum)
        # return tfa.optimizers.SGDW(weight_decay=weight_decay, learning_rate=lr, momentum=momentum)


class SimCLR(models.Model):
    """SimCLR.
    """
    def __init__(self, input_shape, c:Config):
        super().__init__()
        self._config = c
        # self._loss_func = lambda X,Y: NT_Xent_sym(X,Y,tau)
        self._loss_func = lambda X,Y: NT_Xent(X,Y,c.temperature) + NT_Xent(Y,X,c.temperature)
        self.loss_tracker = keras.metrics.Mean(name='loss')
        # self.mae_metric = keras.metrics.MeanAbsoluteError(name="mae")

        # config for the network
        # weights can also be loaded from a path, which takes the same amount of time ~1s
        self._encoder = resnet.ResNet50(include_top=False, weights='imagenet', input_shape=input_shape, pooling='avg')
        self._encoder.trainable = c.train_encoder

        self._projector = models.Sequential([
            layers.Flatten(name='flatten'),
            layers.Dense(4096, activation='relu', name='fc1'),
            layers.BatchNormalization(),
            layers.Dense(256, activation=None, name='fc2'),
        ], name='projector')

    @tf.function
    def call(self, inputs):
        x1, x2 = inputs  # treated as an iterator, not allowed in graph mode
        y1, y2 = self._projector(self._encoder(x1)), self._projector(self._encoder(x2))
        return y1, y2

    # @tf.function
    # def call(self, inputs):
    #     x1 = inputs[0]; x2 = inputs[1]
    #     y1, y2 = self._projector(self._encoder(tf.expand_dims(x1,0))), self._projector(self._encoder(tf.expand_dims(x2,0)))
    #     # https://stackoverflow.com/questions/58387852/what-does-please-wrap-your-loss-computation-in-a-zero-argument-lambda-means
    #     # https://www.tensorflow.org/api_docs/python/tf/keras/layers/Layer?version=nightly#add_loss
    #     self.add_loss(lambda: self._loss_func(y1, y2))
    #     return y1, y2

    def train_step(self, inputs):
        # print(f"Eager execution mode: {tf.executing_eagerly()}")
        # https://keras.io/guides/customizing_what_happens_in_fit
        x1, x2 = inputs  # treated as an iterator, not allowed in graph mode
        with tf.GradientTape() as tape:
            y1, y2 = self._projector(self._encoder(x1)), self._projector(self._encoder(x2))
            loss = self._loss_func(y1, y2)

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

# call and train_step:
# https://github.com/tensorflow/tensorflow/issues/54281
