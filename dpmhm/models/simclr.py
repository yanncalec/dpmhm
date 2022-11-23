"""
SimCLR v1:
- Chen, T., Kornblith, S., Norouzi, M., Hinton, G., 2020. A Simple Framework for Contrastive Learning of Visual Representations, in: International Conference on Machine Learning. Presented at the International Conference on Machine Learning, PMLR, pp. 1597–1607.

SimCLR v2:
- Chen, T., Kornblith, S., Swersky, K., Norouzi, M., Hinton, G.E., 2020. Big Self-Supervised Models are Strong Semi-Supervised Learners, in: Advances in Neural Information Processing Systems. Curran Associates, Inc., pp. 22243–22255.

Code:
https://github.com/google-research/simclr

Note: use large batch size.
"""

import sys
import tensorflow as tf
from tensorflow import keras, linalg
from tensorflow.keras import models, layers, regularizers, callbacks, losses
from tensorflow.keras.applications import resnet

# import numpy as np

"""
Iteration over a tensor is not allowed in graph mode:
[losses.cosine_similarity(x, Y) for x in X], X has shape `(batch, dim)`

use instead:
losses.cosine_similarity(tf.expand_dims(X,1), Y)

Test:
V1 = tf.stack([losses.cosine_similarity(x, Y) for x in X])
V2 = losses.cosine_similarity(tf.expand_dims(X,1), Y)
V1-V2 is all zero
"""

# @tf.function
def _NT_Xent_norm_vec(X, Y, full:bool=False) -> tf.Tensor:
    """Compute the normalization vector of the NT-Ext loss.
    """
    if full:
        Nx = tf.reduce_sum(-losses.cosine_similarity(tf.expand_dims(X,1), Y), axis=-1)
    else:  # original version
        Sx = -losses.cosine_similarity(tf.expand_dims(X,1), Y)
        # Nx = tf.reduce_sum(tf.linalg.set_diag(Sx,tf.zeros(tf.shape(X)[1])), axis=-1)
        Sx -= linalg.diag(linalg.diag_part(Sx))
        Nx = tf.reduce_sum(Sx, axis=-1)
    return Nx


def NT_Xent(X, Y, tau=1e-1):
    """Normalized temperature-scaled cross entropy loss.

    - Sohn, K. Improved deep metric learning with multi-class n-pair loss objective. In Advances in neural information processing systems, pp. 1857–1865, 2016.
    - Oord, A. v. d., Li, Y., and Vinyals, O. Representation learning with contrastive predictive coding. arXiv preprint arXiv:1807.03748, 2018.
    - Wu, Z., Xiong, Y., Yu, S. X., and Lin, D. Unsupervised feature learning via non-parametric instance discrimination. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 3733–3742, 2018.

    Note: `cosine_similarity(x,x)==-1`.
    """
    B = tf.cast(tf.shape(X)[0], tf.float32)  # batch size
    # Original version:
    S = tf.reduce_sum(-losses.cosine_similarity(X, Y)) / tau
    Nx = _NT_Xent_norm_vec(X, Y) / tau
    return -(S - tf.reduce_logsumexp(Nx)) / B
    # # Modified version:
    # S = tf.reduce_sum(losses.cosine_similarity(X, Y)) / tau
    # Nx = - get_NT_Xent_norm_vec(X, Y) / tau
    # return (S - tf.reduce_logsumexp(Nx)) / B


def NT_Xent_sym(X, Y, tau:float=1e-1):
    """Normalized temperature-scaled Cross entropy loss, symmetrized version.
    """
    B = tf.cast(tf.shape(X)[0], tf.float32)  # batch size
    S = tf.reduce_sum(losses.cosine_similarity(X, Y)) / tau
    Nx = - _NT_Xent_norm_vec(X, Y) / tau
    Ny = - _NT_Xent_norm_vec(Y, X) / tau
    return (S - tf.reduce_logsumexp(Nx+Ny)) / B


class SimCLR(models.Model):
    def __init__(self, input_shape, train_encoder:bool=True, tau:float=1e-1, **kwargs):
        super().__init__()
        # self._loss_func = lambda X,Y: NT_Xent_sym(X,Y,tau)
        self._loss_func = lambda X,Y: NT_Xent(X,Y,tau) + NT_Xent(Y,X,tau)
        self.loss_tracker = keras.metrics.Mean(name='loss')
        # self.mae_metric = keras.metrics.MeanAbsoluteError(name="mae")

        # config for the network
        # weights can also be loaded from a path, which takes the same amount of time ~1s
        self._encoder = resnet.ResNet50(include_top=False, weights='imagenet', input_shape=input_shape, pooling='avg')
        self._encoder.trainable = train_encoder

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
        with tf.GradientTape() as tape:
            loss = self._loss_func(*self.call(inputs))

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

class SimCLR_v2:
    pass