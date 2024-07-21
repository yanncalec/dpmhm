""" SimCLR.

References
----------
- Chen, T., Kornblith, S., Norouzi, M., Hinton, G., 2020. A Simple Framework for Contrastive Learning of Visual Representations, in: International Conference on Machine Learning. Presented at the International Conference on Machine Learning, PMLR, pp. 1597–1607.
- Chen, T., Kornblith, S., Swersky, K., Norouzi, M., Hinton, G.E., 2020. Big Self-Supervised Models are Strong Semi-Supervised Learners, in: Advances in Neural Information Processing Systems. Curran Associates, Inc., pp. 22243–22255.

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

# import tensorflow as tf
# # import tensorflow_addons as tfa
# import tensorflow_models as tfm
# # import tensorflow_probability as tfp
# from tensorflow import keras, linalg

# import keras
from keras import models, layers, regularizers, callbacks, losses
# from keras.applications import VGG16, resnet

from dpmhm.models.losses import NT_Xent
# from dpmhm.models.lr_scheduler import WarmupCosineDecay
# from ..config import AbstractConfig
from ..ul import autoencoder
from ..pretrained import get_base_encoder


import logging
logger = logging.getLogger(__name__)


# """
# Iteration over a tensor is not allowed in graph mode:
# [losses.cosine_similarity(x, Y) for x in X], X has shape `(batch, dim)`

# use instead:
# losses.cosine_similarity(tf.expand_dims(X,1), Y)

# Test in eager mode
# V1 = tf.stack([losses.cosine_similarity(x, Y) for x in X])
# V2 = losses.cosine_similarity(tf.expand_dims(X,1), Y)
# V1-V2 is all zero
# """


class SimCLR(models.Model):
    """SimCLR.
    """
    def __init__(self, input_shape, *, name:str='VGG16', tau:float=1.0, **kwargs):
        """
        Parameters
        ----------
        input_shape
            shape of the input
        name, optional
            name of pretrained Keras model, by default 'VGG16'
        tau, optional
            temperature, by default 1.0
        kwargs
            keyword arguments to the Keras pretrain model in `keras.applications`
        """
        super().__init__()
        self._input_shape = input_shape
        self._tau = tau
        # self.loss_tracker = keras.metrics.Mean(name='loss')
        # self.mae_metric = keras.metrics.MeanAbsoluteError(name="mae")

        try:
            self._encoder = get_base_encoder(input_shape, name, **kwargs)
        except:
            cfg = autoencoder.Config(input_shape=input_shape)
            self._encoder = autoencoder.CAES(cfg).encoder

        # self._encoder.trainable = True

        self._projector = models.Sequential([
            # A dense layer applies only on the last dimension while preserving all other dimensions as batch.
            layers.Flatten(name='flatten'),
            layers.Dense(4096, activation='relu', name='fc1'),
            layers.BatchNormalization(),
            layers.Dense(256, activation='relu', name='fc2'),
            layers.BatchNormalization(),
            layers.Dense(64, activation=None, name='fc3'),
        ], name='projector')

    def call(self, inputs, training=True):
        """
        The dataset passed to `model.fit()` must have the element type `((x1, x2), y_fake)` with `y_fake` a fake label, otherwise `inputs` can not be determined correctly from (x1, x2)
        """
        x1, x2 = inputs  # ok if inputs is tuple, error in tensorflow's autograph mode: iterating on a tensor
        # x1, x2 = inputs[0], inputs[1]  # not considered as iterating on a tensor
        #
        # if `inputs` is packed into a single tensor
        # x1 = ops.take(inputs, 0, axis=1)
        # x2 = ops.take(inputs, 1, axis=1)

        y1 = self._projector(self._encoder(x1), training=training)
        y2 = self._projector(self._encoder(x2), training=training)

        # Depending on the way loss is handled, two options:
        # 1. Define a custom loss with fake true label and pass it to `model.compile(loss=...)`
        # ```
        # def _loss(y_true, y_pred):
        #     y1, y2 = y_pred[0], y_pred[1]
        #     # y1 = ops.take(y_pred, 0, axis=0)
        #     # y2 = ops.take(y_pred, 1, axis=0)
        #     return NT_Xent(y1, y2) + NT_Xent(y2, y1)
        # ```
        # and the return of `.call()` shoud be a single tensor
        # `return ops.stack([y1, y2])` which is passed automatically to the custom loss.
        #
        # 2. Add manually the loss using `.add_loss()` inside `.call()`. In this case the loss is not needed in `model.compile()` and `.call()` can return a tuple.
        self.add_loss(
            NT_Xent(y1, y2, self._tau) + NT_Xent(y2, y1, self._tau)
        )
        return y1, y2
