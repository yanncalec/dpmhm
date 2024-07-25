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

from ..losses import InfoNCE, InfoNCE_sim
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
    def on_train_batch_end(self, batch, logs=None):
        # Tricky workout:
        # EMA Variables must be initialized here, after the model has been built. If initialized in the method `on_train_begin()`, it will not get the right number of weights.
        try:
            # Update the value of variables
            for v, w in zip(self._variables, self.model._online.weights):
                v.assign(w)
            # EMA update of the target network
            self.model._target.set_weights(
                [self.model._ema.average(v) for v in self._variables]
            )
        except:
            # Initialization of EMA instance. This should happen only once after the first train on a batch.
            logger.info("Create the EMA instance...")
            # `EMA` is a tensorflow functionality, so it must use `tf.Variable`, not `keras.Variable`.
            self._variables = [tf.Variable(v) for v in self.model._online.weights]  # in Keras `.weights` is identical to `.variables`
            self.model._ema.apply(self._variables)  # create a shadow copy


class MoCo(models.Model):
    def __init__(self, input_shape:tuple, *, output_dim:int=256, tau:float=0.1, sim:bool=False, momentum:float=0.999, memsize:int=100, name:str='VGG16', encoder_kwargs:dict={}):
        """Initializer for MoCo.

        Parameters
        ----------
        input_shape
            shape of the input data in channel last format
        output_dim, optional
            dimension of projector's output, by default 256
        tau, optional
            temperature, by default 0.1
        sim, optional
            use cosine similarity based InfoNCE loss, by default False
        momentum, optional
            momentum for updating the target network, by default 0.999
        memsize, optional
            size of the memory, by default 100
        name, optional
            name of pretrained Keras model for the baseline encoder, by default 'VGG16'
        encoder_kwargs, optional
            keyword arguments for the baseline encoder, by default {}
        """
        super().__init__()
        self._input_shape = input_shape
        self._output_dim = output_dim  # output dimension of the final projection layer
        self._tau = tau
        self._memsize = memsize

        self._memory = keras.Variable(
            ops.zeros((self._memsize, self._output_dim)),
            trainable=False
        )

        if sim:
            self._loss = lambda X,Y,K: InfoNCE_sim(X, Y, K, self._tau)
        else:
            self._loss = lambda X,Y,K: InfoNCE(X, Y, K, self._tau)

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
            layers.BatchNormalization(),
            layers.Dense(output_dim, activation=None, name='fc3'),
        ], name='projector')

        self._online = models.Sequential([
            self._encoder,
            self._projector,
        ], name='online')  # online network

        self._target = models.clone_model(self._online)  # momentum encoder network
        self._target.trainable = False

        # self._ema_rate = momentum
        self._ema = tf.train.ExponentialMovingAverage(momentum)

    # def build(self, input_shape):
    #     batch_size = input_shape[0][0]
    #     # output_dim = self._online.output_shape[-1]  # not available yet
    #     self._memsize = batch_size * self._maxlen
    #     # Variable initialization doesn't play well here with Jax
    #     self._memory = keras.Variable(
    #         ops.zeros((self._memsize, self._output_dim)),
    #         trainable=False
    #     )

    def call(self, inputs, training=True):
        # query and key (positive) instances
        xq, xk = inputs  # treated as an iterator, not allowed in graph mode
        yq = self._online(xq, training=training)
        yk = self._target(xk, training=training)
        # logger.info(f"call: {ops.shape(yq)}, {ops.shape(yk)}")

        self.add_loss(
            # Torch backend:
            # use `ops.copy()` to avoid the runtime error "... is at version 1; expected version 0 instead".
            self._loss(yq, yk, ops.copy(self._memory))
        )

        # `call` function must not have any side effect. To save inner states, we must use `.assign()` available for a Keras variable.
        # Do not use the incremental update scheme for the memory, fix its size instead.
        self._memory.assign(
            ops.take(
                ops.concatenate([yk, self._memory], axis=0),
                range(self._memsize), axis=0
            )
        )

        return yq, yk
