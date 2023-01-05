"""Auto-Encoder"""

import tensorflow as tf
from tensorflow.keras import layers, models #, regularizers
from dataclasses import dataclass
from ..custom import TConv2D, TDense
from .. import AbstractConfig


@dataclass
class Config(AbstractConfig):
    """Global parameters.
    """
    # Dimension of hidden representation
    n_embedding:int = 128

    # Parameters of ConvNet
    kernel_size:tuple = (3,3)
    activation:str = 'relu'
    padding:str = 'same' # 'valid' or 'same'
    pool_size:tuple = (2,2)
    strides:tuple = (2,2)
    # use_bias:bool = False
    # activity_regularizer = None

    def optimizer(self):
        return tf.keras.optimizers.Adam()


class CAES(models.Model):
    """Convolution Auto-Encoder stacks.

    Notes
    -----
    Use more blocks and larger kernel size to get more smoothing in the reconstruction.
    """
    def __init__(self, c:Config):
        self._config = c
        input_shape = c.input_shape  # n_bands, n_frames, c.n_channels
        kernel_size = c.kernel_size
        activation = c.activation
        padding = c.padding
        strides = c.strides
        pool_size = c.pool_size
        n_embedding  = c.n_embedding

        super().__init__()

        layers_encoder = [
            layers.Input(shape=input_shape, name='input_enc'),
            # Block 1
            layers.Conv2D(32, kernel_size=kernel_size, activation=activation, padding=padding, name='conv1_enc'),
            layers.MaxPooling2D(pool_size=pool_size, strides=strides, name='pool1_enc'),
            # Block 2
            layers.Conv2D(64, kernel_size=kernel_size, activation=activation, padding=padding, name='conv2_enc'),
            layers.MaxPooling2D(pool_size=pool_size, strides=strides, name='pool2_enc'),
            # Block 3
            layers.Conv2D(128, kernel_size=kernel_size, activation=activation, padding=padding, name='conv3_enc'),
            layers.MaxPooling2D(pool_size=pool_size, strides=strides, name='pool3_enc'),
            # Block fc
            layers.Flatten(name='flatten'),
            layers.Dense(n_embedding, activation=activation,
            # activity_regularizer=regularizers.L1(1e-5),   # sparse ae
            name='fc1_enc'),
        ]

        self.encoder = models.Sequential(layers_encoder, name='encoder')

        layers_decoder = [
            layers.Input(shape=self.encoder.layers[-1].output_shape[1:], name='input_dec'),
            # Block fc
            layers.Dense(self.encoder.layers[-2].output_shape[-1], activation=activation,
            # activity_regularizer=regularizers.L1(1e-5),
            name='fc1_dec'),
            layers.Reshape(self.encoder.layers[-3].output_shape[1:], name='reshape'),
            # Block 3
            layers.UpSampling2D(strides, name='ups3_dec'),
            layers.Conv2DTranspose(64, kernel_size=kernel_size, activation=activation, padding=padding, name='tconv3_dec'),
            # Block 2
            layers.UpSampling2D(strides, name='ups2_dec'),
            layers.Conv2DTranspose(32, kernel_size=kernel_size, activation=activation, padding=padding, name='tconv2_dec'),
            # Block 1
            layers.UpSampling2D(strides, name='ups1_dec'),
            layers.Conv2DTranspose(input_shape[-1], kernel_size=kernel_size, activation=None, padding=padding, name='tconv1_dec'),
        ]

        self.decoder = models.Sequential(layers_decoder, name='decoder')
        # self.decoder.build()

        self.autoencoder = models.Sequential(layers_encoder+layers_decoder, name='auto-encoder')
        # self.build(input_shape=(None, *input_shape))

    def call(self, x):
        return self.decoder(self.encoder(x))


class CAES_ws(models.Model):
    """Convolution Auto-Encoder stacks with weight sharing.

    Notes:
    - The model fails to train with `decoder.trainable=False`.
    - MSE much larger than the case of no weight sharing.
    """
    def __init__(self, c:Config):
        self._config = c
        input_shape = c.input_shape
        kernel_size = c.kernel_size
        activation = c.activation
        padding = c.padding
        strides = c.strides
        pool_size = c.pool_size
        n_embedding  = c.n_embedding

        super().__init__()

        layers_encoder = [
            layers.Input(shape=input_shape, name='input_enc'),
            # Block 1
            layers.Conv2D(32, kernel_size=kernel_size, activation=activation, padding=padding, name='conv1_enc'),
            layers.MaxPooling2D(pool_size=pool_size, strides=strides, name='pool1_enc'),
            # Block 2
            layers.Conv2D(64, kernel_size=kernel_size, activation=activation, padding=padding, name='conv2_enc'),
            layers.MaxPooling2D(pool_size=pool_size, strides=strides, name='pool2_enc'),
            # Block 3
            layers.Conv2D(128, kernel_size=kernel_size, activation=activation, padding=padding, name='conv3_enc'),
            layers.MaxPooling2D(pool_size=pool_size, strides=strides, name='pool3_enc'),
            # Block 4
            layers.Conv2D(256, kernel_size=kernel_size, activation=activation, padding=padding, name='conv4_enc'),
            layers.MaxPooling2D(pool_size=pool_size, strides=strides, name='pool4_enc'),
            # Block 5
            layers.Conv2D(512, kernel_size=kernel_size, activation=activation, padding=padding, name='conv5_enc'),
            layers.MaxPooling2D(pool_size=pool_size, strides=strides, name='pool5_enc'),
            # Block fc
            layers.Flatten(name='flatten'),
            layers.Dense(n_embedding, activation=activation, name='fc1_enc'),
        ]

        self.encoder = models.Sequential(layers_encoder, name='encoder')

        layers_decoder = [
            layers.Input(shape=self.encoder.layers[-1].output_shape[1:], name='input_dec'),
            # Block fc
            TDense(self.encoder.get_layer('fc1_enc'), name='fc1_dec'),
            layers.Reshape(self.encoder.get_layer('flatten').input_shape[1:], name='reshape'),
            # Block 5
            layers.UpSampling2D(strides, name='ups5_dec'),
            TConv2D(self.encoder.get_layer('conv5_enc'), name='tconv5_dec'),
            # Block 4
            layers.UpSampling2D(strides, name='ups4_dec'),
            TConv2D(self.encoder.get_layer('conv4_enc'), name='tconv4_dec'),
            # Block 3
            layers.UpSampling2D(strides, name='ups3_dec'),
            TConv2D(self.encoder.get_layer('conv3_enc'), name='tconv3_dec'),
            # Block 2
            layers.UpSampling2D(strides, name='ups2_dec'),
            TConv2D(self.encoder.get_layer('conv2_enc'), name='tconv2_dec'),
            # Block 1
            layers.UpSampling2D(strides, name='ups1_dec'),
            TConv2D(self.encoder.get_layer('conv1_enc'), name='tconv1_dec'),
        ]
        self.decoder = models.Sequential(layers_decoder, name='decoder')
        # self.decoder.trainable = False

        self.autoencoder = models.Sequential(layers_encoder+layers_decoder, name='auto-encoder')
        # self.build(input_shape=(None, *input_shape))

    def call(self, x):
        return self.decoder(self.encoder(x))
