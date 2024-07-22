"""Classes of Auto-Encoder.
"""

import keras
from keras import layers, models, regularizers

from ..custom import TConv2D, TDense


class CAES(models.Model):
    """Convolution Auto-Encoder stacks.

    Notes
    -----
    Shape `(H,W)` of the input tensor must be power of 2.
    """
    def __init__(self, input_shape:tuple, *, kernel_size:tuple=(3,3), activation:str='relu', padding:str='same', pool_size:tuple=(2,2), strides:tuple=(2,2), n_embedding:int=128, a_reg:float=0.):
        super().__init__()

        # Use more blocks and larger kernel size to get more smoothing in the reconstruction.
        layers_encoder = [
            layers.Input(shape=input_shape, name='input_enc'),
            # Block 1
            layers.Conv2D(32, kernel_size=kernel_size, activation=activation, padding=padding, name='conv1_enc'),
            layers.MaxPooling2D(pool_size=pool_size, strides=strides, name='pool1_enc'),
            layers.BatchNormalization(name='bn1_enc'), # by default axis=-1 for channel-last

            # Block 2
            layers.Conv2D(64, kernel_size=kernel_size, activation=activation, padding=padding, name='conv2_enc'),
            layers.MaxPooling2D(pool_size=pool_size, strides=strides, name='pool2_enc'),
            layers.BatchNormalization(name='bn2_enc'),

            # Block 3
            layers.Conv2D(128, kernel_size=kernel_size, activation=activation, padding=padding, name='conv3_enc'),
            layers.MaxPooling2D(pool_size=pool_size, strides=strides, name='pool3_enc'),
            layers.BatchNormalization(name='bn3_enc'),

            # Block fc
            layers.Flatten(name='flatten'),
            layers.Dense(n_embedding, activation=activation,activity_regularizer=regularizers.L1(a_reg), name='fc1_enc') if a_reg > 0
            else layers.Dense(n_embedding, activation=activation, name='fc1_enc')
        ]

        self.encoder = models.Sequential(layers_encoder, name='encoder')

        layers_decoder = [
            layers.Input(shape=self.encoder.layers[-1].output.shape[1:], name='input_dec'),
            # Block fc
            layers.Dense(self.encoder.layers[-2].output.shape[-1], activation=activation, activity_regularizer=regularizers.L1(a_reg), name='fc1_dec') if a_reg > 0 else layers.Dense(self.encoder.layers[-2].output.shape[-1], activation=activation, name='fc1_dec'),
            layers.Reshape(self.encoder.layers[-3].output.shape[1:], name='reshape'),

            # Block 3
            layers.BatchNormalization(name='bn3_dec'),
            layers.UpSampling2D(strides, name='ups3_dec'),
            layers.Conv2DTranspose(64, kernel_size=kernel_size, activation=activation, padding=padding, name='tconv3_dec'),

            # Block 2
            layers.BatchNormalization(name='bn2_dec'),
            layers.UpSampling2D(strides, name='ups2_dec'),
            layers.Conv2DTranspose(32, kernel_size=kernel_size, activation=activation, padding=padding, name='tconv2_dec'),

            # Block 1
            layers.BatchNormalization(name='bn1_dec'),
            layers.UpSampling2D(strides, name='ups1_dec'),
            layers.Conv2DTranspose(input_shape[-1], kernel_size=kernel_size, activation=None, padding=padding, name='tconv1_dec'),
        ]

        self.decoder = models.Sequential(layers_decoder, name='decoder')
        # self.decoder.build()

        # self.autoencoder = models.Sequential(layers_encoder+layers_decoder, name='auto-encoder')
        # self.build(input_shape=(None, *input_shape))

    def call(self, x):
        return self.decoder(self.encoder(x))


class CAES_1D(models.Model):
    """Convolution Auto-Encoder stacks for signal.
    """
    def __init__(self, input_shape:tuple, *, kernel_size:tuple=(3,3), activation:str='relu', padding:str='same', pool_size:tuple=(2,2), strides:tuple=(2,2), n_embedding:int=128):
        super().__init__()

        layers_encoder = [
            layers.Input(shape=input_shape, name='input_enc'),
            # Block 1
            layers.Conv1D(32, kernel_size=kernel_size, activation=activation, padding=padding, name='conv1_enc'),
            layers.MaxPooling1D(pool_size=pool_size, strides=strides, name='pool1_enc'),
            # Block 2
            layers.Conv1D(64, kernel_size=kernel_size, activation=activation, padding=padding, name='conv2_enc'),
            layers.MaxPooling1D(pool_size=pool_size, strides=strides, name='pool2_enc'),
            # Block 3
            layers.Conv1D(128, kernel_size=kernel_size, activation=activation, padding=padding, name='conv3_enc'),
            layers.MaxPooling1D(pool_size=pool_size, strides=strides, name='pool3_enc'),
            # Block fc
            layers.Flatten(name='flatten'),
            layers.Dense(n_embedding, activation=activation,
            # activity_regularizer=regularizers.L1(1e-5),   # sparse ae
            name='fc1_enc'),
        ]

        self.encoder = models.Sequential(layers_encoder, name='encoder')

        layers_decoder = [
            layers.Input(shape=self.encoder.layers[-1].output.shape[1:], name='input_dec'),
            # Block fc
            layers.Dense(self.encoder.layers[-2].output.shape[-1], activation=activation,
            # activity_regularizer=regularizers.L1(1e-5),
            name='fc1_dec'),
            layers.Reshape(self.encoder.layers[-3].output.shape[1:], name='reshape'),

            # Block 3
            layers.UpSampling1D(strides, name='ups3_dec'),
            layers.Conv1DTranspose(64, kernel_size=kernel_size, activation=activation, padding=padding, name='tconv3_dec'),

            # Block 2
            layers.UpSampling1D(strides, name='ups2_dec'),
            layers.Conv1DTranspose(32, kernel_size=kernel_size, activation=activation, padding=padding, name='tconv2_dec'),

            # Block 1
            layers.UpSampling1D(strides, name='ups1_dec'),
            layers.Conv1DTranspose(input_shape[-1], kernel_size=kernel_size, activation=None, padding=padding, name='tconv1_dec'),
        ]

        self.decoder = models.Sequential(layers_decoder, name='decoder')
        # self.decoder.build()

        self.autoencoder = models.Sequential(layers_encoder+layers_decoder, name='auto-encoder')
        # self.build(input_shape=(None, *input_shape))

    def call(self, x):
        return self.decoder(self.encoder(x))


class CAES_ws(models.Model):
    """Convolution Auto-Encoder stacks with weight sharing.

    Notes
    -----
    - The model fails to train with `decoder.trainable=False`.
    - MSE much larger than that of no weight sharing.
    """
    def __init__(self, input_shape:tuple, *, kernel_size:tuple=(3,3), activation:str='relu', padding:str='same', pool_size:tuple=(2,2), strides:tuple=(2,2), n_embedding:int=128):
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
            layers.Input(shape=self.encoder.layers[-1].output.shape[1:], name='input_dec'),
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
