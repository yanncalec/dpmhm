"""VGGish model for AudioSet.

For implementations see:
https://github.com/tensorflow/models/tree/master/research/audioset/vggish
https://github.com/antoinemrcr/vggish2Keras
https://github.com/DTaoo/VGGish

For usage see:
https://github.com/DTaoo/VGGish/blob/master/evaluation.py

For Keras implementation see:
https://github.com/keras-team/keras/blob/master/keras/applications/vgg16.py

References
----------
Simonyan, K. and Zisserman, A. (2014) ‘Very Deep Convolutional Networks for Large-Scale Image Recognition’, arXiv:1409.1556 [cs] [Preprint]. Available at: http://arxiv.org/abs/1409.1556 (Accessed: 29 March 2019).

Hershey, S. et al. (2017) ‘CNN architectures for large-scale audio classification’, in 2017 ieee international conference on acoustics, speech and signal processing (icassp). IEEE, pp. 131–135.
"""

import tensorflow as tf
from tensorflow.keras import layers, models #, regularizers
from dataclasses import dataclass
from .. import AbstractConfig


@dataclass
class Config(AbstractConfig):
    """Global parameters for VGGish.

    See also:
    https://github.com/tensorflow/models/blob/master/research/audioset/vggish/params.py
    """
    # Number of classes
    n_classes:int = None

    # Dimension of embedding
    n_embedding:int = 128

    # Parameters of ConvNet
    kernel_size:tuple = (3,3)
    activation:str = None
    activation_classifier:str = None
    padding:str = 'same' # 'valid' or 'same'
    pool_size:tuple = (2,2)
    strides:tuple = (2,2)

    def optimizer(self):
        pass


def VGG11(c:Config) -> models.Model:
    """ConvNet model A or VGG 11 layers.
    """
    input_shape = c.input_shape
    kernel_size = c.kernel_size
    activation = c.activation
    activation_classifier = c.activation_classifier
    padding = c.padding
    strides = c.strides
    pool_size = c.pool_size
    output_dim = c.n_embedding
    n_classes = c.n_classes

    _layers = [
        layers.Input(shape=input_shape, name='input'),
        # Block 1
        layers.Conv2D(64, kernel_size=kernel_size, activation=activation, padding=padding, name='conv1'),
        layers.MaxPooling2D(pool_size=pool_size, strides=strides, name='pool1'),
        # Block 2
        layers.Conv2D(128, kernel_size=kernel_size, activation=activation, padding=padding, name='conv2'),
        layers.MaxPooling2D(pool_size=pool_size, strides=strides, name='pool2'),
        # Block 3
        layers.Conv2D(256, kernel_size=kernel_size, activation=activation, padding=padding, name='conv3_1'),
        layers.Conv2D(256, kernel_size=kernel_size, activation=activation, padding=padding, name='conv3_2'),
        layers.MaxPooling2D(pool_size=pool_size, strides=strides, name='pool3'),
        # Block 4
        layers.Conv2D(512, kernel_size=kernel_size, activation=activation, padding=padding, name='conv4_1'),
        layers.Conv2D(512, kernel_size=kernel_size, activation=activation, padding=padding, name='conv4_2'),
        layers.MaxPooling2D(pool_size=pool_size, strides=strides, name='pool4'),
        # Block 5
        layers.Conv2D(512, kernel_size=kernel_size, activation=activation, padding=padding, name='conv5_1'),
        layers.Conv2D(512, kernel_size=kernel_size, activation=activation, padding=padding, name='conv5_2'),
        layers.MaxPooling2D(pool_size=pool_size, strides=strides, name='pool5'),
        # Block fc
        layers.Flatten(name='flatten'),
        layers.Dense(4096, activation=activation, name='fc1_1'),
        layers.Dense(4096, activation=activation, name='fc1_2'),
        layers.Dense(output_dim, activation=None, name='embedding'),

        # Block classifier
        # layers.Dense(100, activation='relu', name='projector'),
        layers.Dense(n_classes, activation=activation_classifier, name='classifier')
    ]

    return models.Sequential(layers=_layers, name='VGGish-A')


def VGG13(c:Config) -> models.Model:
    """ConvNet model B or VGG 13 layers.
    """
    input_shape = c.input_shape
    kernel_size = c.kernel_size
    activation = c.activation
    activation_classifier = c.activation_classifier
    padding = c.padding
    strides = c.strides
    pool_size = c.pool_size
    output_dim = c.n_embedding
    n_classes = c.n_classes

    _layers = [
        layers.Input(shape=input_shape, name='input'),
        # Block 1
        layers.Conv2D(64, kernel_size=kernel_size, activation=activation, padding=padding, name='conv1_1'),
        layers.Conv2D(64, kernel_size=kernel_size, activation=activation, padding=padding, name='conv1_2'),
        layers.MaxPooling2D(pool_size=pool_size, strides=strides, name='pool1'),
        # Block 2
        layers.Conv2D(128, kernel_size=kernel_size, activation=activation, padding=padding, name='conv2_1'),
        layers.Conv2D(128, kernel_size=kernel_size, activation=activation, padding=padding, name='conv2_2'),
        layers.MaxPooling2D(pool_size=pool_size, strides=strides, name='pool2'),
        # Block 3
        layers.Conv2D(256, kernel_size=kernel_size, activation=activation, padding=padding, name='conv3_1'),
        layers.Conv2D(256, kernel_size=kernel_size, activation=activation, padding=padding, name='conv3_2'),
        layers.MaxPooling2D(pool_size=pool_size, strides=strides, name='pool3'),
        # Block 4
        layers.Conv2D(512, kernel_size=kernel_size, activation=activation, padding=padding, name='conv4_1'),
        layers.Conv2D(512, kernel_size=kernel_size, activation=activation, padding=padding, name='conv4_2'),
        layers.MaxPooling2D(pool_size=pool_size, strides=strides, name='pool4'),
        # Block 5
        layers.Conv2D(512, kernel_size=kernel_size, activation=activation, padding=padding, name='conv5_1'),
        layers.Conv2D(512, kernel_size=kernel_size, activation=activation, padding=padding, name='conv5_2'),
        layers.MaxPooling2D(pool_size=pool_size, strides=strides, name='pool5'),
        # Block fc
        layers.Flatten(name='flatten'),
        layers.Dense(4096, activation=activation, name='fc1_1'),
        layers.Dense(4096, activation=activation, name='fc1_2'),
        layers.Dense(output_dim, activation=None, name='embedding'),

        # Block classifier
        # layers.Dense(100, activation='relu', name='projector'),
        layers.Dense(n_classes, activation=activation_classifier, name='classifier')
    ]

    return models.Sequential(layers=_layers, name='VGGish-B')


__all__ = ['VGG11', 'VGG13']