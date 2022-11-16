"""VGGish model for AudioSet.

For implementations see:
https://github.com/tensorflow/models/tree/master/research/audioset/vggish
https://github.com/antoinemrcr/vggish2Keras
https://github.com/DTaoo/VGGish

For usage see:
https://github.com/DTaoo/VGGish/blob/master/evaluation.py

For Keras implementation see:
https://github.com/keras-team/keras/blob/master/keras/applications/vgg16.py

Reference
---------
Simonyan, K. and Zisserman, A. (2014) ‘Very Deep Convolutional Networks for Large-Scale Image Recognition’, arXiv:1409.1556 [cs] [Preprint]. Available at: http://arxiv.org/abs/1409.1556 (Accessed: 29 March 2019).

Hershey, S. et al. (2017) ‘CNN architectures for large-scale audio classification’, in 2017 ieee international conference on acoustics, speech and signal processing (icassp). IEEE, pp. 131–135.
"""

import keras
from keras import layers, models
# from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten
# from keras.models import Model
from dataclasses import dataclass


@dataclass
class VGGish_Params:
	"""Global parameters for VGGish.

	See also:
	https://github.com/tensorflow/models/blob/master/research/audioset/vggish/params.py
	"""
	# Dimension of input feature (data format: channel last)
	n_channels:int = 3
	n_bands:int = 64
	n_frames:int = 96

	# Dimension of embedding
	n_embedding:int = 128

	# Number of classes, 2 for binary classfication
	n_classes:int = 2

	# Parameters of ConvNet
	kernel_size:tuple = (3,3)
	activation:str = None
	activation_classifier:str = None
	padding:str = 'same' # 'valid' or 'same'
	pool_size:tuple = (2,2)
	strides:tuple = (2,2)


def get_ConvNet_A(params:VGGish_Params) -> keras.models.Model:
	"""ConvNet model A or VGG 11 layers.
	"""
	input_dim = (params.n_bands, params.n_frames, params.n_channels)
	kernel_size = params.kernel_size
	activation = params.activation
	activation_classifier = params.activation_classifier
	padding = params.padding
	strides = params.strides
	pool_size = params.pool_size
	output_dim = params.n_embedding
	n_classes = params.n_classes

	_layers = [
		layers.Input(shape=input_dim, name='input'),
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


def get_ConvNet_B(params:VGGish_Params) -> keras.models.Model:
	"""ConvNet model B or VGG 13 layers.
	"""
	input_dim = (params.n_bands, params.n_frames, params.n_channels)
	kernel_size = params.kernel_size
	activation = params.activation
	activation_classifier = params.activation_classifier
	padding = params.padding
	strides = params.strides
	pool_size = params.pool_size
	output_dim = params.n_embedding
	n_classes = params.n_classes

	_layers = [
		layers.Input(shape=input_dim, name='input'),
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


		# Architectural constants.
	# NUM_FRAMES:int = 96  # Frames in input mel-spectrogram patch.
	# NUM_BANDS:int = 64  # Frequency bands in input mel-spectrogram patch.
	# EMBEDDING_SIZE:int = 128  # Size of embedding layer.

	# # Hyperparameters used in feature and example generation.
	# SAMPLE_RATE:int = 16000
	# STFT_WINDOW_LENGTH_SECONDS:float = 0.025
	# STFT_HOP_LENGTH_SECONDS:float = 0.010
	# NUM_MEL_BINS:int = NUM_BANDS
	# MEL_MIN_HZ:float = 125
	# MEL_MAX_HZ:float = 7500
	# LOG_OFFSET:float = 0.01  # Offset used for stabilized log of input mel-spectrogram.
	# EXAMPLE_WINDOW_SECONDS:float = 0.96  # Each example contains 96 10ms frames
	# EXAMPLE_HOP_SECONDS:float = 0.96     # with zero overlap.

	# # Parameters used for embedding postprocessing.
	# PCA_EIGEN_VECTORS_NAME:str = 'pca_eigen_vectors'
	# PCA_MEANS_NAME:str = 'pca_means'
	# QUANTIZE_MIN_VAL:float = -2.0
	# QUANTIZE_MAX_VAL:float = +2.0

	# # Hyperparameters used in training.
	# INIT_STDDEV:float = 0.01  # Standard deviation used to initialize weights.
	# LEARNING_RATE:float = 1e-4  # Learning rate for the Adam optimizer.
	# ADAM_EPSILON:float = 1e-8  # Epsilon for the Adam optimizer.

	# # Names of ops, tensors, and features.
	# INPUT_OP_NAME:str = 'vggish/input_features'
	# INPUT_TENSOR_NAME:str = INPUT_OP_NAME + ':0'
	# OUTPUT_OP_NAME:str = 'vggish/embedding'
	# OUTPUT_TENSOR_NAME:str = OUTPUT_OP_NAME + ':0'
	# AUDIO_EMBEDDING_FEATURE_NAME:str = 'audio_embedding'

