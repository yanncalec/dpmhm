"""Methods for preprocessing datasets before training.

Convention
----------
The original dataset is channel first and becomes channel last after preprocessing.
"""

import numpy as np
import tensorflow as tf
# import scipy
from tensorflow.keras import layers, models
from tensorflow.data import Dataset

import logging
Logger = logging.getLogger(__name__)


def nested_type_spec(sp):
    tp = {}
    for k,v in sp.items():
        if isinstance(v, dict):
            tp[k] = nested_type_spec(v)
		# elif
        else:
            tp[k] = layers.Input(type_spec=v)
    return tp


def keras_model_supervised(ds:Dataset, labels:list=None, normalize:bool=False):
	"""Initialize a Keras preprocessing model for supervised training.

	The processing model performs the following transformation on a dataset:
	- string label to integer conversion
	- data normalization
	- channel first to channel last conversion

	Args
	----
	ds: input dataset of structure `(data, label)`
		the data field is channel-first.
	labels: list of string labels for lookup
		if not given the labels will be automatically determined from the dataset.
	normalize:
		if True estimate the mean and variance by channel and apply the normalization.

	Returns
	-------
	model: a Keras model
		the model can be applied on a dataset as `ds.map(lambda x: model(x))`
	"""
	# inputs = nested_type_spec(ds.element_spec)
	inputs = (layers.Input(type_spec=v) for v in ds.element_spec)

	# Label conversion: string to int/one-hot
	# Gotcha: by default the converted integer label is zero-based and has zero as the first label which represents the off-class. This increases the number of class by 1 and must be taken into account in the model fitting.
	label_layer = layers.StringLookup(
		# num_oov_indices=0,   # force zero-based integer
		vocabulary=labels,
		# output_mode='one_hot'
	)
	if labels is None:
		# Adaptation depends on the given dataset
		label_layer.adapt([x[-1].numpy() for x in ds])

	label_output = label_layer(inputs[-1])

	# Normalization
	# https://keras.io/api/layers/preprocessing_layers/numerical/normalization/
	if normalize:
		# Manually compute the mean and variance of the data
		X = np.asarray([x[0].numpy() for x in ds]).transpose([1,0,2,3]); X = X.reshape((X.shape[0],-1))
		# X = np.hstack([x[data_name].numpy().reshape((n_channels,-1)) for x in ds])
		mean = X.mean(axis=-1)
		variance = X.var(axis=-1)

		feature_layer = layers.Normalization(axis=0, mean=mean, variance=variance)  # have to provide mean and variance if axis=0 is used
		feature_output = feature_layer(inputs[0])
		#
		# # The following works but will generate extra dimensions when applied on data
		# X = np.asarray([x[data_name].numpy() for x in ds]).transpose([0,2,3,1])
		# layer = keras.layers.Normalization(axis=-1)
		# layer.adapt(X)
		# feature_output = layer(tf.transpose(inputs[data_name], [1,2,0]))
		#
		# # This won't work
		# layer = keras.layers.Normalization(axis=0)
		# layer.adapt(X)  # complain about the unknown shape at dimension 0
	else:
		feature_output = inputs[0]

	# to channel-last
	outputs = (tf.transpose(feature_output, [1,2,0]), label_output)

	model = models.Model(inputs, outputs)
	# ds.map(lambda x: model(x), num_parallel_calls=tf.data.AUTOTUNE)
	return model


__all__ = ['nested_type_spec', 'keras_model_supervised']