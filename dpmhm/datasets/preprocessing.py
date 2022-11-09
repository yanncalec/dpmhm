"""Class for dataset transformer.


Convention
----------
We follow the convention of channel first: The original dataset as well as the transformed dataset has the channel as the first dimension and time as the last dimension.
"""

# from typing import List, Dict
from abc import ABC, abstractmethod, abstractproperty, abstractclassmethod
# from importlib import import_module
# import tempfile
import itertools

import os
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import itertools
import tensorflow as tf
from tensorflow import keras
from tensorflow.data import Dataset

import librosa
# import scipy

import logging
Logger = logging.getLogger(__name__)

# from . import utils, _DTYPE
from dpmhm.datasets import utils, _DTYPE, _ENCLEN

# _SIGNAL = 'signal'
# _FEATURE = 'feature'
# _SAMPLING_RATE = 'sampling_rate'


class AbstractDatasetTransformer(ABC):
	@abstractmethod
	def build(self) -> Dataset:
		"""Build the transformed dataset.
		"""
		pass

	@property
	def dataset(self):
		"""Transformed dataset.
		"""
		try:
			return self._dataset
		except:
			self._dataset = self.build()
			return self._dataset

	# @dataset.setter
	# def dataset(self, df):
	# 	self._dataset = df

	def serialize(self, outdir:str, *, compression:str=None):
		"""Serialize the dataset to disk.

		Serialization consists of saving the dataset and reloading it from a file. This can boost the subsequent performance of a dataset at the cost of storage. Compression by 'GZIP' method can be used.
		"""
		Dataset.save(self.dataset, outdir, compression=compression)
		self._dataset = Dataset.load(outdir, compression=compression)

	# @property
	# def data_dim(self):
	# 	"""Dimension of the data vector.
	# 	"""
	# 	try:
	# 		self._data_dim
	# 	except:
	# 		self._data_dim = tuple(list(self.dataset.take(1))[0][self.data_key].shape)
	# 	return self._data_dim

	# @abstractproperty
	# def data_key(self) -> str:
	# 	"""Key name of the data field.
	# 	"""
	# 	pass

	# @abstractproperty
	# def data_dim(self):
	# 	pass


class DatasetCompactor(AbstractDatasetTransformer):
	"""Class for dataset compactor.

	This class performs the following preprocessing steps on the raw signal:
	- split,
	- resampling of channels,
	- filtration,
	- extraction of new labels,
	- feature transform,
	- sliding window view with downsampling.

	Convention
	----------
	The data included in the field `signal` of the original dataset must be either 1D tensor or 2D tensor of shape `(channel, time)`.
	"""

	def __init__(self, dataset, *, channels:list, keys:list=[], n_trunk:int=1, resampling_rate:int=None, filters:dict={}):
		"""
		Args
		----
		dataset: input
			original dataset
		extractor: callable
			a callable taking arguments (signal, sampling_rate) and returning extracted features.
		channels: list
			channels for extraction of data, if not given all available channels will be used simultaneously (which may yield an empty dataset).
		keys: list
			keys for extraction of new labels, if not given the original labels will be used.
		n_trunk: int
			number of equal pieces that the raw signal is divided into.
		resampling_rate: int
			rate for resampling, if None use the original sampling rate.
		filters: dict
			filters on the field 'metadata'.
		"""
		self._n_trunk = n_trunk
		self._resampling_rate = resampling_rate
		self._filters = filters

		self._channels = channels
		self._keys = keys

		# dictionary for extracted labels, will be populated only after scanning the compacted dataset
		self._label_dict = {}
		# self._dataset_origin = dataset
		# filtered original dataset, of shape (channel, time)
		self._dataset_origin = self.filter_metadata(dataset, self._filters)

	@classmethod
	def filter_metadata(cls, ds, fs:dict):
		"""Filter a dataset by values of its field 'metadata'.

		Args
		----
		ds: Dataset
			input dataset
		fs: dict
			a dictionary of keys and admissible values of the field 'metadata'.
		"""
		@tf.function
		def _filter(X, k, v):
			return tf.reduce_any(tf.equal(X['metadata'][k], v))

		for k,v in fs.items():
			ds = ds.filter(lambda X: _filter(X, k, v))
		return ds

	def build(self):
		ds = self.compact(self.resample(self._dataset_origin, self._resampling_rate))
		if self._n_trunk > 1:
			ds = Dataset.from_generator(
				utils.split_signal_generator(ds, 'signal', self._n_trunk),
				output_signature=ds.element_spec,
			)
		return ds

	@property
	def label_dict(self):
		"""Dictionary of compacted labels.
		"""
		try:
			self._label_dict_scanned
		except:
			self._label_dict = {}
			# make a full scan of the compacted dataset
			for x in self.dataset:
				pass
			self._label_dict_scanned = self._label_dict

		return self._label_dict_scanned

	# @property
	# def label_dict_index(self):
	#   # label index
	#   return {k: n for n, k in enumerate(self.label_dict.keys())}

	def encode_labels(self, *args):
		"""MD5 encoding of a list of labels.

		From:
		https://stackoverflow.com/questions/5417949/computing-an-md5-hash-of-a-data-structure
		"""
		dn = [d.numpy() for d in args]
		# v = [str(d.numpy()) for d in args]
		v = [a.decode('utf-8') if type(a) is (bytes or str) else str(a) for a in dn]

		lb = utils.md5_encoder(*v)[:_ENCLEN]
		# if lb in self._label_dict:
		#   assert self._label_dict[lb] == v
		# else:
		#   self._label_dict[lb] = v
		self._label_dict[lb] = v
		return lb

	@property
	def full_label_dict(self):
		"""Full dictionary of compacted labels.
		"""
		try:
			return self._full_label_dict
		except:
			channels = list(self._dataset_origin.element_spec['signal'].keys())

			for ch in channels:
				compactor = DatasetCompactor(self._dataset_origin, keys=self._keys, channels=[ch])
				try:
					ld.update(compactor.label_dict)
				except:
					ld = compactor.label_dict
			self._full_label_dict = ld
			return ld

	@classmethod
	def resample(cls, dataset, rsr:int):
		"""Resample the dataset to a common target rate.
		"""
		@tf.function
		def _resample(X):
			Y = X.copy()
			if rsr is None:
				try:
					# if the original sampling rate is a dict
					vsr = tf.stack(list(X['sampling_rate'].values()))
					tf.Assert(
						tf.reduce_all(tf.equal(vsr[0], vsr)),  #
						['All channels must have the sampling rate:', vsr]
					)  # must be all constant
					Y['sampling_rate'] = vsr[0]
				except:
					# if the original sampling rate is a number
					Y['sampling_rate'] = X['sampling_rate']
			else:
				xs = {}
				for k in X['signal'].keys():
					if tf.size(X['signal'][k]) > 0:
						try:
							# X['sampling_rate'] has nested structure
							xs[k] = tf.py_function(
								func=lambda x, sr:librosa.resample(x.numpy(), orig_sr=float(sr), target_sr=float(rsr)),
								inp=[X['signal'][k], X['sampling_rate'][k]],
								Tout=_DTYPE
							)
						except KeyError:
							# X['sampling_rate'] is a number
							xs[k] = tf.py_function(
								func=lambda x, sr:librosa.resample(x.numpy(), orig_sr=float(sr), target_sr=float(rsr)),
								inp=[X['signal'][k], X['sampling_rate']],
								Tout=_DTYPE
							)
						except Exception as msg:
							Logger.exception(msg)
							xs[k] = X['signal'][k]
					else:
						xs[k] = X['signal'][k]
				Y['signal'] = xs
				Y['sampling_rate'] = rsr
			return Y

		return dataset.map(_resample, num_parallel_calls=tf.data.AUTOTUNE)

	def compact(self, dataset):
		"""Transform a dataset into a compact form.

		This method compacts the original dataset by
		- stacking the selected channels (must have the same length)
		- renaming the label using selected keys

		Note that the channel-stacking may filter out some samples.	The compacted dataset has a dictionary structure with the fields {'label', 'metadata', 'signal'}.
		"""
		@tf.function  # necessary for infering the size of tensor
		def _has_channels(X):
			"""Test if not empty channels are present in data.
			"""
			flag = True
			for ch in self._channels:
				if tf.size(X['signal'][ch]) == 0:  # to infer the size in graph mode
				# if tf.equal(tf.size(X['signal'][ch]), 0):
				# if X['signal'][ch].shape == 0:  # raise strange "shape mismatch error"
				# if len(X['signal'][ch]) == 0:  # raise TypeError
					flag = False
					# break or return False are not supported by tensorflow
			return flag

		@tf.function
		def _compact(X):
			d = [X['label']] + [X['metadata'][k] for k in self._keys]

			return {
				'label': tf.py_function(func=self.encode_labels, inp=d, Tout=tf.string),
				'metadata': X['metadata'],
				'sampling_rate': X['sampling_rate'],
				'signal': [X['signal'][ch] for ch in self._channels],
			}
		# return dataset.filter(_has_channels)
		ds = dataset.filter(lambda X:_has_channels(X))
		return ds.map(lambda X:_compact(X), num_parallel_calls=tf.data.AUTOTUNE)


class FeatureExtractor(AbstractDatasetTransformer):
	"""Class for feature extractor.

	This class performs the following preprocessing steps:
	- feature transform,
	"""

	def __init__(self, dataset, extractor:callable):
		"""
		Args
		----
		dataset: input
			original dataset
		extractor: callable
			a callable taking arguments (signal, sampling_rate) and returning extracted features.
		"""
		self._dataset_origin = dataset
		self._extractor = extractor

	def build(self):
		return self.to_feature(self._dataset_origin, self._extractor)

	@classmethod
	def to_feature(cls, ds, extractor:callable):
		"""Feature transform of a compacted dataset of signal.

		This method transforms a waveform to spectral features. The transformed database has a dictionary structure which contains the fields {'label', 'metadata', 'feature'}.

		Args
		----
		ds: Dataset
			compacted/resampled signal dataset, with fields {'label', 'metadata', 'signal'}.
		extractor: callable
			method for feature extraction

		Notes
		-----
		Unless the shape of the returned value by self._extractor can be pre-determined, there's no way to make lazy evaluation here (for e.g.faster scanning of the mapped dataset).
		"""
		def _feature_map(X):
			return {
				'label': X['label'],  # string label
				'metadata': X['metadata'],
				'feature': tf.py_function(
					func=lambda x, sr: extractor(x.numpy(), sr),  # makes it a tf callable. x.numpy() must be used inside the method `extractor()`
					inp=[X['signal'], X['sampling_rate']],
					Tout=_DTYPE
					)  # the most compact way
			}

		return ds.map(_feature_map, num_parallel_calls=tf.data.AUTOTUNE)


class WindowSlider(AbstractDatasetTransformer):
	"""Windowed view for dataset.

	This class performs the following preprocessing steps:
	- sliding window view with downsampling.
	"""

	def __init__(self, dataset, *, window_shape:tuple, downsample:tuple):
		"""
		Args
		----
		dataset: input
			original dataset
		window_shape: tuple or int
			either a tuple `(frequency, time)`, i.e. the size of the sliding window in frequency and time axes, or an int which is the size of the sliding window in time axis (the the whole frequency axis is used in this case). No windowed view is created if set to `None`.
		downsample: tuple or int
			downsampling rate in frequency and time axes, either tuple or int, corresponding to the given `frame_size`. No downsampling if set to `None`.
		"""
		self._dataset_origin = dataset
		self._window_shape = window_shape
		self._downsample = downsample

	def build(self):
		return self.to_windows(self._dataset_origin, self._window_shape, self._downsample)

	@classmethod
	def to_windows(cls, dataset, window_shape:tuple, downsample:tuple=None):
		"""Sliding windows of view of a time-frequency feature dataset.

		Windows of view are time-frequency patches of a complete spectral feature. It is obtained by sliding a small window along the time-frequency axes.

		Args
		----
		dataset: Dataset
			feature dataset, must have a dictionary structure and contain the fields {'label', 'info', 'feature'} which corresponds to respectively the label, the context information and the spectral feature of shape (channel, freqeuncy, time).
		window_shape: tuple or int
		downsample: tuple or int

		Returns
		-------
		Transformed dataset of tuple form (label, info, window).

		Notes
		-----
		- The field 'info' should contain context information of the frame, e.g. the orginal signal from which the frame is extracted.
		- We follow the convertion of channel first here for both the input and the output dataset.
		"""
		def _slider(S, ws:tuple, ds:tuple):
			"""Sliding window view of array `S` with window shape `ws` and downsampling rate `ds`.
			"""
			# assert ws is int or tuple, ds
			# assert S.ndim == 3
			if ws is None:
				ws = S.shape[1:]
			elif type(ws) is int:
				ws = (S.shape[1], ws)

			if ds is None:
				return  sliding_window_view(S, (S.shape[0], *ws))[0]
			elif type(ds) is int:
				return  sliding_window_view(S, (S.shape[0], *ws))[0, :, ::ds]
			else:
				return  sliding_window_view(S, (S.shape[0], *ws))[0, ::ds[0], ::ds[1]]

		def _generator(dataset):
			def _get_generator():
				for label, metadata, windows in dataset:
					# `windows` has dimension :
					# (n_view_frequency, n_view_time, n_channel, window_shape[0], window_shape[1])
					for F in windows:  # iteration on frequency axis
						for x in F:  # iteration on time axis
							# if channel_last:
							#   x = tf.transpose(x, [1,2,0])  # convert to channel last
							yield {
								'label': label,
								'metadata': metadata,
								'feature': x,
								# 'feature': tf.cast(x, _DTYPE),
							}
							# yield label, metadata, x
			return _get_generator

		ds = dataset.map(lambda X: (X['label'], X['metadata'], tf.py_function(
			func=lambda S: _slider(S.numpy(), window_shape, downsample),
			inp=[X['feature']],
			Tout=_DTYPE)),
			num_parallel_calls=tf.data.AUTOTUNE)

		tensor_shape = tuple(list(ds.take(1))[0][-1].shape[-3:])  # drop the first two dimensions of sliding view

		# Output signature for the windowed view on the feature dataset.
		_output_signature = {
			'label': dataset.element_spec['label'],
			'metadata': dataset.element_spec['metadata'],
			'feature': tf.TensorSpec(shape=tf.TensorShape(tensor_shape), dtype=_DTYPE),
			# 'feature': tuple([tf.TensorSpec(shape=tf.TensorShape(tensor_shape), dtype=_DTYPE)]*fold),
		}

		# output signature, see:
		# https://www.tensorflow.org/api_docs/python/tf/data/Dataset#from_generator
		return Dataset.from_generator(_generator(ds),
			output_signature=_output_signature
		)

	@property
	def data_dim(self):
		"""Dimension of the data vector.
		"""
		try:
			self._data_dim
		except:
			self._data_dim = tuple(list(self.dataset.take(1))[0]['feature'].shape)
			# self._data_dim = tuple(list(self.dataset.take(1))[0]['feature'][0].shape)
		return self._data_dim


class PairedView(AbstractDatasetTransformer):
	"""Paired view (positive or negative) of a dataset.
	"""
	def __init__(self, dataset, *, keys:list=[], positive:bool=True):
		"""
		Args
		----
		keys: list
			keys of the field `metadata` for pair comparison along with `label`.
		positive: bool
			if True the positive pair (same label and metadata) will be retained.
		"""
		# self._fold = fold
		self._keys = keys
		self._positive = positive
		self._dataset_origin = dataset

	def build(self):
		@tf.function
		def _filter(X, Y):
			v = tf.reduce_all(
				[tf.equal(X['label'], Y['label'])] +
				[tf.equal(X['metadata'][k], Y['metadata'][k]) for k in self._keys]
				)
			if self._positive:
				return v
			else:
				return tf.logical_not(v)

		def _generator():
			for eles in itertools.product(*([self._dataset_origin]*2)):
				yield eles

		ds = tf.data.Dataset.from_generator(
			_generator,
			output_signature=(self._dataset_origin.element_spec,)*2
			)
		return ds.filter(_filter)

		# if self._positive:
		# 	return ds.map(lambda X,Y: {
		# 		'label': X['label'],
		# 		'metadata': X['metadata'],
		# 		'feature': (X['feature'], Y['feature'])
		# 	})
		# else:
		# 	return ds


def get_keras_preprocessing_model(ds, labels:list=None, normalize:bool=False):
	"""Initialize a Keras preprocessing model on a provided dataset.

	The model performs the following transformation:
	- string label to integer
	- normalization by channel

	Args
	----
	ds: training dataset
	labels: list of string labels for lookup
	normalize: if True estimate the mean and variance by channel and apply the normalization.
	"""
	type_spec = ds.element_spec
	inputs = {
		'metadata': {k: keras.Input(type_spec=v) for k,v in type_spec['metadata'].items()},
		'feature': keras.Input(type_spec=type_spec['feature']),
		'label': keras.Input(type_spec=type_spec['label']),
	}

	# Label conversion: string to int/one-hot
	# Gotcha: by default the converted integer label is zero-based and has zero as the first label which represents the off-class. This augments the number of class by 1 and has to be taken into account in the model fitting.
	label_layer = keras.layers.StringLookup(
		# num_oov_indices=0,   # force zero-based integer
		vocabulary=labels,
		# output_mode='one_hot'
	)
	if labels is None:
		# Adaptation depends on the given dataset
		label_layer.adapt([x['label'].numpy() for x in ds])

	label_output = label_layer(inputs['label'])

	# Normalization
	# https://keras.io/api/layers/preprocessing_layers/numerical/normalization/
	if normalize:
		# Manually compute the mean and variance of the data
		X = np.asarray([x['feature'].numpy() for x in ds]).transpose([1,0,2,3]); X = X.reshape((X.shape[0],-1))
		# X = np.hstack([x['feature'].numpy().reshape((n_channels,-1)) for x in ds])
		mean = X.mean(axis=-1)
		variance = X.var(axis=-1)

		feature_layer = keras.layers.Normalization(axis=0, mean=mean, variance=variance)  # have to provide mean and variance if axis=0 is used
		feature_output = feature_layer(inputs['feature'])
		#
		# # The following works but will generate additional dimensions when applied on data
		# X = np.asarray([x['feature'].numpy() for x in ds]).transpose([0,2,3,1])
		# layer = keras.layers.Normalization(axis=-1)
		# layer.adapt(X)
		# feature_output = layer(tf.transpose(inputs['feature'], [1,2,0]))
		#
		# # This won't work
		# layer = keras.layers.Normalization(axis=0)
		# layer.adapt(X)  # complain about the unknown shape at dimension 0
	else:
		feature_output = inputs['feature']

	# to channel-last
	outputs = (tf.transpose(feature_output, [1,2,0]), label_output)
	# outputs = {
	#   'feature': tf.transpose(feature_output, [1,2,0])
	#   'label': label_output,
	# }

	return keras.Model(inputs, outputs)

