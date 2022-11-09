# Obsolete functions
def _wav2feature_pipeline_obslt(ds, module_name, extractor:callable, *,
dc_kwargs:dict, ft_kwargs:dict,
splits:dict=None, sp_mode:str='uniform', sp_kwargs:dict={}):
	"""Transform a dataset of waveform to feature.
	"""
	# module = import_module('..'+module_name, __name__)
	module = import_module('dpmhm.datasets.'+module_name)
	early_mode = 'early' in sp_mode.split('+')
	uniform_mode = 'uniform' in sp_mode.split('+')

	compactor = module.DatasetCompactor(ds, **dc_kwargs)

	transformer = module.FeatureTransformer(compactor.dataset, extractor, **ft_kwargs)
	# df_split[k] = transformer.dataset_feature
	dw = transformer.dataset_windows

	if splits is None:
		return dw
	else:
		if early_mode:
			ds_split = utils.split_dataset(compactor.dataset, splits,
			labels=None if uniform_mode else compactor.label_dict.keys(),
			**sp_kwargs
			)
			# df_split = {}
			dw_split = {}
			for k, ds in ds_split.items():
					transformer = module.FeatureTransformer(ds, extractor, **ft_kwargs)
					# df_split[k] = transformer.dataset_feature
					dw_split[k] = transformer.dataset_windows
		else:
			dw_split = utils.split_dataset(dw, splits,
			labels=None if uniform_mode else compactor.label_dict.keys(),
			**sp_kwargs
			)

		return dw_split


class Preprocessor:
	"""Preprocessor class.
	"""
	def __init__(self, dataset, extractor:callable, *, dc_kwargs:dict, wt_kwargs:dict, outdir:str=None):
		# module = import_module(self.__module__)
		# self._compactor = module.DatasetCompactor(dataset, **dc_kwargs)
		# self._transformer = module.FeatureTransformer(self._compactor.dataset, extractor, **ft_kwargs)
		self._compactor = DatasetCompactor(dataset, **dc_kwargs)
		self._extractor = FeatureExtractor(self._compactor.dataset, extractor)
		self._slider = WindowSlider(**wt_kwargs)
		self.label_dict = self.get_label_dict(dataset, self._compactor._keys)

		# save & reload to boost performance of the dataset
		if outdir is not None:
			# try:
			#   self._transformer.dataset_feature = self.load(outdir, 'feature')
			# except:
			#   self.save(outdir, 'feature')
			#   self._transformer.dataset_feature = self.load(outdir, 'feature')
			try:
				self._transformer.dataset_windows = self.load(outdir, 'windows')
			except:
				self.save(outdir, 'windows')
				self._transformer.dataset_windows = self.load(outdir, 'windows')

	def save(self, outdir:str, name:str):
		if name == 'feature':
			self._transformer.dataset_feature.save(os.path.join(outdir, 'feature'))
		elif name == 'windows':
			self._transformer.dataset_windows.save(os.path.join(outdir, 'windows'))
		# elif name == 'signal':
		#   self._compactor.dataset.save(outdir)
		else:
			raise NameError(name)

	def load(self, outdir:str, name:str) -> Dataset:
		if name == 'feature':
			ds = Dataset.load(os.path.join(outdir, 'feature'))
		elif name == 'windows':
			ds = Dataset.load(os.path.join(outdir, 'windows'))
		# elif name == 'signal':
		#   self._compactor.dataset = Dataset.load(outdir)
		else:
			raise NameError(name)
		return ds

	@property
	def dataset(self):
		return self._compactor.dataset

	@property
	def dataset_feature(self):
		return self._transformer.dataset_feature

	@property
	def dataset_windows(self):
		return self._transformer.dataset_windows

	# def save(self, outdir:str):
	#   self._transformer.dataset_windows.save(outdir)
	#   self._dataset_windows_reloaded = Dataset.load(outdir)

	# @property
	# def dataset_windows(self):
	#   try:
	#     return self._dataset_windows_reloaded
	#   except:
	#     return self._transformer.dataset_windows

	# @abstractproperty
	# def label_dict(self):
	#   pass
		# return import_module('dpmhm.datasets.'+module_name).DatasetCompactor.get_label_dict(dataset, keys)

	# def wav2frames_pipeline(dataset, module_name:str, extractor:callable, *,
	# dc_kwargs:dict, ft_kwargs:dict, export_dir:str=None):
	#   """Transform a dataset of waveform to sliding windowed-view of feature.
	#   """
	#   # module = import_module('..'+module_name, __name__)
	#   module = import_module('dpmhm.datasets.'+module_name)
	#   compactor = module.DatasetCompactor(dataset, **dc_kwargs)
	#   transformer = module.FeatureTransformer(compactor.dataset, extractor, **ft_kwargs)
	#   dw = transformer.dataset_windows
	#   if export_dir is not None:
	#     dw.save(export_dir)
	#     dw = Dataset.load(export_dir)
	#   return dw, compactor, transformer



class FeatureTransformer_Old:
	"""Abstract class for feature transformer.

	This class performs the following preprocessing steps:
	- feature transform,
	- sliding window view with downsampling.

	Convention
	----------
	We follow the convention of channel first: The original dataset (before feature transform) as well as the transformed dataset has the shape `(channel, frequency, time)`.
	"""

	def __init__(self, dataset, extractor:callable, *, window_shape:tuple, downsample:tuple):
		"""
		Args
		----
		dataset: input
			original dataset
		extractor: callable
			a callable taking arguments (signal, sampling_rate) and returning extracted features.
		window_shape: tuple or int
			either a tuple `(frequency, time)`, i.e. the size of the sliding window in frequency and time axes, or an int which is the size of the sliding window in time axis (the the whole frequency axis is used in this case). No windowed view is created if set to `None`.
		downsample: tuple or int
			downsampling rate in frequency and time axes, either tuple or int, corresponding to the given `frame_size`. No downsampling if set to `None`.
		"""
		self._dataset_origin = dataset
		self._extractor = extractor
		self._window_shape = window_shape
		self._downsample = downsample

	@property
	def dataset_feature(self):
		"""Feature-transformed dataset.
		"""
		try:
			return self._dataset_feature
		except:
			ds = self.to_feature(self._dataset_origin)
			# ds.__dpmhm_class__ = self.__class__
			return ds

	@dataset_feature.setter
	def dataset_feature(self, df):
		self._dataset_feature = df
		# self._dataset_feature.__dpmhm_class__ = self.__class__

	@property
	def dataset_windows(self):
		"""Windowed view of the feature dataset.
		"""
		try:
			return self._dataset_windows
		except:
			ds = self.to_windows(self.dataset_feature, self._window_shape, self._downsample)
			# ds.__dpmhm_class__ = self.__class__
			return ds

	@dataset_windows.setter
	def dataset_windows(self, df):
		self._dataset_windows = df
		# self._dataset_windows.__dpmhm_class__ = self.__class__

	def to_feature(self, ds):
		"""Feature transform of a compacted dataset of signal.

		This method transforms a waveform to spectral features. The transformed database has a dictionary structure which contains the fields {'label', 'metadata', 'feature'}.

		Args
		----
		ds: Dataset
			compacted/resampled signal dataset, must have the fields {'label', 'metadata', 'signal'}.

		Notes
		-----
		Unless the shape of the returned value by self._extractor can be pre-determined, there's no way to make lazy evaluation here (for e.g.faster scanning of the mapped dataset).
		"""
		def _feature_map(X):
			return {
				'label': X['label'],  # string label
				# 'label': tf.py_function(
				#   func=lambda s: self.label_dict_index[s.numpy().decode('utf-8')],
				#   inp=[X['label']],
				#   Tout=tf.uint32
				#   ),  # integer label
				# 'info': (X['filename'], X['rpm'], X['rpm_nominal']),
				'metadata': X['metadata'],
				'feature': tf.py_function(
					func=lambda x, sr: self._extractor(x.numpy(), sr),  # makes it a tf callable. x.numpy() must be used inside the method `extractor()`
					inp=[X['signal'], X['sampling_rate']],
					Tout=_DTYPE
					)  # the most compact way
			}

		return ds.map(_feature_map, num_parallel_calls=tf.data.AUTOTUNE)

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
			"""
			ws: tuple
				window shape
			ds: tuple
				downsampling rate
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
		}

		# output signature, see:
		# https://www.tensorflow.org/api_docs/python/tf/data/Dataset#from_generator
		return Dataset.from_generator(_generator(ds),
			output_signature=_output_signature
		)

	@property
	def feature_dim(self):
		"""Dimension of the feature.
		"""
		try:
			self._feature_dim
		except:
			self._feature_dim = tuple(list(self.dataset_feature.take(1))[0]['feature'].shape)
		return self._feature_dim

	@property
	def window_dim(self):
		"""Dimension of the windowed feature.
		"""
		try:
			self._window_dim
		except:
			self._window_dim = tuple(list(self.dataset_windows.take(1))[0]['feature'].shape)
		return self._window_dim
