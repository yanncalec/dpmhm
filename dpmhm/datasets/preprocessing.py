"""Class for dataset transformer.
"""

from abc import ABC, abstractmethod, abstractproperty, abstractclassmethod
from importlib import import_module
from logging import warning
import tempfile

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import tensorflow as tf
from tensorflow import keras

import librosa
# import scipy

from . import utils


def split_signal_generator(dataset, key:str, n_trunk:int):
  """
  dataset:
    input dataset with dictionary structure.
  key: str
    dataset[key] is the signal to be divided.
  """
  def _get_generator():
    for X in dataset:
      truncs = np.array_split(X[key], n_trunk, axis=-1)
      # truncs = tf.split(X[key], num_or_size_splits=n_trunk, axis=-1)
      Y = X.copy()
      for x in truncs:
        Y[key] = x
        yield Y
  return _get_generator


class AbstractDatasetCompactor(ABC):
  """Abstract class for dataset compactor.

  This class performs the following preprocessing steps on the raw signal:
  - split,
  - resampling,
  - filtration,
  - extraction of new labels and channels,
  - feature transform,
  - sliding window view with downsampling.

  Convention
  ----------
  We follow the convention of channel first: The original dataset (before feature transform) as well as the transformed dataset has the shape `(channel, frequency, time)`.
  """

  def __init__(self, dataset, *, n_trunk:int=1, resampling_rate:int=None, filters:dict={}, keys:list=[], channels:list=[]):
    """
    Args
    ----
    dataset: input
      original dataset
    extractor: callable
      a callable taking arguments (signal, sampling_rate) and returning extracted features.
    n_trunk: int
      number of equal pieces that the raw signal is divided into.
    resampling_rate: int
      rate for resampling, if None use the original sampling rate.
    filters: dict
      filters on the field 'metadata'.
    keys: list
      keys for extraction of new labels, if not given the original labels will be used.
    channels: list
      channels for extraction of data.

    Notes
    -----
    """
    self._n_trunk = n_trunk
    self._resampling_rate = resampling_rate
    self._filters = filters
    self._keys = keys
    self._channels = channels
    # dictionary for extracted labels, will be populated only after scanning the compacted dataset
    self._label_dict = {}
    # filtered original dataset, of shape (channel, time)
    self._dataset_filtered = self.filter_metadata(dataset, self._filters)

  def filter_metadata(self, ds, fs:dict):
    """Filter a dataset by values of its field 'metadata'.

    Args
    ----
    ds: tf.data.Dataset
      input dataset
    fs: dict
      a dictionary of lkeys and admissible values of the field 'metadata'.
    """
    @tf.function
    def _filter(X, k, v):
      return tf.reduce_any(tf.equal(X['metadata'][k], v))

    for k,v in fs.items():
      ds = ds.filter(lambda X: _filter(X, k, v))
    return ds

  @property
  def dataset(self):
    """Preprocessed (resampled and compacted) original dataset.
    """
    if self._resampling_rate is None:
      ds = self.compact(self._dataset_filtered)
    else:
      ds = self.resample(self.compact(self._dataset_filtered))
    if self._n_trunk > 1:
      ds = tf.data.Dataset.from_generator(
        split_signal_generator(ds, 'signal', self._n_trunk),
        output_signature=ds.element_spec,
      )
    ds.__dpmhm_class__ = self.__class__
    # ds.__dpmhm_name__ = __name__
    return ds

  @property
  def label_dict(self):
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

    lb = utils.md5_encoder(*v)
    # if lb in self._label_dict:
    #   assert self._label_dict[lb] == v
    # else:
    #   self._label_dict[lb] = v
    self._label_dict[lb] = v
    return lb

  @abstractmethod
  def compact(self):
    """
    Compact the original dataset and make a dictionary structure which contains the fields {'label', 'metadata', 'signal'} among others. The field 'metadata' is again a dictionary which contains the field 'SamplingRate'.
    """
    pass

  def resample(self, dataset):
    """Resample the compacted dataset to a target rate.
    """
    def _resample(X):
      Y = X.copy()
      Y['metadata']['SamplingRate'] = self._resampling_rate
      Y['signal'] = tf.py_function(
        func=lambda x, sr:librosa.resample(x.numpy(), orig_sr=float(sr), target_sr=self._resampling_rate),
        inp=[X['signal'], X['metadata']['SamplingRate']],
        Tout=tf.float64
      )
      return Y

    return dataset.map(_resample, num_parallel_calls=tf.data.AUTOTUNE)


class AbstractFeatureTransformer(ABC):
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
    ds = self.to_feature(self._dataset_origin)
    ds.__dpmhm_class__ = self.__class__
    return ds

  @property
  def dataset_windows(self):
    """Windowed view of the feature dataset.
    """
    ds = self.to_windows(self.dataset_feature, self._window_shape, self._downsample)
    ds.__dpmhm_class__ = self.__class__
    return ds

  def to_feature(self, dataset):
    """Feature transform of a compacted dataset of signal.

    This method transforms a waveform to spectral features. The transformed database has a dictionary structure which contains the fields {'label', 'metadata', 'feature'}.

    Args
    ----
    dataset: tf.data.Dataset
      compacted/resampled signal dataset, must have the fields {'label', 'metadata', 'signal'}.
    """
    # Alternative: define a tf.py_function beforehand
    # py_extractor = lambda x, sr: tf.py_function(
    #   func=lambda x, sr: extractor(x.numpy(), sr),  # makes it a tf callable
    #   inp=[x, sr],
    #   Tout=tf.float64
    # )  # x.numpy() must be used inside the method `extractor()`

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
          func=lambda x, sr: self._extractor(x.numpy(), sr),  # makes it a tf callable
          inp=[X['signal'], X['metadata']['SamplingRate']],
          Tout=tf.float64
          )  # the most compact way
      }

    return dataset.map(_feature_map, num_parallel_calls=tf.data.AUTOTUNE)

  @abstractclassmethod
  def get_output_signature(cls, tensor_shape:tuple):
    """Output signature for the windowed view on the feature dataset.
    """
    pass

  @classmethod
  def to_windows(cls, dataset, window_shape:tuple, downsample:tuple=None):
    """Sliding windows of view of a time-frequency feature dataset.

    Windows of view are time-frequency patches of a complete spectral feature. It is obtained by sliding a small window along the time-frequency axes.

    Args
    ----
    dataset: tf.data.Dataset
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
    def _slider(S, ws, ds):
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
              }
              # yield label, metadata, x
      return _get_generator

    ds = dataset.map(lambda X: (X['label'], X['metadata'], tf.py_function(
      func=lambda S: _slider(S.numpy(), window_shape, downsample),
      inp=[X['feature']],
      Tout=tf.float64)),
      num_parallel_calls=tf.data.AUTOTUNE)

    tensor_shape = tuple(list(ds.take(1))[0][-1].shape[-3:])  # drop the first two dimensions of sliding view

    return tf.data.Dataset.from_generator(_generator(ds),
      output_signature=cls.get_output_signature(tensor_shape)
      # https://www.tensorflow.org/api_docs/python/tf/data/Dataset#from_generator
      # output_types=(tf.string, tf.string, tf.float64))  # not recommended
    )

  @property
  def feature_dim(self):
    try:
      self._feature_dim
    except:
      self._feature_dim = tuple(list(self.dataset_feature.take(1))[0]['feature'].shape)
    return self._feature_dim

  @property
  def window_dim(self):
    try:
      self._window_dim
    except:
      self._window_dim = tuple(list(self.dataset_windows.take(1))[0]['feature'].shape)
    return self._window_dim


def wav2frames_pipeline(dataset, module_name, extractor:callable, *,
dc_kwargs:dict, ft_kwargs:dict, export_dir:str=None):
  """Transform a dataset of waveform to sliding windowed-view of feature.
  """
  # module = import_module('..'+module_name, __name__)
  module = import_module('dpmhm.datasets.'+module_name)
  compactor = module.DatasetCompactor(dataset, **dc_kwargs)
  transformer = module.FeatureTransformer(compactor.dataset, extractor, **ft_kwargs)
  dw = transformer.dataset_windows
  if export_dir is not None:
    dw.save(export_dir)
    dw = tf.data.Dataset.load(export_dir)
  return dw, compactor, transformer


def get_keras_preprocessing_model(dataset, labels:list=None, normalize:bool=False):
  """Get Keras preprocessing model for normalization.
  """
  type_spec = dataset.element_spec
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
    label_layer.adapt([x['label'].numpy() for x in dataset])

  label_output = label_layer(inputs['label'])

  # Normalization
  # https://keras.io/api/layers/preprocessing_layers/numerical/normalization/
  if normalize:
    # Manually compute the mean and variance of the data
    X = np.asarray([x['feature'].numpy() for x in dataset]).transpose([1,0,2,3]); X = X.reshape((X.shape[0],-1))
    # X = np.hstack([x['feature'].numpy().reshape((n_channels,-1)) for x in dataset])
    mean = X.mean(axis=-1)
    variance = X.var(axis=-1)

    feature_layer = keras.layers.Normalization(axis=0, mean=mean, variance=variance)  # have to provide mean and variance if axis=0 is used
    feature_output = feature_layer(inputs['feature'])
    #
    # # The following works but will generate additional dimensions when applied on data
    # X = np.asarray([x['feature'].numpy() for x in dataset]).transpose([0,2,3,1])
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


# Obsolete functions
def _wav2feature_pipeline_obslt(dataset, module_name, extractor:callable, *,
dc_kwargs:dict, ft_kwargs:dict,
splits:dict=None, sp_mode:str='uniform', sp_kwargs:dict={}):
  """Transform a dataset of waveform to feature.
  """
  # module = import_module('..'+module_name, __name__)
  module = import_module('dpmhm.datasets.'+module_name)
  early_mode = 'early' in sp_mode.split('+')
  uniform_mode = 'uniform' in sp_mode.split('+')

  compactor = module.DatasetCompactor(dataset, **dc_kwargs)

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

