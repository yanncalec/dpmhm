"""Class for dataset transformer.
"""

from abc import ABC, abstractmethod, abstractproperty, abstractclassmethod
from tkinter import Y

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import tensorflow as tf
import librosa
# import scipy
import hashlib
import json


def md5_encoder(*args):
  """Encode a list of arguments to a string.
  """
  return hashlib.md5(json.dumps(args, sort_keys=True).encode('utf-8')).hexdigest()


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

  def __init__(self, dataset, n_trunk:int=1, resampling_rate:int=None, filters:dict={}, keys:list=[], channels:list=[]):
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
      _dataset = self.compact(self._dataset_filtered)
    else:
      _dataset = self.resample(self.compact(self._dataset_filtered))
    if self._n_trunk > 1:
      _dataset = tf.data.Dataset.from_generator(
        split_signal_generator(_dataset, 'signal', self._n_trunk),
        output_signature=_dataset.element_spec,
      )
    return _dataset

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

    lb = md5_encoder(*v)
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

  def __init__(self, dataset, extractor:callable, window_shape:tuple, downsample:tuple):
    """
    Args
    ----
    dataset: input
      original dataset
    extractor: callable
      a callable taking arguments (signal, sampling_rate) and returning extracted features.
    window_shape: tuple or int
      either a tuple `(frequency, time)`, i.e. the size of the sliding window in frequency and time axes, or an int which is the size of the sliding window in time axis (the the whole frequency axis is used in this case).
    downsample: tuple or int
      downsampling rate in frequency and time axes, either tuple or int, corresponding to the given `frame_size`.
    """
    self._dataset_origin = dataset
    self._extractor = extractor
    self._window_shape = window_shape
    self._downsample = downsample

  @property
  def dataset_feature(self):
    """Feature-transformed dataset.
    """
    return self.to_feature(self._dataset_origin)

  @property
  def dataset_windows(self):
    """Windowed view of the feature dataset.
    """
    return self.to_windows(self.dataset_feature, self._window_shape, self._downsample)

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
  def get_output_signature(cls):
    """Output signature for the windowed view on the feature dataset.
    """
    pass

  @classmethod
  def to_windows(cls, dataset, window_shape:tuple, downsample:tuple=None, *, channel_last:bool=True):
    """Sliding windows of view of a time-frequency feature dataset.

    Windows of view are time-frequency patches of a complete spectral feature. It is obtained by sliding a small window along the time-frequency axes.

    Args
    ----
    dataset: tf.data.Dataset
      feature dataset, must have a dictionary structure and contain the fields {'label', 'info', 'feature'} which corresponds to respectively the label, the context information and the spectral feature of shape (channel, freqeuncy, time).
    window_shape: tuple or int
    downsample: tuple or int
    channel_last: bool
      if True the output window will have shape `(channel, frequency, time)`.

    Returns
    -------
    Transformed dataset of tuple form (label, info, window).

    Notes
    -----
    - The field 'info' should contain context information of the frame, e.g. the orginal signal from which the frame is extracted.
    - The input dataset is assumed channel first, while the output window can be optionally transformed to channel last.
    """
    def _slider(S, ws, ds):
      # assert ws is int or tuple, ds
      if ws is None:
        ws = S.shape[1:]
      elif type(ws) is int:
          ws = (S.shape[1], ws)
      assert type(ws) is tuple and len(ws)==2

      if ds is None:
        return  sliding_window_view(S, (S.shape[0], *ws))[0]
      elif type(ds) is int:
        return  sliding_window_view(S, (S.shape[0], *ws))[0, :, ::ds]
      else:
        return  sliding_window_view(S, (S.shape[0], *ws))[0, ::ds[0], ::ds[1]]

      # # Regularization for features shorter than frame_size:
      # # return sliding_window_view(S, (S.shape[0], S.shape[1], min(S.shape[-1], window_shape)))[0]
      # if type(window_shape) is int:
      #   window_shape = (S.shape[1], window_shape)
      # Sview =  sliding_window_view(S, (S.shape[0], *wdim))[0]
      # if downsample is None:
      #   return Sview
      # else:
      #   if type(downsample) is int:
      #     return Sview[:, ::downsample]
      #   else:
      #     return Sview[::downsample[0], ::downsample[1]]

    def _generator(dataset):
      def _get_generator():
        for label, metadata, windows in dataset:
          # `windows` has dimension :
          # (n_view_frequency, n_view_time, n_channel, window_shape[0], window_shape[1])
          for F in windows:  # iteration on frequency axis
            for x in F:  # iteration on time axis
              if channel_last:
                x = tf.transpose(x, [1,2,0])  # convert to channel last
              # yield label, metadata, x
              yield {
                'label': label,
                'metadata': metadata,
                'feature': x,
              }
      return _get_generator

    # py_slider = lambda S: tf.py_function(func=_slider, inp=[S], Tout=tf.float64)
    # ds = dataset.map(lambda X: (X['label'], X['info'], py_slider(X['feature'])))
    # or more compactly:

    ds = dataset.map(lambda X: (X['label'], X['metadata'], tf.py_function(
      func=lambda S: _slider(S.numpy(), window_shape, downsample),
      inp=[X['feature']],
      Tout=tf.float64)),
      num_parallel_calls=tf.data.AUTOTUNE)

    return tf.data.Dataset.from_generator(_generator(ds),
      output_signature=cls.get_output_signature()
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


## Feature extractor
def spectrogram(x, sampling_rate:int, *, time_window:float, hop_step:float, n_fft:int=None, to_db:bool=True, normalize:bool=True, **kwargs):
  """Compute the power spectrogram of a multichannel waveform.

  Args
  ----
  x: input
    nd array with the last dimension being time
  sampling_rate: int
    original sampling rate (in Hz) of the input
  time_window: float
    size of short-time Fourier transform (STFT) window, in second
  hop_step: float
    time step of the sliding STFT window, in second
  n_fft: int
    size of FFT, by default automatically determined
  to_db: bool
    power spectrogram in db

  Returns
  -------
  Sxx:
    power spectrogram
  (win_length, hop_length, n_fft):
    real values used by librosa for feature extraction
  (Ts, Fs):
    time and frequency steps of `Sxx`
  """
  sr = float(sampling_rate)

  win_length = int(time_window * sr)
  # hop_length = win_length // 2
  hop_length = int(hop_step * sr)

  if n_fft is None:
    n_fft = 2**(int(np.ceil(np.math.log2(win_length))))

  # Compute the spectrogram
  if normalize:
    y = (x-x.mean())/x.std()
  else:
    y = y
  S = librosa.stft(y, n_fft=n_fft, win_length=win_length, hop_length=hop_length, **kwargs)
  Sxx = librosa.power_to_db(np.abs(S)**2) if to_db else np.abs(S)**2

  Ts = np.arange(S.shape[-1]) * hop_length / sr  # last dimension is time
  Fs = np.arange(S.shape[-2])/S.shape[-2] * sr//2  # before last dimension is frequency
  # elif method == 'scipy':
  #   Fs, Ts, Sxx = scipy.signal.spectrogram(x, sr, nperseg=win_length, noverlap=win_length-hop_length, nfft=n_fft, **kwargs)
  #   Sxx = librosa.power_to_db(Sxx) if to_db else Sxx

  return Sxx, (win_length, hop_length, n_fft), (Ts, Fs)


def melspectrogram(x, sampling_rate:int, *, time_window:float, hop_step:float, n_fft:int=None, n_mels:int=128, normalize:bool=True, **kwargs):
  """Compute the mel-spectrogram of a multichannel waveform.

  Args
  ----
  x: input
    nd array with the last dimension being time
  sampling_rate: int
    original sampling rate (in Hz) of the input
  time_window: float
    size of short-time Fourier transform (STFT) window, in second
  hop_step: float
    time step of the sliding STFT window, in second
  n_fft: int
    size of FFT, by default automatically determined

  Returns
  -------
  Sxx:
    mel-spectrogram
  (win_length, hop_length, n_fft):
    real values used by librosa for feature extraction
  """
  sr = float(sampling_rate)

  win_length = int(time_window * sr)
  # hop_length = win_length // 2
  hop_length = int(hop_step * sr)

  if n_fft is None:
    n_fft = 2**(int(np.ceil(np.math.log2(win_length))))

  # Compute the melspectrogram
  if normalize:
    y = (x-x.mean())/x.std()
  else:
    y = y
  Sxx = librosa.feature.melspectrogram(y=y, n_fft=n_fft, win_length=win_length, hop_length=hop_length, n_mels=n_mels, **kwargs)

  return Sxx, (win_length, hop_length, n_fft)


# Default feature extractors
_EXTRACTOR_SPEC = lambda x, sr: spectrogram(x, sr, time_window=0.025, hop_step=0.01, to_db=True)[0]

_EXTRACTOR_MEL = lambda x, sr: melspectrogram(x, sr, time_window=0.025, hop_step=0.01, n_mels=64)[0]


## Dataset related

def extract_by_category(ds, labels:set=None):
  """Extract from a dataset the sub-datasets corresponding to the given categories.

  Args
  ----
  ds: tf.data.Dataset
    input dataset of element (value, label).
  labels: list
    list of category to be extracted. If not given the labels will be extracted by scanning the dataset (can be time-consuming).

  Returns
  -------
  a dictionary containing sub-dataset of each category.
  """
  if labels is None:
    labels = set([l.numpy() for x,l in ds])

  dp = {}
  for l in labels:
    dp[l] = ds.filter(lambda x: tf.equal(x['label'],l))
  return dp


def get_dataset_size(ds) -> int:
  """Get the number of elements of a dataset.
  """
  # count = ds.reduce(0, lambda x, _: x + 1)  # same efficiency
  count = 0
  for x in ds:
    count += 1
  return count


def random_split_dataset(ds, splits:dict, shuffle_size:int=1, **kwargs):
  """Randomly split a dataset according to the specified ratio.

  Args
  ----
  ds: tf.data.Dataset
    input dataset.
  splits: dict
    dictionary specifying the name and ratio of the splits.
  shuffle_size: int
    size of shuffle, 1 for no shuffle, None for full shuffle.
  kwargs:
    other keywords arguments to the method `shuffle()`, e.g. `reshuffle_each_iteration=False`, `seed=1234`.

  Return
  ------
  A dictionary of datasets with the same keys as `split`.
  """
  assert all([v>=0 for k,v in splits.items()])
  assert tf.reduce_sum(list(splits.values())) == 1.

  try:
    ds_size = len(ds)  # will raise error if length is unknown
  except:
    ds_size = get_dataset_size(ds)
  assert ds_size >= 0
  # assert (ds_size := ds.cardinality()) >= 0

  # Specify seed to always have the same split distribution between runs
  # e.g. seed=1234
  ds = ds.shuffle(ds_size if shuffle_size is None else shuffle_size, **kwargs)

  keys = list(splits.keys())
  sp_size = {k: tf.cast(splits[k]*ds_size, tf.int64) for k in keys[:-1]}
  sp_size[keys[-1]] = ds_size - tf.reduce_sum(list(sp_size.values()))
  assert all([(splits[k]==0.) | (sp_size[k]>0) for k in keys]), "Empty split."

  dp = {}
  s = 0
  for k, v in sp_size.items():
    dp[k] = ds.skip(s).take(v)
    s += v
  return dp


def split_dataset(ds, splits:dict={'train':0.7, 'val':0.2, 'test':0.1}, labels:list=None, *args, **kwargs):
  """Randomly split a dataset either on either global or per category basis.

  Args
  ----
  ds: tf.data.Dataset
    input dataset with element of type (value, label).
  splits: dict
    dictionary specifying the name and ratio of the splits.
  labels: list
    list of categories. If given apply the split per category otherwise apply it globally on the whole dataset.
  *args, **kwargs: arguments for `split_dataset_random()`

  Return
  ------
  A dictionary of datasets with the same keys as `split`.
  """
  if labels is None:
    dp = random_split_dataset(ds, splits, *args, **kwargs)
  else:
    ds = extract_by_category(ds, labels)
    dp = {}
    for n, (k,v) in enumerate(ds.items()):
      # dp[k] = random_split_dataset(v, splits, *args, **kwargs)
      if n == 0:
        dp.update(random_split_dataset(v, splits, *args, **kwargs))
      else:
        dq = random_split_dataset(v, splits, *args, **kwargs)
        for kk in dp.keys():
          dp[kk] = dp[kk].concatenate(dq[kk])

  return dp


def preprocessing_pipeline(dataset, DC, dc_args:dict, FT, ft_args:dict, splits:dict, feature_extractor:callable, outdir:str=None, *, mode:str='global', splits_kwargs:dict={}):
  # step 1: compact & split
  compactor = DC(dataset, *dc_args)

  if mode = 'global':
      labels = None
  else:
      labels = compactor.label_dict.keys()

  ds_split = split_dataset(compactor.dataset, splits=splits, labels=labels, **splits_kwargs)

  # step 2: feature transform
  # df_split = {}
  dw_split = {}

  for k, ds in ds_split.items():
      transformer = FT(ds, feature_extractor, **ft_args)
      # df_split[k] = transformer.dataset_feature
      dw_split[k] = transformer.dataset_windows

  if outdir is not None:
    for k,v in dw_split.items():
        v.save(os.path.joing(outdir, k))

  return dw_split

# def split_dataset_by_category(ds, labels:list=None, split:dict={'train':0.8, 'test':0.1, 'val':0.1}, *args, **kwargs):
#   ds = get_dataset_dictcategory(ds, labels)
#   dp = {}
#   for n, (k,v) in enumerate(ds.items()):
#     dp[k] = split_dataset_random(v, split, *args, **kwargs)

#   return dp


  # @abstractmethod
  # def pipeline(self, dataset):
  #   """Pipeline transform of a dataset.
  #   """
  #   pass

  # @abstractproperty
  # def dataset(self):
  #   """Preprocessed original dataset.
  #   """
  #   pass

  # @abstractproperty
  # def dataset_feature(self):
  #   """Feature-transformed dataset.
  #   """
  #   pass

  # @abstractproperty
  # def dataset_windows(self):
  #   """Windowed view of the feature dataset.
  #   """
  #   pass
