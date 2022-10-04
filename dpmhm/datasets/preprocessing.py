"""Class for dataset transformer.
"""

from abc import ABC, abstractmethod, abstractproperty
from tkinter import Y

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import tensorflow as tf
import librosa
import scipy
import hashlib
import json


def md5_encoder(*args):
  """Encode a list of arguments to a string.
  """
  return hashlib.md5(json.dumps(args, sort_keys=True).encode('utf-8')).hexdigest()


class AbstractDatasetPreprocessing(ABC):
  """Abstract class for dataset preprocessing.

  Convention
  ----------
  The original dataset (before feature transform) has to be channel first, and the transformed dataset is channel last.
  """

  @abstractmethod
  def compact(self):
    """
    Compact the original dataset and make a dictionary structure which contains the fields {'label', 'signal'} among others.
    """
    pass

  @abstractmethod
  def resample(self):
    """
    Resample the compacted dataset to a target rate.
    """
    pass

  @abstractmethod
  def to_feature(self, dataset):
    """Feature transform of a compacted (and resampled) dataset.

    This method transforms a waveform to spectral features. The transformed database has a dictionary structure which contains the fields {'label', 'info', 'feature'}.
    """
    pass

  @abstractproperty
  def output_signature(self):
    """Output signature for the tuple (label, info, frame).
    """
    pass

  def to_windows(self, dataset, window_shape:tuple, downsample:tuple=None):
    """Sliding windows of view of a time-frequency feature dataset.

    Windows of view are time-frequency patches of a complete spectral feature. It is obtained by sliding a small window along the time-frequency axes.

    Args
    ----
    dataset: tf.data.Dataset with a dictionary structure.
      must contain the fields {'label', 'info', 'feature'} which corresponds to respectively the label, the context information and the spectral feature of shape (channel, freqeuncy, time).
    window_shape: tuple
      either a tuple `(frequency, time)`, i.e. the size of the sliding window in frequency and time axes, or an int which is the size of the sliding window in time axis (the the whole frequency axis is used in this case).
    downsample: tuple or int
      downsampling rate in frequency and time axes, either tuple or int, corresponding to the given `frame_size`.

    Returns
    -------
    Transformed dataset of tuple form (label, info, window).

    Notes
    -----
    - The field 'info' should contain context information of the frame, e.g. the orginal signal from which the frame is extracted.
    - The convention of channel first is followed here: the output window has shape `(channel, frequency, time)`.
    """
    def _slider(S, ws, ds):
      if type(ws) is int:
        ws = (S.shape[1], ws)
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
        for label, info, windows in dataset:
          # `windows` has dimension :
          # (n_view_frequency, n_view_time, n_channel, window_shape[0], window_shape[1])
          for F in windows:  # iteration on frequency axis
            for x in F:  # iteration on time axis
              yield label, info, x  # channel first
              # yield label, info, tf.transpose(x, [1,2,0])  # convert to channel last
      return _get_generator

    # py_slider = lambda S: tf.py_function(func=_slider, inp=[S], Tout=tf.float64)
    # ds = dataset.map(lambda X: (X['label'], X['info'], py_slider(X['feature'])))
    # or more compactly:
    ds = dataset.map(lambda X: (X['label'], X['info'], tf.py_function(
      func=lambda S: _slider(S.numpy(), window_shape, downsample),
      inp=[X['feature']],
      Tout=tf.float64)
      ),
      num_parallel_calls=tf.data.AUTOTUNE)

    return tf.data.Dataset.from_generator(_generator(ds),
      output_signature=self.output_signature
      # https://www.tensorflow.org/api_docs/python/tf/data/Dataset#from_generator
      # output_types=(tf.string, tf.string, tf.float64))  # not recommended
    )

  @abstractmethod
  def pipeline(self, dataset):
    """Pipeline transform of a dataset.
    """
    pass

  @abstractproperty
  def dataset(self):
    """Preprocessed original dataset.
    """
    pass

  @abstractproperty
  def dataset_feature(self):
    """Feature-transformed dataset.
    """
    pass

  @abstractproperty
  def dataset_windows(self):
    """Windowed view of the feature dataset.
    """
    pass

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
      self._window_dim = tuple(list(self.dataset_windows.take(1))[0][-1].shape)
    return self._window_dim


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


def get_dataset_dictcategory(ds, labels:set=None):
  if labels is None:
    labels = set([l.numpy() for x,l in ds])

  dp = {}
  for l in labels:
    dp[l] = ds.filter(lambda x,a:tf.equal(a,l))
  return dp


def get_dataset_length(ds)->int:
  # count = ds.reduce(0, lambda x, _: x + 1)  # same efficiency
  count = 0
  for x in ds:
    count += 1
  return count


def split_dataset_random(ds, split:dict, shuffle_size:int=None, **kwargs):
  """Randomly split a dataset according to the specified ratio.

  Args
  ----
  ds: tf.data.Dataset
    input dataset to be split.
  split: dict
    dictionary specifying the name and ratio of the splits.
  shuffle_size: int
    size of shuffle, 1 for no shuffle, None for full shuffle.
  kwargs:
    other keywords arguments to the method `shuffle()`, e.g. `reshuffle_each_iteration=False`, `seed=1234`.

  Return
  ------
  A dictionary of datasets with the same keys as `split`.
  """
  assert tf.reduce_sum(list(split.values())) == 1.
  try:
    ds_size = len(ds)  # will raise error if length is unknown
  except:
    ds_size = get_dataset_length(ds)
  assert ds_size >= 0
  # assert (ds_size := ds.cardinality()) >= 0

  # Specify seed to always have the same split distribution between runs
  # e.g. seed=1234
  ds = ds.shuffle(ds_size if shuffle_size is None else shuffle_size, **kwargs)

  keys = list(split.keys())
  sp_size = {k: tf.cast(split[k]*ds_size, tf.int64) for k in keys[:-1]}
  sp_size[keys[-1]] = ds_size - tf.reduce_sum(list(sp_size.values()))

  dp = {}
  s = 0
  for k, v in sp_size.items():
    dp[k] = ds.skip(s).take(v)
    s += v
  return dp


# def split_dataset_by_category(ds, labels:list=None, split:dict={'train':0.8, 'test':0.1, 'val':0.1}, *args, **kwargs):
#   ds = get_dataset_dictcategory(ds, labels)
#   dp = {}
#   for n, (k,v) in enumerate(ds.items()):
#     dp[k] = split_dataset_random(v, split, *args, **kwargs)

#   return dp


def split_dataset(ds, split:dict={'train':0.7, 'val':0.2, 'test':0.1}, labels:list=None, *args, **kwargs):
  if labels is None:
    dp = split_dataset_random(ds, split, *args, **kwargs)
  else:
    ds = get_dataset_dictcategory(ds, labels)
    dp = {}
    for n, (k,v) in enumerate(ds.items()):
      if n == 0:
        dp.update(split_dataset_random(v, split, *args, **kwargs))
      else:
        dq = split_dataset_random(v, split, *args, **kwargs)
        for kk in dp.keys():
          dp[kk] = dp[kk].concatenate(dq[kk])

  return dp
