"""CWRU bearing dataset.

Predefined task
---------------
Channels: subset of ['DE', 'FE', 'BA']
Labels: [FaultComponent, FaultSize]
Domains: load force [0,1,2,3]
Training/Test split: domain split

"""

import os
from pathlib import Path
# import itertools
# from attr import frozen
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import pandas as pd
# import scipy
# import mat4py
import librosa

from dpmhm.datasets.preprocessing import AbstractDatasetPreprocessing, md5_encoder, spectrogram, melspectrogram


# Data type
try:
  _DTYPE = tf.as_dtype(os.environ['DPMHM_DTYPE'])  # from the environment variable
except:
  _DTYPE = tf.float64


_DESCRIPTION = """
Case Western Reserve University bearing dataset.

Description
===========
Motor bearings were seeded with faults using electro-discharge machining (EDM). Faults ranging from 0.007 inches in diameter to 0.040 inches in diameter were introduced separately at the inner raceway, rolling element (i.e. ball) and outer raceway. Faulted bearings were reinstalled into the test motor and vibration data was recorded for motor loads of 0 to 3 horsepower (motor speeds of 1797 to 1720 RPM).

Further Information
-------------------
Two possible locations of bearings in the test rig: namely the drive-end and the fan-end.

For one location, a fault of certain diameter is introduced at one of the three components: inner race, ball, outer race. The inner race and the ball are rolling elements thus the fault does not have a fixed position, while the outer race is a fixed frame and the fault has three different angular positions: 3, 6 and 12.

The vibration signals are recorded by an accelerometer at (one or more of) the drive-end (DE), fan-end (FE) or base (BA) positions at the sampling rate 12 or 48 kHz, for a certain motor load.

Homepage
--------
https://engineering.case.edu/bearingdatacenter

Original Data
=============
Date of acquisition: 2000
Format: Matlab
Channels: up to 3 acclerometers, which are named Drive-End (DE), Fan-End (FE), Base (BA)
Split: not specified
Sampling rate: 12 kHz or 48 kHz, not fixed
Recording duration: ~ 10 seconds, not fixed
Label: Normal, Drive-end fault, Fan-end fault
Data fields:
- DE: drive end accelerometer data, contained by all files
- FE: fan end accelerometer data, contained by most files except those with name '300x'
- BA: base accelerometer data, contained by some files, in particular not those of normal data (faultless)
- RPM: real rpm during testing, contained by most files

Processed data
==============
Split: ['train']

Features
--------
'label': ['None', 'DriveEnd', 'FanEnd'], location of fault.
'metadata': {
    'FaultComponent': {InnerRace, Ball, OuterRace6, OuterRace3, OuterRace12}, component of fault.
    'FaultLocation': {DriveEnd, FanEnd}, same as 'label'. This is the location of fault not that of the accelerometer.
    'FaultSize': {0.007, 0.014, 0.021, 0.028}, diameter of fault in inches.
    'FileName': original filename.
    'LoadForce': {0, 1, 2, 3} nominal motor load in horsepower, corresponding to the nominal RPM: [1797, 1772, 1750, 1730].
    'RotatingSpeed': nominal RPM.
    'SamplingRate': {12000, 48000} in Hz.
},
'rpm': real RPM of experiment, contained by most files.
'signal': {
    'BA': base accelerometer data,
    'DE': drive end accelerometer data,
    'FE': fan end accelerometer data,
}

Notes
=====
- Convention for a normal experiment: FaultLocation and FaultComponent are None and FaultSize is 0.
"""

_CITATION = """
@misc{case_school_of_engineering_cwru_2000,
	title = {{CWRU} {Bearing} {Data} {Center}},
	copyright = {Case Western Reserve University},
	shorttitle = {{CWRU} {Bearing} {Dataset}},
	url = {https://engineering.case.edu/bearingdatacenter},
	language = {English},
	publisher = {Case Western Reserve University},
	author = {Case School of Engineering},
	year = {2000},
}
"""

# Load meta-information of all datafiles
_METAINFO = pd.read_csv(Path(__file__).parent / 'metainfo.csv')

# [['SamplingRate', 'Load', 'Location', 'Component', 'Diameter', 'FileName']]
# Make a label by cartesian product
# _CLASS = list(['-'.join(a) for a in itertools.product(_RPM, _LOCATION, _COMPONENT, _DIAMETER)])

# _DATA_URLS = ('https://engineering.case.edu/sites/default/files/'+_METAINFO['FileName']).tolist()


class CWRU(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for CWRU dataset.
  """

  VERSION = tfds.core.Version('1.0.0')

  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }

  def _info(self) -> tfds.core.DatasetInfo:
    return tfds.core.DatasetInfo(
        builder=self,
        description=_DESCRIPTION,
        features=tfds.features.FeaturesDict({
          # The sampling rate and the shape of a signal are not fixed.
          'signal':{  # three possible channels: drive-end, fan-end and base
            'DE': tfds.features.Tensor(shape=(None,), dtype=_DTYPE),
            'FE': tfds.features.Tensor(shape=(None,), dtype=_DTYPE),
            'BA': tfds.features.Tensor(shape=(None,), dtype=_DTYPE),
          },
          # Can not save all channels in a tensor with unknown dimension, like
          # 'signal': tfds.features.Tensor(shape=(None,1), dtype=tf.float64),

          'rpm': tf.uint32, # real rpm of the signal, not the nominal one (motor load).

          'label': tfds.features.ClassLabel(names=['None', 'DriveEnd', 'FanEnd']),

          'metadata': {
              'SamplingRate': tf.uint32, # {12000, 48000} Hz
              'RotatingSpeed': tf.uint32,  # average RPM: [1797, 1772, 1750, 1730]
              'LoadForce': tf.uint32, # {0, 1, 2 ,3}, corresponding average RPM: [1797, 1772, 1750, 1730]
              'FaultLocation': tf.string, # {'DriveEnd', 'FanEnd', 'None'}
              'FaultComponent': tf.string,  # {'InnerRace', 'Ball', 'OuterRace3', 'OuterRace6', 'OuterRace12', 'None'}
              'FaultSize': tf.float32,  # {0.007, 0.014, 0.021, 0.028, 0}
              'FileName': tf.string,
          },

          # Another possibility is to use class labels (string)
          # This implies conversion in the decoding part.
          # 'metadata': {
          #     'SamplingRate': tfds.features.ClassLabel(names=['12000', '48000']),  # Hz
          #     'LoadForce': tfds.features.ClassLabel(names=['0', '1', '2', '3']),  # Horsepower
          #     'RotatingSpeed': tfds.features.ClassLabel(names=['1797', '1772', '1750', '1730']),
          #     'FaultLocation': tfds.features.ClassLabel(names=['DriveEnd', 'FanEnd', 'None']),
          #     'FaultComponent': tfds.features.ClassLabel(names=['InnerRace', 'Ball', 'OuterRace3', 'OuterRace6', 'OuterRace12', 'None']),
          #     'FaultSize': tfds.features.ClassLabel(names=['0.007', '0.014', '0.021', '0.028', '0.0']),  # inches
          #     'FileName': tf.string,
          # },
        }),
        supervised_keys=None,
        # supervised_keys=('signal', 'label'),  # Set to `None` to disable
        homepage='https://engineering.case.edu/bearingdatacenter',
        citation=_CITATION,
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Return SplitGenerators.
    """
    # If class labels are used for `meta`, add `.astype(str)` in the following before `to_dict()`
    if dl_manager._manual_dir.exists():  # prefer to use manually downloaded data
      datadir = Path(dl_manager._manual_dir)

      # _data_files = dl_manager._manual_dir.glob('*.mat')
      # fp_dict = {str(fp): _METAINFO.loc[_METAINFO['FileName'] == fp.name].iloc[0].to_dict() for fp in _data_files}
    else:  # automatically download data
      # Parallel download (may result in corrupted files):
      # zipfile = dl_manager.download(_DATA_URLS)   # urls must be a list

      # Sequential download:
      # _data_files = [dl_manager.download(url) for url in _DATA_URLS]

      raise NotImplemented()

    return {
        # 'train': self._generate_examples(fp_dict),
        'train': self._generate_examples(datadir),
    }

  def _generate_examples(self, datadir):
    # for fn, fobj in tfds.download.iter_archive(zipfile, tfds.download.ExtractMethod.ZIP):
    for fp in datadir.glob('*.mat'):
      fname = fp.parts[-1]
      # print(fname)
      metadata = _METAINFO.loc[_METAINFO['FileName'] == fname].iloc[0].to_dict()

      try:
        dm = {k: np.asarray(v).squeeze() for k,v in tfds.core.lazy_imports.scipy.io.loadmat(fp).items() if k[:2]!='__'}
        # using mat4py.loadmat
        # dm = {k:np.asarray(v).squeeze() for k,v in mat4py.loadmat(fp).items()}
      except Exception as msg:
        raise Exception(f"Error in processing {fp}: {msg}")
        # continue

      # If filename is e.g. `112.mat`, then the matlab file  contains some of the fields `X112_DE_time` (always), `X112RPM`, as well as `X112_FE_time` and `X112_BA_time`.
      # Exception: the datafiles `300?.mat`` do not contain fields starting with `X300?` nor the field `RPM`.

      fn = f"{int(metadata['FileName'].split('.mat')[0]):03d}"
      if f'X{fn}_DE_time' in dm:  # find regular fields
        kde = f'X{fn}_DE_time'
        # xde = dm[kde].squeeze()
        xde = dm[kde]
        kfe = f'X{fn}_FE_time'
        # xfe = dm[kfe].squeeze() if kfe in dm else []
        xfe = dm[kfe] if kfe in dm else np.empty(0)
        kba = f'X{fn}_BA_time'
        # xba = dm[kba].squeeze() if kba in dm else []
        xba = dm[kba] if kba in dm else np.empty(0)
        krpm = f'X{fn}RPM'
      else:   # datafiles 300x.mat are special
        # print(dm.keys())
        kde = [k for k in dm.keys() if k[0]=='X'][0]
        # xde, xfe, xba = dm[kde].squeeze(), [], []
        xde, xfe, xba = dm[kde], np.empty(0), np.empty(0)

      # Use nominal value if the real value is not given.
      rpm = int(dm[krpm]) if krpm in dm else metadata['RotatingSpeed']

      yield hash(frozenset(metadata.items())), {
        'signal': {
          'DE': xde.astype(_DTYPE.as_numpy_dtype),
          'FE': xfe.astype(_DTYPE.as_numpy_dtype),
          'BA': xba.astype(_DTYPE.as_numpy_dtype),
        },  # if not empty all three have the same length
        # 'signal': {
        #   'DE': xde,
        #   'FE': xfe,
        #   'BA': xba,
        # },  # if not empty all three have the same length
        'rpm': rpm,
        'label': metadata['FaultLocation'],
        'metadata': metadata,
      }

  @staticmethod
  def get_references():
    try:
      with open(Path(__file__).parent / 'Exported Items.bib') as fp:
        return fp.read()
    except:
      pass


# Default feature extractors
_EXTRACTOR_SPEC = lambda x, sr: spectrogram(x, sr, time_window=0.025, hop_step=0.01, to_db=True)[0]

_EXTRACTOR_MEL = lambda x, sr: melspectrogram(x, sr, time_window=0.025, hop_step=0.01, n_mels=64)[0]


class DatasetPreprocessing(AbstractDatasetPreprocessing):
  """Preprocessing for CWRU dataset.

  This class performs the following preprocessing steps:
  - resampling,
  - filtration,
  - extraction of new labels and channels,
  - feature transform,
  - sliding window view with downsampling.
  """

  def __init__(self, dataset, extractor:callable=_EXTRACTOR_SPEC, *, resampling_rate:int=None, filters:dict={}, keys:list=[], channels:list=['DE'], window_shape, downsample):
    """
    Args
    ----
    dataset: input
      original CWRU dataset
    extractor: callable
      a callable taking arguments (signal, sampling_rate) and returning extracted features.
    resampling_rate: int
      rate for resampling, if None use the original sampling rate.
    filters: dict
      filters on the field 'metadata'.
    keys: list
      keys for extraction of new labels, subset of ['LoadForce', 'FaultComponent', 'FaultSize']. If not given the original labels will be used.
    channels: list
      channels for extraction of data, subset of ['DE', 'FE', 'BA].
    window_shape: tuple
      either a tuple `(frequency, time)`, i.e. the size of the sliding window in frequency and time axes, or an int which is the size of the sliding window in time axis (the the whole frequency axis is used in this case).
    downsample: tuple or int
      downsampling rate in frequency and time axes, either tuple or int, corresponding to the given `frame_size`.

    Notes
    -----
    """
    self._extractor = extractor
    self._resampling_rate = resampling_rate
    self._filters = filters
    # for k in keys:
    #   assert k in ['LoadForce', 'FaultComponent', 'FaultSize']
    self._keys = keys
    self._channels = channels
    self._window_shape = window_shape
    self._downsample = downsample
    # dictionary for extracted labels, will be populated only after scanning the compacted dataset
    self._label_dict = {}
    # filtered original dataset, of shape (channel, time)
    self._dataset_filtered = self.filter_metadata(dataset, self._filters)

  def filter_metadata(self, ds, fs):
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
    return _dataset

  @property
  def dataset_feature(self):
    """Feature-transformed dataset.
    """
    return self.to_feature(self.dataset)

  @property
  def dataset_windows(self):
    """Windowed view of the feature dataset.
    """
    return self.to_windows(self.dataset_feature, self._window_shape, self._downsample)

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

  @property
  def label_dict_index(self):
    # label index
    return {k: n for n, k in enumerate(self.label_dict.keys())}

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

  def compact(self, dataset):
    @tf.function  # necessary for infering the size of tensor
    def _has_channels(X):
      """Test if channels are present in data."""
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

      # signal = [X['signal'][ch] for ch in self._channels]
      # if len(self._channels) > 0:
      #   signal = [X['signal'][ch] for ch in self._channels]
      # else:
      #   # signal = [X['signal'][k] for k in ['DE', 'FE', 'BA']]
      #   signal = []
      #   for v in X['signal'].values():
      #     tf.cond(tf.size(v)>0, true_fn=lambda: signal.append(v), false_fn=lambda: None)
      #   # signal = [v for v in X['signal'].values() if tf.size(v)>0]

      return {
        'label': tf.py_function(func=self.encode_labels, inp=d, Tout=tf.string),
        'rpm': X['rpm'],
        'rpm_nominal': X['metadata']['RotatingSpeed'],
        'filename': X['metadata']['FileName'],
        'sampling_rate': X['metadata']['SamplingRate'],
        'signal': [X['signal'][ch] for ch in self._channels],
        # 'signal': signal
      }
    # return dataset.filter(_has_channels)
    ds = dataset.filter(_has_channels)
    return ds.map(_compact, num_parallel_calls=tf.data.AUTOTUNE)

  def resample(self, dataset):
    def _resample(X):
      Y = X.copy()
      Y['sampling_rate'] = self._resampling_rate
      Y['signal'] = tf.py_function(
        func=lambda x, sr:librosa.resample(x.numpy(), orig_sr=float(sr), target_sr=self._resampling_rate),
        inp=[X['signal'], X['sampling_rate']],
        Tout=_DTYPE
      )
      return Y

    return dataset.map(_resample, num_parallel_calls=tf.data.AUTOTUNE)

  def to_feature(self, dataset):
    """Feature transform of a compacted dataset.

    Args
    ----
    dataset:
      compacted/resampled dataset.
    """
    # Alternative: define a tf.py_function beforehand
    # py_extractor = lambda x, sr: tf.py_function(
    #   func=lambda x, sr: extractor(x.numpy(), sr),  # makes it a tf callable
    #   inp=[x, sr],
    #   Tout=tf.float64
    # )  # x.numpy() must be used inside the method `extractor()`

    def _feature_map(X):
      return {
        # 'label': X['label'],  # string label
        'label': tf.py_function(
          func=lambda s: self.label_dict_index[s.numpy().decode('utf-8')],
          inp=[X['label']],
          Tout=tf.uint32
          ),  # integer label
        'info': (X['filename'], X['rpm'], X['rpm_nominal']),
        'feature': tf.py_function(
          func=lambda x, sr: self._extractor(x.numpy(), sr),  # makes it a tf callable
          inp=[X['signal'], X['sampling_rate']],
          Tout=_DTYPE
          )  # the mose compact way
      }

    return dataset.map(_feature_map, num_parallel_calls=tf.data.AUTOTUNE)

  @property
  def output_signature(self):
    # return (
    #   tf.TensorSpec(shape=(), dtype=tf.string, name=None),
    #   (
    #     tf.TensorSpec(shape=(), dtype=tf.string, name=None),  # filename
    #     tf.TensorSpec(shape=(), dtype=tf.uint32, name=None),  # real rpm
    #     tf.TensorSpec(shape=(), dtype=tf.uint32, name=None)  # nominal rpm
    #     ),
    #   tf.TensorSpec(shape=(None, None, None), dtype=_DTYPE, name=None)
    # )
    return (
      tf.TensorSpec(shape=(), dtype=tf.uint32, name='label'),
      (
        tf.TensorSpec(shape=(), dtype=tf.string, name='filename'),  # filename
        tf.TensorSpec(shape=(), dtype=tf.uint32, name='rpm'),  # real rpm
        tf.TensorSpec(shape=(), dtype=tf.uint32, name='rpm_nominal')  # nominal rpm
        ),
      tf.TensorSpec(shape=(None, None, None), dtype=_DTYPE, name='feature')
    )

  def pipeline(self, extractor, frame_size:int):
    return self.to_frames(self.to_feature(extractor), frame_size)
