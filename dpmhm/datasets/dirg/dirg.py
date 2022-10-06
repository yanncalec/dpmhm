"""DIRG dataset.
"""

import os
from pathlib import Path
# import itertools
# import json
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
# import pandas as pd
# from scipy.io import loadmat
import librosa

from dpmhm.datasets.preprocessing import AbstractDatasetPreprocessing, md5_encoder, _EXTRACTOR_SPEC
from dpmhm.datasets import _DTYPE


_DESCRIPTION = """
The Politecnico di Torino rolling bearing test rig dataset.

Description
===========
Data aquired on the rolling bearing test rig of the Dynamic and Identification Research Group (DIRG), in the Department of Mechanical and Aerospace Engineering at Politecnico di Torino.

The test rig contains two accelerometers at the position `A1` and `A2` and the shaft with its three roller bearings `B1-B2-B3`. Faults of different size are introduced in `B1` on the inner ring or the roller.

Two types of experiments are conducted on the test rig:
- variable speed and load test: with a variation of the fault size, nominal speed of the shaft and nominal load.
- endurance test: with the fault of type `4A` with the nominal speed at 300 Hz and nominal load at 1800 N.

More details can be found in the original publication.

Original data
=============
Date of acquisition: 2016
Format: Matlab
Channels: 6, for two accelerometers in the x-y-z axis
Split: 'Variable speed and load' test, 'Endurance' test
Sampling rate: 51200 Hz for `Variable speed and load` test and 102400 Hz for `Endurance` test
Recording duration: 10 seconds for `Variable speed and load` test and 8 seconds for `Endurance` test
Label: normal and faulty

Download
--------
ftp://ftp.polito.it/people/DIRG_BearingData/

Processed data
==============
Split: ['variation', 'endurance'].

Features
--------
'signal': of shape (6, time)
'label': [Normal, Faulty, Unknown]
'load': real load in N
'metadata': {
  'SamplingRate': 51200 Hz for Variation test or 102400 Hz for Endurance test
  'RotatingSpeed': Nominal speed of the shaft in Hz [100, 200, 300, 400, 500] for the variation test and 300 for the endurance test.
  'LoadForce': Nominal load in N [0, 1000, 1400, 1800]
  'FaultComponent': {'Roller', 'InnerRing'}
  'FaultSize': [450, 250, 150, 0] um
  'OriginalSplit': {'Variation', 'Endurance'}
  'FileName': original file name,
}

Notes
=====
- Conversion: load is converted from mV to N using the sensitivity factor 0.499 mV/N
- The endurance test was originally with the fault type 4A but in the processed data we marked its label as "unknown".
"""

_CITATION = """
@article{DAGA2019252,
title = {The Politecnico di Torino rolling bearing test rig: Description and analysis of open access data},
journal = {Mechanical Systems and Signal Processing},
volume = {120},
pages = {252-273},
year = {2019},
issn = {0888-3270},
doi = {https://doi.org/10.1016/j.ymssp.2018.10.010},
url = {https://www.sciencedirect.com/science/article/pii/S0888327018306800},
author = {Alessandro Paolo Daga and Alessandro Fasana and Stefano Marchesiello and Luigi Garibaldi},
}
"""

_DATA_URLS = []

# _SENSOR_LOCATION = ['A1', 'A2']

# _FAULT_LOCATION = ['B1']

_NOMINAL_LOAD = np.asarray([0, 1000, 1400, 1800])

# coding of fault component and diameter (in um)
_FAULT_TYPE_MATCH = {
  '0A': ('None', 0),
  '1A': ('InnerRing', 450),
  '2A': ('InnerRing', 250),
  '3A': ('InnerRing', 150),
  '4A': ('Roller', 450),
  '5A': ('Roller', 250),
  '6A': ('Roller', 150),
}

# _DATA_URLS = 'ftp://ftp.polito.it/people/DIRG_BearingData'


class DIRG(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for dirg dataset."""

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }

  MANUAL_DOWNLOAD_INSTRUCTIONS = """
  Due to the access limitation of the ftp server, automatic download is not supported in this package. Please download all data from

    ftp://ftp.polito.it/people/DIRG_BearingData/

  and proceed the installation manually.
  """

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    # TODO(dirg): Specifies the tfds.core.DatasetInfo object
    return tfds.core.DatasetInfo(
        builder=self,
        description=_DESCRIPTION,
        features=tfds.features.FeaturesDict({
            # These are the features of your dataset like images, labels ...
            'signal': tfds.features.Tensor(shape=(6, None), dtype=_DTYPE),

            'label': tfds.features.ClassLabel(names=['Normal', 'Faulty', 'Unknown']),

            'load': tf.float32,  # Real load in N

            'metadata': {
              'SamplingRate': tf.uint32,  # 51200 Hz for Variation test or 102400 Hz for Endurance test
              'RotatingSpeed': tf.uint32,  # Nominal speed of the shaft in Hz
              'LoadForce': tf.uint32,  # Nominal load in N
              'FaultComponent': tf.string, # {'Roller', 'InnerRing'}
              'FaultSize': tf.uint32,  # 450, 250, 150, 0 um
              'OriginalSplit': tf.string,  # {'Variation', 'Endurance'}
              'FileName': tf.string,
            },
        }),

        # If there's a common (input, target) tuple from the
        # features, specify them here. They'll be used if
        # `as_supervised=True` in `builder.as_dataset`.
        supervised_keys=None,
        homepage='',
        citation=_CITATION,
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""
    if dl_manager._manual_dir.exists():  # prefer to use manually downloaded data
      datadir = Path(dl_manager._manual_dir)
    else:
      # Parallel download (may result in corrupted files):
      # _data_files = dl_manager.download(_DATA_URLS)   # urls must be a list

      # Sequential download:
      # _data_files = [dl_manager.download(url) for url in _DATA_URLS]

      # fp_dict = {}
      # for fp in _data_files:
      #   with open(str(fp)+'.INFO') as fu:
      #     fp_dict[str(fp)] = _METAINFO.loc[_METAINFO['FileName'] == json.load(fu)['original_fname']].iloc[0].to_dict()
      raise NotImplemented()

    return {
        'variation': self._generate_examples(datadir/'VariableSpeedAndLoad'),
        'endurance': self._generate_examples(datadir/'EnduranceTest'),
    }

  def _generate_examples(self, datadir):
    """Yields examples."""
    for fp in datadir.glob('*.mat'):
      fname = fp.parts[-1]

      try:
        dm = tfds.core.lazy_imports.scipy.io.loadmat(fp)
        # dm = loadmat(fp)
      except Exception as msg:
        raise Exception(f"Error in processing {fp}: {msg}")

      if fname.upper()[0] == 'C':
        ss = fname.upper().split('_')
        # self._fname_parser(fname.name)
        _component, _diameter = _FAULT_TYPE_MATCH[ss[0][1:]]
        _samplingrate = 51200
        _shaftrate = int(ss[1])
        _load_real = float(ss[2])/0.499
        _load = _NOMINAL_LOAD[np.argmin(np.abs(_load_real-_NOMINAL_LOAD))]
        _label = 'Normal' if _component=='None' else 'Faulty'
        _datalabel = 'Variation'
      elif fname.upper()[:3] == 'E4A':
        _component, _diameter = _FAULT_TYPE_MATCH['4A']
        _samplingrate = 102400
        _shaftrate = 300
        _load = 1800
        _load_real = 1800
        _label = 'Unknown'
        _datalabel = 'Endurance'
      else:
        continue

      metadata = {
          'SamplingRate': _samplingrate,
          'RotatingSpeed': _shaftrate,
          'LoadForce': _load,
          'FaultComponent': _component,
          'FaultSize': _diameter,
          'OriginalSplit': _datalabel,
          'FileName': os.path.join(*fp.parts[-2:])
      }

      yield hash(frozenset(metadata.items())), {
        'signal': dm[fname[:-4]].T.astype(_DTYPE.as_numpy_dtype),  # transpose to the shape (channel, time)
        'label': _label,
        'load': _load_real,
        'metadata': metadata
      }


class DatasetCompactor(AbstractDatasetCompactor):
  """Preprocessing for DIRG dataset.
  """

  def __init__(self, *args, **kwargs):
    """
    Notes
    -----
    """
    super().__init__(*args, **kwargs)

    for k in self._keys:
      assert k in ['FaultComponent', 'FaultSize']
    # self._channels is not used here.

  def compact(self, dataset):
    @tf.function
    def _compact(X):
      d = [X['label']] + [X['metadata'][k] for k in self._keys]

      return {
        'label': tf.py_function(func=self.encode_labels, inp=d, Tout=tf.string),
        'metadata': {
          'Load': X['load'],
          'Load_Nominal': X['metadata']['LoadForce'],
          'RPM_Nominal': X['metadata']['RotatingSpeed'],
          'FileName': X['metadata']['FileName'],
        }
        'signal': X['signal'],
      }
    return dataset.map(_compact, num_parallel_calls=tf.data.AUTOTUNE)


class FeatureTransformer(AbstractFeatureTransformer):
  """Feature transform for DIRG dataset.
  """

  @classmethod
  def get_output_signature(cls):
    return {
      'label': tf.TensorSpec(shape=(), dtype=tf.string),
      'metadata': {
        'Load': tf.TensorSpec(shape=(), dtype=tf.uint32),  # real load
        'Load_Nominal': tf.TensorSpec(shape=(), dtype=tf.uint32),  # nominal load
        'RPM_Nominal': tf.TensorSpec(shape=(), dtype=tf.uint32),  # nominal rpm
        'FileName': tf.TensorSpec(shape=(), dtype=tf.string),  # filename
      },
      'feature': tf.TensorSpec(shape=tf.TensorShape(None), dtype=tf.float64),
    }