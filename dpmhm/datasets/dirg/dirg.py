"""DIRG dataset.
"""

import os
from pathlib import Path
import itertools
import json
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import pandas as pd
# from scipy.io import loadmat


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
'signal':
'label': [Normal, Faulty, Unknown]
'metadata': {
  'SamplingRate': 51200 Hz for Variation test or 102400 Hz for Endurance test
  'RotatingSpeed': Nominal speed of the shaft in Hz
  'LoadForce': Load in N, conversion from mV: mV/0.499 with 0.499 being the sensitivity
  'FaultComponent': {'Roller', 'InnerRing'}
  'FaultSize': 450, 250, 150, 0 um
  'OriginalSplit': {'Variation', 'Endurance'}
  'FileName': original file name,
}

Notes
=====
- Conversion: load is converted from mV to N using the sensitivity factor 0.499 mV/N
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
            'signal': tfds.features.Tensor(shape=(None,6), dtype=tf.float64),

            'label': tfds.features.ClassLabel(names=['Normal', 'Faulty', 'Unknown']),

            'metadata': {
              'SamplingRate': tf.uint32,  # 51200 Hz for Variation test or 102400 Hz for Endurance test
              'RotatingSpeed': tf.float32,  # Nominal speed of the shaft in Hz
              'LoadForce': tf.float32,  # Load in N, conversion from mV: mV/0.499 with 0.499 being the sensitivity
              'FaultComponent': tf.string, # {'Roller', 'InnerRing'}
              'FaultSize': tf.float32,  # 450, 250, 150, 0 um
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
        _shaftrate = float(ss[1])
        _load = float(ss[2])/0.499
        _label = 'Normal' if _component=='None' else 'Faulty'
        _datalabel = 'Variation'
      elif fname.upper()[:3] == 'E4A':
        _component, _diameter = _FAULT_TYPE_MATCH['4A']
        _samplingrate = 102400
        _shaftrate = 300
        _load = 1800
        _label = 'Faulty'
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
        'signal': dm[fname[:-4]],
        'label': _label,
        'metadata': metadata
      }

