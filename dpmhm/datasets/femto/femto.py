"""FEMTO dataset."""

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
FEMTO-ST bearing dataset used in the IEEE PHM 2012 Data Challenge for RUL (remaining useful lifetime) estimation.

Description
===========
This dataset consists of run-to-failure experiments carried on the PRONOSTIA platform. Data provided by this platform corresponds to normally degraded bearings, which means that the defects are not initially initiated on the bearings and that each degraded bearing contains almost all the types of defects (balls, rings and cage).

Data are acquired under three operating conditions (rotating speed and load force):
Condition 1. 1800 rpm and 4000 N: folders Bearing1_x
Condition 2. 1650 rpm and 4200 N: folders Bearing2_x
Condition 3. 1500 rpm and 5000 N: folders Bearing3_x

In order to avoid propagation of damages to the whole test bed (and for security reasons), all tests were stopped when the amplitude of the vibration signal overpassed 20g which is used as definition of the RUL.

Provided data contains the records of two acclerometers and one temperature sensor, and are splitted into the learning set of 6 experiments and the test set (truncated + full) of 11 experiments. The goal is to estimate the RUL on the test set. The learning set was quite small while the spread of the life duration of all bearings was very wide (from 1h to 7h).

Actual RULs (in second) of Test set
-----------------------------------
- Bearing1_3: 5730
- Bearing1_4: 339
- Bearing1_5: 1619
- Bearing1_6: 1460
- Bearing1_7: 7570
- Bearing2_3: 7530
- Bearing2_4: 1390
- Bearing2_5: 3090
- Bearing2_6: 1290
- Bearing2_7: 580
- Bearing3_3: 820

Homepage
--------
https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/#femto

http://www.femto-st.fr/

https://github.com/Lucky-Loek/ieee-phm-2012-data-challenge-dataset

Original data
=============
Format: CSV files
Number of channels: not fixed, up to 3
Vibration signals (horizontal and vertical)
- Sampling frequency: 25.6 kHz
- Recordings: 2560 (i.e. 1/10 s) are recorded each 10 seconds
Temperature signals
- Sampling frequency: 10 Hz
- Recordings: 600 samples are recorded each minute

Download
--------
https://github.com/Lucky-Loek/ieee-phm-2012-data-challenge-dataset
"""

_CITATION = """
P. Nectoux, R. Gouriveau, K. Medjaher, E. Ramasso, B. Morello, N. Zerhouni, C. Varnier. PRONOSTIA: An Experimental Platform for Bearings Accelerated Life Test. IEEE International Conference on Prognostics and Health Management, Denver, CO, USA, 2012
"""

_SPLIT_PATH_MATCH = {
  'train': 'Learning_set',
  'test': 'Test_set',
  'full_test': 'Full_Test_Set'
}

_PARSER_MATCH = {
  'Bearing1': 1, # 'condition 1',
  'Bearing2': 2, # 'condition 2',
  'Bearing3': 3, # 'condition 3',
}

_DATA_URLS = 'https://github.com/Lucky-Loek/ieee-phm-2012-data-challenge-dataset/archive/refs/heads/master.zip'


class FEMTO(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for FEMTO dataset."""

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    return tfds.core.DatasetInfo(
        builder=self,
        description=_DESCRIPTION,
        features=tfds.features.FeaturesDict({
          # Number of channels is named or not fixed
          'signal': {
            'vibration': tfds.features.Tensor(shape=(None, 2), dtype=tf.float64),
            'temperature': tfds.features.Tensor(shape=(None, ), dtype=tf.float64),
          },

          # 'label': tfds.features.ClassLabel(names=['Healthy', 'Faulty', 'Unknown']),

          'metadata': {
            'OperatingCondition': tf.int32,  # More operating conditions
            # 'ID': tf.string,  # ID of the bearing
            # 'Lifetime': tf.float32,  # Time of the run-to-failure experiment
            'OriginalSplit': tf.string,  # Original split
            'FileName': tf.string,  # Original filename with path
          }
        }),
        # If there's a common (input, target) tuple from the
        # features, specify them here. They'll be used if
        # `as_supervised=True` in `builder.as_dataset`.
        # supervised_keys=('signal', 'label'),  # Set to `None` to disable
        supervised_keys=None,
        homepage='',
        citation=_CITATION,
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    if dl_manager._manual_dir.exists():  # prefer to use manually downloaded data
      datadir = dl_manager._manual_dir
    else:  # automatically download data
      _resource = tfds.download.Resource(url=_DATA_URLS, extract_method=tfds.download.ExtractMethod.ZIP)  # in case that the extraction method cannot be deduced automatically from files
      datadir = dl_manager.download_and_extract(_resource)

    return {
      sp: self._generate_examples(datadir/fn, sp) for sp, fn in _SPLIT_PATH_MATCH.items()
    }

  def _generate_examples(self, path, split):
    # assert path.exists()
    # If the download files are extracted
    for fp in path.rglob('*.csv'):  # do not use `rglob` if path has no subfolders
      print(fp)
      # print(fp.parent)

      with open(fp,'r') as ff:
        sep = ';' if ';' in ff.readline() else ','

      dm = pd.read_csv(fp, sep=sep, header=None)
      assert dm.shape[1] >= 5

      if fp.name[:3] == 'acc':
        _signal = {
          'vibration': dm.iloc[:,-2:].values,
          'temperature': np.array([])
        }
      elif fp.name[:4] == 'temp':
        _signal = {
          'vibration': np.array([]).reshape((-1,2)),
          'temperature': dm.iloc[:,-1].values,
        }
      else:
        continue

      metadata = {
        'OperatingCondition': _PARSER_MATCH[fp.parts[-2].split('_')[0]],
        'OriginalSplit': split,
        'FileName': os.path.join(*fp.parts[-3:])
      }

      yield hash(frozenset(metadata.items())), {
        'signal': _signal,
        'metadata': metadata
      }

  @staticmethod
  def get_references():
    try:
      with open(Path(__file__).parent / 'Exported Items.bib') as fp:
        return fp.read()
    except:
      pass
