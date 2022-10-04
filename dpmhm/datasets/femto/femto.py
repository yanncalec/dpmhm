"""FEMTO dataset.

Instruction
-----------
If manually downloaded the data file need first to be unzipped.
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
- Bearing1_5: 1610
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

Further Information
-------------------
Monitoring data of the 11 test bearings were truncated so that participants were supposed to predict the remaining life, and thereby perform RUL estimates. Also, no assumption on the type of failure to be occurred was given (nothing known about the nature and the origin of the degradation: balls, inner or outer races, cage...)

Original data
=============
Format: CSV files
Channels: not fixed, up to 3 (2 vibrations and 1 temperature)
  Vibration signals (horizontal and vertical)
    - Sampling frequency: 25.6 kHz
    - Recordings: 2560 (i.e. 1/10 s) are recorded each 10 seconds
  Temperature signals
    - Sampling frequency: 10 Hz
    - Recordings: 600 samples are recorded each minute
Split: ['Learning_set', 'Test_set', 'Full_Test_Set']
Label: None

Download
--------
https://github.com/Lucky-Loek/ieee-phm-2012-data-challenge-dataset

Processed data
==============
Split: ['train', 'test', 'full_test']

Features
--------
'signal': {
  'vibration',
  'temperature'
},

'metadata': {
  'OperatingCondition': ['Bearing1','Bearing2','Bearing3']
  'OriginalSplit': ['Learning_set', 'Test_set', 'Full_Test_Set']
  'FileName': Original filename with path
}

Notes
=====
- The split 'test' is the truncation of 'full_test' used for RUL.
"""

_CITATION = """
P. Nectoux, R. Gouriveau, K. Medjaher, E. Ramasso, B. Morello, N. Zerhouni, C. Varnier. PRONOSTIA: An Experimental Platform for Bearings Accelerated Life Test. IEEE International Conference on Prognostics and Health Management, Denver, CO, USA, 2012
"""

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
            'OperatingCondition': tf.string,  # More operating conditions
            # 'SamplingRate': tf.uint32,
            # 'LoadForce': tf.uint32,
            # 'RotatingSpeed': tf.uint32,
            # 'ID': tf.string,  # ID of the bearing
            # 'Lifetime': tf.uint32,  # Time of the run-to-failure experiment, or Remaining Useful Life
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
      datadir = Path(dl_manager._manual_dir)  # force conversion
      # zipfile = dl_manager._manual_dir / 'femto.zip'

    else:  # automatically download data
      # _resource = tfds.download.Resource(url=_DATA_URLS, extract_method=tfds.download.ExtractMethod.ZIP)  # in case that the extraction method cannot be deduced automatically from files
      # datadir = dl_manager.download_and_extract(_resource)
      raise NotImplemented()

    return {
      'train': self._generate_examples(datadir, 'Learning_set'),
      'test': self._generate_examples(datadir, 'Test_set'),
      'full_test': self._generate_examples(datadir, 'Full_Test_Set')
    }

  def _generate_examples(self, datadir, split):
    for fp in (datadir/split).rglob('*.csv'):
      fname = fp.parts[-1]

      # Delimiter used by csv files is not uniform: both ',' and ';' are encountered => let pandas detect automatically
      try:
        dm = pd.read_csv(fp, sep=None, header=None, engine='python')
        assert dm.shape[1] >= 5
      except:
        raise Exception(f'Cannot parse CSV file {fname}')

      if fname[:3] == 'acc':
        _signal = {
          'vibration': dm.iloc[:,-2:].values,
          'temperature': np.array([])
        }
      elif fname[:4] == 'temp':
        _signal = {
          'vibration': np.array([]).reshape((-1,2)),
          'temperature': dm.iloc[:,-1].values,
        }
      else:
        continue

      metadata = {
        'OperatingCondition': fp.parts[-2].split('_')[0],
        'OriginalSplit': split,
        'FileName': os.path.join(*fp.parts[-2:])  # full path file name
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
