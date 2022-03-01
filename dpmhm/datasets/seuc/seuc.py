"""Gearbox dataset from Southeast University, China (SEUC).
"""

import os
import pathlib
import itertools
import json
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import pandas as pd

_DESCRIPTION = """
Gearbox dataset from Southeast University, China (SEUC).

Data acquired by Dr. Siyu Shao in the research group of Pf. Ruqiang Yan, Southeaset University, China.

Description
===========
This dataset contains 2 subdatasets, including bearing data and gear data, which are both acquired on Drivetrain Dynamics Simulator (DDS). Two different working conditions are investigated with the rotating speed system load set to be either 20 HZ-0V or 30 HZ-2V.

Homepage
--------
http://mlmechanics.ics.uci.edu/

Original data files
===================
Format: csv
Sampling rate: not specified
Recording duration: 1048560 samples
Number of channels: 8
Splits: `bearingset` and `gearset`
Label: normal, faulty
Size: 1.6 Gb

Each file contains 8 channels of signals representing:
- channel 1: motor vibration,
- channels 2,3,4: vibration of planetary gearbox in the x-y-z directions
- channel 5: motor torque,
- channels 6,7,8: vibration of parallel gearbox in the x-y-z directions
Signals of rows 2,3,4 are all effective.

Bearing dataset
---------------
- Ball
- Inner ring
- Outer ring
- Combination of inner ring and outer ring
- Health woring state

Gear dataset
------------
- Chipped: Crack occurs in the gear feet
- Missing: Missing one of feet in the gear
- Root: Crack occurs in the root of gear feet
- Surface: Wear occurs in the surface of gear
- Health working state

URLs
----
https://github.com/cathysiyu/Mechanical-datasets

Modifications
=============
- The original splits `bearingset` and `gearset` are renamed to `bearing` and `gearbox` respectively.
- The original dataset contains also a subset of CWRU, which is excluded from the current package.

References
==========
- Highly Accurate Machine Fault Diagnosis Using Deep Transfer Learning, Shao et al., 2019.
"""

# TODO(seuc): BibTeX citation
_CITATION = """
@ARTICLE{8432110,
  author={Shao, Siyu and McAleer, Stephen and Yan, Ruqiang and Baldi, Pierre},
  journal={IEEE Transactions on Industrial Informatics},
  title={Highly Accurate Machine Fault Diagnosis Using Deep Transfer Learning},
  year={2019},
  volume={15},
  number={4},
  pages={2446-2455},
  doi={10.1109/TII.2018.2864759}}
"""

_URL = 'https://github.com/cathysiyu/Mechanical-datasets/archive/refs/heads/master.zip'

# Components of fault
_FAULT_GEARBOX = ['Chipped', 'Missing', 'Root', 'Surface']

_FAULT_BEARING = ['Ball', 'Inner', 'Outer', 'Combination']

_FAULT_COMPONENT = ['None'] + _FAULT_BEARING + _FAULT_GEARBOX

# Dictionary for name conversion
_FAULT_MATCH = {
  'comb': 'Combination',
  'health': 'None',
  'miss': 'Missing',
}


class SEUC(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for SEUC dataset.
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
          'signal': tfds.features.Tensor(shape=(None, 8), dtype=tf.float64),

          'label': tfds.features.ClassLabel(names=['Normal', 'Faulty']),

          'metadata': {
            'LoadForce':  tf.string,
            'FaultComponent': tf.string,
            'OriginalSplit': tf.string,  # the original split
            'FileName': tf.string,
          },

          # 'metadata': {
          #   'Load':  tfds.features.ClassLabel(names=['20Hz-0V', '30Hz-2V']),
          #   'Location': tfds.features.ClassLabel(names=['Bearing', 'Gearbox']),
          #   'Component': tfds.features.ClassLabel(names=_FAULT_COMPONENT),
          # },
        }),
        # If there's a common (input, target) tuple from the
        # features, specify them here. They'll be used if
        # `as_supervised=True` in `builder.as_dataset`.
        supervised_keys=None,
        # supervised_keys=('signal', 'label'),  # Set to `None` to disable
        homepage='https://github.com/cathysiyu/Mechanical-datasets',
        citation=_CITATION,
    )

  @classmethod
  def _fname_parser(cls, fname):
    foos = fname[:-4].split('_')  # remove '.csv'
    f = foos[0].lower()
    _component = _FAULT_MATCH[f] if f in _FAULT_MATCH else f.capitalize()
    # _component = f.capitalize()
    _load = f'{foos[1]}Hz-{foos[2]}V'
    return _component, _load

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Return SplitGenerators.
    """
    if dl_manager._manual_dir.exists():  # prefer to use manually downloaded data
      datadir = dl_manager._manual_dir / 'gearbox'
    else:  # automatically download data
      _path = dl_manager.download_and_extract(_URL)
      datadir = _path/'Mechanical-datasets-master'/'gearbox'
    # print(datadir)

    return {
      # Use the original splits
      'gearbox': self._generate_examples(datadir/'gearset'),
      'bearing': self._generate_examples(datadir/'bearingset'),
      # 'train': self._generate_examples(datadir)  # this will rewrite on precedent splits
    }

  def _generate_examples(self, path):
    # assert path.exists()
    # Recursive glob `path.rglob` may not behave as expected
    for fp in path.glob('*.csv'):
      # print(fp)
      _component, _load = self._fname_parser(fp.name)
      try:
        df = pd.read_csv(fp,skiprows=15, sep='\t').iloc[:,:-1]
        if df.shape[1] != 8:
          raise Exception
      except:
        df = pd.read_csv(fp,skiprows=15, sep=',').iloc[:,:-1]
        if df.shape[1] != 8:
          raise Exception

      metadata = {
        'LoadForce': _load,
        'FaultComponent': _component,
        'OriginalSplit': 'Gearbox' if fp.parent.name=='gearset' else 'Bearing',
        'FileName': fp.name,
      }

      yield hash(frozenset(metadata.items())), {
          'signal': df.values,
          'label': 'Normal' if _component=='None' else 'Faulty',
          'metadata': metadata
      }
