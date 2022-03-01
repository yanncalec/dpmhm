"""Phmap2021 dataset."""

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
PHM Asia Pacific 2021 Data Challenge.

Description
===========
Data has been acquired under a variety of environments, not only fault-free condition, but also seeded faults under controlled conditions with the support of domain expertise. Hence, the dataset comprises signals in five categories: Normal, Unbalance, Belt-looseness, Belt-looseness (High), and Bearing fault. As two sensors equipped on the compressor monitor the signals, the dataset contains two signals from each channel.

System Information
------------------
Type of Equipment: Oil-injection screw compressor
Motor: 15kW
Axis rotating speed of Motor: 3,600 rpm
Axis rotating speed of Screw: 7,200 rpm

Homepage
--------
http://phmap.org/data-challenge/

Original data
=============
Format: CSV
Channels: 2 vibration channels, from motor and screw
Sampling rate: 10544 Hz
Recording duration: not fixed
Split: not specified
Operating conditions: fixed rotating speed at 3600 rpm for the motor and 7200 rpm for the screw
Faults: 5 Classes
- Normal: Fault-free operating condition
- Unbalance: Unbalance between centers of mass and axis
- Belt-Looseness: Looseness of V‐belt connecting between motor pully and screw pully
- Belt-Looseness High: High Looseness of V-belt
- Bearing fault: Removing grease of Ball Bearing on Motor, which induces its wear-out

Download
--------
https://drive.google.com/drive/folders/1Zcth6UhPfP3vM8YadHhCSmK6v4aEyOA7

Notes
=====
Two test sets (for the task of classification and regression) were also provided along with the original datset, but cannot be downloaded. See:
https://www.kaggle.com/c/phmap21-regression-task/data
"""

_CITATION = """
@misc{noauthor_data_nodate,
	title = {Data {Challenge} – {PHM} {Asia} {Pacific} 2021},
	url = {http://phmap.org/data-challenge/},
	language = {en-US},
	urldate = {2022-03-01},
}
"""

_SPLIT_PATH_MATCH = {
  'Normal': ['train_1st_Normal.csv', 'train_3rd_Normal.csv'],
  'Unbalance': ['train_1st_Unbalance.csv', 'train_2nd_Unbalance.csv', 'train_3rd_Unbalance.csv'],
  'Looseness': ['train_1st_Looseness.csv', 'train_2nd_Looseness.csv'],
  'High': ['train_1st_high.csv'],
  'Bearing': ['train_1st_Bearing.csv', 'train_2nd_Bearing.csv']
}


class Phmap2021(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for Phmap2021 dataset."""

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }

  MANUAL_DOWNLOAD_INSTRUCTIONS = """
  Automatic download is not supported in this package. Please download (optionally extract) all data via the link

    https://drive.google.com/drive/folders/1Zcth6UhPfP3vM8YadHhCSmK6v4aEyOA7

  Extract all files and check that all filenames follows the pattern `train_xxx_Yyy.csv` (since some files may be automatically renamed during the download). Then proceed the installation manually.
  """

  def _info(self) -> tfds.core.DatasetInfo:
    return tfds.core.DatasetInfo(
        builder=self,
        description=_DESCRIPTION,
        features=tfds.features.FeaturesDict({
          # Both number of channels and length are fixed
          'signal': tfds.features.Tensor(shape=(None, 2), dtype=tf.float64),

          'label': tfds.features.ClassLabel(names=list(_SPLIT_PATH_MATCH.keys())),

          'metadata': {
            'OriginalSplit': tf.string,  # Original split that the signal belongs to
            'FileName': tf.string,  # Original filename with path in the dataset
          },
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
      # For too large dataset or unsupported format
      raise FileNotFoundError(self.MANUAL_DOWNLOAD_INSTRUCTIONS)

    return {
        # sp.lower(): self._generate_examples(datadir, fn, sp) for sp, fn in _SPLIT_PATH_MATCH.items()
        'train': self._generate_examples(datadir),
    }

  def _generate_examples(self, path):
    # assert path.exists()

    for sp, fnames in _SPLIT_PATH_MATCH.items():
      for fn in fnames:
        fp = path / fn
        # print(fp)

        dm = pd.read_csv(fp, index_col=0)
        metadata = {
          'OriginalSplit': sp,
          'FileName': fp.name,
        }

        yield hash(frozenset(metadata.items())), {
          'signal': dm.values,
          'label': sp,
          'metadata': metadata
        }

  @staticmethod
  def get_references():
    try:
      with open(Path(__file__).parent / 'Exported Items.bib') as fp:
        return fp.read()
    except:
      pass
