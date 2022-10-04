"""Fraunhofer_151 dataset."""

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
Vibration Measurements on a Rotating Shaft at Different Unbalance Strengths

Description
===========
This dataset contains vibration data recorded on a rotating drive train. This drive train consists of an electronically commutated DC motor and a shaft driven by it, which passes through a roller bearing. With the help of a 3D-printed holder, unbalances with different weights and different radii were attached to the shaft. Besides the strength of the unbalances, the rotation speed of the motor was also varied. This dataset can be used to develop and test algorithms for the automatic detection of unbalances on drive trains. Datasets for 4 differently sized unbalances and for the unbalance-free case were recorded. The vibration data was recorded at a sampling rate of 4096 values per second. Datasets for development (ID "D[0-4]") as well as for evaluation (ID "E[0-4]") are available for each unbalance strength. The rotation speed was varied between approx. 630 and 2330 RPM in the development datasets and between approx. 1060 and 1900 RPM in the evaluation datasets. For each measurement of the development dataset there are approx. 107min of continuous measurement data available, for each measurement of the evaluation dataset 28min. Details of the recorded measurements and the used unbalance strengths are documented in the README.md file.

Homepage
--------
https://fordatis.fraunhofer.de/handle/fordatis/151.2

Original Data
=============
Format: CSV files
Date of acquisition: 2020
Channels: measured rpm and 3 vibrations
Split: development and evaluation
Operating conditions: 630~2330 rpm for the development set and 1060~1900 rpm for the evaluation set, continuous record.
Sampling rate: 4096 Hz
Recording duration: 107 minutes for development set and 28 minutes for evaluation set.
Faults: 4 differently sized unbalances and unbalance-free case

Download
--------
https://fordatis.fraunhofer.de/bitstream/fordatis/151.2/1/fraunhofer_eas_dataset_for_unbalance_detection_v1.zip

https://fordatis.fraunhofer.de/bitstream/fordatis/151.2/3/README.md

Notes
=====
The original record consists of two periods where the rotation speed increases linearly. In this package these two periods are separated. Moreover, the first 5 seconds of record seem to correspond to initialization hence are removed.
"""

_CITATION = """
@article{mey_vibration_2020,
	title = {Vibration {Measurements} on a {Rotating} {Shaft} at {Different} {Unbalance} {Strengths}},
	copyright = {https://creativecommons.org/licenses/by/4.0/},
	url = {https://fordatis.fraunhofer.de/handle/fordatis/151.2},
	doi = {10.24406/fordatis/65.2},
	language = {en},
	urldate = {2022-03-01},
	author = {Mey, Oliver and Neudeck, Willi and Schneider, André and Enge-Rosenblatt, Olaf},
	month = mar,
	year = {2020},
	note = {Accepted: 2020-04-14T10:59:09Z},
}
"""

_DATA_URLS = 'https://fordatis.fraunhofer.de/bitstream/fordatis/151.2/1/fraunhofer_eas_dataset_for_unbalance_detection_v1.zip'


class Fraunhofer151(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for Fraunhofer_151 dataset."""

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }

  def _info(self) -> tfds.core.DatasetInfo:
    return tfds.core.DatasetInfo(
        builder=self,
        description=_DESCRIPTION,
        features=tfds.features.FeaturesDict({
          'signal': {
            'V_in': tfds.features.Tensor(shape=(None,), dtype=tf.float64),
            'Measured_RPM': tfds.features.Tensor(shape=(None,), dtype=tf.float64),
            'Vibrations': tfds.features.Tensor(shape=(None, 3), dtype=tf.float64),
          },

          'label': tfds.features.ClassLabel(names=['Normal', 'Unbalanced']),

          'metadata': {
            'StartIndex': tf.int32,
            'FaultExtend': tf.int32,
            'FileName': tf.string,  # Original filename with path in the dataset
          },
        }),
        # If there's a common (input, target) tuple from the
        # features, specify them here. They'll be used if
        # `as_supervised=True` in `builder.as_dataset`.
        # supervised_keys=('signal', 'label'),  # Set to `None` to disable
        supervised_keys=None,
        homepage='https://fordatis.fraunhofer.de/handle/fordatis/151',
        citation=_CITATION,
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    if dl_manager._manual_dir.exists():  # prefer to use manually downloaded data
      datadir = dl_manager._manual_dir
    else:  # automatically download data
      _resource = tfds.download.Resource(url=_DATA_URLS, extract_method=tfds.download.ExtractMethod.ZIP)  # in case that the extraction method cannot be deduced automatically from files
      datadir = dl_manager.download_and_extract(_resource)

    return {
        'train': self._generate_examples(datadir, 'train'),
        'test': self._generate_examples(datadir, 'test'),
    }

  def _generate_examples(self, path, split):
    # assert path.exists()

    fpaths = path.glob('*D.csv') if split=='train' else path.glob('*E.csv')

    for fp in fpaths:
      # print(fp)

      metadata = {
        'FaultExtend': int(fp.name[0]),
        'FileName': fp.name
      }
      label = 'Normal' if metadata['FaultExtend']==0 else 'Unbalanced'

      df0 = pd.read_csv(fp)

      # The original signal is truncated into two parts
      sl = 4096*3200 if split=='train' else 4096*835
      t0s = [4096*5, 4096*3225] if split=='train' else [4096*5, 4096*845]

      for t0 in t0s:
        metadata['StartIndex'] = t0
        df = df0.loc[t0:(t0+sl)]

        yield hash(frozenset(metadata.items())), {
          'signal': {
            'V_in': df.loc[t0:(t0+sl)]['V_in'].values,
            'Measured_RPM': df['Measured_RPM'].values,
            'Vibrations': df[['Vibration_1', 'Vibration_2', 'Vibration_3']].values,
          },
          'label': label,
          'metadata': metadata
        }

  @staticmethod
  def get_references():
    try:
      with open(Path(__file__).parent / 'Exported Items.bib') as fp:
        return fp.read()
    except:
      pass
