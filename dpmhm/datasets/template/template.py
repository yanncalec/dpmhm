"""TEMPLATE dataset."""

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
Title here, with the information of author and license.

Description
===========
A short summary of the dataset.

Homepage
--------

Original data
=============
Format:
Date of acquisition:
Channels:
Split:
Operating conditions:
Faults:
Size:
Sampling rate:
Recording duration:
Recording period:

Download
--------

Notes
=====
"""

_CITATION = """
"""

_SPLIT_PATH_MATCH = {
  'name of the split': 'folder of the original data',
}

_PARSER_MATCH = {
  # 'file name pattern':
}


_DATA_URLS = ''


class TEMPLATE(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for TEMPLATE dataset."""

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }

  MANUAL_DOWNLOAD_INSTRUCTIONS = """
  Due to ????, automatic download is not supported in this package. Please download (optionally extract) all data and proceed the installation manually.
  """

  def _info(self) -> tfds.core.DatasetInfo:
    return tfds.core.DatasetInfo(
        builder=self,
        description=_DESCRIPTION,
        features=tfds.features.FeaturesDict({
          # Both number of channels and length are fixed
          'signal': tfds.features.Tensor(shape=(None, 2), dtype=tf.float64),
          # # Number of channels is not fixed but length is fixed
          # 'signal': tfds.features.Tensor(shape=(None, None, 1), dtype=tf.float64),
          # # Number of channels is named or not fixed
          # 'signal': {
          #   'Channel 1': tfds.features.Tensor(shape=(None, ), dtype=tf.float64),
          #   'Channel 2': tfds.features.Tensor(shape=(None, ), dtype=tf.float64),
          # },

          'label': tfds.features.ClassLabel(names=['Healthy', 'Faulty', 'Unknown']),

          'metadata': {
            'ID': tf.string,  # ID of the bearing or experiment
            'SamplingRate': tf.uint32,  # Sampling rate of the signal
            'RotatingSpeed': tf.float32,  # Rotation speed of the shaft
            'LoadForce': tf.float32,  # Load force
            'OperatingCondition': tf.string,  # More details about the operating conditions
            'SensorName': tf.string,  # name of sensor
            'SensorLocation': tf.string,  # Location of the sensor, e.g. {'Gearbox', 'Bearing'}
            'FaultName': tf.string,  # Name of the fault, e.g. {'Unbalance', 'Lossness'}
            'FaultComponent': tf.string, # Component of the fault, e.g. {'Ball', 'Cage' ,'InnerRace', 'OuterRace'}
            'FaultSize': tf.float32,  # Size of the fault
            'FaultExtend': tf.int32,  # Extend of the fault, e.g. mild, severe etc
            'FaultType': tf.string,  # Type of damage: e.g. EDM, Engraver, Lifetime (fatigue, plastic indentation),
            'Lifetime': tf.float32,  # Time of the run-to-failure experiment
            'OriginalSplit': tf.string,  # Original split that the signal belongs to
            'DataLabel': tf.string, # Other uncategorized information
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
      # raise FileNotFoundError(self.MANUAL_DOWNLOAD_INSTRUCTIONS)

      _resource = tfds.download.Resource(url=_DATA_URLS, extract_method=tfds.download.ExtractMethod.ZIP)  # in case that the extraction method cannot be deduced automatically from files
      datadir = dl_manager.download_and_extract(_resource)

    return {
        sp.lower(): self._generate_examples(datadir, fn, sp.lower()) for sp, fn in _SPLIT_PATH_MATCH.items()
        # 'train': self._generate_examples(datadir),
    }

  def _generate_examples(self, path, fn):
    # assert path.exists()

    # for sp, fnames in _SPLIT_PATH_MATCH.items():

    # If the download files are not extracted
    for fname0, fobj in tfds.download.iter_archive(path, tfds.download.ExtractMethod.ZIP):
      fname = Path(fname0).parts[1]

    # If the download files are extracted
    for fp in path.glob('*.xxx'):  # do not use `rglob` if path has no subfolders
      # print(fp)
      dm = pd.read_csv(fp)  # csv file
      try:
        dm = tfds.core.lazy_imports.scipy.io.loadmat(fp)
      except Exception as msg:
        raise Exception()(f'Error in processing {fp}: {msg}')
        # print(f'Error in processing {fp}: {msg}')
        pass

      x = np.stack([dm['Channel_1'].squeeze(), dm['Channel_2'].squeeze()]).T

      metadata = {
        'FileName': fp.name,
      }

      yield hash(frozenset(metadata.items())), {
        'signal': x,
        'label': 'Normal' if metadata['Split']=='Healthy' else 'Faulty',
        'metadata': metadata
      }

  @staticmethod
  def get_references():
    try:
      with open(Path(__file__).parent / 'Exported Items.bib') as fp:
        return fp.read()
    except:
      pass
