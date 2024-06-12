"""MFPT Condition Based Maintenance Fault Database.
"""

import os
import pathlib
import itertools
import json
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import pandas as pd
from scipy.io import loadmat

from dpmhm.datasets import _DTYPE, _ENCODING

_DESCRIPTION = """
MFPT Condition Based Maintenance Fault Database.

Data Assembled and Prepared on behalf of MFPT by Dr Eric Bechhoefer, Chief Engineer, NRG Systems

Description
===========
The data set comprises the following
- 3 baseline conditions: 270 lbs of load, input shaft rate of 25 Hz, sample rate of 97,656 sps, for 6 seconds
- 3 outer race fault conditions: 270 lbs of load, input shaft rate of 25 Hz, sample rate of 97,656 sps for 6 seconds
- 7 outer race fault conditions: 25, 50, 100, 150, 200, 250 and 300 lbs of load, input shaft rate 25 Hz, sample rate of 48,828 sps for 3 seconds (bearing resonance was found be less than 20 kHz)
- 7 inner race fault conditions: 0, 50, 100, 150, 200, 250 and 300 lbs of load, input shaft rate of 25 Hz, sample rate of 48,828 sps for 3 seconds
- 3 real world example files are also included: an intermediate shaft bearing from a wind turbine (data structure holds bearing rates and shaft rate), an oil pump shaft bearing from a wind turbine, and a real world planet bearing fault).

Homepage
--------
https://www.mfpt.org/fault-data-sets/

Original data files
===================
Format: Matlab
Sampling rate: not fixed
Number of channels: 1
Label: normal, faulty and unknown
Year of acquisition: 2013
Size: 59 Mb

Download
--------
https://www.mfpt.org/wp-content/uploads/2020/02/MFPT-Fault-Data-Sets-20200227T131140Z-001.zip

Modifications
=============
The original split is not used in this package. Use the field of `DataLabel` or `FileName` to recover the information of split.
"""

_CITATION = """
@misc{bechhoefer_condition_2013,
	title = {Condition {Based} {Maintenance} {Fault} {Database} for {Testing} of {Diagnostic} and {Prognostics} {Algorithms}},
	shorttitle = {{MFPT} {Bearing} {Fault} {Dataset}},
	url = {https://www.mfpt.org/fault-data-sets/},
	abstract = {The goal of the Condition Based Maintenance Fault Database is to provide various data sets of known good and faulted conditions for both bearings and gears. This dataset is hereby freely distributed with example processing code with the hope that researchers and CBM practitioners will improve upon the techniques, and consequently, mature CBM systems, faster.},
	publisher = {Society for Machinery Failure Prevention Technology},
	author = {Bechhoefer, Eric},
	year = {2013},
}
"""

_URL = 'https://www.mfpt.org/wp-content/uploads/2020/02/MFPT-Fault-Data-Sets-20200227T131140Z-001.zip'


_META_MATCH = {
  'rate': ('RotatingSpeed', lambda x: float(x)),
  'load': ('LoadForce', lambda x: int(x)),
  # 'gs': 'Signal',
  'sr': ('SamplingRate', lambda x: int(x)),
  # 'ball', 'cage', 'outer', 'inner'
}

_TYPE_MATCH = {
  '1': 'Baseline',
  '2': 'OuterRace',
  '3': 'OuterRace',
  '4': 'InnerRace',
  '6': 'RealWorld'
}


class MFPT(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for MFPT dataset."""

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    # Specifies the tfds.core.DatasetInfo object
    return tfds.core.DatasetInfo(
        builder=self,
        description=_DESCRIPTION,
        features=tfds.features.FeaturesDict({
            # These are the features of your dataset like images, labels ...
            'signal': {'sig':tfds.features.Tensor(shape=(None,), dtype=tf.float32)},
            'sampling_rate': tf.uint32,
            'metadata': {
              'Label': tfds.features.Text(),
              'ShaftRate': tf.float32, # 25 Hz
              'LoadForce': tf.float32,  # in [0, 300] lbs
              'FileName': tf.string,
              'Dataset': tf.string,
            }
        }),
        # If there's a common (input, target) tuple from the
        # features, specify them here. They'll be used if
        # `as_supervised=True` in `builder.as_dataset`.
        supervised_keys=None,  # Set to `None` to disable
        homepage='https://www.mfpt.org/fault-data-sets/',
        citation=_CITATION,
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    def _get_split_dict(datadir):
      directories = [
          '1 - Three Baseline Conditions',
          '2 - Three Outer Race Fault Conditions',
          '3 - Seven More Outer Race Fault Conditions',
          '4 - Seven Inner Race Fault Conditions',
          # '6 - Real World Examples'
      ]
      mat_files = []

      for dir_name in directories:
        dir_path = datadir / dir_name
        if dir_path.exists():
          mat_files.extend(dir_path.glob('*.mat'))
        else:
          print(f"The directory {dir_path} does not exist.")
      
      return {
        'train': mat_files,
      }
    if dl_manager._manual_dir.exists():  # prefer to use manually downloaded data
      datadir = dl_manager._manual_dir
    else:  # automatically download data
      _path = dl_manager.download_and_extract(_URL)
      datadir = _path/'MFPT Fault Data Sets'

    return {sp: self._generate_examples(files) for sp, files in _get_split_dict(datadir).items()}

  def _generate_examples(self, path):
    def convert_to_float32(data):
      """Convert numpy arrays in the data to float32."""
      if isinstance(data, dict):
        return {k: convert_to_float32(v) for k, v in data.items()}
      elif isinstance(data, np.ndarray):
        return data.astype(np.float32)
      else:
        return data
    """Yield examples."""
    counter = itertools.count()
    for fp in path:
      try:
        mat_data = loadmat(fp)
      except Exception as e:
        raise ValueError(f"Error loading MAT file {fp}: {e}")

      # Process signals
      signals = mat_data['bearing'][0]
      for signal in signals:
        # Extract metadata
        metadata = {}

        # Determine the data label based on the parent directory name
        data_label = _TYPE_MATCH.get(fp.parent.name[0])
        if data_label=='Baseline':
          metadata['DataLabel'] = data_label
        elif data_label=='OuterRace':
          metadata['DataLabel'] = 'OuterRace'
        elif data_label=='InnerRace':
          metadata['DataLabel'] = 'InnerRace'
        else:
          metadata['DataLabel'] = 'RealWorld'

        if metadata['DataLabel'] == 'Baseline':
          sample_rate = signal[0][0][0]
          accel_values = signal[1][:, 0]
          load = signal[2][0][0] 
          shaft_rate = signal[3][0][0]
        elif metadata['DataLabel'] == 'OuterRace' or metadata['DataLabel'] == 'InnerRace':
          sample_rate = signal[3][0][0]
          accel_values = signal[2][:, 0]
          load = signal[1][0][0]
          if load != 270.0:
            load = [25.0, 50.0, 100.0, 150.0, 200.0, 250.0, 300.0][int(load)-1]
          shaft_rate = signal[0][0][0]
        elif metadata['DataLabel'] == 'RealWorld':
          sample_rate=48828
          accel_values = []
          for element in signal:
              if element.size > 0:
                  # Aplatir l'array et ajouter ses valeurs à la liste
                  accel_values.extend(element.flatten())
          load=270
          shaft_rate=25

        metadata['ShaftRate'] = shaft_rate
        metadata['LoadForce'] = load

        key = f"{next(counter)}"
        # Yield example
        yield key, {
          'signal': {'sig':convert_to_float32(accel_values)},
          'sampling_rate':sample_rate,
          'metadata': {
              'Label': metadata.get('DataLabel', ''),
              'ShaftRate': metadata.get('ShaftRate', ''),
              'LoadForce': metadata.get('LoadForce', ''),
              'FileName': fp.name,
              'Dataset': 'MFPT',
          }
        }


  # def _split_generators(self, dl_manager: tfds.download.DownloadManager):
  #   """Returns SplitGenerators."""
  #   # TODO(mfpt): Downloads the data and defines the splits
  #   path = dl_manager.download_and_extract('https://todo-data-url')

  #   # TODO(mfpt): Returns the Dict[split names, Iterator[Key, Example]]
  #   return {
  #       'train': self._generate_examples(path / 'train_imgs'),
  #   }

  # def _generate_examples(self, path):
  #   """Yields examples."""
  #   # TODO(mfpt): Yields (key, example) tuples from the dataset
  #   for f in path.glob('*.jpeg'):
  #     yield 'key', {
  #         'image': f,
  #         'label': 'yes',
  #     }
