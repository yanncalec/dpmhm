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
    # TODO(mfpt): Specifies the tfds.core.DatasetInfo object
    return tfds.core.DatasetInfo(
        builder=self,
        description=_DESCRIPTION,
        features=tfds.features.FeaturesDict({
            # These are the features of your dataset like images, labels ...
            'signal': tfds.features.Tensor(shape=(None,), dtype=tf.float64),

            'label': tfds.features.ClassLabel(names=['Normal', 'Faulty', 'Unknown']),

            'metadata': {
              'SamplingRate': tf.uint32,  # up to 48,828 Hz
              'RotatingSpeed': tf.float32,  # up to 25 Hz
              'LoadForce': tf.float32,  # in [0, 300] lbs
              'DataLabel': tf.string, # {'Baseline', 'OuterRace', 'InnerRace', 'RealWorld}
              'FileName': tf.string,
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
    """Return SplitGenerators.
    """
    if dl_manager._manual_dir.exists():  # prefer to use manually downloaded data
      datadir = dl_manager._manual_dir
    else:  # automatically download data
      _path = dl_manager.download_and_extract(_URL)
      datadir = _path/'MFPT Fault Data Sets'

    return {
        'train': self._generate_examples(datadir),
    }

  def _generate_examples(self, path):
    """Yield examples.
    """
    for fp in path.rglob('*.mat'):
      # print(fp)
      if fp.parent.name[0] in ['1', '2', '3', '4', '6']:
        try:
          dm = tfds.core.lazy_imports.scipy.io.loadmat(fp)
          # dm = loadmat(fp)
        except Exception as msg:
          raise Exception(f"Error in processing {fp}: {msg}")

        metadata = {}
        for nn,vv in zip (dm['bearing'][0].dtype.names, dm['bearing'][0][0]):
          try:
            foo = _META_MATCH[nn.lower()]
            try:
              metadata[foo[0]] = foo[1](vv)
            except:
              metadata[foo[0]] = np.nan
          except:
            if nn.lower() == 'gs':
              x = vv.squeeze()

        metadata['DataLabel'] = _TYPE_MATCH[fp.parent.name[0]]
        if metadata['DataLabel'] == 'Baseline':
          label = 'Normal'
        elif metadata['DataLabel'] in ['InnerRace', 'OuterRace']:
          label = 'Faulty'
        elif metadata['DataLabel'] == 'RealWorld':
          label = 'Unknown'

        metadata['FileName'] = fp.name

        yield hash(frozenset(metadata.items())), {
          'signal': x,
          'label': label,
          'metadata': metadata
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
