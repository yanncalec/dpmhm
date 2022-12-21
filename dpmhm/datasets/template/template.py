"""
Title here, with the information of author and license.

Description
===========
A short summary of the dataset.

Further information
-------------------

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

Processed data
==============

Split:

Features
--------

Notes
=====
"""

# import os
from pathlib import Path
# import itertools
# import json
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import pandas as pd
# from scipy.io import loadmat
# import mat4py
# import librosa

# from dpmhm.datasets.preprocessing import DatasetCompactor, FeatureTransformer, Preprocessor
from dpmhm.datasets import _DTYPE, _ENCODING


_CITATION = """
"""
# _METAINFO =

# Data urls, can be list or string.
_DATA_URLS = ['']
# _DATA_URLS = ''


class TEMPLATE(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for TEMPLATE dataset."""

    VERSION = tfds.core.Version('1.0.0')

    RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
    }

    # MANUAL_DOWNLOAD_INSTRUCTIONS =

    def _info(self) -> tfds.core.DatasetInfo:
        return tfds.core.DatasetInfo(
            builder=self,
            description=__doc__,
            features=tfds.features.FeaturesDict({
                # Both number of channels and length are fixed
                'signal': tfds.features.Tensor(shape=(None, 2), dtype=tf.float64),

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
                    'FaultLocation': tf.string,  # Location of the fault, e.g. {'FanEnd', 'DriveEnd'}
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
            datadir = Path(dl_manager._manual_dir)
        else:  # automatically download data
            # For too large dataset or unsupported format
            # raise NotImplementedError("Automatic download not supported.")

            # Parallel download (may result in corrupted files):
            #   _resource = tfds.download.Resource(url=_DATA_URLS, extract_method=tfds.download.ExtractMethod.ZIP)  # in case that the extraction method cannot be deduced automatically from files
            # datadir = dl_manager.download_and_extract(_resource)

            datadir = dl_manager.download_and_extract(_DATA_URLS) / 'name_of_the_ds'

            # Sequential download:
            # datadir = [Path(dl_manager.download_and_extract(url) for url in _DATA_URLS]

        return {
            sp.lower(): self._generate_examples(datadir, fn, sp.lower()) for sp, fn in _SPLIT_PATH_MATCH.items()
        # 'train': self._generate_examples(datadir),
    }

    def _generate_examples(self, path, fn):
        pass
        yield hash(frozenset(metadata.items())), {
            'signal': x,
            'label': 'Normal' if metadata['Split']=='Healthy' else 'Faulty',
            'metadata': metadata
        }