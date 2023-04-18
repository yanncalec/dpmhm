"""
Title here, with the information of author and license.

Description
===========
A short summary of the dataset.

Further information
-------------------

Homepage
--------

Original Dataset
================
- Type of experiments: initiated faults or run-to-failure
- Format:
- Date of acquisition:
- Size:
- Channels:
- Split:
- Sampling rate:
- Recording duration:
- Recording period:
- Operating conditions:
- Data fields:

Download
--------

Built Dataset
=============

- Split:

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

from dpmhm.datasets import _DTYPE, _ENCODING, extract_zenodo_urls


_CITATION = """
"""
# _METAINFO =

# Data urls, can be list or string.
_DATA_URLS = ['']
# _DATA_URLS = ''


class TEMPLATE(tfds.core.GeneratorBasedBuilder):
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
                'signal': tfds.features.Tensor(shape=(None, 3), dtype=_DTYPE, encoding=_ENCODING),

                # 'signal': {
                #     'Channel_1': tfds.features.Tensor(shape=(None, ), dtype=_DTYPE, encoding=_ENCODING),
                #     'Channel_2': tfds.features.Tensor(shape=(None, 2), dtype=_DTYPE, encoding=_ENCODING),
                # }

                # 'label': tfds.features.ClassLabel(names=['Healthy', 'Faulty', 'Unknown']),

                'sampling_rate': tf.uint32,  # Sampling rate of the signal

                'metadata': {
                    'Label': tf.string, # label information
                    'ID': tf.string,  # ID of the bearing or experiment
                    'RotatingSpeed': tf.float32,  # Rotation speed of the shaft
                    'RPM': tf.float32, # real rpm
                    'NominalRPM': tf.float32, # nominal rpm
                    'LoadForce': tf.float32,  # Load force
                    'OperatingCondition': tf.string,  # More details about the operating conditions
                    'SensorName': tf.string,  # name of sensor
                    'SensorLocation': tf.string,  # Location of the sensor, e.g. {'Gearbox', 'Bearing'}
                    'FaultName': tf.string,  # Name of the fault, e.g. {'Unbalance', 'Lossness'}
                    'FaultLocation': tf.string,  # Location of the fault, e.g. {'FanEnd', 'DriveEnd'}
                    'FaultComponent': tf.string, # Component of the fault, e.g. {'Ball', 'Cage' ,'InnerRace', 'OuterRace', 'Imbalance', 'Misalignment'}
                    'FaultSize': tf.float32,  # Size of the fault
                    'FaultExtend': tf.int32,  # Extend of the fault, e.g. mild, severe etc
                    'FaultType': tf.string,  # Type of damage: e.g. EDM, Engraver, Lifetime (fatigue, plastic indentation)
                    'Lifetime': tf.float32,  # Time of the run-to-failure experiment
                    'OriginalSplit': tf.string,  # Original split that the signal belongs to
                    'FileName': tf.string,  # Original filename with path in the dataset
                    'Dataset': tf.string, # name of the dataset
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
        def _get_split_dict(datadir):
            return {
                'train': next(datadir.rglob('train_folder')).rglob('*.mat'),
                'test': next(datadir.rglob('test_folder')).rglob('*.mat'),
            }

        if dl_manager._manual_dir.exists():  # prefer to use manually downloaded data
            datadir = Path(dl_manager._manual_dir)
        elif dl_manager._extract_dir.exists(): # automatically downloaded & extracted data
            datadir = Path(dl_manager._extract_dir)
        # elif dl_manager._download_dir.exists(): # automatically downloaded data
        #     datadir = Path(dl_manager._download_dir)
        #     tfds.download.iter_archive(fp, tfds.download.ExtractMethod.ZIP)
        else:
            raise FileNotFoundError()

        return {sp: self._generate_examples(files, sp) for sp, files in _get_split_dict(datadir).items()}


    def _generate_examples(self, files, split):
        yield hash(frozenset(metadata.items())), {
            'signal': x,
            'sampling_rate': sr,
            'metadata': metadata
        }

    # @staticmethod
    # def get_references():
    #     try:
    #         with open(Path(__file__).parent / 'references.bib') as fp:
    #             return fp.read()
    #     except:
    #         pass
