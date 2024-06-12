"""PHM2022

A Dataset for Fault Classification in Rock Drills, a Fast Oscillating Hydraulic System

Description
===========
The data is collected from a carefully instrumented hydraulic rock drill, operating in a test cell while inducing a number of faults. Hydraulic pressures are measured at 50kHz at three different locations, resulting in a detailed pressure signature for each
fault. Due to wave propagation phenomena, the system is sensitive to individual differences between different rock drills, drills rigs and configurations. Such differences are introduced in the data by altering certain parameters in the test setup. An important part of the data is therefore the availability of No-fault reference cycles, which are supplied for all individuals. These reference cycles give information on how individuals differ from each other, and can be used to improve classification.

Further information
-------------------

Homepage
--------

Original Dataset
================
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

Built Dataset
=============

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
import csv
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


class PHM2022(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for PHM dataset."""

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
                'signal': {'sig':tfds.features.Tensor(shape=(None,), dtype=_DTYPE, encoding=_ENCODING)},
                'sampling_rate': tf.uint32,
                'metadata': {
                    'Label': tfds.features.Text(),
                    'SensorType': tf.string,
                    'FileName': tf.string,
                    'Dataset': tf.string,
                },
            }),
            supervised_keys=None,
            homepage='',
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        def _get_split_dict(datadir):
            return {
                'test': next(datadir.rglob('Data_Challenge_PHM2022_testing_data')).glob('*.csv'),
                'train': next(datadir.rglob('Data_Challenge_PHM2022_training_data')).glob('*.csv'),
                'val': next(datadir.rglob('Data_Challenge_PHM2022_validation_data')).glob('*.csv'),
            }
        
        if dl_manager._manual_dir.exists():  # prefer to use manually downloaded data
            datadir = Path(dl_manager._manual_dir)
        else:
            datadir = dl_manager.download_and_extract(_DATA_URLS) / 'name_of_the_ds'

            # Sequential download:
            # datadir = [Path(dl_manager.download_and_extract(url) for url in _DATA_URLS]

        return {sp: self._generate_examples(files) for sp, files in _get_split_dict(datadir).items()}

    def _generate_examples(self, files):
        for file_path in files:
            with open(file_path, newline='') as csvfile:
                reader = csv.reader(csvfile)
                for i, row in enumerate(reader):
                    label_num = int(row[0])  # Extract numeric label from the first column
                    label = self._convert_label(label_num)  # Convert numeric label to corresponding string label
                    signal = [float(val) for val in row[1:]]
                    metadata = self._extract_metadata(file_path, label)

                    yield f"{file_path.stem}_{i}", {
                        'signal': {'sig':signal},
                        'sampling_rate': 50000,  
                        'metadata': metadata
                    }

    def _convert_label(self, label_num):
        label_list = [
            'No-fault',
            'Thicker drill steel',
            'A-seal missing. Leakage from high pressure channel to control channel',
            'B-seal missing. Leakage from control channel to return channel',
            'Return accumulator, damaged',
            'Longer drill steel',
            'Damper orifice is larger than usual',
            'Low flow to the damper circuit',
            'Valve damage. A small wear-flat on one of the valve lands',
            'Orifice on control line outlet larger than usual',
            'Charge level in high pressure accumulator is low'
        ]
        return label_list[label_num - 1]
    
    def _extract_metadata(self, file_path, label):
        file_name = file_path.stem
        dataset_name = file_path.parent.name
        sensor_type = file_name.split('_')[1][:-1]  # Assuming file name format is 'data_{sensor_type}_{individual_number}'
        
        if sensor_type == 'pin':
            sensor_type = 'pin'
        elif sensor_type == 'pdmp':
            sensor_type = 'pdin'
        elif sensor_type == 'po':
            sensor_type = 'po'
    
        return {
            'Label': label,
            'SensorType': sensor_type,
            'FileName': file_name,
            'Dataset': dataset_name,
        }