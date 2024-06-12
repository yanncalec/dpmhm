"""
PHM Asia Pacific 2021 Data Challenge.

Description
===========
Data has been acquired under a variety of environments, not only fault-free condition, but also seeded faults under controlled conditions with the support of domain expertise. Hence, the dataset comprises signals in five categories: Normal, Unbalance, Belt-looseness, Belt-looseness (High), and Bearing fault. As two sensors equipped on the compressor monitor the signals, the dataset contains two signals from each channel.

System Information
------------------
- Type of Equipment: Oil-injection screw compressor
- Motor: 15kW
- Axis rotating speed of Motor: 3,600 rpm
- Axis rotating speed of Screw: 7,200 rpm

Homepage
--------
http://phmap.org/data-challenge/

Original Dataset
================
- Type of experiments: initiated faults
- Format: CSV
- Size: ~ 10.8 Gb unzipped
- Channels: 2 vibration channels, from motor and screw
- Sampling rate: 10544 Hz
- Recording duration: not fixed
- Split: not specified
- Operating conditions: fixed rotating speed at 3600 rpm for the motor and 7200 rpm for the screw
- Faults: 5 Classes
    - Normal: Fault-free operating condition
    - Unbalance: Unbalance between centers of mass and axis
    - Belt-Looseness: Looseness of V-belt connecting between motor pully and screw pully
    - Belt-Looseness High: High Looseness of V-belt
    - Bearing fault: Removing grease of Ball Bearing on Motor, which induces its wear-out

Download
--------
https://drive.google.com/drive/folders/1Zcth6UhPfP3vM8YadHhCSmK6v4aEyOA7

Notes
=====
- Two test sets (for the task of classification and regression) were also provided along with the original datset, but cannot be downloaded. See:
    https://www.kaggle.com/c/phmap21-regression-task/data
- Extract all files and check that all filenames follows the pattern `train_xxx_Yyy.csv`. In fact, some files may be automatically renamed during the download.
"""

# import os
from pathlib import Path
# import itertools
# import json
# import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import pandas as pd
# from scipy.io import loadmat
from dpmhm.datasets import _DTYPE, _ENCODING
import csv


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

_DATA_URLS = ['https://sandbox.zenodo.org/record/1184362/files/phmap.zip']


class Phmap2021(tfds.core.GeneratorBasedBuilder):
    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
        '1.0.0': 'Initial release.',
    }

    def _info(self) -> tfds.core.DatasetInfo:
        return tfds.core.DatasetInfo(
            builder=self,
            description=__doc__,
            features=tfds.features.FeaturesDict({
                'signal': {
                    'sig1': tfds.features.Tensor(shape=(None,), dtype=_DTYPE, encoding=_ENCODING),
                    'sig2': tfds.features.Tensor(shape=(None,), dtype=_DTYPE, encoding=_ENCODING),
                },
                'sampling_rate': tf.uint32,
                'metadata': {
                    'Label': tf.string,
                    'FileName': tf.string,
                    'Dataset': tf.string,
                },
            }),
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
        files_dict = {
            'train': [datadir / filename for filenames in _SPLIT_PATH_MATCH.values() for filename in filenames]
        }
        return {split: self._generate_examples(files) for split, files in files_dict.items()}

    def _generate_examples(self, files):
        for file_path in files:
            with open(file_path, newline='') as csvfile:
                reader = csv.reader(csvfile)
                next(reader)
                signal_col1 = []
                signal_col2 = []
                for i, row in enumerate(reader):
                    signal_col1.extend([float(val) for val in row[1].split(',')])
                    signal_col2.extend([float(val) for val in row[2].split(',')])
                label = file_path.stem.split('_')[2]
                metadata = {
                    'Label': label,
                    'FileName': file_path.name,
                    'Dataset': 'phm2021',
                }
                yield f"{file_path.stem}", {
                    'signal': {'sig1': signal_col1, 'sig2': signal_col2},
                    'sampling_rate': 50000, 
                    'metadata': metadata
                }

    @staticmethod
    def get_references():
        try:
            with open(Path(__file__).parent / 'Exported Items.bib') as fp:
                return fp.read()
        except:
            pass
