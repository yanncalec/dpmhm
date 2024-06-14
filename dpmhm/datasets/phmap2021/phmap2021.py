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
    - Belt-Looseness: Looseness of V‐belt connecting between motor pully and screw pully
    - Belt-Looseness High: High Looseness of V-belt
    - Bearing fault: Removing grease of Ball Bearing on Motor, which induces its wear-out

Download
--------
https://drive.google.com/drive/folders/1Zcth6UhPfP3vM8YadHhCSmK6v4aEyOA7

Notes
=====
- Two test sets (for the task of classification and regression) were also provided along with the original datset, but cannot be downloaded. See:
    https://www.kaggle.com/c/phmap21-regression-task/data
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

_DATA_URLS = [
    # 'https://sandbox.zenodo.org/record/1184362/files/phmap.zip'
    'https://zenodo.org/records/11546285/files/phmap.zip?download=1'
    ]


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
                    'vibration': tfds.features.Tensor(shape=(2,None), dtype=_DTYPE, encoding=_ENCODING),
                },

                # 'label': tfds.features.ClassLabel(names=list(_SPLIT_PATH_MATCH.keys())),

                'sampling_rate': tf.uint32,

                'metadata': {
                    'Label': tf.string,
                    # 'OriginalSplit': tf.string,  # Original split that the signal belongs to
                    'FileName': tf.string,  # Original filename with path in the dataset
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

        return {
            # sp.lower(): self._generate_examples(datadir/fn) for sp, fn in _SPLIT_PATH_MATCH.items()
            'train': self._generate_examples(datadir),
        }

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        def _get_split_dict(datadir):
            # This doesn't work:
            # return {sp: (datadir/fn).rglob('*.csv') for sp, fn in _SPLIT_PATH_MATCH.items()}
            return {
                'train': datadir.rglob('*.csv'),
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

        return {sp: self._generate_examples(files) for sp, files in _get_split_dict(datadir).items()}

    # def _generate_examples(self, path):
    #     for sp, fnames in _SPLIT_PATH_MATCH.items():
    def _generate_examples(self, files):
        for fp in files:
            # for fn in fnames:
            #     fp = path / fn

            _signal = pd.read_csv(fp, index_col=0).T.values.astype(_DTYPE)
            sp = fp.stem.split('_')[-1].capitalize()

            metadata = {
                'Label': sp,
                # 'OriginalSplit': sp,
                'FileName': fp.name,
                'Dataset': 'PHMAP2021',
            }

            yield hash(frozenset(metadata.items())), {
                'signal': {'vibration': _signal},
                # 'label': sp,
                'sampling_rate': 10544,
                'metadata': metadata
            }

    # @staticmethod
    # def get_references():
    #     try:
    #         with open(Path(__file__).parent / 'Exported Items.bib') as fp:
    #             return fp.read()
    #     except:
    #         pass
