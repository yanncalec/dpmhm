"""
Bearing Vibration Data under Time-varying Rotational Speed Conditions.

Description
===========
The data contain vibration signals collected from bearings of different health conditions under time-varying rotational speed conditions.  There are 36 datasets in total. For each dataset, there are two experimental settings: bearing health condition and varying speed condition. The health conditions of the bearing include (i) healthy, (ii) faulty with an inner race defect, and (iii) faulty with an outer race defect. The operating rotational speed conditions are (i) increasing speed, (ii) decreasing speed, (iii) increasing then decreasing speed, and (iv) decreasing then increasing speed. Therefore, there are 12 different cases for the setting. To ensure the authenticity of the data, 3 trials are collected for each experimental setting which results in 36 datasets in total. Each dataset contains two channels: 'Channel_1' is vibration data measured by the accelerometer and 'Channel_2' is the rotational speed data measured by the encoder. All these data are sampled at 200,000Hz and the sampling duration is 10 seconds.

Original Dataset
================
- Type of experiments: initiated faults
- Year of acquisition: 2020
- Format: Matlab
- Size: ~ 763 Mb, unzipped
- Channels: 2, vibration and rotational speed
- Sampling rate: 200 kHz
- Recording duration: 10 seconds
- Label: labelled by fault component and rotational speed

Download
--------
https://data.mendeley.com/datasets/v43hmbwxpm/2

Built Dataset
=============
- Split: ['Healthy', 'InnerRace', 'OuterRace', 'Ball', 'Combination']

Features
--------
- 'signal': 2 channels
- 'metadata':
    - 'RotatingSpeed': ['Increasing', 'Decreasing', 'Increasing-Decreasing',
    'Decreasing-Increasing']
    - 'FaultComponent': ['None', 'InnerRace', 'OuterRace', 'Ball', 'Combination']

Installation
============
Download and unzip all files into a folder `LOCAL_DIR`, from terminal run

```sh
$ tfds build Ottawa --imports dpmhm.datasets.ottawa --manual_dir LOCAL_DIR
```
"""

# import os
from pathlib import Path
# import itertools
# import json
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
# import pandas as pd
# from scipy.io import loadmat

from dpmhm.datasets import _DTYPE, _ENCODING

_CITATION = """
@article{huang_bearing_2019,
title = {Bearing {Vibration} {Data} under {Time}-varying {Rotational} {Speed} {Conditions}},
volume = {2},
url = {https://data.mendeley.com/datasets/v43hmbwxpm/2},
doi = {10.17632/v43hmbwxpm.2},
language = {en},
urldate = {2022-02-24},
author = {Huang, Huan and Baddour, Natalie},
month = feb,
year = {2019},
note = {Publisher: Mendeley Data},
}
"""

_SPLIT_PATH_MATCH = {
    'Healthy': '1 Data collected from a healthy bearing',
    'InnerRace': '2 Data collected from a bearing with inner race fault',
    'OuterRace': '3 Data collected from a bearing with outer race fault',
    'Ball': '4 Data collected from a bearing with ball fault',
    'Combination': '5 Data collected from a bearing with a combination of faults',
}

_PARSER_MATCH_TYPE = {
    'H': 'None',  # Healthy
    'I': 'InnerRace',
    'O': 'OuterRace',
    'B': 'Ball',
    'C': 'Combination',
}

_PARSER_MATCH_SPEED = {
    'A': 'Increasing',
    'B': 'Decreasing',
    'C': 'Increasing-Decreasing',
    'D': 'Decreasing-Increasing',
}

_DATA_URLS = 'https://data.mendeley.com/api/datasets-v2/datasets/v43hmbwxpm/zip/download?version=2'


class Ottawa(tfds.core.GeneratorBasedBuilder):
    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
        '1.0.0': 'Initial release.',
    }

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""
        return tfds.core.DatasetInfo(
            builder=self,
            description=__doc__,
            features=tfds.features.FeaturesDict({
                'signal': {
                    'channels': tfds.features.Tensor(shape=(2,None,), dtype=_DTYPE, encoding=_ENCODING),
                    # 'speed': tfds.features.Tensor(shape=(None,), dtype=_DTYPE, encoding=_ENCODING),
                },

                # 'label': tfds.features.ClassLabel(names=['Normal', 'Faulty']),

                'sampling_rate': tf.uint32,

                'metadata': {
                    'RotatingSpeed': tf.string,
                    'FaultComponent': tf.string,
                    'FileName': tf.string,
                    'Dataset': tf.string,
                }
            }),
            supervised_keys=('signal', 'label'),  # Set to `None` to disable
            homepage='https://data.mendeley.com/datasets/v43hmbwxpm/2',
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        if dl_manager._manual_dir.exists():  # prefer to use manually downloaded data
            datadir = Path(dl_manager._manual_dir)
        else:  # automatically download data
            raise NotImplementedError()
            # datadir = dl_manager.download(_URL)
            _resource = tfds.download.Resource(url=_DATA_URLS, extract_method=tfds.download.ExtractMethod.ZIP)
            datadir = dl_manager.download_and_extract(_resource)

        return {
            sp.lower(): self._generate_examples(datadir/fn) for sp, fn in _SPLIT_PATH_MATCH.items()
            # 'train': self._generate_examples(datadir),
        }

    def _generate_examples(self, path):
        assert path.exists()

        for fp in path.rglob('*.mat'):  #  will raise error if `path` not converted to pathlib.Path object
            try:
                dm = tfds.core.lazy_imports.scipy.io.loadmat(fp)
                # dm = loadmat(fp)
            except Exception as msg:
                raise Exception(f"Error in processing {fp}: {msg}")

            x = np.stack([dm['Channel_1'].squeeze(), dm['Channel_2'].squeeze()]).astype(_DTYPE.as_numpy_dtype)

            metadata = {
                'RotatingSpeed': _PARSER_MATCH_SPEED[fp.name[2]],
                'FaultComponent': _PARSER_MATCH_TYPE[fp.name[0]],
                'FileName': fp.name,
                'Dataset': 'Ottawa'
            }

            yield hash(frozenset(metadata.items())), {
                'signal': {'channels': x},
                # 'label': 'Normal' if metadata['FaultComponent']=='None' else 'Faulty',
                'sampling_rate': 200000,
                'metadata': metadata,
            }

    @staticmethod
    def get_references():
        try:
            with open(Path(__file__).parent / 'Exported Items.bib') as fp:
                return fp.read()
        except:
            pass
