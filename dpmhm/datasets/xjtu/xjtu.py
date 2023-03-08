"""XJTU-SY Bearing Datasets.

Description
===========
XJTU-SY bearing datasets are provided by the Institute of Design Science and Basic Component at Xi’an Jiaotong University (XJTU), Shaanxi, P.R. China (http://gr.xjtu.edu.cn/web/yaguolei) and the Changxing Sumyoung Technology Co.,
Ltd. (SY), Zhejiang, P.R. China (https://www.sumyoungtech.com.cn). The datasets contain complete run-to-failure data of 15 rolling element bearings that were acquired by conducting many accelerated degradation experiments. These datasets are publicly available and anyone can use them to validate prognostics algorithms of rolling element bearings. Publications making use of the XJTU-SY bearing datasets are requested to cite the following paper.

Citation
--------
Biao Wang, Yaguo Lei, Naipeng Li, Ningbo Li, “A Hybrid Prognostics Approach for Estimating Remaining Useful Life of Rolling Element Bearings”, IEEE Transactions on Reliability, pp. 1-12, 2018. DOI: 10.1109/TR.2018.2882682.

Homepage
--------
https://biaowang.tech/xjtu-sy-bearing-datasets/

Original Dataset
================
- Type of experiments: run-to-failure, with identification of fault element.
- Size: ~ 11.4 Gb unzipped
- Sampling rate: 25.6 kHz
- Duration: 1.28 seconds
- Signal length: 32768
- Sampling period: 1 minute
- Channels: 2, horizontal and vertical acceleration
- Original split: 3 operating conditions (rotating speed and radial force) on 5 bearings
    - 1) 2100 rpm (35 Hz) and 12 kN;
    - 2) 2250 rpm (37.5 Hz) and 11 kN;
    - 3) 2400 rpm (40 Hz) and 10 kN.

Download
--------
https://www.dropbox.com/sh/qka3b73wuvn5l7a/AADdQk8ZCsNkzjz11JewU7cBa/Data?dl=0&subfolder_nav_tracking=1

or via any other links listed on the author's [website](https://biaowang.tech/xjtu-sy-bearing-datasets/).


Built Dataset
=============
- Split: ['condition1', 'condition2', 'condition3']

Features
--------
- 'signal':
    - 'vibration': horizontal & vertical vibration signals
- 'sampling_rate: 25600 Hz
- 'metadata':
    - 'OperatingCondition': Operating condition 1,2,3
    - 'BearingID': ID of the bearing
    - 'FaultComponent': ['Inner', 'Ball', 'Cage', 'Outer', 'Inner+Outer', 'Inner+Ball+Cage+Outer']
    - 'Lifetime': in minutes

Notes
=====
- The original dataset contains 6 rar files which needs to be extracted all together (extract the first one `XJTU-SY_Bearing_Datasets.part01.rar` will automatically extract all other files).

Installation
============
Download and unzip all files into a folder `LOCAL_DIR`, from terminal run

```sh
$ tfds build XJTU --imports dpmhm.datasets.xjtu --manual_dir LOCAL_DIR
```
"""

import os
from pathlib import Path
# import itertools
# import json
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import pandas as pd
# from scipy.io import loadmat
from dpmhm.datasets import _DTYPE, _ENCODING

_METAINFO = pd.read_csv(Path(__file__).parent/'metainfo.csv')

_CITATION = """
@article{wang2018hybrid,
  title={A hybrid prognostics approach for estimating remaining useful life of rolling element bearings},
  author={Wang, Biao and Lei, Yaguo and Li, Naipeng and Li, Ningbo},
  journal={IEEE Transactions on Reliability},
  volume={69},
  number={1},
  pages={401--412},
  year={2018},
  publisher={IEEE}
}
"""

_SPLIT_PATH_MATCH = {
    'condition1': '35Hz12kN',
    'condition2': '37.5Hz11kN',
    'condition3': '40Hz10kN',
}

# _PARSER_MATCH = {
#   # 'file name pattern':
# }

# _DATA_URLS = ''

class XJTU(tfds.core.GeneratorBasedBuilder):
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
            # Number of channels is fixed and length is fixed
                'signal': {
                    'vibration': tfds.features.Tensor(shape=(2,None), dtype=_DTYPE, encoding=_ENCODING),
                },

                'sampling_rate': tf.uint32,

                # 'label': tfds.features.ClassLabel(names=['Unknown']),

                'metadata': {
                    'OperatingCondition': tf.int32,  # Operating condition
                    'BearingID': tf.int32,  # ID of the bearing
                    'FaultComponent': tf.string, # Component of the fault, e.g. {'Roller', 'InnerRing'}
                    'Lifetime': tf.float32,
                    'FileName': tf.string,  # Original filename with path
                    'Dataset': tf.string,
                },
            }),
            supervised_keys=None,
            homepage='https://biaowang.tech/xjtu-sy-bearing-datasets/',
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        if dl_manager._manual_dir.exists():  # prefer to use manually downloaded data
            datadir = Path(dl_manager._manual_dir)
        else:  # automatically download data
            raise NotImplementedError()

        return {
            sp: self._generate_examples(datadir/fn) for sp, fn in _SPLIT_PATH_MATCH.items()
            # 'train': self._generate_examples_all(datadir),
        }

    def _generate_examples(self, path):
        for fp in path.rglob('*.csv'):
            # parse the filename
            _condition, _bearing = fp.parent.name[7:].split('_')
            # print(int(_condition), int(_bearing))

            metadata = _METAINFO.loc[(_METAINFO['OperatingCondition']==int(_condition)) & (_METAINFO['BearingID']==int(_bearing))].iloc[0].to_dict()
            metadata['FileName'] = os.path.join(*fp.parts[-3:])
            metadata['Dataset'] = 'XJTU'

            x = pd.read_csv(fp).T.values

            yield hash(frozenset(metadata.items())), {
                'signal': {'vibration': x.astype(_DTYPE.as_numpy_dtype)},
                # 'label': 'Unknown',
                'sampling_rate': 25600,
                'metadata': metadata
            }

    @staticmethod
    def get_references():
        try:
            with open(Path(__file__).parent / 'Exported Items.bib') as fp:
                return fp.read()
        except:
            pass
