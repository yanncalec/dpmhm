"""The Politecnico di Torino rolling bearing test rig dataset.

Description
===========
Data aquired on the rolling bearing test rig of the Dynamic and Identification Research Group (DIRG), in the Department of Mechanical and Aerospace Engineering at Politecnico di Torino.

The test rig contains two accelerometers at the position `A1` and `A2` and the shaft with its three roller bearings `B1-B2-B3`. Faults of different size are introduced in `B1` on the inner ring or the roller.

Two types of experiments are conducted on the test rig:
- variable speed and load test: with a variation of the fault size, nominal speed of the shaft and nominal load.
- endurance test: with the fault of type `4A` with the nominal speed at 300 Hz and nominal load at 1800 N.

More details can be found in the original publication.

Original Dataset
================
- Type of experiments: initiated faults and run-to-failure.
- Date of acquisition: 2016
- Format: Matlab
- Channels: 6, for two accelerometers in the x-y-z axis
- Split: 'Variable speed and load' test, 'Endurance' test
- Sampling rate: 51200 Hz for `Variable speed and load` test and 102400 Hz for `Endurance` test
- Recording duration: 10 seconds for `Variable speed and load` test and 8 seconds for `Endurance` test
- Label: normal and faulty
- Size: ~ 3Gb unzipped

Download
--------
ftp://ftp.polito.it/people/DIRG_BearingData

Built Dataset
=============
- Split: ['variation', 'endurance'].

Features
--------
- 'signal': {'A1', 'A2'}
- 'label': [Normal, Faulty, Unknown]
- 'sampling_rate': dict, 51200 Hz for Variation test or 102400 Hz for Endurance test
- 'metadata':
    - 'RotatingSpeed': Nominal speed of the shaft in Hz [100, 200, 300, 400, 500] for the variation test and 300 for the endurance test.
    - 'NominalLoadForce': Nominal load in N [0, 1000, 1400, 1800]
    - 'LoadForce': Real load in N
    - 'FaultComponent': {'Roller', 'InnerRing'}
    - 'FaultSize': [450, 250, 150, 0] um
    - 'OriginalSplit': {'Variation', 'Endurance'}
    - 'FileName': original file name,

Notes
=====
- Conversion: load is converted from mV to N using the sensitivity factor 0.499 mV/N
- The endurance test was originally with the fault type 4A but in the processed data we marked its label as "unknown".
"""


import os
from pathlib import Path
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
# import pandas as pd
# from scipy.io import loadmat
# import librosa

# from dpmhm.datasets.preprocessing import AbstractDatasetCompactor, AbstractFeatureTransformer, AbstractPreprocessor
from dpmhm.datasets import _DTYPE, _ENCODING


_CITATION = """
@article{DAGA2019252,
title = {The Politecnico di Torino rolling bearing test rig: Description and analysis of open access data},
journal = {Mechanical Systems and Signal Processing},
volume = {120},
pages = {252-273},
year = {2019},
issn = {0888-3270},
doi = {https://doi.org/10.1016/j.ymssp.2018.10.010},
url = {https://www.sciencedirect.com/science/article/pii/S0888327018306800},
author = {Alessandro Paolo Daga and Alessandro Fasana and Stefano Marchesiello and Luigi Garibaldi},
}
"""

# _SENSOR_LOCATION = ['A1', 'A2']

# _FAULT_LOCATION = ['B1']

_NOMINAL_LOAD = np.asarray([0, 1000, 1400, 1800])

# coding of fault component and diameter (in um)
_FAULT_TYPE_MATCH = {
    '0A': ('None', 0),
    '1A': ('InnerRing', 450),
    '2A': ('InnerRing', 250),
    '3A': ('InnerRing', 150),
    '4A': ('Roller', 450),
    '5A': ('Roller', 250),
    '6A': ('Roller', 150),
}

_DATA_URLS = ['https://sandbox.zenodo.org/record/1183545/files/dirg.zip']


class DIRG(tfds.core.GeneratorBasedBuilder):
    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
        '1.0.0': 'Initial release.',
    }

    def _info(self) -> tfds.core.DatasetInfo:
        # TODO(dirg): Specifies the tfds.core.DatasetInfo object
        return tfds.core.DatasetInfo(
            builder=self,
            description=__doc__,
            features=tfds.features.FeaturesDict({
                # These are the features of your dataset like images, labels ...
                'signal': {
                    'A1': tfds.features.Tensor(shape=(3, None), dtype=_DTYPE, encoding=_ENCODING),
                    'A2': tfds.features.Tensor(shape=(3, None), dtype=_DTYPE, encoding=_ENCODING),
                },

                'sampling_rate': tf.uint32,

                # 'label': tfds.features.ClassLabel(names=['Normal', 'Faulty', 'Unknown']),

                # 'load': tf.float32,  # Real load in N

                'metadata': {
                    'Label': tf.string,
                    'RotatingSpeed': tf.uint32,  # Nominal speed of the shaft in Hz
                    'NominalLoadForce': tf.uint32,  # Nominal load in N
                    'LoadForce': tf.float32,  # Real load in N
                    'FaultComponent': tf.string, # {'Roller', 'InnerRing'}
                    'FaultSize': tf.uint32,  # 450, 250, 150, 0 um
                    'OriginalSplit': tf.string,  # {'Variation', 'Endurance'}
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
                'variation': next(datadir.rglob('VariableSpeedAndLoad')).glob('*.mat'),
                'endurance': next(datadir.rglob('EnduranceTest')).glob('*.mat'),
            }

        if dl_manager._manual_dir.exists():  # prefer to use manually downloaded data
            datadir = Path(dl_manager._manual_dir)
        elif dl_manager._extract_dir.exists(): # automatically download & extracted data
            datadir = Path(dl_manager._extract_dir)
        else:
            raise FileNotFoundError()

        return {sp: self._generate_examples(files) for sp, files in _get_split_dict(datadir).items()}


    def _generate_examples(self, files):
        for fp in files:
            fname = fp.parts[-1]

            try:
                dm = tfds.core.lazy_imports.scipy.io.loadmat(fp)
                # dm = loadmat(fp)
            except Exception as msg:
                raise Exception(f"Error in processing {fp}: {msg}")

            if fname.upper()[0] == 'C':
                ss = fname.upper().split('_')
                # self._fname_parser(fname.name)
                _component, _diameter = _FAULT_TYPE_MATCH[ss[0][1:]]
                _samplingrate = 51200
                _shaftrate = int(ss[1])
                _load_real = float(ss[2])/0.499
                _load = _NOMINAL_LOAD[np.argmin(np.abs(_load_real-_NOMINAL_LOAD))]
                _label = 'Normal' if _component=='None' else 'Faulty'
                _datalabel = 'Variation'
            elif fname.upper()[:3] == 'E4A':
                _component, _diameter = _FAULT_TYPE_MATCH['4A']
                _samplingrate = 102400
                _shaftrate = 300
                _load = 1800
                _load_real = 1800
                _label = 'Unknown'
                _datalabel = 'Endurance'
            else:
                continue

            metadata = {
                'Label': _label,
                'RotatingSpeed': _shaftrate,
                'NominalLoadForce': _load,
                'LoadForce': _load_real,
                'FaultComponent': _component,
                'FaultSize': _diameter,
                'OriginalSplit': _datalabel,
                'FileName': os.path.join(*fp.parts[-2:]),
                'Dataset': 'DIRG',
            }

            Xs = dm[fname[:-4]].T #.astype(_DTYPE),

            yield hash(frozenset(metadata.items())), {
                # 'signal': Xs,  # transpose to the shape (channel, time)
                'signal': {
                    'A1': Xs[:3,:].astype(_DTYPE),
                    'A2': Xs[3:6,:].astype(_DTYPE),
                    # '6': Xs[5,:] #.astype(_DTYPE),
                },
                'sampling_rate': _samplingrate,
                # 'label': _label,
                'metadata': metadata
            }

