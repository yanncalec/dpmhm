"""
Paderborn University Bearing DataCenter.

Experimental bearing data sets for condition monitoring (CM) based on vibration and motor current signals, provided by the Chair of Design and Drive Technology, Paderborn University, Germany.

Description
===========
The main characteristic of the data set are:

- Synchronously measured motor currents and vibration signals with high resolution and sampling rate of 26 damaged bearing states and 6 undamaged (healthy) states for reference.
- Supportive measurement of speed, torque, radial load, and temperature.
- Four different operating conditions (see operating conditions).
- 20 measurements of 4 seconds each for each setting, saved as a MatLab file with a name consisting of the code of the operating condition and the four-digit bearing code (e.g. N15_M07_F10_KA01_1.mat).
- Systematic description of the bearing damage by uniform fact sheets and a measuring log, which can be downloaded with the data.

In total, experiments with 32 different bearing damages in ball bearings of type 6203 were performed:

- Undamaged (healthy) bearings (6x), see Table 6 in (pdf).
- Artificially damaged bearings (12x), see Table 4 in (pdf).
- Bearings with real damages caused by accelerated lifetime tests, (14x) see Table 5 in (pdf)

Homepage
--------
https://mb.uni-paderborn.de/kat/forschung/datacenter/bearing-datacenter

Original Dataset
================
- Type of experiments: initiated faults & run-to-failure
- Format: Matlab
- Size: ~ 20.8 Gb, unzipped
- Year of acquisition: 2015
- Channels & Sampling rate:
    - motor current sampling frequency: 64 kHz, 2 channels ['phase_current_1', 'phase_current_2']
    - vibration sampling frequency: 64 kHz, 1 channel ['vibration_1']
    - sampling frequency of mechanic parameters (load force, load torque, speed): 4 kHz, 3 channels ['force', 'torque', 'speed']
    - temperature sampling frequency: 1 Hz, 1 channel ['temp_2_bearing_module']
- Recording duration: 4 seconds
- Load and Rotational Speed:
    - For undamaged and artificially damaged bearings test: not used
    - For lifetime test: 3800 N and 2900 rpm
- Split: Healthy + Artificially damaged, Accelerated lifetime test

Download
--------
https://mb.uni-paderborn.de/en/kat/main-research/datacenter/bearing-datacenter/data-sets-and-download

Built Dataset
=============
- Split: ['artificial', 'lifetime'] for initiated faults (merged with 'healthy') & run-to-failure experiments respectively.

Features
--------
- 'signal':
    - 'vibration': 1 channel
    - 'current': 2 channels ['phase_current_1', 'phase_current_2']
    - 'mechanic': 3 channels ['force', 'speed', 'torque']
    - 'temperature': 1 channel
- 'sampling_rate':
    - 'vibration': 64 kHz
    - 'current': 64 kHz
    - 'mechanic': 4 kHz
    - 'temperature': 1 Hz
- 'metadata':
    - 'FaultComponent': 'None', 'Inner Ring', 'Outer Ring', 'Inner Ring+Outer Ring'
    - 'FaultExtend': 0,1,2,3,
    - 'DamageMethod': 'Healthy', 'Aritificial', 'Lifetime'
    - 'FaultType': how fault is introduced
    - 'FileName': original filename.

References
==========
- Lessmeier, Christian; Kimotho, J. Kimotho; Zimmer, Detmar; Sextro, Walter:
A Benchmark Data Set for Data-Driven Classification, European Conference of the Prognostics and Health Management Society (PHM Society), Bilbao (Spain), 05.-08.07.2016.  (pdf)
- ISO 15243:2017 Rolling bearings — Damage and failures — Terms, characteristics and causes
https://www.iso.org/standard/59619.html

Notes
=====
- The file `KA08/N15_M01_F10_KA08_2.mat` cannot be loaded by `scipy.io.loadmat`.
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

# import patoolib  # extraction of rar file
# patoolib.extract_archive(fp, outdir=datadir, interactive=False)

import logging
Logger = logging.getLogger(__name__)

from dpmhm.datasets import _DTYPE, _ENCODING


_CITATION = """
Christian Lessmeier et al., KAt-DataCenter: mb.uni-paderborn.de/kat/datacenter, Chair of Design and Drive Technology, Paderborn University.
"""

_METAINFO = pd.read_csv(Path(__file__).parent / 'metainfo.csv', index_col=0)  # use 'Bearing Code' as index

# _DATA_URLS = ('http://groups.uni-paderborn.de/kat/BearingDataCenter/' + _METAINFO.index+'.rar').tolist()

_DATA_URLS = [
    'https://sandbox.zenodo.org/record/1184342/files/paderborn.zip'
]

# _SPLIT_PATH_MATCH = {k: _METAINFO.loc[_METAINFO['DamageMethod']==k].index.tolist() for k in ['Healthy', 'Artificial', 'Lifetime']}

_SPLIT_PATH_MATCH = {
  'healthy': ['K001', 'K002', 'K003', 'K004', 'K005', 'K006'],
  'artificial': ['KA01', 'KA03', 'KA05', 'KA06', 'KA07', 'KA08', 'KA09', 'KI01', 'KI03', 'KI05', 'KI07', 'KI08'],
#   'artificial': ['K001', 'K002', 'K003', 'K004', 'K005', 'K006'] + ['KA01', 'KA03', 'KA05', 'KA06', 'KA07', 'KA08', 'KA09', 'KI01', 'KI03', 'KI05', 'KI07', 'KI08'],  # Merge healthy and artificial set
  'lifetime': ['KA04', 'KA15', 'KA16', 'KA22', 'KA30', 'KB23', 'KB24', 'KB27', 'KI04', 'KI14', 'KI16', 'KI17', 'KI18', 'KI21']
}


class Paderborn(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for Paderborn dataset."""

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
                'vibration': tfds.features.Tensor(shape=(None,), dtype=_DTYPE, encoding=_ENCODING),
                'current': tfds.features.Tensor(shape=(2,None,), dtype=_DTYPE, encoding=_ENCODING),
                'mechanic': tfds.features.Tensor(shape=(3,None,), dtype=_DTYPE, encoding=_ENCODING),
                'temperature': tfds.features.Tensor(shape=(None,), dtype=_DTYPE, encoding=_ENCODING),
            },

            'sampling_rate': {
                'vibration': tf.uint32,  # 64 kHz
                'current': tf.uint32,  # 64 kHz
                'mechanic': tf.uint32,  # 4 kHz
                'temperature': tf.uint32,  # 1 Hz
            },

            # 'signal': {
            #     'force': tfds.features.Tensor(shape=(None,), dtype=_DTYPE, encoding=_ENCODING),
            #     'phase_current_1': tfds.features.Tensor(shape=(None,), dtype=_DTYPE, encoding=_ENCODING),
            #     'phase_current_2': tfds.features.Tensor(shape=(None,), dtype=_DTYPE, encoding=_ENCODING),
            #     'speed': tfds.features.Tensor(shape=(None,), dtype=_DTYPE, encoding=_ENCODING),
            #     'temp_2_bearing_module': tfds.features.Tensor(shape=(None,), dtype=_DTYPE, encoding=_ENCODING),
            #     'torque': tfds.features.Tensor(shape=(None,), dtype=_DTYPE, encoding=_ENCODING),
            #     'vibration_1': tfds.features.Tensor(shape=(None,), dtype=_DTYPE, encoding=_ENCODING),
            # },

            # 'sampling_rate': {
            #     'force': tf.uint32,
            #     'phase_current_1': tf.uint32,
            #     'phase_current_2': tf.uint32,
            #     'speed': tf.uint32,
            #     'temp_2_bearing_module': tf.uint32,
            #     'torque': tf.uint32,
            #     'vibration_1': tf.uint32,
            # },

            # 'label': tfds.features.ClassLabel(names=['Healthy', 'Faulty', 'Unknown']),

            'metadata': {
                'FaultComponent': tf.string, # Component of the fault
                'FaultExtend': tf.int32,  # Extend of the fault
                'DamageMethod': tf.string,  # Method of damage
                'FaultType': tf.string,   # Type of damage
                # 'Condition':  # Operating conditions (RPM, Torque, Force) see Table 6.
                'FileName': tf.string,  # Original filename (with path)
                'Dataset': tf.string,
            }
            }),
            supervised_keys=None,
            homepage='https://mb.uni-paderborn.de/kat/forschung/datacenter/bearing-datacenter',
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        def _get_split_dict(datadir):
            d = {}
            for sp, fl in _SPLIT_PATH_MATCH.items():
                files_dict = {}
                for fn in fl:
                    files_dict[fn] = list(next(datadir.rglob(fn)).glob('*.mat'))
                files = []
                for k,vals in files_dict.items():
                    files += [(v,k) for v in vals]
                d[sp] = files
            return d

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

    def _generate_examples(self, files):
        for fp, key in files:
            metadata = _METAINFO.loc[key][['FaultComponent', 'FaultExtend', 'DamageMethod', 'FaultType']].to_dict()
            # print(fp)
            metadata['FileName'] = fp.name
            metadata['Dataset'] = 'Paderborn'

            # if metadata['DamageMethod']=='Healthy':
            #     _label = 'Healthy'
            # elif metadata['DamageMethod']=='Artificial':
            #     _label = 'Faulty'
            # else:  # Lifetime
            #     _label = 'Unknown'

            # dm=tfds.core.lazy_imports.scipy.io.loadmat(fp)[fp.name.split('.mat')[0]][0]
            # dm = loadmat(fp)[fp.name.split('.mat')[0]][0]

            # xd = {}
            # for dd in dm[0][2][0]:
            #     xd[dd[0][0]] = dd[2].squeeze()

            try:
                dm = tfds.core.lazy_imports.scipy.io.loadmat(fp,mat_dtype=True, squeeze_me=True,)
            except Exception as msg:
                # The file KA08/N15_M01_F10_KA08_2.mat cannot be processed.
                Logger.error(f'Error in processing {fp}: {msg}')
                continue

            xd = {}
            for dd in np.atleast_1d(dm[fp.name.split('.mat')[0]])[0][2]:
                xd[dd[0]] = dd[2].astype(_DTYPE)

            _signal = {
                'vibration': xd['vibration_1'],
                'current': np.stack([xd['phase_current_1'], xd['phase_current_2']]),
                'mechanic': np.stack([xd['force'], xd['speed'], xd['torque']]),
                'temperature': xd['temp_2_bearing_module'],
            }

            yield hash(frozenset(metadata.items())), {
                'signal': _signal,
                'sampling_rate': {'vibration': 64000, 'current': 64000, 'mechanic': 4000, 'temperature': 1},
                # 'label': _label,
                'metadata': metadata
            }
