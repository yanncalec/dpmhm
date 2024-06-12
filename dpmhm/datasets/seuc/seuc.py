"""Gearbox dataset from Southeast University, China (SEUC).

Data acquired by Dr. Siyu Shao in the research group of Pf. Ruqiang Yan, Southeaset University, China.

Description
===========
This dataset contains 2 subdatasets, including bearing data and gear data, which are both acquired on Drivetrain Dynamics Simulator (DDS). Two different working conditions are investigated with the rotating speed system load set to be either 20 HZ-0V or 30 HZ-2V.

Homepage
--------
http://mlmechanics.ics.uci.edu/

Original Dataset
================
- Type of experiments: initiated faults
- Size: 1.6 Gb unzipped
- Year of acquisition: 2018
- Format: csv
- Sampling rate: not specified
- Recording duration: 1048560 samples
- Number of channels: 8
- Splits: `bearingset` and `gearset`
- Label: normal, faulty

Each file contains 8 channels of signals representing:

- channel 1: motor vibration,
- channels 2,3,4: vibration of planetary gearbox in the x-y-z directions
- channel 5: motor torque,
- channels 6,7,8: vibration of parallel gearbox in the x-y-z directions

Bearing dataset
---------------
- Ball
- Inner ring
- Outer ring
- Combination of inner ring and outer ring
- Health woring state

Gear dataset
------------
- Chipped: Crack occurs in the gear feet
- Missing: Missing one of feet in the gear
- Root: Crack occurs in the root of gear feet
- Surface: Wear occurs in the surface of gear
- Health working state

URLs
----
https://github.com/cathysiyu/Mechanical-datasets

Notes
=====
- The original splits `bearingset` and `gearset` are renamed to `bearing` and `gearbox` respectively.
- The original dataset contains also a subset of CWRU, which is excluded from the current package.

References
==========
- Highly Accurate Machine Fault Diagnosis Using Deep Transfer Learning, Shao et al., 2019.

Installation
============
Download and unzip all files into a folder `LOCAL_DIR`, from terminal run

```sh
$ tfds build SEUC --imports dpmhm.datasets.seuc --manual_dir LOCAL_DIR
```
"""

# import os
# import pathlib
# import itertools
# import json
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import pandas as pd
from dpmhm.datasets import _DTYPE, _ENCODING
import csv

_CITATION = """
@ARTICLE{8432110,
author={Shao, Siyu and McAleer, Stephen and Yan, Ruqiang and Baldi, Pierre},
journal={IEEE Transactions on Industrial Informatics},
title={Highly Accurate Machine Fault Diagnosis Using Deep Transfer Learning},
year={2019},
volume={15},
number={4},
pages={2446-2455},
doi={10.1109/TII.2018.2864759}}
"""

_DATA_URLs = 'https://github.com/cathysiyu/Mechanical-datasets/archive/refs/heads/master.zip'

# Components of fault
_FAULT_GEARBOX = ['Chipped', 'Missing', 'Root', 'Surface']

_FAULT_BEARING = ['Ball', 'Inner', 'Outer', 'Combination']

_FAULT_COMPONENT = ['None'] + _FAULT_BEARING + _FAULT_GEARBOX

# Dictionary for name conversion
_FAULT_MATCH = {
    'comb': 'Combination',
    'health': 'None',
    'miss': 'Missing',
}


class SEUC(tfds.core.GeneratorBasedBuilder):
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
                    'motor': tfds.features.Tensor(shape=(None,), dtype=_DTYPE, encoding=_ENCODING),
                    'planetary_x': tfds.features.Tensor(shape=(None,), dtype=_DTYPE, encoding=_ENCODING),
                    'planetary_y': tfds.features.Tensor(shape=(None,), dtype=_DTYPE, encoding=_ENCODING),
                    'planetary_z': tfds.features.Tensor(shape=(None,), dtype=_DTYPE, encoding=_ENCODING),
                    'torque': tfds.features.Tensor(shape=(None,), dtype=_DTYPE, encoding=_ENCODING),
                    'parallel_x': tfds.features.Tensor(shape=(None,), dtype=_DTYPE, encoding=_ENCODING),
                    'parallel_y': tfds.features.Tensor(shape=(None,), dtype=_DTYPE, encoding=_ENCODING),
                    'parallel_z': tfds.features.Tensor(shape=(None,), dtype=_DTYPE, encoding=_ENCODING),
                },
                'sampling_rate': tf.uint32,
                'metadata': {
                    'LoadForce':  tf.string,
                    'FaultKind': tf.string,
                    'FaultSize': tf.string,
                    'OriginalSplit': tf.string,  # the original split
                    'FileName': tf.string,
                    'Dataset': tf.string,
                },
            }),

            supervised_keys=None,
            homepage='https://github.com/cathysiyu/Mechanical-datasets',
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        if dl_manager._manual_dir.exists():  # prefer to use manually downloaded data
            datadir = dl_manager._manual_dir / 'gearbox'
        else:  # automatically download data
            datadir = list(dl_manager.download_and_extract(_DATA_URLs).iterdir())[0] / 'gearbox'
        return {
            # Use the original splits
            'gearbox': self._generate_examples(datadir/'gearset'),
            'bearing': self._generate_examples(datadir/'bearingset'),
            # 'train': self._generate_examples(datadir)  # this will rewrite on precedent splits
        }

    def _generate_examples(self, path):
        for fp in path.glob('*.csv'):
            filename = fp.stem  # get filename without extension
            # Assume the filename structure to extract the required information
            parts = filename.split('_')
            if len(parts) == 3:
                fault, size, load = parts
            else:
                continue  # Skip files that don't match the expected format
            
            if fp.parent.name=='gearset':
                if fault == 'Chipped':
                    fault='Chipped tooth'
                if fault == 'Missing':
                    fault='Missing tooth'
                if fault == 'Root':
                    fault='Root fault'
                if fault == 'Surface':
                    fault='Surface fault'
                if fault == 'Health':
                    fault='Health working state'
            else:
                if fault=='comb':
                    fault='Combination fault on both inner ring and outer ring'
                if fault=='health':
                    fault='Health woring state'
                if fault=='ball':
                    fault='Ball fault'
                if fault=='inner':
                    fault='Inner ring fault'
                if fault=='outer':
                    fault='Outer ring fault'
            
            # Read the CSV file
            with open(fp, newline='') as csvfile:
                reader = csv.reader(csvfile)
                for _ in range(16):
                   next(reader)
                signal = [[] for _ in range(8)]
                for i, row in enumerate(reader):
                    for j,val in enumerate(row[0].split('\t')[:-1]):
                        signal[j].append(float(val))
            
            metadata = {
                'LoadForce': load,
                'FaultKind': fault,
                'FaultSize': '0.'+ size,
                'OriginalSplit': 'Gearbox' if fp.parent.name == 'gearset' else 'Bearing',
                'FileName': filename,
                'Dataset': 'SEUC',
            }

            yield hash(frozenset(metadata.items())), {
                'signal': {
                    'motor': signal[0],
                    'planetary_x': signal[1], 
                    'planetary_y': signal[2], 
                    'planetary_z': signal[3], 
                    'torque': signal[4],
                    'parallel_x': signal[5],
                    'parallel_y': signal[6],
                    'parallel_z': signal[7],
                },
                'sampling_rate': 10000,  # Original sampling rate is not specified
                'metadata': metadata
            }
