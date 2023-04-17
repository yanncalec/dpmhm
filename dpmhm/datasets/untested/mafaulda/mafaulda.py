"""
Mafaulda Machinery Fault Database.

This database is composed of 1951 multivariate time-series acquired by sensors on a SpectraQuest's Machinery Fault Simulator (MFS) Alignment-Balance-Vibration (ABVT). The 1951 comprises six different simulated states: normal function, imbalance fault, horizontal and vertical misalignment faults and, inner and outer bearing faults.

For more information please contact Felipe M. L. Ribeiro (felipe.ribeiro@smt.ufrj.br).

Description
===========

Fault Types
-----------
- Normal Sequences: There are 49 sequences without any fault, each with a fixed rotation speed within the range from 737 rpm to 3686 rpm with steps of approximately 60 rpm.
- Imbalance Faults: Simulated with load values within the range from 6 g to 35 g. For each load value below 30 g, the rotation frequency assumed in the same 49 values employed in the normal operation case. For loads equal to or above 30 g, however, the resulting vibration makes impracticable for the system to achieve rotation frequencies above 3300 rpm, limiting the number of distinct rotation frequencies. The table below presents the number of sequences per weight.
- Horizontal Parallel Misalignment: This type of fault was induced into the MFS by shifting the motor shaft horizontally of 0.5 mm, 1.0 mm, 1.5 mm, and 2.0 mm. Using the same range for the rotation frequency as in the normal operation for each horizontal shift, the table below presents the number of sequences per degree of misalignment.
- Vertical Parallel Misalignment: This type of fault was induced into the MFS by shifting the motor shaft horizontally of 0.51 mm, 0.63 mm, 1.27 mm, 1.40 mm, 17.8 mm and 1.90 mm. Using the same range for the rotation frequency as in the normal operation for each vertical shift, the table below presents the number of sequences per degree of misalignment.
- Bearing Faults: This type of fault was induced into one of the three components (outer track, rolling elements, and inner track) and at either the underhang position (between the rotor and the motor) or the overhang position (between the bearing and the motor). Bearing faults are practically imperceptible when there is no imbalance. So, three masses of 6 g, 20 g, and 35 g were added to induce a detectable effect, with different rotation frequencies as before.

Data Acquisition System
-----------------------
- Three Industrial IMI Sensors, Model 601A01 accelerometers on the radial, axial and tangencial directions:
    - Sensibility: (±20%) 100 mV per g (10.2 mV per m/s2);
    - Frequency range: (±3 dB) 16-600000 CPM (0.27-10.000 Hz);
    - Measurement range: ±50 g (±490 m/s2).
- One IMI Sensors triaxial accelerometer, Model 604B31, returning data over the radial, axial and tangencial directions:
    - Sensibility: (±20%) 100 mV per g (10.2 mV per m/s2);
    - Frequency range: (±3 dB) 30-300000 CPM (0.5-5.000 Hz);
    - Measurement range: ±50 g (±490 m/s2)
- Monarch Instrument MT-190 analog tachometer
- Shure SM81 microphone with frequency range of 20-20.000 Hz
- Two National Instruments NI 9234 4 channel analog acquisition modules, with sample rate of 51.2 kHz

Homepage
--------
http://www02.smt.ufrj.br/~offshore/mfs/page_01.html

Original Dataset
================
- Type of experiments: initiated faults
- Sampling rate: 50 kHz
- Year of acquisition: 2017
- Format: CSV files
- Recording duration: 5 seconds, 250000 samples
- Size: ~ 31 Gb, unzipped

The database is composed by several CSV (Comma-Separated Values) files, each one with 8 columns, one column for each sensor, according to:
- column 1: tachometer signal that allows to estimate rotation frequency;
- columns 2 to 4: underhang bearing accelerometer (axial, radiale tangential direction);
- columns 5 to 7: overhang bearing accelerometer (axial, radiale tangential direction);
- column 8: microphone.

Built Dataset
=============
Split: ['normal', 'horizontal-misalignment', 'vertical-misalignment', 'imbalance', 'underhang', 'overhang']

Features
--------
- 'signal':
    - 'tachometer': 1 channel
    - 'underhang': 3 channels vibration
    - 'overhang': 3 channels vibration
    - 'microphone': 1 channel vibration
- 'metadata':
    - 'NominalRPM': rpm of the record
    - 'FaultName': ['normal', 'imbalance', 'horizontal-misalignment', 'vertical-misalignment', 'overhang', 'underhang']
    - 'FaultSize': Description of fault. Bearing fault component ['none', 'outer_race', 'cage_fault', 'ball_fault'] + load force [0g, 6g, 10g, 15g, 20g, 25g, 30g, 35g] + misalignment [0.5mm, 1.0mm, 1.5mm, 2.0mm] + [0.51mm, 0.63mm, 1.27mm, 1.40mm, 1.78mm, 1.90mm]

Notes
=====
For bearing faults the original data used the lables [`ball_fault`, `cage_fault`, `outer_race`] respectively for the components ['rolling elements', 'inner track', 'outer track']. Some of the labels seem to be misspecified when comparing to the list from the original description showing the number of files of each class.


Installation
============
Download and unzip all files into a folder `LOCAL_DIR`, from terminal run

```sh
$ tfds build Mafaulda --imports dpmhm.datasets.mafaulda --manual_dir LOCAL_DIR
```
"""

# import os
from pathlib import Path
# import itertools
# import json
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import pandas as pd

from dpmhm.datasets import _DTYPE, _ENCODING

_CITATION = """
@misc{ribeiro_mafaulda_2014,
title = {{MAFAULDA}: {Machinery} {Fault} {Database}},
url = {http://www02.smt.ufrj.br/~offshore/mfs/},
author = {Ribeiro, Felipe M.L.},
year = {2014},
}
"""

_FAULT_TYPE = ['normal', 'horizontal-misalignment', 'vertical-misalignment', 'imbalance', 'underhang', 'overhang']

_DATA_URLS = ['http://www02.smt.ufrj.br/~offshore/mfs/database/mafaulda/'+f+'.zip' for f in _FAULT_TYPE]  # The original '.tgz' files provided cannot be read by tfds


class Mafaulda(tfds.core.GeneratorBasedBuilder):

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
                    'tachometer': tfds.features.Tensor(shape=(None,), dtype=_DTYPE, encoding=_ENCODING),
                    'underhang': tfds.features.Tensor(shape=(3,None,), dtype=_DTYPE, encoding=_ENCODING),
                    'overhang': tfds.features.Tensor(shape=(3,None,), dtype=_DTYPE, encoding=_ENCODING),
                    'microphone': tfds.features.Tensor(shape=(None,), dtype=_DTYPE, encoding=_ENCODING),
                },

                # 'label': tfds.features.ClassLabel(names=['Normal', 'Faulty']),

                'sampling_rate': tf.uint32, # 50 kHz

                'metadata': {
                    # 'Label': tf.string,
                    'NominalRPM': tf.float32,
                    'FaultName': tf.string,  # ['normal', 'imbalance', 'horizontal-misalignment', 'vertical-misalignment', 'overhang', 'underhang']
                    'FaultSize': tf.string,  # Description of fault. Bearing fault component ['none', 'outer_race', 'cage_fault', 'ball_fault'] + load force [0g, 6g, 10g, 15g, 20g, 25g, 30g, 35g] + misalignment [0.5mm, 1.0mm, 1.5mm, 2.0mm] + [0.51mm, 0.63mm, 1.27mm, 1.40mm, 1.78mm, 1.90mm]
                    'FileName': tf.string,
                    'Dataset': tf.string,
                },
            }),
            supervised_keys = None,  # Set to `None` to disable
            homepage='http://www02.smt.ufrj.br/~offshore/mfs/page_01.html',
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Return SplitGenerators.
        """
        if dl_manager._manual_dir.exists():  # prefer to use manually downloaded data
            datadir = Path(dl_manager._manual_dir)
            # print(datadir)
            return {
                # 'train': self._generate_examples(datadir)
                ft: self._generate_examples(datadir/ft) for ft in _FAULT_TYPE
            }
        else:  # automatically download data
            raise NotImplementedError()
            # datadir = dl_manager.download(_DATA_URLS)
            datadir = [dl_manager.download(url) for url in _DATA_URLS]
            fdict = self._fname_parser(datadir)

            return {
                ft: self._generate_examples(fdict[ft]) for ft in _FAULT_TYPE
                # 'train': self._generate_examples(datadir),
            }

    def _generate_examples(self, path):
        assert path.exists()
        for fp in path.rglob('*.csv'):
            x = pd.read_csv(fp).T.values.astype(_DTYPE.as_numpy_dtype)
            _signal = {
                'tachometer': x[0],
                'underhang': x[[1,2,3]],
                'overhang': x[[4,5,6]],
                'microphone': x[7],
            }

            metadata = {
                'NominalRPM': float(fp.name.split('.csv')[0])*60,
                'FaultName': fp.parts[-3],
                'FaultSize': fp.parts[-2],
                'FileName': '/'.join(fp.parts[-3:]),
                'Dataset': 'Mafaulda',
            }

            yield hash(frozenset(metadata.items())), {
                'signal': _signal,
                # 'label': 'Normal' if metadata['FalutType']=='normal' else 'Faulty',
                'sampling_rate': 50000,
                'metadata': metadata,
            }

    # def _fname_parser(self, path):
    #     fd = {}
    #     for zp in path:
    #         # print('parser', zp)
    #         with open(str(zp)+'.INFO') as fp:
    #             dd=json.load(fp)
    #             fd[dd['original_fname'].split('.zip')[0]] = zp

    #     return fd

    # def _generate_examples(self, path):
    #     for fname, fobj in tfds.download.iter_archive(path, tfds.download.ExtractMethod.ZIP):
    #         x = pd.read_csv(fobj).values

    #         _dscrp = Path(fname).parent.parts
    #         metadata = {
    #             'DataLabel': ':'.join(_dscrp),
    #             'FileName': fname,
    #         }

    #         yield hash(frozenset(metadata.items())), {
    #             'signal': x,
    #             'label': 'Normal' if _dscrp[0]=='normal' else 'Faulty',
    #             'metadata': metadata,
    #         }
