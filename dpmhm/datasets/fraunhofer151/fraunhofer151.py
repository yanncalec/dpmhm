"""
Fraunhofer_151 dataset.

Vibration Measurements on a Rotating Shaft at Different Unbalance Strengths

Description
===========
This dataset contains vibration data recorded on a rotating drive train. This drive train consists of an electronically commutated DC motor and a shaft driven by it, which passes through a roller bearing. With the help of a 3D-printed holder, unbalances with different weights and different radii were attached to the shaft. Besides the strength of the unbalances, the rotation speed of the motor was also varied.

This dataset can be used to develop and test algorithms for the automatic detection of unbalances on drive trains. Datasets for 4 differently sized unbalances and for the unbalance-free case were recorded. The vibration data was recorded at a sampling rate of 4096 values per second. Datasets for development (ID "D[0-4]") as well as for evaluation (ID "E[0-4]") are available for each unbalance strength. The rotation speed was varied between approx. 630 and 2330 RPM in the development datasets and between approx. 1060 and 1900 RPM in the evaluation datasets. For each measurement of the development dataset there are approx. 107min of continuous measurement data available, for each measurement of the evaluation dataset 28min.

Details of the recorded measurements and the used unbalance strengths are documented in the README.md file.

Overview of the dataset components:

|   ID	 |	Radius [mm]  | Mass [g] |
|--------|---------------|----------|
| 0D/ 0E | -		     | -        |
| 1D/ 1E | 14		     | 3.281    |
| 2D/ 2E | 18.5	         | 3.281    |
| 3D/ 3E | 23		     | 3.281    |
| 4D/ 4E | 23		     | 6.614    |

Homepage
--------
https://fordatis.fraunhofer.de/handle/fordatis/151.2

Original Dataset
================
- Type of experiments: labelled data.
- Format: CSV files
- Date of acquisition: 2020
- Channels: voltage(?), measured rpm and 3 vibrations
- Split: development and evaluation
- Operating conditions: 630~2330 rpm for the development set and 1060~1900 rpm for the evaluation set, continuous record.
- Sampling rate: 4096 Hz
- Recording duration: 107 minutes for development set and 28 minutes for evaluation set.
- Faults: 4 differently sized unbalances and unbalance-free case
- Size: ~ 11 Gb, unzipped

Download
--------
https://fordatis.fraunhofer.de/bitstream/fordatis/151.2/1/fraunhofer_eas_dataset_for_unbalance_detection_v1.zip

https://fordatis.fraunhofer.de/bitstream/fordatis/151.2/3/README.md

Built Dataset
=============
Split: ['train', 'test'].

Features
--------
- 'signal': {'V_in', 'Measured_RPM', 'Vibration'},
- 'sampling_rate: 4096,
- 'metadata':
    - 'LoadRadius': radius of the load,
    - 'Label': ['Normal', 'Unbalanced']
    - 'LoadMass': mass of the load,
    - 'TrunkIndex': time index of the signal in the original file.
    - 'FileName': original file name,

Notes
=====
The original record consists of two periods where the rotation speed increases linearly. Their durations are almost the same but depend on the split set. In the processed data these two periods are separated. Moreover, we truncate the few seconds (~10s) near the beginning and the end of each period which seem to correspond to initialization. The field `TrunkIndex` was added in `metadata` for record.
"""

from pathlib import Path
import tensorflow as tf
import tensorflow_datasets as tfds
import pandas as pd
# from scipy.io import loadmat

from dpmhm.datasets import _DTYPE, _ENCODING


_CITATION = """
@article{mey_vibration_2020,
    title = {Vibration {Measurements} on a {Rotating} {Shaft} at {Different} {Unbalance} {Strengths}},
    copyright = {https://creativecommons.org/licenses/by/4.0/},
    url = {https://fordatis.fraunhofer.de/handle/fordatis/151.2},
    doi = {10.24406/fordatis/65.2},
    language = {en},
    urldate = {2022-03-01},
    author = {Mey, Oliver and Neudeck, Willi and Schneider, André and Enge-Rosenblatt, Olaf},
    month = mar,
    year = {2020},
    note = {Accepted: 2020-04-14T10:59:09Z},
}
"""

_DATA_URLS = [
    'https://fordatis.fraunhofer.de/bitstream/fordatis/151.2/1/fraunhofer_eas_dataset_for_unbalance_detection_v1.zip'
    ]


_RADIUS = {'0': 0., '1': 14., '2': 18.5, '3':23., '4':23.}

_MASS = {'0': 0., '1': 3.281, '2': 3.281, '3':3.281, '4':6.614}


class Fraunhofer151(tfds.core.GeneratorBasedBuilder):
    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
        '1.0.0': 'Initial release.',
    }

    def _info(self) -> tfds.core.DatasetInfo:
        return tfds.core.DatasetInfo(
            builder=self,
            description=__doc__,
            features=tfds.features.FeaturesDict({
                # 'signal': tfds.features.Tensor(shape=(5, None), dtype=_DTYPE),

                'signal': {
                    'V_in': tfds.features.Tensor(shape=(None,), dtype=_DTYPE, encoding=_ENCODING),
                    'Measured_RPM': tfds.features.Tensor(shape=(None,), dtype=_DTYPE, encoding=_ENCODING),
                    # 'Vibration_1': tfds.features.Tensor(shape=(None,), dtype=_DTYPE, encoding=_ENCODING),
                    # 'Vibration_2': tfds.features.Tensor(shape=(None,), dtype=_DTYPE, encoding=_ENCODING),
                    # 'Vibration_3': tfds.features.Tensor(shape=(None,), dtype=_DTYPE, encoding=_ENCODING),
                    'Vibration': tfds.features.Tensor(shape=(3, None), dtype=_DTYPE, encoding=_ENCODING),
                },

                # 'label': tfds.features.ClassLabel(names=['Normal', 'Unbalanced']),

                'sampling_rate': tf.uint32,

                'metadata': {
                    # 'SamplingRate': tf.uint32,
                    'Label': tf.string,
                    'LoadRadius': tf.float32,
                    'LoadMass': tf.float32,
                    'TrunkIndex': tf.uint32,
                    'FileName': tf.string,  # Original filename with path in the dataset
                    'Dataset': tf.string,
                },
            }),
            supervised_keys=None,
            homepage='https://fordatis.fraunhofer.de/handle/fordatis/151',
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        def _get_split_dict(datadir):
            return {
                'train': datadir.glob('*D.csv'),
                'test': datadir.glob('*E.csv'),
            }

        if dl_manager._manual_dir.exists():  # prefer to use manually downloaded data
            datadir = Path(dl_manager._manual_dir)
        elif dl_manager._extract_dir.exists(): # automatically download & extracted data
            datadir = Path(dl_manager._extract_dir)
        else:
            raise FileNotFoundError()
        return {sp: self._generate_examples(files, sp) for sp, files in _get_split_dict(datadir).items()}

        # return {
        #     'train': self._generate_examples(datadir, 'train'),
        #     'test': self._generate_examples(datadir, 'test'),
        # }

    def _generate_examples(self, files, split):
        # assert path.exists()

        for fp in files:
            metadata = {
                # 'SamplingRate': 4096,
                'Label': 'Normal' if fp.name[0]=='0' else 'Unbalanced',
                'LoadRadius': _RADIUS[fp.name[0]],
                'LoadMass': _MASS[fp.name[0]],
                'FileName': fp.name,
                'Dataset': 'Fraunhofer151',
            }

            df0 = pd.read_csv(fp)

            # The original signal is truncated into two parts using their starting index and the duration
            sl = 107/2*60-10 if split=='train' else 28//2*60-10

            for t0 in [5, sl+15]:
                # metadata['StartIndex'] = t0
                df = df0.loc[(4096*t0):(4096*(t0+sl))]
                metadata['TrunkIndex'] = 4096*t0

                yield hash(frozenset(metadata.items())), {
                    # 'signal': df[['V_in', 'Measured_RPM', 'Vibration_1', 'Vibration_2', 'Vibration_3']].values.astype(_DTYPE),

                    'signal': {
                        'V_in': df['V_in'].values.astype(_DTYPE),
                        'Measured_RPM': df['Measured_RPM'].values.astype(_DTYPE),
                        # 'Vibration_1': df['Vibration_1'].values.astype(_DTYPE),
                        # 'Vibration_2': df['Vibration_2'].values.astype(_DTYPE),
                        # 'Vibration_3': df['Vibration_3'].values.astype(_DTYPE),
                        'Vibration': df[['Vibration_1', 'Vibration_2', 'Vibration_3']].values.T.astype(_DTYPE),
                    },
                    'sampling_rate': 4096,
                    # 'label': label,
                    'metadata': metadata
                }
