"""
DCASE2020 Task2 dataset:

Unsupervised Anomalous Sound Detection for Machine Condition Monitoring under Domain Shifted Conditions.

Description
===========
The main challenge of this task is to detect unknown anomalous sounds under the condition that only normal sound samples have been provided as training data. In real-world factories, actual anomalous sounds rarely occur and are highly diverse. Therefore, exhaustive patterns of anomalous sounds are impossible to deliberately make and/or collect. This means we have to detect unknown anomalous sounds that were not observed in the given training data. This point is one of the major differences in premise between ASD for industrial equipment and the past supervised DCASE tasks for detecting defined anomalous sounds such as gunshots or a baby crying.

Homepage
--------
https://dcase.community/challenge2020/task-unsupervised-detection-of-anomalous-sounds


Original Dataset
================
- Type of experiments: labelled data
- Format: wav file, int16
- Size: ~ 16.5 Gb, unzipped
- Sampling rate: 16000 Hz
- Recording duration: 10 seconds
- Number of channels: 1
- Label: 'normal', 'anomaly' or 'unknown'
- Split: train, test, query

Built Dataset
=============
Split: ['train', 'test', 'query']

Features
--------
- 'signal': {'channel': audio},
- 'sampling_rate': 16000,
- 'metadata': {
    'Machine': type of machine,
    'ID': machine ID,
    'FileName': original file name,
    }

Notes
=====
The training data contains only normal samples while the test data contains both normal and anomal samples.

Installation
============
Download and unzip all files into a folder `LOCAL_DIR`, from terminal run

```sh
$ tfds build Dcase2020 --imports dpmhm.datasets.dcase2020 --manual_dir LOCAL_DIR
```
"""

import os
import numpy as np
# import json
import tensorflow as tf
import tensorflow_datasets as tfds
from pathlib import Path
import itertools

from dpmhm.datasets import _DTYPE, _ENCODING, extract_zenodo_urls


_CITATION = """
@inproceedings{Koizumi_DCASE2020_01,
        Author = "Koizumi, Yuma and Kawaguchi, Yohei and Imoto, Keisuke and Nakamura, Toshiki and Nikaido, Yuki and Tanabe, Ryo and Purohit, Harsh and Suefusa, Kaori and Endo, Takashi and Yasuda, Masahiro and Harada, Noboru",
        title = "Description and Discussion on {DCASE}2020 Challenge Task2: Unsupervised Anomalous Sound Detection for Machine Condition Monitoring",
        year = "2020",
        booktitle = "Proceedings of the Detection and Classification of Acoustic Scenes and Events 2020 Workshop (DCASE2020)",
        month = "November",
        pages = "81--85",
}
"""

_Zenodo_URLS = [
    'https://zenodo.org/record/3678171',  # Development Dataset
    'https://zenodo.org/record/3727685',  # Additional Training Dataset
    'https://zenodo.org/record/3841772'  # Evaluation Dataset
    ]

# # Flatten nested list
# _DATA_URLS = list(itertools.chain.from_iterable([extract_zenodo_urls(url) for url in _Zenodo_URLS]))

_DATA_URLS = ['https://zenodo.org//record/3678171/files/dev_data_fan.zip',
 'https://zenodo.org//record/3678171/files/dev_data_fan.zip',
 'https://zenodo.org//record/3678171/files/dev_data_pump.zip',
 'https://zenodo.org//record/3678171/files/dev_data_pump.zip',
 'https://zenodo.org//record/3678171/files/dev_data_slider.zip',
 'https://zenodo.org//record/3678171/files/dev_data_slider.zip',
 'https://zenodo.org//record/3678171/files/dev_data_ToyCar.zip',
 'https://zenodo.org//record/3678171/files/dev_data_ToyCar.zip',
 'https://zenodo.org//record/3678171/files/dev_data_ToyConveyor.zip',
 'https://zenodo.org//record/3678171/files/dev_data_ToyConveyor.zip',
 'https://zenodo.org//record/3678171/files/dev_data_valve.zip',
 'https://zenodo.org//record/3678171/files/dev_data_valve.zip',
 'https://zenodo.org//record/3727685/files/eval_data_train_fan.zip',
 'https://zenodo.org//record/3727685/files/eval_data_train_fan.zip',
 'https://zenodo.org//record/3727685/files/eval_data_train_pump.zip',
 'https://zenodo.org//record/3727685/files/eval_data_train_pump.zip',
 'https://zenodo.org//record/3727685/files/eval_data_train_slider.zip',
 'https://zenodo.org//record/3727685/files/eval_data_train_slider.zip',
 'https://zenodo.org//record/3727685/files/eval_data_train_ToyCar.zip',
 'https://zenodo.org//record/3727685/files/eval_data_train_ToyCar.zip',
 'https://zenodo.org//record/3727685/files/eval_data_train_ToyConveyor.zip',
 'https://zenodo.org//record/3727685/files/eval_data_train_ToyConveyor.zip',
 'https://zenodo.org//record/3727685/files/eval_data_train_valve.zip',
 'https://zenodo.org//record/3727685/files/eval_data_train_valve.zip',
 'https://zenodo.org//record/3841772/files/eval_data_test_fan.zip',
 'https://zenodo.org//record/3841772/files/eval_data_test_fan.zip',
 'https://zenodo.org//record/3841772/files/eval_data_test_pump.zip',
 'https://zenodo.org//record/3841772/files/eval_data_test_pump.zip',
 'https://zenodo.org//record/3841772/files/eval_data_test_slider.zip',
 'https://zenodo.org//record/3841772/files/eval_data_test_slider.zip',
 'https://zenodo.org//record/3841772/files/eval_data_test_ToyCar.zip',
 'https://zenodo.org//record/3841772/files/eval_data_test_ToyCar.zip',
 'https://zenodo.org//record/3841772/files/eval_data_test_ToyConveyor.zip',
 'https://zenodo.org//record/3841772/files/eval_data_test_ToyConveyor.zip',
 'https://zenodo.org//record/3841772/files/eval_data_test_valve.zip',
 'https://zenodo.org//record/3841772/files/eval_data_test_valve.zip']


class Dcase2020(tfds.core.GeneratorBasedBuilder):
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
                    'channel': tfds.features.Audio(file_format='wav', shape=(None,), sample_rate=None, dtype=np.int16, encoding=tfds.features.Encoding.BYTES)
                },

                # 'signal': tfds.features.Audio(file_format='wav', shape=(None,), sample_rate=None, dtype=tf.int16, encoding=tfds.features.Encoding.BYTES),  # shape=(1, None) doesn't work

                # 'signal': tfds.features.Tensor(shape=(1,None), dtype=tf.float64),  # much slower on wav files

                'sampling_rate': tf.uint32,

                # 'label': tfds.features.ClassLabel(names=['normal', 'anomaly', 'unknown']),

                'metadata': {
                    'Machine': tf.string,
                    'ID': tf.string,
                    'Label': tf.string,
                    'FileName': tf.string,
                    'Dataset': tf.string,
                },
            }),

            supervised_keys=None,
            homepage='https://dcase.community/challenge2020/task-unsupervised-detection-of-anomalous-sounds',
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        def _get_split_dict(datadir):
            train_list = list(datadir.rglob('*/train/*.wav'))
            # separate query data (not labelled) from test data
            test_list = list(datadir.rglob('*/test/anomaly*.wav')) + list(datadir.rglob('*/test/normal*.wav'))
            query_list = [x for x in datadir.rglob('*/test/*.wav') if x not in test_list]

            # train_list = [str(x) for x in datadir.rglob('*/train/*.wav')]
            # # separate query data (not labelled) from test data
            # test_list = [str(x) for x in datadir.rglob('*/test/anomaly*.wav')] +  [str(x) for x in datadir.rglob('*/test/normal*.wav')]

            # aa = [str(x) for x in datadir.rglob('*/test/*.wav')]
            # query_list = [x for x in aa if x not in test_list]

            # print(len(train_list), len(test_list), len(query_list))
            return {
                'train': train_list,
                'test': test_list,
                'query': query_list
            }

        if dl_manager._manual_dir.exists():  # prefer to use manually downloaded data
            datadir = Path(dl_manager._manual_dir)
        elif dl_manager._extract_dir.exists(): # automatically download & extracted data
            datadir = Path(dl_manager._extract_dir)
        else:
            raise FileNotFoundError()

        return {sp: self._generate_examples(files) for sp, files in _get_split_dict(datadir).items()}

        # return {
        #     'train': self._generate_examples(train_list),
        #     'test': self._generate_examples(test_list),
        #     'query': self._generate_examples(query_list),
        # }

    @classmethod
    def _fname_parser(cls, fname):
        """Parse the filename and extract relevant information.

        Examples of filename
        --------------------
        train: fan/train/normal_id_00_00000000.wav
        test: ToyCar/test/anomaly_id_01_00000005.wav
        query: ToyCar/test/id_07_00000496.wav
        """
        _machine = fname.parts[0]
        _mode = fname.parts[1]
        idx = fname.parts[-1].find('id_')
        _id = fname.parts[-1][idx+3:idx+5]
        _label = fname.parts[-1].split('_')[0]

        if _label not in ['normal', 'anomaly']:
            _label = 'unknown'
            _mode = 'query'

        return _machine, _mode, _id, _label

    def _generate_examples(self, files):
        # for zf in path.glob('*.zip'):
        #   # flat iteration being transparent to sub folders of zip_path
        #   for fname, fobj in tfds.download.iter_archive(zf, tfds.download.ExtractMethod.ZIP):

        for fp in files:
            _machine, _mode, _id, _label = self._fname_parser(Path(*fp.parts[-3:]))
            # _, x = tfds.core.lazy_imports.scipy.io.wavfile.read(fp)

            metadata = {
                'Machine': _machine,
                'ID': _id,
                'Label': _label,
                'FileName': os.path.join(*fp.parts[-3:]),
                'Dataset': 'DCASE2020',
            }

            yield hash(frozenset(metadata.items())), {
                'signal': {'channel': fp},
                'sampling_rate': 16000,
                # 'label': _label,
                'metadata': metadata
            }

