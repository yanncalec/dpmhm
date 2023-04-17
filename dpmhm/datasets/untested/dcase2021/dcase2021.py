"""
DCASE2021 Task2 dataset:

Unsupervised Anomalous Sound Detection for Machine Condition Monitoring under Domain Shifted Conditions.

Description
===========
The scope of this task is to identify whether the sound emitted from a target machine is normal or anomalous via an anomaly detector trained using only normal sound data. The main difference from the DCASE 2020 Task 2 is that the participants have to solve the domain shift problem, i.e., the condition where the acoustic characteristics of the training and test data are different.

Homepage
--------
http://dcase.community/challenge2021/task-unsupervised-detection-of-anomalous-sounds


Original Dataset
================
- Type of experiments: labelled data
- Format: wav file, int16
- Sampling rate: 16000 Hz
- Recording duration: 10 seconds
- Number of channels: 1
- Domain: source, target
- Label: normal, anomaly or unknown
- Split: train, test, query
- Size: ~ 17.7 Gb, unzipped

Built Dataset
=============
Split: ['train', 'test', 'query']

Features
--------
- 'signal': {'channel': audio},
- 'sampling_rate': 16000
- 'metadata':
    - 'Machine': name of machine,
    - 'Section': section ID,
    - 'Domain': source or target domain,
    - 'FileName': original file name,

Installation
============
Download and unzip all files into a folder `LOCAL_DIR`, from terminal run

```sh
$ tfds build Dcase2021 --imports dpmhm.datasets.dcase2021 --manual_dir LOCAL_DIR
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
@article{Kawaguchi_arXiv2021_01,
        author = "Kawaguchi, Yohei and Imoto, Keisuke and Koizumi, Yuma and Harada, Noboru and Niizumi, Daisuke and Dohi, Kota and Tanabe, Ryo and Purohit, Harsh and Endo, Takashi",
        title = "Description and Discussion on {DCASE} 2021 Challenge Task 2: Unsupervised Anomalous Sound Detection for Machine Condition Monitoring under Domain Shifted Conditions",
        journal = "In arXiv e-prints: 2106.04492, 1–5",
        year = "2021"
}
"""

_Zenodo_URLS = [
    # Development Dataset:
    'https://zenodo.org/record/4562016',
    #Additional Training Dataset:
    'https://zenodo.org/record/4660992',
    # Evaluation Dataset:
    'https://zenodo.org/record/4884786'
]

# # Flatten nested list
# _DATA_URLS = list(itertools.chain.from_iterable([extract_zenodo_urls(url) for url in _Zenodo_URLS]))

_DATA_URLS = ['https://zenodo.org//record/4562016/files/dev_data_fan.zip',
 'https://zenodo.org//record/4562016/files/dev_data_fan.zip',
 'https://zenodo.org//record/4562016/files/dev_data_gearbox.zip',
 'https://zenodo.org//record/4562016/files/dev_data_gearbox.zip',
 'https://zenodo.org//record/4562016/files/dev_data_pump.zip',
 'https://zenodo.org//record/4562016/files/dev_data_pump.zip',
 'https://zenodo.org//record/4562016/files/dev_data_slider.zip',
 'https://zenodo.org//record/4562016/files/dev_data_slider.zip',
 'https://zenodo.org//record/4562016/files/dev_data_ToyCar.zip',
 'https://zenodo.org//record/4562016/files/dev_data_ToyCar.zip',
 'https://zenodo.org//record/4562016/files/dev_data_ToyTrain.zip',
 'https://zenodo.org//record/4562016/files/dev_data_ToyTrain.zip',
 'https://zenodo.org//record/4562016/files/dev_data_valve.zip',
 'https://zenodo.org//record/4562016/files/dev_data_valve.zip',
 'https://zenodo.org//record/4660992/files/eval_data_fan_train.zip',
 'https://zenodo.org//record/4660992/files/eval_data_fan_train.zip',
 'https://zenodo.org//record/4660992/files/eval_data_gearbox_train.zip',
 'https://zenodo.org//record/4660992/files/eval_data_gearbox_train.zip',
 'https://zenodo.org//record/4660992/files/eval_data_pump_train.zip',
 'https://zenodo.org//record/4660992/files/eval_data_pump_train.zip',
 'https://zenodo.org//record/4660992/files/eval_data_slider_train.zip',
 'https://zenodo.org//record/4660992/files/eval_data_slider_train.zip',
 'https://zenodo.org//record/4660992/files/eval_data_ToyCar_train.zip',
 'https://zenodo.org//record/4660992/files/eval_data_ToyCar_train.zip',
 'https://zenodo.org//record/4660992/files/eval_data_ToyTrain_train.zip',
 'https://zenodo.org//record/4660992/files/eval_data_ToyTrain_train.zip',
 'https://zenodo.org//record/4660992/files/eval_data_valve_train.zip',
 'https://zenodo.org//record/4660992/files/eval_data_valve_train.zip',
 'https://zenodo.org//record/4884786/files/eval_data_fan_test.zip',
 'https://zenodo.org//record/4884786/files/eval_data_fan_test.zip',
 'https://zenodo.org//record/4884786/files/eval_data_gearbox_test.zip',
 'https://zenodo.org//record/4884786/files/eval_data_gearbox_test.zip',
 'https://zenodo.org//record/4884786/files/eval_data_pump_test.zip',
 'https://zenodo.org//record/4884786/files/eval_data_pump_test.zip',
 'https://zenodo.org//record/4884786/files/eval_data_slider_test.zip',
 'https://zenodo.org//record/4884786/files/eval_data_slider_test.zip',
 'https://zenodo.org//record/4884786/files/eval_data_ToyCar_test.zip',
 'https://zenodo.org//record/4884786/files/eval_data_ToyCar_test.zip',
 'https://zenodo.org//record/4884786/files/eval_data_ToyTrain_test.zip',
 'https://zenodo.org//record/4884786/files/eval_data_ToyTrain_test.zip',
 'https://zenodo.org//record/4884786/files/eval_data_valve_test.zip',
 'https://zenodo.org//record/4884786/files/eval_data_valve_test.zip']

# # Information labels relevant to this dataset
# _MACHINE = [
#   'fan',
#   'gearbox',
#   'pump',
#   'slider',  # or 'Slide rail'
#   'toycar',
#   'toytrain',
#   'valve',
# ]

# _SECTION = ['00', '01', '02', '03', '04', '05']

# _DOMAIN = ['source', 'target']

# _MODE = ['train', 'test', 'query']

# _CONDITION = ['normal', 'anomaly', 'unknown']

# _DATA_URLS = []


class Dcase2021(tfds.core.GeneratorBasedBuilder):
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
                    'channel': tfds.features.Audio(file_format='wav', shape=(None,), sample_rate=None, dtype=np.int16, encoding=tfds.features.Encoding.BYTES),
                },

                # 'signal': tfds.features.Tensor(shape=(None,), dtype=tf.int16),

                # 'label': tfds.features.ClassLabel(names=['normal', 'anomaly', 'unknown']),

                'sampling_rate': tf.uint32,

                'metadata': {
                    'Machine': tf.string,
                    'Label': tf.string,
                    'Section': tf.string,
                    'Domain': tf.string,
                    'FileName': tf.string,
                    'Dataset': tf.string,
                },

                # 'metadata': {
                #   'Machine': tfds.features.ClassLabel(names=_MACHINE),
                #   'Section': tfds.features.ClassLabel(names=_SECTION),
                #   'Domain': tfds.features.ClassLabel(names=_DOMAIN),
                # },
            }),
            supervised_keys=None,

            homepage='http://dcase.community/challenge2021/task-unsupervised-detection-of-anomalous-sounds',
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        def _get_split_dict(datadir):
            train_list = list(datadir.rglob('*train*.wav'))
            # separate query data (not labelled) from test data
            test_list = list(datadir.rglob('*test*anomaly*.wav')) + list(datadir.rglob('*test*normal*.wav'))
            query_list = [x for x in datadir.rglob('*test*.wav') if x not in test_list]

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

    @classmethod
    def _fname_parser(cls, fname):
        """Parse the filename and extract relevant information.

        Examples of filename
        --------------------
        train: 'pump/train/section_03_source_train_normal_0004_serial_no_030_water.wav'
        test: 'ToyTrain/source_test/section_00_source_test_anomaly_0090.wav'
        query: 'ToyTrain/target_test/section_05_target_test_0162.wav'
        """
        _machine = fname.parts[0] #.lower()

        goo = fname.parts[-1].split('_')
        _section, _domain, _mode, _label = goo[1], goo[2], goo[3], goo[4]

        if _label not in ['normal', 'anomaly']:
            _label = 'unknown'
            _mode = 'query'

        return _machine, _section, _domain, _mode, _label

    def _generate_examples(self, files):
        for fp in files:
            _machine, _section, _domain, _mode, _label = self._fname_parser(Path(*fp.parts[-3:]))
            # assert _machine in _MACHINE
            # assert _mode == mode

            # wavfiles.append(os.path.join(*fp.parts[-3:]))
            # _, x = tfds.core.lazy_imports.scipy.io.wavfile.read(fp)

            metadata = {
                'Machine': _machine,
                'Label': _label,
                'Section': _section,
                'Domain': _domain,
                'FileName': os.path.join(*fp.parts[-3:]),
                'Dataset': 'DCASE2021',
            }

            yield hash(frozenset(metadata.items())), {
                'signal': {'channel': fp},
                'sampling_rate': 16000,
                # 'label': _label,
                'metadata': metadata
            }

