"""
DCASE2022 Task2 dataset:

Unsupervised Anomalous Sound Detection for Machine Condition Monitoring under Domain Shifted Conditions.

Description
===========
This task is the follow-up to DCASE 2020 Task 2 and DCASE 2021 Task 2. The task this year is to detect anomalous sounds under three main conditions:

1. Only normal sound clips are provided as training data (i.e., unsupervised learning scenario). In real-world factories, anomalies rarely occur and are highly diverse. Therefore, exhaustive patterns of anomalous sounds are impossible to create or collect and unknown anomalous sounds that were not observed in the given training data must be detected. This condition is the same as in DCASE 2020 Task 2 and DCASE 2021 Task 2.

2. Factors other than anomalies change the acoustic characteristics between training and test data (i.e., domain shift). In real-world cases, operational conditions of machines or environmental noise often differ between the training and testing phases. For example, the operation speed of a conveyor can change due to seasonal demand, or environmental noise can fluctuate depending on the states of surrounding machines. This condition is the same as in DCASE 2021 Task 2.

3. In test data, samples unaffected by domain shifts (source domain data) and those affected by domain shifts (target domain data) are mixed, and the source/target domain of each sample is not specified. Therefore, the model must detect anomalies with the same threshold value regardless of the domain (i.e., domain generalization).

Homepage
--------
https://dcase.community/challenge2022/task-unsupervised-anomalous-sound-detection-for-machine-condition-monitoring

Original Dataset
================
- Format: wav file, int16
- Sampling rate: 16000 Hz
- Recording duration: 10 seconds
- Number of channels: 1
- Domain: source, target
- Label: normal, anomaly or unknown
- Split: train, test, query
- Size: ~ 15 Gb, unzipped

Processed Dataset
=================
Split: ['train', 'test', 'query']

Features
--------
- 'signal': {'channel': audio},
- 'sampling_rate': 16000,
- 'metadata':
    - 'Machine': name of machine,
    - 'Section': section ID,
    - 'Domain': source or target domain,
    - 'Attribute': attribute of domain,
    - 'FileName': original file name,

Installation
============
Download and unzip all files into a folder `LOCAL_DIR`, from terminal run

```sh
$ tfds build Dcase2022 --imports dpmhm.datasets.dcase2022 --manual_dir LOCAL_DIR
```
"""

import os
# import json
import tensorflow as tf
import tensorflow_datasets as tfds
from pathlib import Path
# import pandas as pd
# import numpy as np

# from dpmhm.datasets.preprocessing import AbstractDatasetCompactor, AbstractFeatureTransformer, AbstractPreprocessor
from dpmhm.datasets import _DTYPE


_CITATION = """
@article{Dohi_arXiv2022_02,
        author = "Dohi, Kota and Imoto, Keisuke and Harada, Noboru and Niizumi, Daisuke and Koizumi, Yuma and Nishida, Tomoya and Purohit, Harsh and Endo, Takashi and Yamamoto, Masaaki and Kawaguchi, Yohei",
        title = "Description and Discussion on {DCASE} 2022 Challenge Task 2: Unsupervised Anomalous Sound Detection for Machine Condition Monitoring Applying Domain Generalization Techniques",
        journal = "In arXiv e-prints: 2206.05876",
        year = "2022"
}
"""

_Zenodo_URLS = [
    # Development Dataset:
    'https://zenodo.org/record/6355122',
    # Additional Training Dataset:
    'https://zenodo.org/record/6462969',
    # Evaluation Dataset:
    'https://zenodo.org/record/6586456'
]

# # Flatten nested list
# _DATA_URLS = list(itertools.chain.from_iterable([extract_zenodo_urls(url) for url in _Zenodo_URLS]))

_DATA_URLS = ['https://zenodo.org//record/6355122/files/dev_bearing.zip',
 'https://zenodo.org//record/6355122/files/dev_bearing.zip',
 'https://zenodo.org//record/6355122/files/dev_fan.zip',
 'https://zenodo.org//record/6355122/files/dev_fan.zip',
 'https://zenodo.org//record/6355122/files/dev_gearbox.zip',
 'https://zenodo.org//record/6355122/files/dev_gearbox.zip',
 'https://zenodo.org//record/6355122/files/dev_slider.zip',
 'https://zenodo.org//record/6355122/files/dev_slider.zip',
 'https://zenodo.org//record/6355122/files/dev_ToyCar.zip',
 'https://zenodo.org//record/6355122/files/dev_ToyCar.zip',
 'https://zenodo.org//record/6355122/files/dev_ToyTrain.zip',
 'https://zenodo.org//record/6355122/files/dev_ToyTrain.zip',
 'https://zenodo.org//record/6355122/files/dev_valve.zip',
 'https://zenodo.org//record/6355122/files/dev_valve.zip',
 'https://zenodo.org//record/6462969/files/eval_data_bearing_train.zip',
 'https://zenodo.org//record/6462969/files/eval_data_bearing_train.zip',
 'https://zenodo.org//record/6462969/files/eval_data_fan_train.zip',
 'https://zenodo.org//record/6462969/files/eval_data_fan_train.zip',
 'https://zenodo.org//record/6462969/files/eval_data_gearbox_train.zip',
 'https://zenodo.org//record/6462969/files/eval_data_gearbox_train.zip',
 'https://zenodo.org//record/6462969/files/eval_data_slider_train.zip',
 'https://zenodo.org//record/6462969/files/eval_data_slider_train.zip',
 'https://zenodo.org//record/6462969/files/eval_data_ToyCar_train.zip',
 'https://zenodo.org//record/6462969/files/eval_data_ToyCar_train.zip',
 'https://zenodo.org//record/6462969/files/eval_data_ToyTrain_train.zip',
 'https://zenodo.org//record/6462969/files/eval_data_ToyTrain_train.zip',
 'https://zenodo.org//record/6462969/files/eval_data_valve_train.zip',
 'https://zenodo.org//record/6462969/files/eval_data_valve_train.zip',
 'https://zenodo.org//record/6586456/files/eval_data_bearing_test.zip',
 'https://zenodo.org//record/6586456/files/eval_data_bearing_test.zip',
 'https://zenodo.org//record/6586456/files/eval_data_fan_test.zip',
 'https://zenodo.org//record/6586456/files/eval_data_fan_test.zip',
 'https://zenodo.org//record/6586456/files/eval_data_gearbox_test.zip',
 'https://zenodo.org//record/6586456/files/eval_data_gearbox_test.zip',
 'https://zenodo.org//record/6586456/files/eval_data_slider_test.zip',
 'https://zenodo.org//record/6586456/files/eval_data_slider_test.zip',
 'https://zenodo.org//record/6586456/files/eval_data_ToyCar_test.zip',
 'https://zenodo.org//record/6586456/files/eval_data_ToyCar_test.zip',
 'https://zenodo.org//record/6586456/files/eval_data_ToyTrain_test.zip',
 'https://zenodo.org//record/6586456/files/eval_data_ToyTrain_test.zip',
 'https://zenodo.org//record/6586456/files/eval_data_valve_test.zip',
 'https://zenodo.org//record/6586456/files/eval_data_valve_test.zip']


class Dcase2022(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for DCASE 2022 task2 dataset."""

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
                # 'signal': tfds.features.Audio(file_format='wav', shape=(None,), sample_rate=None, dtype=tf.int16, encoding=tfds.features.Encoding.ZLIB),  # for more compression

                # 'signal': tfds.features.Tensor(shape=(None,), dtype=tf.int16),

                'sampling_rate': tf.uint32,

                # 'label': tfds.features.ClassLabel(names=['normal', 'anomaly', 'unknown']),

                'metadata': {
                    # 'SamplingRate': tf.uint32,
                    'Machine': tf.string,
                    'Label': tf.string,
                    'Section': tf.string,
                    'Domain': tf.string,
                    'Attribute': tf.string,
                    'FileName': tf.string,
                    'Dataset': tf.string,
                },
            }),
            supervised_keys=None,
            homepage='https://dcase.community/challenge2022/task-unsupervised-anomalous-sound-detection-for-machine-condition-monitoring',
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        def _get_split_dict(datadir):
            train_list = list(datadir.rglob('*train*.wav'))
            # separate query data (not labelled) from test data
            test_list = list(datadir.rglob('*test*anomaly*.wav')) + list(datadir.rglob('*test*normal*.wav'))
            query_list = [x for x in datadir.rglob('*/test/*.wav') if x not in test_list]

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
        train: 'ToyCar/train/section_00_source_train_normal_0009_car_A1_spd_28V_mic_1_noise_1.wav'
        test: 'gearbox/test/section_00_source_test_anomaly_0007_volt_2.5.wav'
        query: 'bearing/test/section_05_0196.wav'
        """
        _machine = fname.parts[0] #.lower()

        goo = fname.parts[-1].split('_')
        try:
            _section, _domain, _mode, _label, _attribute = goo[1], goo[2], goo[3], goo[4], goo[6:].join('_')
        except:
            _section = goo[1]
            _domain, _mode, _label, _attribute = '', 'query', 'unknown', ''

        return _machine, _section, _domain, _mode, _label, _attribute

    def _generate_examples(self, files):
        for fp in files:
            _machine, _section, _domain, _mode, _label, _attribute = self._fname_parser(Path(*fp.parts[-3:]))

            # sr, x = tfds.core.lazy_imports.scipy.io.wavfile.read(fp)
            # xs = pd.Series(x, dtype=np.int16)
            # print(len(x))
            # print(x.dtype)

            metadata = {
                'Machine': _machine,
                'Label': _label,
                'Section': _section,
                'Domain': _domain,
                'Attribute': _attribute,
                'FileName': os.path.join(*fp.parts[-3:]),
                'Dataset': 'DCASE2022',
            }

            yield hash(frozenset(metadata.items())), {
                'signal': {'channel': fp},
                'sampling_rate': 16000,
                'metadata': metadata
            }

