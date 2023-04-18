"""
DCASE2023 Task2 dataset:


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
"""

_Zenodo_URLS = [
    'https://zenodo.org/record/7690157',  # Development Dataset
    'https://zenodo.org/record/7830345',  # Additional Training Dataset
    # 'https://zenodo.org/record/'  # Evaluation Dataset
    ]

# # Flatten nested list
# _DATA_URLS = list(itertools.chain.from_iterable([extract_zenodo_urls(url) for url in _Zenodo_URLS]))

_DATA_URLS = ['https://zenodo.org//record/7690157/files/dev_bearing.zip',
 'https://zenodo.org//record/7690157/files/dev_bearing.zip',
 'https://zenodo.org//record/7690157/files/dev_fan.zip',
 'https://zenodo.org//record/7690157/files/dev_fan.zip',
 'https://zenodo.org//record/7690157/files/dev_gearbox.zip',
 'https://zenodo.org//record/7690157/files/dev_gearbox.zip',
 'https://zenodo.org//record/7690157/files/dev_slider.zip',
 'https://zenodo.org//record/7690157/files/dev_slider.zip',
 'https://zenodo.org//record/7690157/files/dev_ToyCar.zip',
 'https://zenodo.org//record/7690157/files/dev_ToyCar.zip',
 'https://zenodo.org//record/7690157/files/dev_ToyTrain.zip',
 'https://zenodo.org//record/7690157/files/dev_ToyTrain.zip',
 'https://zenodo.org//record/7690157/files/dev_valve.zip',
 'https://zenodo.org//record/7690157/files/dev_valve.zip',
 'https://zenodo.org//record/7830345/files/eval_data_bandsaw_train.zip',
 'https://zenodo.org//record/7830345/files/eval_data_bandsaw_train.zip',
 'https://zenodo.org//record/7830345/files/eval_data_grinder_train.zip',
 'https://zenodo.org//record/7830345/files/eval_data_grinder_train.zip',
 'https://zenodo.org//record/7830345/files/eval_data_shaker_train.zip',
 'https://zenodo.org//record/7830345/files/eval_data_shaker_train.zip',
 'https://zenodo.org//record/7830345/files/eval_data_ToyDrone_train.zip',
 'https://zenodo.org//record/7830345/files/eval_data_ToyDrone_train.zip',
 'https://zenodo.org//record/7830345/files/eval_data_ToyNscale_train.zip',
 'https://zenodo.org//record/7830345/files/eval_data_ToyNscale_train.zip',
 'https://zenodo.org//record/7830345/files/eval_data_ToyTank_train.zip',
 'https://zenodo.org//record/7830345/files/eval_data_ToyTank_train.zip',
 'https://zenodo.org//record/7830345/files/eval_data_Vacuum_train.zip',
 'https://zenodo.org//record/7830345/files/eval_data_Vacuum_train.zip']


class Dcase2023(tfds.core.GeneratorBasedBuilder):
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
            homepage='https://dcase.community/challenge2023/task-first-shot-unsupervised-anomalous-sound-detection-for-machine-condition-monitoring',
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

