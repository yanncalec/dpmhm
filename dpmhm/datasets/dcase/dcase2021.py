"""DCASE2021 Task2 dataset."""

import os
import json
import tensorflow as tf
import tensorflow_datasets as tfds
from pathlib import Path

_DESCRIPTION = """
DCASE2021 Task2:
Unsupervised Anomalous Sound Detection for Machine Condition Monitoring under Domain Shifted Conditions.

Description
===========
The scope of this task is to identify whether the sound emitted from a target machine is normal or anomalous via an anomaly detector trained using only normal sound data. The main difference from the DCASE 2020 Task 2 is that the participants have to solve the domain shift problem, i.e., the condition where the acoustic characteristics of the training and test data are different.

Homepage
--------
http://dcase.community/challenge2021/task-unsupervised-detection-of-anomalous-sounds

Original data
=============
Format: wav file, int16
Sampling rate: 16000 Hz
Recording duration: 10 seconds
Number of channels: 1
Domain: source, target
Label: normal, anomaly or unknown
Split: train, test, query

Processed data
==============
Split: ['train', 'test', 'query']

Features
--------
'signal': audio,
'label': ['normal', 'anomaly', 'unknown'],
'metadata': {
  'Machine': name of machine,
  'Section': section ID,
  'Domain': source or target domain,
  'FileName': original file name,
}
"""

_CITATION = """
@article{Kawaguchi_arXiv2021_01,
    author = "Kawaguchi, Yohei and Imoto, Keisuke and Koizumi, Yuma and Harada, Noboru and Niizumi, Daisuke and Dohi, Kota and Tanabe, Ryo and Purohit, Harsh and Endo, Takashi",
    title = "Description and Discussion on {DCASE} 2021 Challenge Task 2: Unsupervised Anomalous Sound Detection for Machine Condition Monitoring under Domain Shifted Conditions",
    journal = "In arXiv e-prints: 2106.04492, 1–5",
    year = "2021"
}
"""

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

# Manual installation:
# Pass the path of downloaded data to the command `tfds build dcase2021 --manual_dir ${path}` or use the keyword argument `download_and_prepare_kwargs` in the method `tfds.load()`.


class DCASE2021(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for DCASE 2021 task2 dataset."""

  VERSION = tfds.core.Version('1.0.0')

  MANUAL_DOWNLOAD_INSTRUCTIONS = """
  Automatic download is disabled. Please download manually all zip files (without extraction) via the following links

  - Development dataset: https://zenodo.org/record/4562016#.YdanYooo9hE
  - Additional training dataset: https://zenodo.org/record/4660992#.YdanZIoo9hE
  - Evaluation dataset: https://zenodo.org/record/4884786#.YdanZooo9hE

  and proceed the installation manually.
  """

  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }

  def _info(self) -> tfds.core.DatasetInfo:
    return tfds.core.DatasetInfo(
        builder=self,
        description=_DESCRIPTION,
        features=tfds.features.FeaturesDict({
            'signal': tfds.features.Audio(file_format='wav', shape=(None,), sample_rate=None, dtype=tf.int16, encoding=tfds.features.Encoding.BYTES),

            # 'signal': tfds.features.Tensor(shape=(None,), dtype=tf.int16),

            'label': tfds.features.ClassLabel(names=['normal', 'anomaly', 'unknown']),

            'metadata': {
              'Machine': tf.string,
              'Section': tf.string,
              'Domain': tf.string,
              'FileName': tf.string,
            },

            # 'metadata': {
            #   'Machine': tfds.features.ClassLabel(names=_MACHINE),
            #   'Section': tfds.features.ClassLabel(names=_SECTION),
            #   'Domain': tfds.features.ClassLabel(names=_DOMAIN),
            # },
        }),

        # supervised_keys=('signal', 'label'),  # Set to `None` to disable
        supervised_keys=None,

        homepage='http://dcase.community/challenge2021/task-unsupervised-detection-of-anomalous-sounds',
        citation=_CITATION,
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    if dl_manager._manual_dir.exists():  # prefer to use manually downloaded data
      datadir = Path(dl_manager._manual_dir)
    else:
      raise FileNotFoundError(self.MANUAL_DOWNLOAD_INSTRUCTIONS)

    train_list = [str(x) for x in datadir.rglob('*train*.wav')]
    test_list = [str(x) for x in datadir.rglob('*test*anomaly*.wav')] + [str(x) for x in datadir.rglob('*test*norm*.wav')]
    aa = [str(x) for x in datadir.rglob('*test*.wav')]
    query_list = [x for x in aa if x not in test_list]

    return {
        'train': self._generate_examples(train_list),
        'test': self._generate_examples(test_list),
        'query': self._generate_examples(query_list),
    }

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

  def _generate_examples(self, fnames):
    # wavfiles = []

    # for zf in path.glob('*.zip'):
    #   # flat iteration being transparent to sub folders of zip_path
    #   for fname, fobj in tfds.download.iter_archive(zf, tfds.download.ExtractMethod.ZIP):
    for fn in fnames:
      fp = Path(fn)

      _machine, _section, _domain, _mode, _label = self._fname_parser(Path(*fp.parts[-3:]))
      # assert _machine in _MACHINE
      # assert _mode == mode

      # wavfiles.append(os.path.join(*fp.parts[-3:]))
      # _, x = tfds.core.lazy_imports.scipy.io.wavfile.read(fp)

      metadata = {
        'Machine': _machine,
        'Section': _section,
        'Domain': _domain,
        'FileName': os.path.join(*fp.parts[-3:])
      }

      yield hash(frozenset(metadata.items())), {
        'signal': fp,
        # 'signal': x,
        'label': _label,
        'metadata': metadata
      }

    # with open(self._dl_manager._extract_dir/'wavfiles_extract.json', 'w') as fp:

    # with open(f'/home/han/tmp/dcase2021/wavfiles_extract_{mode}.json', 'w') as fp:
    #   json.dump(wavfiles, fp)


