"""DCASE2021 Task2 dataset."""

import os
import json
import tensorflow as tf
import tensorflow_datasets as tfds


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
Format: wav file
Sampling rate: 16000 Hz
Recording duration: 10 seconds
Number of channels: 1
Domain: source, target
Label: normal, anomaly or unknown
Split: train, test, query
"""

_CITATION = """
@article{Kawaguchi_arXiv2021_01,
    author = "Kawaguchi, Yohei and Imoto, Keisuke and Koizumi, Yuma and Harada, Noboru and Niizumi, Daisuke and Dohi, Kota and Tanabe, Ryo and Purohit, Harsh and Endo, Takashi",
    title = "Description and Discussion on {DCASE} 2021 Challenge Task 2: Unsupervised Anomalous Sound Detection for Machine Condition Monitoring under Domain Shifted Conditions",
    journal = "In arXiv e-prints: 2106.04492, 1–5",
    year = "2021"
}
"""

# Information labels relevant to this dataset
_MACHINE = [
  'fan',
  'gearbox',
  'pump',
  'slider',  # or 'Slide rail'
  'toycar',
  'toytrain',
  'valve',
]

_SECTION = ['00', '01', '02', '03', '04', '05']

_DOMAIN = ['source', 'target']

_CONDITION = ['normal', 'anomaly', 'unknown']

_MODE = ['train', 'test', 'query']

# Manual installation:
# Pass the path of downloaded data to the command `tfds build dcase2021_task2 --manual_dir ${path}` or use the keyword argument `download_and_prepare_kwargs` in the method `tfds.load()`.


class Dcase2021Task2(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for dcase2021_task2 dataset."""

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
            'signal': tfds.features.Audio(file_format='wav', shape=(160000,), sample_rate=16000),

            'label': tfds.features.ClassLabel(names=_CONDITION),

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
      datadir = dl_manager._manual_dir
    else:
      raise FileNotFoundError(self.MANUAL_DOWNLOAD_INSTRUCTIONS)

    print(datadir)

    return {
        'train': self._generate_examples(datadir, 'train'),
        'test': self._generate_examples(datadir, 'test'),
        'query': self._generate_examples(datadir, 'query'),
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
    foo = fname.split('/')
    _machine = foo[0].lower()

    goo = foo[-1].split('_')
    _section, _domain, _mode, _label = goo[1], goo[2], goo[3], goo[4]

    if _label not in ['normal', 'anomaly']:
      _label = 'unknown'
      _mode = 'query'

    return _machine, _section, _domain, _mode, _label

  def _generate_examples(self, path, mode):
    wavfiles = {'train':[], 'test':[], 'query':[]}

    for zf in path.glob('*.zip'):
      # flat iteration being transparent to sub folders of zip_path
      for fname, fobj in tfds.download.iter_archive(zf, tfds.download.ExtractMethod.ZIP):
        # print(fname, fobj.name)
        _machine, _section, _domain, _mode, _label = self._fname_parser(fname)
        # assert _machine in _MACHINE

        if _mode == mode:
          wavfiles[_mode].append(str(zf/fname))
          # _, x = tfds.core.lazy_import.scipy.io.wavfile.read(fname)

          metadata = {
            'Machine': _machine,
            'Section': _section,
            'Domain': _domain,
            'FileName': fname
          }
          yield hash(frozenset(metadata.items())), {
            'signal': fobj,
            'label': _label,
            'metadata': metadata
        }
        else:
          continue

    # with open(self._dl_manager._extract_dir/'wavfiles_extract.json', 'w') as fp:
    with open(path/'wavfiles_extract.json', 'w') as fp:
      json.dump(wavfiles, fp)


