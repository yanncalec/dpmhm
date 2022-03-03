"""Paderborn dataset."""

import os
from pathlib import Path
import itertools
import json
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import pandas as pd
# from scipy.io import loadmat
import patoolib  # extraction of rar file


_DESCRIPTION = """
Paderborn University Bearing DataCenter.

Experimental bearing data sets for condition monitoring (CM) based on vibration and motor current signals, provided by the Chair of Design and Drive Technology, Paderborn University, Germany.

License
-------
The data is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-nc/4.0/

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

Original data
=============
Sampling rate:
- motor current sampling frequency: 64 kHz, 2 channels
- vibration sampling frequency: 64 kHz, 1 channel
- sampling frequency of mechanic parameters (load force, load torque, speed): 4 kHz, 3 channels
- temperature sampling frequency: 1 Hz, 1 channel
Duration: 4 seconds
Number of channels: 7, ['force', 'phase_current_1', 'phase_current_2', 'speed', 'temp_2_bearing_module', 'torque', 'vibration_1']
Load and Rotational Speed:
- For undamaged and artificially damaged bearings test: not used
- For lifetime test: 3800 N and 2900 rpm
Split: Healthy, Artificially damaged, Accelerated lifetime test
Size: 20.8 Gb

Download
--------
http://groups.uni-paderborn.de/kat/BearingDataCenter/

References
==========
- Lessmeier, Christian; Kimotho, J. Kimotho; Zimmer, Detmar; Sextro, Walter:
A Benchmark Data Set for Data-Driven Classification, European Conference of the Prognostics and Health Management Society (PHM Society), Bilbao (Spain), 05.-08.07.2016.  (pdf)
- ISO 15243:2017 Rolling bearings — Damage and failures — Terms, characteristics and causes
https://www.iso.org/standard/59619.html

Notes
=====
The file `KA08/N15_M01_F10_KA08_2.mat` seems corrupted and cannot be loaded by `scipy.io.loadmat`.
"""

_CITATION = """
Christian Lessmeier et al., KAt-DataCenter: mb.uni-paderborn.de/kat/datacenter, Chair of Design and Drive Technology, Paderborn University.
"""

_METAINFO = pd.read_csv(Path(__file__).parent / 'metainfo.csv', index_col=0)  # use 'Bearing Code' as index

_DATA_URLS = ('http://groups.uni-paderborn.de/kat/BearingDataCenter/' + _METAINFO.index+'.rar').tolist()

_SPLIT = ['healthy', 'artificial', 'lifetime']

_SPLIT_PATH_MATCH = {k: _METAINFO.loc[_METAINFO['DamageMethod']==k.capitalize()].index.tolist() for k in _SPLIT}

# _SPLIT_PATH_MATCH = {
#   'healthy': ['K001', 'K002', 'K003', 'K004', 'K005', 'K006'],
#   'artificial': ['KA01', 'KA03', 'KA05', 'KA06', 'KA07', 'KA08', 'KA09', 'KI01', 'KI03', 'KI05', 'KI07', 'KI08'],
#   'lifetime': ['KA04', 'KA15', 'KA16', 'KA22', 'KA30', 'KB23', 'KB24', 'KB27', 'KI04', 'KI14', 'KI16', 'KI17', 'KI18', 'KI21']
# }

class Paderborn(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for Paderborn dataset."""

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }

  MANUAL_DOWNLOAD_INSTRUCTIONS = """
  Due to ????, automatic download is not supported in this package. Please download (optionally extract) all data and proceed the installation manually.
  """

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    return tfds.core.DatasetInfo(
        builder=self,
        description=_DESCRIPTION,
        features=tfds.features.FeaturesDict({
          # Number of channels is named or not fixed
          'signal': {
            'force': tfds.features.Tensor(shape=(None,), dtype=tf.float64),
            'phase_current_1': tfds.features.Tensor(shape=(None, ), dtype=tf.float64),
            'phase_current_2': tfds.features.Tensor(shape=(None, ), dtype=tf.float64),
            'speed': tfds.features.Tensor(shape=(None, ), dtype=tf.float64),
            'temp_2_bearing_module': tfds.features.Tensor(shape=(None, ), dtype=tf.float64),
            'torque': tfds.features.Tensor(shape=(None, ), dtype=tf.float64),
            'vibration_1': tfds.features.Tensor(shape=(None, ), dtype=tf.float64),
          },

          'label': tfds.features.ClassLabel(names=['Healthy', 'Faulty', 'Unknown']),

          'metadata': {
            'FaultComponent': tf.string, # Component of the fault
            'FaultExtend': tf.int32,  # Extend of the fault
            'DamageMethod': tf.string,  # Method of damage
            'FaultType': tf.string,   # Type of damage
            # 'Condition':  # Operating conditions (RPM, Torque, Force) see Table 6.
            'FileName': tf.string,  # Original filename (with path)
          }
        }),
        # If there's a common (input, target) tuple from the
        # features, specify them here. They'll be used if
        # `as_supervised=True` in `builder.as_dataset`.
        # supervised_keys=('signal', 'label'),  # Set to `None` to disable
        supervised_keys=None,
        homepage='https://mb.uni-paderborn.de/kat/forschung/datacenter/bearing-datacenter',
        citation=_CITATION,
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    if dl_manager._manual_dir.exists():  # prefer to use manually downloaded data
      datadir = dl_manager._manual_dir
    else:  # automatically download data
      # Parallel download results in incomplete and corrupted files
      # datadir0 = dl_manager.download(_DATA_URLS)
      datadir0 =[dl_manager.download(url) for url in _DATA_URLS]
      # print(dl_manager._extract_dir)
      # print(datadir0)
      datadir = dl_manager._extract_dir
      # extract rar files
      try:
        os.makedirs(datadir)
      except FileExistsError:
        pass

      for fp in datadir0: #.glob('*.INFO'):
        # print(fp)
        patoolib.extract_archive(fp, outdir=datadir, interactive=False)

    return {
        sp: self._generate_examples(datadir, fn, sp) for sp, fn in _SPLIT_PATH_MATCH.items()
    }

  def _generate_examples(self, path, files, mode):
    # assert path.exists()

    for file in files:
      metadata = _METAINFO.loc[file][['FaultComponent', 'FaultExtend', 'DamageMethod', 'FaultType']].to_dict()
      for fp in (path/file).glob('*.mat'):  # do not use `rglob`
        # print(fp)
        metadata['FileName'] = fp.name

        # dm=tfds.core.lazy_imports.scipy.io.loadmat(fp)[fp.name.split('.mat')[0]][0]
        # dm = loadmat(fp)[fp.name.split('.mat')[0]][0]

        # xd = {}
        # for dd in dm[0][2][0]:
        #     xd[dd[0][0]] = dd[2].squeeze()

        try:
          dm = tfds.core.lazy_imports.scipy.io.loadmat(fp,
                      mat_dtype=True,
                      squeeze_me=True,
                      )
        except Exception as msg:
          # The file KA08/N15_M01_F10_KA08_2.mat
          # cannot be processed for unknown reasons.
          print(f'Error in processing {fp}: {msg}')
          pass

        xd = {}
        for dd in np.atleast_1d(dm[fp.name.split('.mat')[0]])[0][2]:
          xd[dd[0]] = dd[2]

        yield hash(frozenset(metadata.items())), {
          'signal': xd,
          'label': 'Healthy' if metadata['DamageMethod']=='Healthy' else 'Faulty',
          'metadata': metadata
        }

  @staticmethod
  def get_references():
    try:
      with open(Path(__file__).parent / 'Exported Items.bib') as fp:
        return fp.read()
    except:
      pass
