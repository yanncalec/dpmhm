"""CWRU bearing dataset.

Information about of the dataset
================================
Two possible locations of bearings in the test rig: namely the drive-end and the fan-end.

For one location, a fault of certain diameter is introduced at one of the three components: inner race, ball, outer race. The inner race and the ball are rolling elements thus the fault does not have a fixed position, while the outer race is a fixed frame and the fault has three different angular positions: 3, 6 and 12.

The vibration signals are recorded by an accelerometer at (one or more of) the drive-end (DE), fan-end (FE) or base (BA) positions at the sampling rate 12 or 48 kHz, for a certain motor load.

Labels and meta-Information
---------------------------
Each experiment has the label {Normal, Faulty} and is described by the tuple (SamplingRate, Load, RPM, Location, Component, Diameter, FileName), with
- SamplingRate: {12000, 48000} in Hz
- Load: {0, 1, 2, 3} motor load in horsepower, corresponding to the average RPM: [1797, 1772, 1750, 1730]. Note that this is an average value and the true rpm of each experiment is given in its datafile.
- Location: {DriveEnd, FanEnd}. Note that this is the location of fault not that of the accelerometer.
- Component: {InnerRace, Ball, OuterRace6, OuterRace3, OuterRace12}
- Diameter: {0.007, 0.014, 0.021, 0.028}
- FileName: original filename

Convention for a normal experiment: Location and Component are None and Diameter is 0.
"""

import os
import pathlib
import itertools
import json
from attr import frozen
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import pandas as pd
from scipy.io import loadmat


_DESCRIPTION = """
Case Western Reserve University bearing dataset.

Description
===========
Motor bearings were seeded with faults using electro-discharge machining (EDM). Faults ranging from 0.007 inches in diameter to 0.040 inches in diameter were introduced separately at the inner raceway, rolling element (i.e. ball) and outer raceway. Faulted bearings were reinstalled into the test motor and vibration data was recorded for motor loads of 0 to 3 horsepower (motor speeds of 1797 to 1720 RPM).

Homepage
--------
https://engineering.case.edu/bearingdatacenter

Original Data
=============
Format: Matlab
Date of acquisition: 2000
Channels: up to 3 acclerometer data, Drive-End (DE), Fan-End (FE), Base (BA)
Split: not specified
Sampling rate: 12 kHz or 48 kHz, not fixed
Recording duration: not fixed
Label: normal and faulty data
Data fields:
- DE: drive end accelerometer data, contained by all files
- FE: fan end accelerometer data, contained by most files
- BA: base accelerometer data, contained by some files
- RPM: rpm during testing, contained by most files
"""

_CITATION = """
@misc{case_school_of_engineering_cwru_2000,
	title = {{CWRU} {Bearing} {Data} {Center}},
	copyright = {Case Western Reserve University},
	shorttitle = {{CWRU} {Bearing} {Dataset}},
	url = {https://engineering.case.edu/bearingdatacenter},
	language = {English},
	publisher = {Case Western Reserve University},
	author = {Case School of Engineering},
	year = {2000},
}
"""

# Load meta-information of all datafiles
_METAINFO = pd.read_csv(pathlib.Path(__file__).parent / 'metainfo.csv')
# [['SamplingRate', 'Load', 'Location', 'Component', 'Diameter', 'FileName']]
# Make a label by cartesian product
# _CLASS = list(['-'.join(a) for a in itertools.product(_RPM, _LOCATION, _COMPONENT, _DIAMETER)])

_DATA_URLS = ('https://engineering.case.edu/sites/default/files/'+_METAINFO['FileName']).tolist()


class CWRU(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for CWRU dataset.
  """

  VERSION = tfds.core.Version('1.0.0')

  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }

  def _info(self) -> tfds.core.DatasetInfo:
    return tfds.core.DatasetInfo(
        builder=self,
        description=_DESCRIPTION,
        features=tfds.features.FeaturesDict({
          # The sampling rate and the shape of a signal are not fixed.
          'signal':{  # three possible channels: drive-end, fan-end and base
            'DE': tfds.features.Tensor(shape=(None,), dtype=tf.float64),
            'FE': tfds.features.Tensor(shape=(None,), dtype=tf.float64),
            'BA': tfds.features.Tensor(shape=(None,), dtype=tf.float64),
          },
          # No easy to to save all channels in a tensor with unknown dimension, like
          # 'signal': tfds.features.Tensor(shape=(None,1), dtype=tf.float64),

          'rpm': tf.float32, # true rpm of the signal, not the nominal one (motor load).
          # 'rpm': tfds.features.Tensor(shape=None, dtype=tf.int32),

          'label': tfds.features.ClassLabel(names=['Normal', 'Faulty']),

          'metadata': {
              'SamplingRate': tf.uint32, # {12000, 48000} Hz
              'RotatingSpeed': tf.uint32,  # average RPM: [1797, 1772, 1750, 1730]
              'LoadForce': tf.uint32, # {0, 1, 2 ,3}, corresponding average RPM: [1797, 1772, 1750, 1730]
              'SensorLocation': tf.string, # {'DriveEnd', 'FanEnd', 'None'}
              'FaultComponent': tf.string,  # {'InnerRace', 'Ball', 'OuterRace3', 'OuterRace6', 'OuterRace12', 'None'}
              'FaultSize': tf.float32,  # {0.007, 0.014, 0.021, 0.028, 0}
              'FileName': tf.string,
          },

          # Another possibility is to use class labels (string)
          # This implies conversion in the decoding part.
          # 'metadata': {
          #     'SamplingRate': tfds.features.ClassLabel(names=['12000', '48000']),  # Hz
          #     'Load': tfds.features.ClassLabel(names=['0', '1', '2', '3']),  # Horsepower
          #     'RPM': tfds.features.ClassLabel(names=['1797', '1772', '1750', '1730']),
          #     'Location': tfds.features.ClassLabel(names=['DriveEnd', 'FanEnd', 'None']),
          #     'Component': tfds.features.ClassLabel(names=['InnerRace', 'Ball', 'OuterRace3', 'OuterRace6', 'OuterRace12', 'None']),
          #     'Diameter': tfds.features.ClassLabel(names=['0.007', '0.014', '0.021', '0.028', '0.0']),  # inches
          #     'FileName': tf.string,
          # },
        }),
        supervised_keys=None,
        # supervised_keys=('signal', 'label'),  # Set to `None` to disable
        homepage='https://engineering.case.edu/bearingdatacenter',
        citation=_CITATION,
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Return SplitGenerators.
    """
    # If class labels are used for `metadata`, add `.astype(str)` in the following before `to_dict()`
    if dl_manager._manual_dir.exists():  # prefer to use manually downloaded data
      _data_files = dl_manager._manual_dir.glob('*.mat')
      fp_dict = {str(fp): _METAINFO.loc[_METAINFO['FileName'] == fp.name].iloc[0].to_dict() for fp in _data_files}
    else:  # automatically download data
      # _data_files = dl_manager.download(_DATA_URLS)   # urls must be a list
      _data_files = [dl_manager.download(url) for url in _DATA_URLS]   # parallel download may result in corrupted files

      fp_dict = {}
      for fp in _data_files:
        with open(str(fp)+'.INFO') as fu:
          fp_dict[str(fp)] = _METAINFO.loc[_METAINFO['FileName'] == json.load(fu)['original_fname']].iloc[0].to_dict()

    return {
        'train': self._generate_examples(fp_dict),
    }

  def _generate_examples(self, fp_dict):
    for fp, metadata in fp_dict.items():
      dm = tfds.core.lazy_imports.scipy.io.loadmat(fp)
      # dm = loadmat(fp)

      # If filename is e.g. `112.mat`, then the matlab file  contains some of the fields `X112_DE_time` (always), `X112RPM`, as well as `X112_FE_time` and `X112_BA_time`.
      # Exception: the datafiles `300?.mat`` do not contain fields starting with `X300?` nor the field `RPM`.

      fn = f"{int(metadata['FileName'].split('.mat')[0]):03d}"
      if f'X{fn}_DE_time' in dm:  # find regular fields
        kde = f'X{fn}_DE_time'
        xde = dm[kde].squeeze()
        kfe = f'X{fn}_FE_time'
        xfe = dm[kfe].squeeze() if kfe in dm else []
        kba = f'X{fn}_BA_time'
        xba = dm[kba].squeeze() if kba in dm else []
        krpm = f'X{fn}RPM'
      else:   # datafiles 300x.mat are special
        kde = [k for k in dm.keys() if k[0]=='X'][0]
        xde, xfe, xba = dm[kde].squeeze(), [], []

      # Use nan if the real value is not given.
      rpm = float(dm[krpm]) if krpm in dm else np.nan # metadata['RPM']

      yield hash(frozenset(metadata.items())), {
        'signal': {
          'DE': xde,
          'FE': xfe,
          'BA': xba,
        },  # if not empty all three have the same length
        'rpm': rpm,
        'label': 'Normal' if metadata['SensorLocation']=='None' else 'Faulty',
        'metadata': metadata,
      }

  @staticmethod
  def get_references():
    try:
      with open(pathlib.Path(__file__).parent / 'Exported Items.bib') as fp:
        return fp.read()
    except:
      pass

