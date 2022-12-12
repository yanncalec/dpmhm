"""IMS dataset.

Test-to-failure experiments on bearings. The data set was provided by the Center for Intelligent Maintenance Systems (IMS), University of Cincinnati.

Description
===========
An AC motor, coupled by a rub belt, keeps the rotation speed constant. The four bearings are in the same shaft and are forced lubricated by a circulation system that regulates the flow and the temperature. It is announced on the provided “Readme Document for IMS Bearing Data” in the downloaded file, that the test was stopped when the accumulation of debris on a magnetic plug exceeded a certain level indicating the possibility of an impending failure. The four bearings are all of the same type. There are double range pillow blocks rolling elements bearing.

Three (3) data sets are included in the data packet. Each data set describes a test-to-failure experiment. Each data set consists of individual files that are 1-second vibration signal snapshots recorded at specific intervals. Each file consists of 20,480 points with the sampling rate set at 20 kHz. The file name indicates when the data was collected. Each record (row) in the data file is a data point. Data collection was facilitated by NI DAQ Card 6062E. Larger intervals of time stamps (showed in file names) indicate resumption of the experiment in the next working day.

For more details, see the descriptions in
- `Readme Document for IMS Bearing Data.pdf` included in the downloaded data.

Homepage
--------
http://imscenter.net/
http://ti.arc.nasa.gov/project/prognostic-data-repository

Download
--------
https://ti.arc.nasa.gov/c/3/

Original Data
=============
Format: text
Sampling rate: 20480 Hz
Size: 6.1 Gb

Notes
=====
- There is no `label` in this package since this is a run-to-failure dataset.
- The original data has a single `.7z` file which extracts to three subsets.
- The extracted subfolder named `4th_test` corresponds actually to the 3rd test.
"""


import os
import pathlib
import itertools
import json
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import pandas as pd
# from scipy.io import loadmat

# TODO(ims): Markdown description  that will appear on the catalog page.
_DESCRIPTION = """
IMS Bearing Data Set.

Test-to-failure experiments on bearings. The data set was provided by the Center for Intelligent Maintenance Systems (IMS), University of Cincinnati.

Description
===========
An AC motor, coupled by a rub belt, keeps the rotation speed constant. The four bearings are in the same shaft and are forced lubricated by a circulation system that regulates the flow and the temperature. It is announced on the provided “Readme Document for IMS Bearing Data” in the downloaded file, that the test was stopped when the accumulation of debris on a magnetic plug exceeded a certain level indicating the possibility of an impending failure. The four bearings are all of the same type. There are double range pillow blocks rolling elements bearing.

Three (3) data sets are included in the data packet. Each data set describes a test-to-failure experiment. Each data set consists of individual files that are 1-second vibration signal snapshots recorded at specific intervals. Each file consists of 20,480 points with the sampling rate set at 20 kHz. The file name indicates when the data was collected. Each record (row) in the data file is a data point. Data collection was facilitated by NI DAQ Card 6062E. Larger intervals of time stamps (showed in file names) indicate resumption of the experiment in the next working day.

For more details, see the descriptions in
- `Readme Document for IMS Bearing Data.pdf` included in the downloaded data.

Homepage
--------
http://imscenter.net/
http://ti.arc.nasa.gov/project/prognostic-data-repository

Download
--------
https://ti.arc.nasa.gov/c/3/

Original Data
=============
Format: text
Sampling rate: 20480 Hz
Size: 6.1 Gb

Notes
=====
- There is no `label` in this package since this is a run-to-failure dataset.
- The original data has a single `.7z` file which extracts to three subsets.
- The extracted subfolder named `4th_test` corresponds actually to the 3rd test.
"""

_CHARACTERISTICS_BEARING = {
  'Pitch diameter': 71.5, # mm
  'Rolling element diameter': 8.4,  # mm
  'Number of rolling element per row': 16,
  'Contact angle': 15.17,  # degree
  'Static load': 26690  # N
}

_CHARACTERISTICS_TESTRIG = {
  'Shaft frequency': 	33.3,  # Hz
  'Ball Pass Frequency Outer race (BPFO)': 	236,  # Hz
  'Ball Pass Frequency Inner race (BPFI)': 	297,  # Hz
  'Ball Spin Frequency (BSF)': 	278,  # Hz
  'Fundamental Train Frequency (FTF)': 	15  # Hz
}


_CITATION = """
- Hai Qiu, Jay Lee, Jing Lin. “Wavelet Filter-based Weak Signature Detection Method and its Application on Roller Bearing Prognostics.” Journal of Sound and Vibration 289 (2006) 1066-1090
- J. Lee, H. Qiu, G. Yu, J. Lin, and Rexnord Technical Services (2007). IMS, University of Cincinnati. "Bearing Data Set", NASA Ames Prognostics Data Repository (http://ti.arc.nasa.gov/project/prognostic-data-repository), NASA Ames Research Center, Moffett Field, CA
"""


class IMS(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for ims dataset."""

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }

  MANUAL_DOWNLOAD_INSTRUCTIONS = """
  Due to unsupported file format (7zip, rar), automatic download & extraction is not supported in this package. Please download all data from

    https://www.kaggle.com/datasets/vinayak123tyagi/bearing-dataset?resource=download

  or
    https://www.nasa.gov/content/prognostics-center-of-excellence-data-set-repository

  extract all files, then proceed the installation manually.
  """

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    # TODO(ims): Specifies the tfds.core.DatasetInfo object
    return tfds.core.DatasetInfo(
        builder=self,
        description=_DESCRIPTION,
        features=tfds.features.FeaturesDict({
          'signal': tfds.features.Tensor(shape=(None,None,1), encoding='zlib', dtype=tf.float64),
          # {
          #   'dataset1': tfds.features.Tensor(shape=(None,8), dtype=tf.float64),
          #   'dataset2': tfds.features.Tensor(shape=(None,4), dtype=tf.float64),
          #   'dataset3': tfds.features.Tensor(shape=(None,4), dtype=tf.float64),
          # },

          # Note: there is no 'label' here since this is a run-to-failure dataset.

          'metadata': {
            # 'SamplingRate': tf.uint32,
            # 'RotatingSpeed': tf.float32,
            'OriginalSplit': tf.string,
            'FileName': tf.string,
          },
        }),
        # If there's a common (input, target) tuple from the
        # features, specify them here. They'll be used if
        # `as_supervised=True` in `builder.as_dataset`.
        supervised_keys=None,
        homepage='http://imscenter.net/',
        citation=_CITATION,
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    if dl_manager._manual_dir.exists():  # prefer to use manually downloaded data
      datadir = dl_manager._manual_dir
    else:  # automatically download data
      # datadir = dl_manager.download_and_extract(_URL)
      raise FileNotFoundError(self.MANUAL_DOWNLOAD_INSTRUCTIONS)

    return {
        'dataset1': self._generate_examples(datadir / '1st_test', 'dataset1'),
        'dataset2': self._generate_examples(datadir / '2nd_test', 'dataset2'),
        'dataset3': self._generate_examples(datadir / '4th_test' / 'txt', 'dataset3'),
    }

  def _generate_examples(self, path, mode):
    """Yields examples."""
    for fp in path.glob('*'):
      x = pd.read_csv(fp, sep='\t',header=None).values[:,:,np.newaxis]

      metadata = {
        # 'SamplingRate': 20480,
        # 'RotatingSpeed': 33.3,
        'OriginalSplit': mode,
        'FileName': fp.name,
      }

      yield hash(frozenset(metadata.items())), {
          'signal': x,
          'metadata': metadata,
      }
