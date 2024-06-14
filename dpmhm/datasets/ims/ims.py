"""IMS dataset.

Test-to-failure experiments on bearings. The data set was provided by the Center for Intelligent Maintenance Systems (IMS), University of Cincinnati.

Description
===========
An AC motor, coupled by a rub belt, keeps the rotation speed constant. The four bearings are in the same shaft and are forced lubricated by a circulation system that regulates the flow and the temperature. It is announced on the provided “Readme Document for IMS Bearing Data” in the downloaded file, that the test was stopped when the accumulation of debris on a magnetic plug exceeded a certain level indicating the possibility of an impending failure. The four bearings are all of the same type. There are double range pillow blocks rolling elements bearing.

Three data sets are included in the data packet. Each data set describes a test-to-failure experiment. Each data set consists of individual files that are 1-second vibration signal snapshots recorded at specific intervals. Each file consists of 20,480 points with the sampling rate set at 20 kHz. The file name indicates when the data was collected. Each record (row) in the data file is a data point. Data collection was facilitated by NI DAQ Card 6062E. Larger intervals of time stamps (showed in file names) indicate resumption of the experiment in the next working day.

For more details, see the description pdf file included in the downloaded data.

Homepage
--------
https://www.nasa.gov/content/prognostics-center-of-excellence-data-set-repository

https://data.phmsociety.org/nasa/

https://phm-datasets.s3.amazonaws.com/NASA/4.+Bearings.zip

https://www.kaggle.com/datasets/vinayak123tyagi/bearing-dataset

Original Dataset
================
- Type of experiments: run-to-failure
- Year of acquisition: 2003-2004
- Format: text
- Sampling rate: 20480 Hz
- Channels: 8 for set 1 (2 per bearing), 4 for set 2, 4 for set 3 (1 per bearing)
- Recording duration: ~ 1 second
- Size: ~ 6.1 Gb, unzipped

Notes
=====
- The original dataset contains a folder named `4th_test`, which corresponds actually to the 3rd test. This has been modified in the zip file.
"""


# import os
from pathlib import Path
# import itertools
# import json
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import pandas as pd
# from scipy.io import loadmat

from dpmhm.datasets import _DTYPE, _ENCODING


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

# _DATA_URLS = 'https://phm-datasets.s3.amazonaws.com/NASA/4.+Bearings.zip'
_DATA_URLS = [
    # 'https://sandbox.zenodo.org/record/1184320/files/ims.zip'
    'https://zenodo.org/records/11545355/files/ims.zip?download=1'
    ]

_CITATION = """
- Hai Qiu, Jay Lee, Jing Lin. “Wavelet Filter-based Weak Signature Detection Method and its Application on Roller Bearing Prognostics.” Journal of Sound and Vibration 289 (2006) 1066-1090
- J. Lee, H. Qiu, G. Yu, J. Lin, and Rexnord Technical Services (2007). IMS, University of Cincinnati. "Bearing Data Set", NASA Ames Prognostics Data Repository (http://ti.arc.nasa.gov/project/prognostic-data-repository), NASA Ames Research Center, Moffett Field, CA
"""


class IMS(tfds.core.GeneratorBasedBuilder):
    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
        '1.0.0': 'Initial release.',
    }

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""
        return tfds.core.DatasetInfo(
            builder=self,
            description=__doc__,
            features=tfds.features.FeaturesDict({
                'signal': {
                    # tfds.features.Tensor(shape=(None,None,1), encoding='zlib', dtype=tf.float64),
                    'B1C1': tfds.features.Tensor(shape=(None,), dtype=_DTYPE, encoding=_ENCODING),
                    'B1C2': tfds.features.Tensor(shape=(None,), dtype=_DTYPE, encoding=_ENCODING),
                    'B2C1': tfds.features.Tensor(shape=(None,), dtype=_DTYPE, encoding=_ENCODING),
                    'B2C2': tfds.features.Tensor(shape=(None,), dtype=_DTYPE, encoding=_ENCODING),
                    'B3C1': tfds.features.Tensor(shape=(None,), dtype=_DTYPE, encoding=_ENCODING),
                    'B3C2': tfds.features.Tensor(shape=(None,), dtype=_DTYPE, encoding=_ENCODING),
                    'B4C1': tfds.features.Tensor(shape=(None,), dtype=_DTYPE, encoding=_ENCODING),
                    'B4C2': tfds.features.Tensor(shape=(None,), dtype=_DTYPE, encoding=_ENCODING),
                },

                'sampling_rate': tf.uint32,

                'metadata': {
                    # 'RotatingSpeed': tf.float32,
                    'OriginalSplit': tf.string,
                    'FileName': tf.string,
                    'Dataset': tf.string,
                },
            }),
            supervised_keys=None,
            homepage='https://data.phmsociety.org/nasa/',
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        def _get_split_dict(datadir):
            return {
                'dataset1': next(datadir.rglob('1st_test')).glob('*'),
                'dataset2': next(datadir.rglob('2nd_test')).glob('*'),
                'dataset3': next(datadir.rglob('3rd_test')).glob('*'),
                # 'dataset2': (datadir/'2nd_test').glob('*'),
                # 'dataset3': (datadir/'3rd_test').glob('*'),
            }

        if dl_manager._manual_dir.exists():  # prefer to use manually downloaded data
            datadir = Path(dl_manager._manual_dir)
        elif dl_manager._extract_dir.exists(): # automatically download & extracted data
            datadir = Path(dl_manager._extract_dir)
        else:
            raise FileNotFoundError()

        return {sp: self._generate_examples(files, sp) for sp, files in _get_split_dict(datadir).items()}

    def _generate_examples(self, files, split):
        for fp in files:
            x = pd.read_csv(fp, sep='\t',header=None).values.astype(_DTYPE) #[:,:,np.newaxis]
            if x.shape[1] == 4:
                xd = {
                    'B1C1': x[:,0],
                    # 'B1C2': np.empty(0).astype(_DTYPE),
                    'B1C2': [],
                    'B2C1': x[:,1],
                    'B2C2': [],
                    'B3C1': x[:,2],
                    'B3C2': [],
                    'B4C1': x[:,3],
                    'B4C2': [],
                }
            else:
                xd = {
                    'B1C1': x[:,0],
                    'B1C2': x[:,1],
                    'B2C1': x[:,2],
                    'B2C2': x[:,3],
                    'B3C1': x[:,4],
                    'B3C2': x[:,5],
                    'B4C1': x[:,6],
                    'B4C2': x[:,7],
                }

            metadata = {
                # 'RotatingSpeed': 33.3,
                'OriginalSplit': split,
                'FileName': fp.name,
                'Dataset': 'IMS',
            }

            yield hash(frozenset(metadata.items())), {
                'signal': xd,
                'sampling_rate': 20480,
                # 'label': 'Unknown',
                'metadata': metadata,
            }
