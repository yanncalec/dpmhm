"""
Case Western Reserve University bearing dataset.

Description
===========
Motor bearings were seeded with faults using electro-discharge machining (EDM). Faults ranging from 0.007 inches in diameter to 0.040 inches in diameter were introduced separately at the inner raceway, rolling element (i.e. ball) and outer raceway. Faulted bearings were reinstalled into the test motor and vibration data was recorded for motor loads of 0 to 3 horsepower (motor speeds of 1797 to 1720 RPM).

Further Information
-------------------
Two possible locations of bearings in the test rig: namely the drive-end and the fan-end.

For one location, a fault of certain diameter is introduced at one of the three components: inner race, ball, outer race. The inner race and the ball are rolling elements thus the fault does not have a fixed position, while the outer race is a fixed frame and the fault has three different angular positions: 3, 6 and 12.

The vibration signals are recorded by an accelerometer at (one or more of) the drive-end (DE), fan-end (FE) or base (BA) positions at the sampling rate 12 or 48 kHz, for a certain motor load.

Homepage
--------
https://engineering.case.edu/bearingdatacenter

Original Dataset
================
- Type of experiments: initiated faults
- Year of acquisition: 2000
- Format: Matlab
- Size: ~ 656 Mb, unzipped
- Channels: up to 3 accelerometers, named Drive-End (DE), Fan-End (FE), Base (BA).
- Split: not specified.
- Sampling rate: 12 kHz or 48 kHz, not fixed.
- Recording duration: ~ 10 seconds, not fixed.
- Label: 'Normal', 'Drive-end fault', 'Fan-end fault'.
- Data fields:
    - DE: drive end accelerometer data, contained by all files.
    - FE: fan end accelerometer data, contained by most files except those with name '300x'.
    - BA: base accelerometer data, contained by some files, in particular not those of normal data (faultless).
    - RPM: real rpm during testing, contained by most files.

Built Dataset
=============
- Split: ['train']

Features
--------
- 'signal':
    - 'BA': base accelerometer data.
    - 'DE': drive end accelerometer data.
    - 'FE': fan end accelerometer data.
- 'sampling_rate': 12 or 48 kHz.
- 'metadata':
    - 'NominalRPM': nominal RPM.
    - 'RPM': real RPM.
    - 'LoadForce': {0, 1, 2, 3} nominal motor load in horsepower, corresponding to the nominal RPM: {1797, 1772, 1750, 1730}.
    - 'FaultLocation': {'None', 'DriveEnd', 'FanEnd'}. Note that this is the location of fault not that of accelerometer.
    - 'FaultComponent': {'None', 'InnerRace', 'Ball', 'OuterRace6', 'OuterRace3', 'OuterRace12'}, component of fault.
    - 'FaultSize': {0, 0.007, 0.014, 0.021, 0.028}, diameter of fault in inches.
    - 'FileName': original filename.

Notes
=====
- Experiment with normal operating condition can be identified by any of the following clauses: `FaultLocation==None`, `FaultComponent==None`, `FaultSize==0`.
- The value of 'RPM' (real RPM) is available for most experiments, otherwise it is replaced by the nominal value.

References
==========
Review of developments based on CWRU:

- Wei, X. and Söffker, D. (2021) ‘Comparison of CWRU Dataset-Based Diagnosis Approaches: Review of Best Approaches and Results’, in P. Rizzo and A. Milazzo (eds) European Workshop on Structural Health Monitoring. Cham: Springer International Publishing (Lecture Notes in Civil Engineering), pp. 525–532. Available at: https://doi.org/10.1007/978-3-030-64594-6_51.
"""

# import os
from pathlib import Path
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import pandas as pd
# import scipy
# import mat4py
# import librosa

from dpmhm.datasets import _DTYPE, _ENCODING, extract_zenodo_urls

_ZIP_ENABLED = True  # The builder can iterate directly over zip files

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
# Use the option `keep_default_na` to preserving `None` (for e.g. 99.mat) as a string and not converting it to nan.
# https://stackoverflow.com/questions/10867028/get-pandas-read-csv-to-read-empty-values-as-empty-string-instead-of-nan
_METAINFO = pd.read_csv(Path(__file__).parent / 'metainfo.csv', keep_default_na=False)

# URL to the zip file
# _DATA_URLS = ('https://engineering.case.edu/sites/default/files/'+_METAINFO['FileName']).tolist()
_DATA_URLS = [
    'https://zenodo.org/records/7457149/files/cwru.zip?download=1'
    ]


class CWRU(tfds.core.GeneratorBasedBuilder):
    VERSION = tfds.core.Version('1.0.0')

    RELEASE_NOTES = {
            '1.0.0': 'Initial release.',
    }

    def _info(self) -> tfds.core.DatasetInfo:
        return tfds.core.DatasetInfo(
            builder=self,
            description=__doc__,
            features=tfds.features.FeaturesDict({
                # The sampling rate and the shape of a signal are not fixed.
                'signal':{  # three possible channels: drive-end, fan-end and base
                    'DE': tfds.features.Tensor(shape=(None,), dtype=_DTYPE, encoding=_ENCODING),
                    'FE': tfds.features.Tensor(shape=(None,), dtype=_DTYPE, encoding=_ENCODING),
                    'BA': tfds.features.Tensor(shape=(None,), dtype=_DTYPE, encoding=_ENCODING),
                },
                # Can not save all channels in a tensor with unknown dimension, like
                # 'signal': tfds.features.Tensor(shape=(None,1), dtype=tf.float64),

                # 'label': tfds.features.ClassLabel(names=['None']),  # not used

                'sampling_rate': tf.uint32,  # {12000, 48000} Hz

                'metadata': {
                    'NominalRPM': tf.uint32,  # nominal RPM: [1797, 1772, 1750, 1730]
                    'RPM': tf.uint32,  # real RPM
                    'LoadForce': tf.uint32, # {0, 1, 2 ,3}, corresponding nominal RPM: [1797, 1772, 1750, 1730]
                    'FaultLocation': tf.string, # {'DriveEnd', 'FanEnd', 'None'}
                    'FaultComponent': tf.string,  # {'InnerRace', 'Ball', 'OuterRace3', 'OuterRace6', 'OuterRace12', 'None'}
                    'FaultSize': tf.float32,  # {0.007, 0.014, 0.021, 0.028, 0}
                    'FileName': tf.string,
                    'Dataset': tf.string,
                },

                # Another possibility is to use class labels (string),
                # e.g.
                # 'metadata': {
                #     'LoadForce': tfds.features.ClassLabel(names=['0', '1', '2', '3']),  # Horsepower
                # },
                # This implies conversion in the decoding part.
            }),
            supervised_keys=None,
            # supervised_keys=('signal', 'label'),  # Set to `None` to disable
            homepage='https://engineering.case.edu/bearingdatacenter',
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        def _get_split_dict(datadir):
            return {
                'train': datadir.rglob('*.mat'),
            }

        if dl_manager._manual_dir.exists():  # prefer to use manually downloaded data
            datadir = Path(dl_manager._manual_dir)
        elif dl_manager._extract_dir.exists(): # automatically downloaded & extracted data
            datadir = Path(dl_manager._extract_dir)
        # elif dl_manager._download_dir.exists(): # automatically downloaded data
        #     datadir = Path(dl_manager._download_dir)
        #     tfds.download.iter_archive(fp, tfds.download.ExtractMethod.ZIP)
        else:
            raise FileNotFoundError()

        return {sp: self._generate_examples(files) for sp, files in _get_split_dict(datadir).items()}

    def _generate_examples(self, files):
        for fp in files:
            fname = fp.parts[-1]
            # print(fname)
            metadata = _METAINFO.loc[_METAINFO['FileName'] == fname].iloc[0].to_dict()

            try:
                dm = {k: np.asarray(v).squeeze() for k,v in tfds.core.lazy_imports.scipy.io.loadmat(fp).items() if k[:2]!='__'}
                # using mat4py.loadmat
                # dm = {k:np.asarray(v).squeeze() for k,v in mat4py.loadmat(fp).items()}
            except Exception as msg:
                raise Exception(f"Error in processing {fp}: {msg}")
                # continue

            # If filename is e.g. `112.mat`, then the matlab file  contains the fields
            # - `X112_DE_time` (always),
            # - `X112RPM`, `X112_FE_time` and `X112_BA_time` (not aways).
            # Exception: the files `300?.mat`` do not contain fields of name starting with `X300?` nor the field `RPM`.

            fn = f"{int(metadata['FileName'].split('.mat')[0]):03d}"
            if f'X{fn}_DE_time' in dm:  # find regular fields
                kde = f'X{fn}_DE_time'
                # xde = dm[kde].squeeze()
                xde = dm[kde]
                kfe = f'X{fn}_FE_time'
                # xfe = dm[kfe].squeeze() if kfe in dm else []
                xfe = dm[kfe] if kfe in dm else np.empty(0)
                kba = f'X{fn}_BA_time'
                # xba = dm[kba].squeeze() if kba in dm else []
                xba = dm[kba] if kba in dm else np.empty(0)
                krpm = f'X{fn}RPM'
            else:   # datafiles 300x.mat are special
                # print(dm.keys())
                kde = [k for k in dm.keys() if k[0]=='X'][0]
                # xde, xfe, xba = dm[kde].squeeze(), [], []
                xde, xfe, xba = dm[kde], np.empty(0), np.empty(0)

            # Use nominal value if the real value is not given.
            metadata['RPM'] = int(dm[krpm]) if krpm in dm else metadata['NominalRPM']
            metadata['Dataset'] = 'CWRU'

            sr = metadata.pop('SamplingRate')  # pop out the field from metadata

            yield hash(frozenset(metadata.items())), {
                'signal': {
                    # 'DE': xde.astype(_DTYPE.as_numpy_dtype),
                    # 'FE': xfe.astype(_DTYPE.as_numpy_dtype),
                    # 'BA': xba.astype(_DTYPE.as_numpy_dtype),
                    # tfds (>=4.9.2) now accepts Numpy's DType
                    'DE': xde.astype(_DTYPE),
                    'FE': xfe.astype(_DTYPE),
                    'BA': xba.astype(_DTYPE),
                },  # if not empty all three have the same length
                'sampling_rate': sr,
                # 'label': metadata['FaultLocation'],
                'metadata': metadata,
            }
