"""Fraunhofer_205 dataset."""

# import os
from pathlib import Path
# import itertools
# import json
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import pandas as pd
# from scipy.io import loadmat

from dpmhm.datasets.preprocessing import AbstractDatasetCompactor, AbstractFeatureTransformer, AbstractPreprocessor
from dpmhm.datasets import _DTYPE


_DESCRIPTION = """
Condition Monitoring of Drive Trains by Data Fusion of Acoustic Emission and Vibration Sensors: Measurements of Vibrations and Acoustic Emissions on a Rotating Shaft.

Description
===========
The dataset contains the measured raw data (data/<measurement>/<sensor>.csv) as well as metadata that describes the vibration and acoustic emission sensors (sensors/<sensor>.json) and the configuration of all measurements (measurements/<measurement>.json). A total of 29 measurements are included. For each measurement the the file vb.csv contains the vibration data: a timestamp in the first column followed by 8192 sensor values. It is sampled with a sampling rate of 8192 Hz. One line thus corresponds to a measurement time of one second. The ae.csv files with the acoustic emission data also contain a timestamp in the first column followed by 8000 sensor values. It is sampled at a sampling rate of 390625Hz. One line therefore corresponds to a measurement time of 20.48 ms. A file w.csv with the speeds was also recorded for each measurement. These files contain the time frames (begin and end timestamp) for the five rotational speeds: 600rpm, 1000rpm, 1400rpm, 1800rpm, 2200rpm). While the measurement for the vibration and the acoustic emission was carried out continuously for the different speeds, the phases with an almost constant speed can be extracted with the help of the time ranges based on the w.csv files.

Homepage
--------
https://fordatis.fraunhofer.de/handle/fordatis/205

https://github.com/deepinsights-analytica/mdpi-arci2021-paper

Original Data
=============
Format: CSV files
Date of acquisition: 2021
Channels: vibration and acoustic emission
Split: not specified
Operating conditions: five rotational speeds at 600rpm, 1000rpm, 1400rpm, 1800rpm, 2200rpm recorded continuously.
Faults: Artificial damages of two categories (small and large) induced by electrical discharge machining (EDM) at the inner race, the outer race, and the rolling elements.

Vibration signal
----------------
Sampling rate: 8192 Hz
Recording duration: 1 second, 8192 samples
Recording period: 1 second

Acoustic emission signal
------------------------
Sampling rate: 390625 Hz
Recording duration: 20.48 milliseconds, 8000 samples
Recording period: ~ 41 milliseconds

Download
--------
https://fordatis.fraunhofer.de/bitstream/fordatis/205/1/fraunhofer_iis_eas_dataset_vibrations_acoustic_emissions_of_drive_train_v1.zip

Notes
=====
- In the processed data the records are separated by their RPM.
"""

_CITATION = """
@article{mey_condition_2021,
	title = {Condition {Monitoring} of {Drive} {Trains} by {Data} {Fusion} of {Acoustic} {Emission} and {Vibration} {Sensors}},
	volume = {9},
	copyright = {http://creativecommons.org/licenses/by/3.0/},
	issn = {2227-9717},
	url = {https://www.mdpi.com/2227-9717/9/7/1108},
	doi = {10.3390/pr9071108},
	language = {en},
	number = {7},
	urldate = {2022-02-28},
	journal = {Processes},
	author = {Mey, Oliver and Schneider, André and Enge-Rosenblatt, Olaf and Mayer, Dirk and Schmidt, Christian and Klein, Samuel and Herrmann, Hans-Georg},
	month = jul,
	year = {2021},
	note = {Number: 7
Publisher: Multidisciplinary Digital Publishing Institute},
	keywords = {acoustic emission, condition monitoring, data fusion, drive train, machine learning, vibration},
	pages = {1108},
}
"""

_SPEED = [600, 1000, 1400, 1800, 2200]

_COMPONENT = ['Ball', 'InnerRace', 'OuterRace', 'None']

_METAINFO = pd.read_csv(Path(__file__).parent/'metainfo.csv', index_col=0)

_DATA_URLS = 'https://fordatis.fraunhofer.de/bitstream/fordatis/205/1/fraunhofer_iis_eas_dataset_vibrations_acoustic_emissions_of_drive_train_v1.zip'


class Fraunhofer205(tfds.core.GeneratorBasedBuilder):
	"""DatasetBuilder for Fraunhofer_205 dataset."""

	VERSION = tfds.core.Version('1.0.0')
	RELEASE_NOTES = {
			'1.0.0': 'Initial release.',
	}

	def _info(self) -> tfds.core.DatasetInfo:
		return tfds.core.DatasetInfo(
				builder=self,
				description=_DESCRIPTION,
				features=tfds.features.FeaturesDict({
					# Number of channels is named or not fixed
					'signal': {
						'Vibration': tfds.features.Tensor(shape=(None,), dtype=_DTYPE),
						'AcousticEmission': tfds.features.Tensor(shape=(None,), dtype=_DTYPE),
					},

					'label': tfds.features.ClassLabel(names=_COMPONENT),

					'metadata': {
						'SamplingRate': tf.uint32,
						'RotatingSpeed': tf.uint32,  # Rotation speed of the shaft
						'FaultComponent': tf.string, # Component of the fault, e.g. {'Ball', 'Cage' ,'InnerRace', 'OuterRace', 'None}
						'FaultExtend': tf.uint32,  # Extend of the fault, [0,1,2]
						'FileName': tf.string,  # Original filename with path in the dataset
					},
				}),
				# If there's a common (input, target) tuple from the
				# features, specify them here. They'll be used if
				# `as_supervised=True` in `builder.as_dataset`.
				# supervised_keys=('signal', 'label'),  # Set to `None` to disable
				supervised_keys=None,
				homepage='https://fordatis.fraunhofer.de/handle/fordatis/205',
				citation=_CITATION,
		)

	def _split_generators(self, dl_manager: tfds.download.DownloadManager):
		if dl_manager._manual_dir.exists():  # prefer to use manually downloaded data
			datadir = dl_manager._manual_dir
		else:  # automatically download data
			# _resource = tfds.download.Resource(url=_DATA_URLS, extract_method=tfds.download.ExtractMethod.ZIP)  # in case that the extraction method cannot be deduced automatically from files
			# datadir = dl_manager.download_and_extract(_resource)
			raise NotImplementedError()

		return {
				# cp.lower(): self._generate_examples(datadir) for cp in _COMPONENT
				'train': self._generate_examples(datadir/'data'),
		}

	def _generate_examples(self, path):
		for fp in path.iterdir():
			# print(fp)
			dv = pd.read_csv(fp/'vb.csv', header=None, sep=' ', index_col=0).astype(np.float64)
			da = pd.read_csv(fp/'ae.csv', header=None, sep=' ', index_col=0).astype(np.float64)
			dw = pd.read_csv(fp/'w.csv', header=None, sep=' ', index_col=0)

			# dv.index = pd.DatetimeIndex(dv.index)
			# dv = dv.resample('1s').mean()
			# da.index = pd.DatetimeIndex(da.index)
			# da = da.resample(pd.Timedelta('20480us')).mean()

			for rpm in _SPEED:
				t0,t1 = dw.index[dw[1]==rpm]  # time range of the constant RPM
				xvb = dv.loc[t0:t1]
				xae = da.loc[t0:t1]

				metadata = _METAINFO.loc[fp.name].to_dict()
				metadata['RotatingSpeed'] = rpm
				metadata['FileName'] = fp.name

				yield hash(frozenset(metadata.items())), {
					'signal': {
						'Vibration': dv.loc[t0:t1].values.astype(_DTYPE.as_numpy_dtype),
						'AcousticEmission': da.loc[t0:t1].values.astype(_DTYPE.as_numpy_dtype),
					},
					'label': metadata['FaultComponent'],
					'metadata': metadata
				}

	@staticmethod
	def get_references():
		try:
			with open(Path(__file__).parent / 'Exported Items.bib') as fp:
				return fp.read()
		except:
			pass


class DatasetCompactor(AbstractDatasetCompactor):
	_all_keys = ['FaultExtend']  # FaultComponent is redundant with the label
	_all_channels = ['Vibration', 'AcousticEmission']

	def compact(self, dataset):
		@tf.function
		def _compact(X):
			d = [X['label']] + [X['metadata'][k] for k in self._keys]

			return {
				'label': tf.py_function(func=self.encode_labels, inp=d, Tout=tf.string),
				'metadata': # X['metadata'],
				{
					'SamplingRate': X['metadata']['SamplingRate'],
					'Fault': X['metadata']['LoadMass'],
					'FileName': X['metadata']['FileName'],
				},
				'signal': [X['signal'][ch] for ch in self._channels],
			}
		return dataset.map(lambda X:_compact(X), num_parallel_calls=tf.data.AUTOTUNE)


class FeatureTransformer(AbstractFeatureTransformer):
	@classmethod
	def get_output_signature(cls, tensor_shape:tuple=None):
		return {
			'label': tf.TensorSpec(shape=(), dtype=tf.string),
			'metadata': {
				'SamplingRate': tf.TensorSpec(shape=(), dtype=tf.uint32),
				'LoadRadius': tf.TensorSpec(shape=(), dtype=tf.float32),
				'LoadMass': tf.TensorSpec(shape=(), dtype=tf.float32),
				'FileName': tf.TensorSpec(shape=(), dtype=tf.string),  # filename
			},
			'feature': tf.TensorSpec(shape=tf.TensorShape(tensor_shape), dtype=_DTYPE),
		}


class Preprocessor(AbstractPreprocessor):
	pass


__all__ = ['DatasetCompactor', 'FeatureTransformer', 'Preprocessor']

