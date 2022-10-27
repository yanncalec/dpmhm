"""Fraunhofer_151 dataset."""

import os
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
Vibration Measurements on a Rotating Shaft at Different Unbalance Strengths

Description
===========
This dataset contains vibration data recorded on a rotating drive train. This drive train consists of an electronically commutated DC motor and a shaft driven by it, which passes through a roller bearing. With the help of a 3D-printed holder, unbalances with different weights and different radii were attached to the shaft. Besides the strength of the unbalances, the rotation speed of the motor was also varied.

This dataset can be used to develop and test algorithms for the automatic detection of unbalances on drive trains. Datasets for 4 differently sized unbalances and for the unbalance-free case were recorded. The vibration data was recorded at a sampling rate of 4096 values per second. Datasets for development (ID "D[0-4]") as well as for evaluation (ID "E[0-4]") are available for each unbalance strength. The rotation speed was varied between approx. 630 and 2330 RPM in the development datasets and between approx. 1060 and 1900 RPM in the evaluation datasets. For each measurement of the development dataset there are approx. 107min of continuous measurement data available, for each measurement of the evaluation dataset 28min.

Details of the recorded measurements and the used unbalance strengths are documented in the README.md file.

Overview of the dataset components:

|   ID	 |	Radius [mm]  | Mass [g] |
|--------|---------------|----------|
| 0D/ 0E | -		         | -        |
| 1D/ 1E | 14		         | 3.281    |
| 2D/ 2E | 18.5	         | 3.281    |
| 3D/ 3E | 23		         | 3.281    |
| 4D/ 4E | 23		         | 6.614    |

Homepage
--------
https://fordatis.fraunhofer.de/handle/fordatis/151.2

Original Data
=============
Format: CSV files
Date of acquisition: 2020
Channels: voltage(?), measured rpm and 3 vibrations
Split: development and evaluation
Operating conditions: 630~2330 rpm for the development set and 1060~1900 rpm for the evaluation set, continuous record.
Sampling rate: 4096 Hz
Recording duration: 107 minutes for development set and 28 minutes for evaluation set.
Faults: 4 differently sized unbalances and unbalance-free case

Download
--------
https://fordatis.fraunhofer.de/bitstream/fordatis/151.2/1/fraunhofer_eas_dataset_for_unbalance_detection_v1.zip

https://fordatis.fraunhofer.de/bitstream/fordatis/151.2/3/README.md

Processed data
==============
Split: ['train', 'test'].

Features
--------
'signal': {'V_in', 'Measured_RPM', 'Vibration_1', 'Vibration_2', 'Vibration_3'}
'label': ['Normal', 'Unbalanced']
'metadata': {
	'SamplingRate': 4096 Hz,
	'FileName': original file name,
}

Notes
=====
The original record consists of two periods where the rotation speed increases linearly. Their durations are almost the same but depend on the split set. In the processed data these two periods are separated. Moreover, we truncate the few seconds (~10s) near the beginning and the end of each period which seem to correspond to initialization.
"""

_CITATION = """
@article{mey_vibration_2020,
	title = {Vibration {Measurements} on a {Rotating} {Shaft} at {Different} {Unbalance} {Strengths}},
	copyright = {https://creativecommons.org/licenses/by/4.0/},
	url = {https://fordatis.fraunhofer.de/handle/fordatis/151.2},
	doi = {10.24406/fordatis/65.2},
	language = {en},
	urldate = {2022-03-01},
	author = {Mey, Oliver and Neudeck, Willi and Schneider, André and Enge-Rosenblatt, Olaf},
	month = mar,
	year = {2020},
	note = {Accepted: 2020-04-14T10:59:09Z},
}
"""

_DATA_URLS = 'https://fordatis.fraunhofer.de/bitstream/fordatis/151.2/1/fraunhofer_eas_dataset_for_unbalance_detection_v1.zip'


_RADIUS = {'0': 0., '1': 14., '2': 18.5, '3':23., '4':23.}

_MASS = {'0': 0., '1': 3.281, '2': 3.281, '3':3.281, '4':6.614}


class Fraunhofer151(tfds.core.GeneratorBasedBuilder):
	"""DatasetBuilder for Fraunhofer_151 dataset."""

	VERSION = tfds.core.Version('1.0.0')
	RELEASE_NOTES = {
			'1.0.0': 'Initial release.',
	}

	def _info(self) -> tfds.core.DatasetInfo:
		return tfds.core.DatasetInfo(
				builder=self,
				description=_DESCRIPTION,
				features=tfds.features.FeaturesDict({
					# 'signal': tfds.features.Tensor(shape=(5, None), dtype=_DTYPE),

					'signal': {
						'V_in': tfds.features.Tensor(shape=(None,), dtype=_DTYPE),
						'Measured_RPM': tfds.features.Tensor(shape=(None,), dtype=_DTYPE),
						'Vibration_1': tfds.features.Tensor(shape=(None,), dtype=_DTYPE),
						'Vibration_2': tfds.features.Tensor(shape=(None,), dtype=_DTYPE),
						'Vibration_3': tfds.features.Tensor(shape=(None,), dtype=_DTYPE),
					},

					# 	# 'Vibration': tfds.features.Tensor(shape=(3, None), dtype=_DTYPE),

					'label': tfds.features.ClassLabel(names=['Normal', 'Unbalanced']),

					'metadata': {
						'SamplingRate': tf.uint32,
						'LoadRadius': tf.float32,
						'LoadMass': tf.float32,
						'TrunkIndex': tf.uint32,
						'FileName': tf.string,  # Original filename with path in the dataset
					},
				}),
				# If there's a common (input, target) tuple from the
				# features, specify them here. They'll be used if
				# `as_supervised=True` in `builder.as_dataset`.
				# supervised_keys=('signal', 'label'),  # Set to `None` to disable
				supervised_keys=None,
				homepage='https://fordatis.fraunhofer.de/handle/fordatis/151',
				citation=_CITATION,
		)

	def _split_generators(self, dl_manager: tfds.download.DownloadManager):
		if dl_manager._manual_dir.exists():  # prefer to use manually downloaded data
			datadir = Path(dl_manager._manual_dir)
		else:  # automatically download data
			# _resource = tfds.download.Resource(url=_DATA_URLS, extract_method=tfds.download.ExtractMethod.ZIP)  # in case that the extraction method cannot be deduced automatically from files
			# datadir = dl_manager.download_and_extract(_resource)
			raise NotImplementedError()

		return {
				'train': self._generate_examples(datadir, 'train'),
				'test': self._generate_examples(datadir, 'test'),
		}

	def _generate_examples(self, path, split):
		# assert path.exists()

		fpaths = path.glob('*D.csv') if split=='train' else path.glob('*E.csv')

		for fp in fpaths:
			metadata = {
				'SamplingRate': 4096,
				'LoadRadius': _RADIUS[fp.name[0]],
				'LoadMass': _MASS[fp.name[0]],
				'FileName': fp.name
			}
			label = 'Normal' if fp.name[0]=='0' else 'Unbalanced'
			df0 = pd.read_csv(fp)

			# The original signal is truncated into two parts using their starting index and the duration
			sl = 107/2*60-10 if split=='train' else 28//2*60-10

			for t0 in [5, sl+15]:
				# metadata['StartIndex'] = t0
				df = df0.loc[(4096*t0):(4096*(t0+sl))]
				metadata['TrunkIndex'] = 4096*t0

				yield hash(frozenset(metadata.items())), {
					# 'signal': df[['V_in', 'Measured_RPM', 'Vibration_1', 'Vibration_2', 'Vibration_3']].values.astype(_DTYPE.as_numpy_dtype),

					'signal': {
						'V_in': df['V_in'].values.astype(_DTYPE.as_numpy_dtype),
						'Measured_RPM': df['Measured_RPM'].values.astype(_DTYPE.as_numpy_dtype),
						'Vibration_1': df['Vibration_1'].values.astype(_DTYPE.as_numpy_dtype),
						'Vibration_2': df['Vibration_2'].values.astype(_DTYPE.as_numpy_dtype),
						'Vibration_3': df['Vibration_3'].values.astype(_DTYPE.as_numpy_dtype),
						# 'Vibration': df[['Vibration_1', 'Vibration_2', 'Vibration_3']].values.astype(_DTYPE.as_numpy_dtype),
					},

					'label': label,
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
	_all_keys = []
	_all_channels = ['V_in', 'Measured_RPM', 'Vibration_1', 'Vibration_2', 'Vibration_3']

	def compact(self, dataset):
		@tf.function
		def _compact(X):
			d = [X['label']] + [X['metadata'][k] for k in self._keys]

			return {
				'label': tf.py_function(func=self.encode_labels, inp=d, Tout=tf.string),
				'metadata': # X['metadata'],
				{
					'SamplingRate': X['metadata']['SamplingRate'],
					'LoadRadius': X['metadata']['LoadRadius'],
					'LoadMass': X['metadata']['LoadMass'],
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

