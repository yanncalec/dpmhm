"""FEMTO dataset.

Type of experiments: run-to-failure.
"""

import os
from pathlib import Path
# import itertools
# import json
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import pandas as pd
# from scipy.io import loadmat
from datetime import datetime

# from dpmhm.datasets.preprocessing import AbstractDatasetCompactor, AbstractFeatureTransformer, AbstractPreprocessor
from dpmhm.datasets import _DTYPE


_DESCRIPTION = """
FEMTO-ST bearing dataset used in the IEEE PHM 2012 Data Challenge for RUL (remaining useful lifetime) estimation.

Description
===========
This dataset consists of run-to-failure experiments carried on the PRONOSTIA platform. Data provided by this platform corresponds to normally degraded bearings, which means that the defects are not initially initiated on the bearings and that each degraded bearing contains almost all the types of defects (balls, rings and cage).

Data are acquired under three operating conditions (rotating speed and load force):
Condition 1. 1800 rpm and 4000 N: folders Bearing1_x
Condition 2. 1650 rpm and 4200 N: folders Bearing2_x
Condition 3. 1500 rpm and 5000 N: folders Bearing3_x

In order to avoid propagation of damages to the whole test bed (and for security reasons), all tests were stopped when the amplitude of the vibration signal overpassed 20g which is used as definition of the RUL.

Provided data contains the records of two acclerometers and one temperature sensor, and are splitted into the learning set of 6 experiments and the test set (truncated + full) of 11 experiments. The goal is to estimate the RUL on the test set. The learning set was quite small while the spread of the life duration of all bearings was very wide (from 1h to 7h).

Actual RULs (in second) of Test set
-----------------------------------
- Bearing1_3: 5730
- Bearing1_4: 339
- Bearing1_5: 1610
- Bearing1_6: 1460
- Bearing1_7: 7570
- Bearing2_3: 7530
- Bearing2_4: 1390
- Bearing2_5: 3090
- Bearing2_6: 1290
- Bearing2_7: 580
- Bearing3_3: 820

Homepage
--------
https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/#femto

http://www.femto-st.fr/

https://github.com/Lucky-Loek/ieee-phm-2012-data-challenge-dataset

Further Information
-------------------
Monitoring data of the 11 test bearings were truncated so that participants were supposed to predict the remaining life, and thereby perform RUL estimates. Also, no assumption on the type of failure to be occurred was given (nothing known about the nature and the origin of the degradation: balls, inner or outer races, cage...)

Original data
=============
Format: CSV files
Channels: not fixed, up to 3 (2 vibrations and 1 temperature)
	Vibration signals (horizontal and vertical)
		- Sampling frequency: 25.6 kHz
		- Recordings: 2560 (i.e. 1/10 s) are recorded each 10 seconds
	Temperature signals
		- Sampling frequency: 10 Hz
		- Recordings: 600 samples are recorded each minute
Split: ['Learning_set', 'Test_set', 'Full_Test_Set']
Label: None

Download
--------
https://github.com/Lucky-Loek/ieee-phm-2012-data-challenge-dataset

Processed data
==============
Split: ['train', 'test', 'full_test']

Features
--------
'signal': {
	'vibration',
	'temperature'
},

'label': 'Unknown',

'sampling_rate': {
	'vibration': 25600,
	'temperature': 10
},

'metadata': {
	'ID': ['Bearing1','Bearing2','Bearing3']
	'OriginalSplit': ['Learning_set', 'Test_set', 'Full_Test_Set']
	'FileName': Original filename with path
}

Notes
=====
- The split 'test' is the truncation of 'full_test' used for RUL.
- When both being recorded, 'vibration' and 'temperature' correspond to the same experiment and should be understood as two channels. However, being sampled at  different sampling rate they cannot be extracted simultaneously in form of a 3-channels signal.
- The original timestamp of sampling is not regular. We made the choice to not include the timestamp data.
- RUL cannot be extracted due to a strange error occurs, see the comments in the source code.
"""

# Code snippet showing the original timestamp is not regular.
# fname = '.../ieee-phm-2012-data-challenge-dataset-master/Test_set/Bearing1_6/acc_00136.csv'
# df0 = pd.read_csv(fname, header=0, sep=',', index_col=[0,1,2,3])
# ts  = [datetime.datetime(2012,1,1,*np.int32(a)).timestamp() for a in df0.index]
# # the time difference is not regular!
# np.diff(ts) # not constant

_CITATION = """
P. Nectoux, R. Gouriveau, K. Medjaher, E. Ramasso, B. Morello, N. Zerhouni, C. Varnier. PRONOSTIA: An Experimental Platform for Bearings Accelerated Life Test. IEEE International Conference on Prognostics and Health Management, Denver, CO, USA, 2012
"""

_DATA_URLS = 'https://github.com/Lucky-Loek/ieee-phm-2012-data-challenge-dataset/archive/refs/heads/master.zip'


# Date of experiment
_DATE = {
	'Bearing1_1': datetime(2010,12,1),
	'Bearing1_2': datetime(2011,4,6),
	'Bearing1_3': datetime(2010,11,17),
	'Bearing1_4': datetime(2010,12,7),
	'Bearing1_5': datetime(2011,4,13),
	'Bearing1_6': datetime(2011,4,14),
	'Bearing1_7': datetime(2011,4,15),
	'Bearing2_1': datetime(2011,5,6),
	'Bearing2_2': datetime(2011,6,17),
	'Bearing2_3': datetime(2011,5,19),
	'Bearing2_4': datetime(2011,5,26),
	'Bearing2_5': datetime(2011,5,27),
	'Bearing2_6': datetime(2011,6,7),
	'Bearing2_7': datetime(2011,6,8),
	'Bearing3_1': datetime(2011,4,7),
	'Bearing3_2': datetime(2011,6,28),
	'Bearing3_3': datetime(2011,4,8),
}

# Remaining useful life
_RUL = {
	'Bearing1_3': 5730,
	'Bearing1_4': 339,
	'Bearing1_5': 1610,
	'Bearing1_6': 1460,
	'Bearing1_7': 7570,
	'Bearing2_3': 7530,
	'Bearing2_4': 1390,
	'Bearing2_5': 3090,
	'Bearing2_6': 1290,
	'Bearing2_7': 580,
	'Bearing3_3': 820
}

# Load force
_LOAD = {
	'Bearing1': 4000,
	'Bearing2': 4200,
	'Bearing3': 5000,
}

# Nominal RPM
_RPM = {
	'Bearing1': 1800,
	'Bearing2': 1650,
	'Bearing3': 1500,
}


class FEMTO(tfds.core.GeneratorBasedBuilder):
	"""DatasetBuilder for FEMTO dataset."""

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
					'vibration': tfds.features.Tensor(shape=(2, None), dtype=_DTYPE),
					'temperature': tfds.features.Tensor(shape=(None,), dtype=_DTYPE),
				},

				# Timestamp not included
				# 'timestamp': {
				# 	'vibration': tfds.features.Tensor(shape=(None), dtype=tf.float64),
				# 	'temperature': tfds.features.Tensor(shape=(None,), dtype=tf.float64),
				# },

				'label': tfds.features.ClassLabel(names=['Healthy', 'Faulty', 'Unknown']),

				'sampling_rate':{
					'vibration': tf.uint32,
					'temperature': tf.uint32,
				},

				'metadata': {
					# 'SamplingRate': tf.uint32,
					'ID': tf.string,  # ID of the bearing, also its operating conditions
					'RotatingSpeed': tf.uint32,
					'LoadForce': tf.uint32,
					# 'RemainingUsefulLife': tf.float32,  # Time of the run-to-failure experiment
					'OriginalSplit': tf.string,  # Original split
					'FileName': tf.string,  # Original filename with path
				}
			}),
			# If there's a common (input, target) tuple from the
			# features, specify them here. They'll be used if
			# `as_supervised=True` in `builder.as_dataset`.
			# supervised_keys=('signal', 'label'),  # Set to `None` to disable
			supervised_keys=None,
			homepage='',
			citation=_CITATION,
		)

	def _split_generators(self, dl_manager: tfds.download.DownloadManager):
		if dl_manager._manual_dir.exists():  # prefer to use manually downloaded data
			datadir = Path(dl_manager._manual_dir)  # force conversion
			# zipfile = dl_manager._manual_dir / 'femto.zip'

		else:  # automatically download data
			# _resource = tfds.download.Resource(url=_DATA_URLS, extract_method=tfds.download.ExtractMethod.ZIP)  # in case that the extraction method cannot be deduced automatically from files
			# datadir = dl_manager.download_and_extract(_resource)
			raise NotImplementedError()

		return {
			'train': self._generate_examples(datadir, 'Learning_set'),
			'test': self._generate_examples(datadir, 'Test_set'),
			'full_test': self._generate_examples(datadir, 'Full_Test_Set')
		}

	def _generate_examples(self, datadir, split):
		for fp in (datadir/split).rglob('*.csv'):
			fname = fp.parts[-1]

			# Delimiter used by csv files is not uniform: both ',' and ';' are encountered => let pandas detect automatically
			try:
				dm = pd.read_csv(fp, sep=None, header=None, engine='python', index_col=[0,1,2,3])
				assert dm.shape[1] >= 1
			except:
				raise Exception(f'Cannot parse CSV file {fname}')

			if fname[:3] == 'acc':
				_signal = {
					'vibration': dm.iloc[:,-2:].values.T.astype(_DTYPE.as_numpy_dtype), # channel-first
					'temperature': np.array([], dtype=_DTYPE.as_numpy_dtype) #.reshape((1,-1))
				}
				# _timestamp = {
				# }
				_sr = 25600
			elif fname[:4] == 'temp':
				_signal = {
					'vibration': np.array([], dtype=_DTYPE.as_numpy_dtype).reshape((2,-1)), #.astype(_DTYPE.as_numpy_dtype),
					'temperature': dm.iloc[:,-1].values.astype(_DTYPE.as_numpy_dtype) #.reshape((1,-1)),
				}
				_sr = 10
			else:
				continue

			bid = fp.parts[-2]  # bearing experiment id, e.g. 'Bearing1_x'
			gid = bid.split('_')[0]  # bearing group id, e.g. 'Bearing1'
			# A strange error occurs during encoding 'Bearing1_7/acc_00520.csv': RUL is encoded as tuple '(7570,)' but not as number
			#
			# try:
			#   rul = _RUL[bid],
			# except:
			#   rul = np.inf
			#   # rul = -1
			# assert isinstance(rul, float), f"{rul}, {fp}"

			metadata = {
				# 'SamplingRate': _sr,
				'ID': bid,
				'RotatingSpeed': _RPM[gid],
				'LoadForce': _LOAD[gid],
				# 'RemainingUsefulLife': rul,
				'OriginalSplit': split,
				'FileName': os.path.join(*fp.parts[-2:])  # full path file name
			}

			yield hash(frozenset(metadata.items())), {
				'signal': _signal,
				'sampling_rate': {
					'vibration': 25600,
					'temperature': 10
				},
				'label': 'Unknown',
				'metadata': metadata
			}

	@staticmethod
	def get_references():
		try:
			with open(Path(__file__).parent / 'Exported Items.bib') as fp:
				return fp.read()
		except:
			pass


# class DatasetCompactor(AbstractDatasetCompactor):
# 	_all_keys = ['ID']  # the fields 'RotatingSpeed' and 'LoadForce' are redundant.
# 	_all_channels = ['vibration', 'temperature']  # cannot be extracted at the same time

# 	@classmethod
# 	def get_label_dict(cls, dataset, keys:list):
# 		compactor = cls(dataset, keys=keys, channels=['vibration'])
# 		return compactor.label_dict

# 	def compact(self, dataset):
# 		@tf.function  # necessary for infering the size of tensor
# 		def _has_channels(X):
# 			"""Test if channels are present in data."""
# 			flag = True
# 			for ch in self._channels:
# 				if tf.size(X['signal'][ch]) == 0:
# 					flag = False
# 					# break or return False are not supported by tensorflow
# 			return flag

# 		@tf.function
# 		def _compact(X):
# 			d = [X['metadata'][k] for k in self._keys]

# 			return {
# 				'label': tf.py_function(func=self.encode_labels, inp=d, Tout=tf.string),
# 				'metadata': {
# 					'RPM_Nominal': X['metadata']['RotatingSpeed'],
# 					'LoadForce': X['metadata']['LoadForce'],
# 					'SamplingRate': X['metadata']['SamplingRate'],
# 					'FileName': X['metadata']['FileName'],
# 				},
# 				'signal': [X['signal'][ch] for ch in self._channels],
# 			}
# 		# return dataset.filter(_has_channels)
# 		ds = dataset.filter(lambda X:_has_channels(X))
# 		return ds.map(lambda X:_compact(X), num_parallel_calls=tf.data.AUTOTUNE)


# class FeatureTransformer(AbstractFeatureTransformer):
# 	"""Feature transform for CWRU dataset.
# 	"""
# 	@classmethod
# 	def get_output_signature(cls, tensor_shape:tuple=None):
# 		return {
# 			'label': tf.TensorSpec(shape=(), dtype=tf.string),
# 			'metadata': {
# 				'RPM_Nominal': tf.TensorSpec(shape=(), dtype=tf.uint32),  # real rpm
# 				'LoadForce': tf.TensorSpec(shape=(), dtype=tf.uint32),
# 				'SamplingRate': tf.TensorSpec(shape=(), dtype=tf.uint32),
# 				'FileName': tf.TensorSpec(shape=(), dtype=tf.string),  # filename
# 			},
# 			# 'feature': tf.TensorSpec(shape=(None, None, None), dtype=tf.float64, name='feature'),
# 			'feature': tf.TensorSpec(shape=tf.TensorShape(tensor_shape), dtype=_DTYPE),
# 		}


# class Preprocessor(AbstractPreprocessor):
# 	pass


# __all__ = ['DatasetCompactor', 'FeatureTransformer', 'Preprocessor']

