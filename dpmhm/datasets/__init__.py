"""Collection of open source datasets.
"""

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import os
from pathlib import Path
import requests
from tensorflow.data import Dataset

# import pycurl
# from io import BytesIO
# from zipfile import ZipFile

import logging
Logger = logging.getLogger(__name__)

from .. import cli

# Data type
_FLOAT16 = np.float16
_FLOAT32 = np.float32
_FLOAT64 = np.float64
_UINT32 = np.uint32
_STRING = tf.string

try:
    _DTYPE = tf.as_dtype(os.environ['DPMHM_DTYPE']).as_numpy_dtype
except:
    _DTYPE = tf.float32.as_numpy_dtype

# Encoding length for class label
try:
    _ENCLEN = int(os.environ['DPMHM_ENCLEN'])
except:
    _ENCLEN = 16

try:
    _ENCODING = os.environ['DPMHM_ENCODING']
    assert _ENCODING in {'zlib', 'bytes', 'none'}
except:
    _ENCODING = 'none'
    # _ENCODING = tfds.features.Encoding.NONE

# Location of tfds datasets
try:
    TFDS_DATA_DIR = Path(os.environ['TFDS_DATA_DIR'])
except:
    TFDS_DATA_DIR = Path(os.path.expanduser('~/tensorflow_datasets'))


def get_dataset_list():
    """Get the full list of supported datasets.
    """
    return list(cli._DATASET_DICT.values())


def install(ds:str, *, data_dir:str=None, download_dir:str=None, extract_dir:str=None, manual_dir:str=None, dl_kwargs:dict={}, **kwargs) -> Dataset:
    """Install a dataset.

    Parameters
    ----------
    ds
        name of the dataset to be installed.
    data_dir
        location of tensorflow datasets, by default the environment variable `TFDS_DATA_DIR.
    download_dir
        location of download folder, by default `data_dir/dpmhm/downloads/ds`.
    extract_dir
        location of extraction folder, by default `data_dir/dpmhm/extracted/ds`.
    manual_dir
        location of manually downloaded & extracted files.
    dl_kwargs
        keyword arguments for `tfds.download.DownloadManager()`
    kwargs
        other keyword arguments to `tfds.load()`

    Returns
    -------
    a `tf.data.Dataset` object.

    Notes
    -----
    Once installed, the dataset can be reloaded using `tfds.load()`.
    """
    # register the dataset in the namespace
    cli.import_dataset_module(ds)

    dsl = ds.lower()
    dataset_name = cli._DATASET_DICT[dsl]
    data_dir = TFDS_DATA_DIR if data_dir is None else Path(data_dir)

    # download & extract only if manual files not provided
    if manual_dir is None:
        download_dir = data_dir / 'dpmhm' / 'downloads'/ dsl if download_dir is None else Path(download_dir) / dsl
        extract_dir = data_dir / 'dpmhm' / 'extracted' / dsl if extract_dir is None else Path(extract_dir) / dsl

        dl_manager = tfds.download.DownloadManager(
            dataset_name=dataset_name,
            download_dir=download_dir,
            extract_dir=extract_dir,
            manual_dir=manual_dir,
            # max_simultaneous_downloads=2,
            **dl_kwargs
        )

        Logger.debug('Downloading data files...')
        _ = dl_manager.download_and_extract(cli.get_urls(dsl))
    else:
        assert os.path.exists(manual_dir)

    Logger.debug('Building the dataset...')

    return tfds.load(
        dataset_name,
        data_dir=data_dir,
        download_and_prepare_kwargs = {
            'download_dir': download_dir,  # currently not used by `_split_generators()`
            'download_config': tfds.download.DownloadConfig(
                extract_dir=extract_dir,
                manual_dir=manual_dir,
                # download_mode=tfds.GenerateMode.REUSE_DATASET_IF_EXISTS
            )
        },
        **kwargs
    )


def extract_zenodo_urls(url:str) -> list:
    """Extract from a Zenodo page the urls containing downloadable files.

    Parameters
    ----------
    url
        url of a Zenodo page, e.g. https://zenodo.org/record/3727685/ or https://sandbox.zenodo.org/record/1183527/

    Returns
    -------
    a list of extracted urls.
    """
    from bs4 import BeautifulSoup

    header = url.split('/record/')[0]
    # Logger.debug(header)
    reqs = requests.get(url)
    soup = BeautifulSoup(reqs.text, 'html.parser')

    urls = []
    for link in soup.find_all('a'):
        s = link.get('href')
#         urls.append(s)
#         print(s)
        try:
            if '?download=1' in s:
                urls.append(header+'/'+s.split('?download=1')[0])
        except:
            pass
    return urls


def query_parameters(ds_name:str) -> dict:
    """Query the parameters of a dataset.
    """

    parms = {
        'signal':dict(),
        'sampling_rate': None,
        'keys':dict(),
        'filters':dict(),
        'type':None,
        }

    match ds_name.upper():
        case 'CWRU':
            parms['signal'] = {'DE':1, 'FE':1, 'BA':1}
            parms['split'] = ['train']
            parms['sampling_rate'] = [12000, 48000]
            parms['keys'] = {
                    'FaultLocation': {'None', 'DriveEnd', 'FanEnd'},
                    'FaultComponent': {'None', 'InnerRace', 'Ball', 'OuterRace6', 'OuterRace3', 'OuterRace12'},
                    'FaultSize': {0, 0.007, 0.014, 0.021, 0.028}
                }
            parms['filters'] = {'LoadForce': {0, 1, 2, 3}}
            parms['type'] = 'initiated'
        case 'DIRG':
            parms['signal'] = {'A1':3, 'A2':3}
            parms['split'] = ['vibration', 'endurance']
            parms['sampling_rate'] = [51200, 102400]
            parms['keys'] = {
                    'FaultComponent': {'Roller', 'InnerRing'},
                    'FaultSize': {0, 150, 250, 450}
                }
            parms['filters'] = {
                'RotatingSpeed': {100, 200, 300, 400, 500},
                'NominalLoadForce': {0, 1000, 1400, 1800}
            }
            parms['type'] = 'initiated+failure'
        case 'FEMTO':
            parms['signal'] = {'vibration':2, 'temperature':1}
            parms['split'] = ['train', 'test', 'full_test']
            parms['sampling_rate'] = {'vibration':25600, 'temperature':10}
            parms['keys'] = {
                'ID': {
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
                    'Bearing3_3': 820,
                }
            }
            parms['filters'] = {
                'RotatingSpeed': {1800, 1650, 1500},
            }
            parms['type'] = 'failure'
        case 'FRAUNHOFER151':
            parms['signal'] = {
                'Measured_RPM': 1,
                'V_in': 1,
                'Vibration': 3
            }
            parms['split'] = ['train']
            parms['sampling_rate'] = [4096]
            parms['keys'] = {
                # 'FileName': {'?D.csv', '?E.csv'},
                'Label': {'Normal', 'Unbalanced'},
                'LoadMass': {3.281, 6.614},
                'LoadRadius': {14., 18.5, 23.},
            }
            parms['type'] = 'failure'
        case 'FRAUNHOFER205':
            parms['signal'] = {
                'Vibration': 1,
                'AcousticEmission': 1
            }
            parms['sampling_rate'] = {
                'Vibration': 8192,
                'AcousticEmission': 390625
            }
            parms['keys'] = {
                'FaultComponent': {'Ball', 'InnerRace', 'OuterRace', 'None'},
                'FaultExtend': {0, 1, 2}
            }
            parms['filters'] = {
                'RotatingSpeed': {600, 1000, 1400, 1800, 2200},
            }
            parms['type'] = 'initiated'
        case 'IMS':
            parms['signal'] = {
                'dataset1': ['B1C1', 'B1C2', 'B2C1', 'B2C2', 'B3C1', 'B3C2', 'B4C1', 'B4C2'],
                'dataset2': ['B1C1', 'B2C1', 'B3C1', 'B4C1'],
                'dataset3': ['B1C1', 'B2C1', 'B3C1', 'B4C1'],
            }
            parms['split'] = ['dataset1', 'dataset2', 'dataset3']
            parms['sampling_rate'] = 20480
            parms['type'] = 'failure'
        case 'MAFAULDA':
            parms['signal'] = {
                'tachometer': 1,
                'underhang': 3,
                'overhang': 3,
                'microphone': 1
            }
            parms['split'] = ['normal', 'horizontal-misalignment', 'vertical-misalignment', 'imbalance', 'underhang', 'overhang']
            parms['sampling_rate'] = 50000
            parms['keys'] = {
                'FaultName': ['normal', 'imbalance', 'horizontal-misalignment', 'vertical-misalignment', 'overhang', 'underhang'],
                'FaultSize':{
                    'normal': 'normal',
                    'horizontal-misalignment': [0.5, 1.0, 1.5, 2.0],
                    'vertical-misalignment': [0.51, 0.63, 1.27, 1.40, 1.78, 1.90],
                    'underhang': (['none', 'outer_race', 'cage_fault', 'ball_fault'], [0, 6, 20, 35]),
                    'overhang': (['none', 'outer_race', 'cage_fault', 'ball_fault'], [0, 6, 20, 35])
                }
            }
            parms['filters'] = {
                'NominalRPM': None
            }
            parms['type'] = 'initiated'
        case 'OTTAWA':
            parms['signal'] = {
                'channels': 2
            }
            parms['split'] = ['train']
            parms['sampling_rate'] = 200000
            parms['keys'] = {
                'FaultComponent': {'None', 'InnerRace', 'OuterRace', 'Ball', 'Combination'}
            }
            parms['filters'] = {
                'RotatingSpeed':  {'Increasing', 'Decreasing', 'Increasing-Decreasing', 'Decreasing-Increasing'}
            }
            parms['type'] = 'initiated'
        case 'PADERBORN':
            parms['signal'] = {
                'vibration': 1,
                'current': 2,
                'mechanic': 3,
                'temperature': 1
            }
            parms['split'] = ['healthy', 'artificial', 'lifetime']
            parms['keys'] = {
                'FaultComponent': {'None', 'Inner Ring', 'Outer Ring', 'Inner Ring+Outer Ring'},
                'FaultExtend': {0, 1, 2, 3},
                'DamageMethod': {'Healthy', 'Aritificial', 'Lifetime'},
                'FaultType':  {'None', 'Electrical Discharge Machining', 'Electric Engraver', 'Fatigue: Pitting', 'Drilling'}
            }
            parms['filters'] = {
            }
            parms['type'] = 'initiated+failure'
        case 'PHMAP2021':
            parms['signal'] = {
                'vibration': 2
            }
            parms['split'] = ['train']
            parms['sampling_rate'] = 10544
            parms['keys'] = {'Normal', 'Unbalance', 'Looseness', 'High', 'Bearing'}
            parms['type'] = 'initiated'

        case 'SEUC':
            parms['signal'] = {
                'motor':1, 'parallel':3, 'planetary':3, 'torque':1
            }
            parms['split'] = ['gearbox', 'bearing']
            parms['sampling_rate'] = None
            parms['keys'] = {
                'gearbox': {'Chipped', 'Missing', 'Root', 'Surface', 'None'},
                'bearing': {'Ball', 'Inner', 'Outer', 'Combination', 'None'},
            }
            parms['filters'] = {'LoadForce'}
            parms['type'] = 'initiated'
        case 'XJTU':
            parms['signal'] = {'vibration': 2}
            parms['split'] = ['condition1', 'condition2', 'condition3']
            parms['sampling_rate'] = 25600
            parms['keys'] = {
                'FaultComponent': ['Inner', 'Ball', 'Cage', 'Outer', 'Inner+Outer', 'Inner+Ball+Cage+Outer'],
                'Lifetime': None
            }
            parms['filters'] = {'BearingID', 'OperatingCondition'}
            parms['type'] = 'failure'
        # case '':
        #     parms['signal'] = {}
        #     parms['sampling_rate'] = {}
        #     parms['keys'] = {}
        #     parms['filters'] = {}
        #     parms['type'] = ''
        case _:
            raise NameError(f"Unknown dataset: {ds_name}")

    return parms


# def load_compact(ds_name:str, split:str|list, **kwargs):
#     from .transformer import DatasetCompactor
#     ds0 = tfds.load(ds_name, split=split)
#     parms = query_parameters(ds_name)
#     compactor = DatasetCompactor(
#         ds0, **kwargs
#     )
#     return compactor

