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

# from .cwru import CWRU
# from .dcase import DCASE2021
# from .seuc import SEUC
# from .mfpt import MFPT
# from .dirg import DIRG
# from .mafaulda import MAFAULDA
# from .ims import IMS
# from .ottawa import Ottawa
# from .paderborn import Paderborn
# from .femto import FEMTO
# from .fraunhofer import Fraunhofer205, Fraunhofer151
# from .phmdc import Phmap2021

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


def install(ds:str, *, data_dir:str=None, download_dir:str=None, extract_dir:str=None, manual_dir:str=None, dl_kwargs:dict={}, **kwargs) -> Dataset:
    """Install a dataset.

    Args
    ----
    ds:
        name of the dataset to be installed.
    data_dir:
        location of tensorflow datasets, by default the environment variable `TFDS_DATA_DIR.
    download_dir:
        location of download folder, by default `data_dir/dpmhm/downloads/ds`.
    extract_dir:
        location of extraction folder, by default `data_dir/dpmhm/extracted/ds`.
    manual_dir:
        location of manually downloaded & extracted files.
    dl_kwargs:
        keyword arguments for `tfds.download.DownloadManager()`
    kwargs:
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

    Args
    ----
    url:
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