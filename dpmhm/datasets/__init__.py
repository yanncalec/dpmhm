"""Collection of open source datasets.
"""

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

_DATASET_LIST = ['CWRU',
    'DCASE2020',
    'DCASE2021',
    'DCASE2022',
    'DIRG',
    'FEMTO',
    'FRAUNHOFER151',
    'FRAUNHOFER205',
    # 'IMS',
    # 'MAFAULDA',
    # 'MFPT',
    # 'OTTAWA',
    # 'PADERBORN',
    # 'PHMAP2021',
    # 'SEUC',
    # 'XJTU'
]

def get_dataset_list():
	return _DATASET_LIST

import tensorflow as tf
import os

# Data type
try:
    _DTYPE = tf.as_dtype(os.environ['DPMHM_DTYPE'])  # from the environment variable
except:
    _DTYPE = tf.float32

# Encoding length for class label
try:
    _ENCLEN = int(os.environ['DPMHM_ENCLEN'])
except:
    _ENCLEN = 8

try:
    _ENCODING = os.environ['DPMHM_ENCODING']
    assert _ENCODING in {'zlib', 'bytes', 'none'}
except:
    _ENCODING = 'none'
    # _ENCODING = tfds.features.Encoding.NONE
