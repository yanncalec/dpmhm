"""
Collection of open source datasets.

Installation
------------
Run from the subfolder `datasets`:
```
$ tfds build $NAME_OF_DATABASE --manual_dir $PATH_OF_UNZIPPED_FILES
```
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
 'DCASE2021',
 'DIRG',
 'FEMTO',
 'FRAUNHOFER151',
 'FRAUNHOFER205',
 'IMS',
 'MAFAULDA',
 'MFPT',
 'OTTAWA',
 'PADERBORN',
 'PHMAP2021',
 'SEUC',
 'XJTU']

def get_dataset_list():
    return _DATASET_LIST
