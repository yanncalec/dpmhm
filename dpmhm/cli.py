"""Methods for CLI usage.
"""

import click
from importlib import import_module
# TF logging level
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


# List of supported dataset
_DATASET_DICT = {
    'cwru': 'CWRU',
    'dcase2020': 'Dcase2020',
    # 'dcase2021': 'Dcase2021',
    # 'dcase2022': 'Dcase2022',
    'dirg': 'DIRG',
    'femto': 'FEMTO',
    'fraunhofer151': 'Fraunhofer151',
    'fraunhofer205': 'Fraunhofer205',
    'ims': 'IMS',
    'mafaulda': 'Mafaulda',
    'ottawa': 'Ottawa',
    'paderborn': 'Paderborn',
    'phmap2021': 'Phmap2021',
    'seuc': 'SEUC',
    'xjtu': 'XJTU'
}


def import_dataset_module(ds:str):
    dsl = ds.lower()
    return import_module('.'+dsl, f'dpmhm.datasets.{dsl}')

def get_info(ds:str):
    """Retrieve information of a dataset.
    """
    return import_dataset_module(ds).__doc__

def get_urls(ds:str):
    """Retrieve data urls of a dataset.
    """
    return import_dataset_module(ds)._DATA_URLS


@click.group()
def dpmhm_datasets():
    """CLI for DPMHM datasets.
    """
    pass

@click.command("install")
@click.argument('ds')
@click.option('--data_dir', default=None, help='location of tensorflow datasets')
@click.option('--download_dir', default=None, help='location of download folder, optional for automatic installation')
@click.option('--extract_dir', default=None, help='location of extraction folder, optional for automatic installation')
@click.option('--manual_dir', default=None, help='location of manual folder, mandatory for manual installation')
@click.option('--max_sim_dl', default=1, help='maximum number of simultaneous downloads')
def _install(ds:str, *, data_dir:str, download_dir:str, extract_dir:str, manual_dir:str, max_sim_dl:int=1):
    """Install a dataset.

    \b
    Example of usage
    ----------------

    For manual installation, download and unzip first the dataset `DS` into `~/tmp/DS`, and run

        dpmhm-datasets install DS --manual_dir=~/tmp/DS
    """
    from dpmhm import datasets

    dl_kwargs = {'max_simultaneous_downloads': max_sim_dl}

    return datasets.install(ds, data_dir=data_dir, download_dir=download_dir, extract_dir=extract_dir, manual_dir=manual_dir, dl_kwargs=dl_kwargs)


@click.command("info")
@click.argument('ds', default='')
# @click.option('--url', is_flag=True, help='Print data source urls.')
def _info(ds:str, *, url:bool=False):
    """Print information of installable datasets.
    """
    if ds != '':
        if url:
            for _url in get_urls(ds):
                print(_url)
        else:
            print(get_info(ds))
    else:
        print("Installable datasets:\n")
        for ds in _DATASET_DICT.values():
            print(ds)

dpmhm_datasets.add_command(_info)
dpmhm_datasets.add_command(_install)
