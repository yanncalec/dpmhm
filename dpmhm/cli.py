import click

@click.group()
def dpmhm():
    """CLI for DPMHM.
    """
    pass

@click.command()
@click.argument('ds')
@click.option('--data_dir', default=None, help='location of tensorflow datasets')
@click.option('--download_dir', default=None, help='location of download folder')
@click.option('--extract_dir', default=None, help='location of extraction folder')
@click.option('--manual_dir', default=None, help='location of manual folder')
@click.option('--max_sim_dl', default=1, help='maximum number of simultaneous downloads')
def install(ds:str, *, data_dir:str, download_dir:str, extract_dir:str, manual_dir:str, max_sim_dl:int=1):
    """Install a dataset."""
    from . import datasets

    dl_kwargs = {'max_simultaneous_downloads': max_sim_dl}

    return datasets.install(ds, data_dir=data_dir, download_dir=download_dir, extract_dir=extract_dir, manual_dir=manual_dir, dl_kwargs=dl_kwargs)

@click.command()
@click.argument('ds', default='')
def info(ds:str):
    from . import datasets
    if ds == '':
        print(datasets.get_dataset_list())
    else:
        print(datasets.get_info(ds))

# dpmhm.add_command(info)

dpmhm.add_command(install)
