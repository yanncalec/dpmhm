::: dpmhm.datasets.dirg.dirg

## Installation
Download and unzip all files into a folder e.g. `~/tmp/dirg`, which will look like
```sh
.
├── DefectOnRoller_4A_Photos.pdf
├── Description and analysis of open access data.pdf
├── EnduranceTest
└── VariableSpeedAndLoad
```

Build the dataset either from python
```python
import tensorflow_datasets as tfds
import dpmhm.datasets.dirg

_ = tfds.load(
    'DIRG',
    download_and_prepare_kwargs = {
        'download_config': tfds.download.DownloadConfig(
            manual_dir='~/tmp/dirg/'
        )
    }
)
```

or from terminal
```sh
$ tfds build dirg --imports dpmhm.datasets.dirg --manual_dir ~/tmp/dirg
```

## Tutorial


<!-- ::: notebooks/CWRU.ipynb -->