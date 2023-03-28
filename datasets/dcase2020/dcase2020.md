::: dpmhm.datasets.dcase2020
    options:
      show_source: false

## Installation
Only manual installation is supported. Download all zip files from the provided link. Unzip them into a same folder e.g. `~/tmp/dcase2020`
```sh
.
├── fan
├── pump
├── slider
├── ToyCar
├── ToyConveyor
└── valve
```

Build the dataset from terminal
```sh
$ tfds build dcase2020 --imports dpmhm.datasets.dcase.dcase2020 --manual_dir ~/tmp/dcase2020
```

or with the python code
```python
import tensorflow_datasets as tfds
import dpmhm.datasets.dcase.dcase2020

_ = tfds.load(
    'DCASE2020',
    download_and_prepare_kwargs = {
        'download_config': tfds.download.DownloadConfig(
            manual_dir='~/tmp/dcase2020/'
        )
    }
)
```

## Tutorial


<!-- ::: notebooks/CWRU.ipynb -->