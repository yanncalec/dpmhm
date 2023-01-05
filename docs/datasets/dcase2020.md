::: dpmhm.datasets.dcase.dcase2020
    options:
      show_source: false

## Installation
Due to the large size, fully automatic installation is not supported.
Manually download all zip files from the provided link. Unzip all files into a same folder e.g. `~/tmp/dcase2020`, which will look like
```sh
.
├── fan
├── pump
├── slider
├── ToyCar
├── ToyConveyor
└── valve
```

Build the dataset either from python
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

or from terminal
```sh
$ tfds build dcase2020 --imports dpmhm.datasets.dcase.dcase2020 --manual_dir ~/tmp/dcase2020
```

## Tutorial


<!-- ::: notebooks/CWRU.ipynb -->