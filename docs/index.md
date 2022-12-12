# DPMHM

DPMHM stands for **Diagnosis and Prognosis in Machine Health Monitoring (MHM)**. It is a library written in Python and Tensorflow for benchmark test of deep-learning based algorithms in the context of MHM, featuring

- A large collection of open-access database of vibration signals, standardized and formatted using [`tensorflow-datasets`](https://www.tensorflow.org/datasets/overview)
- Automatic preprocessing & transformation of data
- State-of-the-arts (SOTA) machine learning algorithms: Supervised/Unsupervised/Self-Supervised representation learning, Remaining Useful Life estimator...

### About the name
> A diagnosis is an identification of a disease via examination. What follows is a prognosis, which is a prediction of the course of the disease as well as the treatment and results. A helpful trick is that a diagnosis comes before a prognosis, and diagnosis is before prognosis alphabetically. Additionally, diagnosis and detection both start with "d" whereas prognosis and prediction both start with "p."
> -- <cite>Merriam-Webster dictionary</cite>

### Purpose
- Why this package?
- What we can do with it?


## Installation
```shell
pip install dpmhm
```
This will also install the dependencies, notably `tensorflow-datasets`.

### Installation of datasets
This package does not provide the actual datasets but only the facilities of preprocessing. A dataset must be installed before it can be loaded to memory. The general procedure is:

1. Download the original dataset manually (for some datasets automatic download & preprocessing are supported), unzip all compressed files.
2. Build & install the dataset using `tensorflow-datasets`.

Instructions of installation can be found in the document page of each dataset.

#### Example of [CWRU](datasets/cwru.md) bearing dataset

In the terminal
```shell
$ tfds build cwru --manual_dir ~/Download/cwru
```

Load the dataset into memory
```python
import tensorflow_datasets as tfds

ds = tfds.load('CWRU')
```
### Machine learning models
No extra installation needed for machine learning models:
```python
from dpmhm.models.ssl import simclr
```

## Concepts and design principles
By building and installation, all information of the original dataset are preserved by `tensorflow_datasets`. Once loaded into memory, a raw dataset may need some extra setups before it can be fed to a ML model, which may include:

1. File-level preprocessing: data selection, label modification, signal resampling & truncation etc.
2. Data-level preprocessing: feature extraction, windowed view, data augmentation etc.
3. Algorithm-level preprocessing: modification to meet the specification of the algorithm of training.

Let's take example of a dataset consisting of 3-channels acoustic records of 10 seconds each file. We apply these preprocessing steps:

1. Select only the first channel. Split the long signal into trunks of 1 second.
2. Transform wav to spectrogram and split into patches of shape (64, 64).
3. Make batches of paired view $(x,x')$ with $x,x'$ two patches, for the training of a contrastive learning model.

$$
fx
$$

### Dataset structure
Installed datasets have standard interface and can be loaded into memory by `tfds.load()`. Conventions:

- `signal`:
- `label`:
- `metadata`:


## Models