# DPMHM

**DPMHM** stands for **Diagnosis and Prognosis in Machine Health Monitoring (MHM)**. It is a library written in Python and Tensorflow for benchmark test of deep-learning based algorithms in the context of MHM, featuring

- A large collection of open-access database of vibration signals, standardized and formatted using [`tensorflow-datasets`](https://www.tensorflow.org/datasets/overview)
- Automatic preprocessing & transformation of data
- State-of-the-arts (SOTA) machine learning algorithms: Supervised/Unsupervised/Self-Supervised representation learning, Remaining Useful Life (RUL) estimator...

### About the name
> A diagnosis is an identification of a disease via examination. What follows is a prognosis, which is a prediction of the course of the disease as well as the treatment and results. A helpful trick is that a diagnosis comes before a prognosis, and diagnosis is before prognosis alphabetically. Additionally, diagnosis and detection both start with "d" whereas prognosis and prediction both start with "p."
> -- <cite>Merriam-Webster dictionary</cite>

In DPMHM we understand *diagnosis* as the detection of anomalies and *prognosis* as the prediction of RUL.

### Purpose
- Why this package?
- What we can do with it?


## Installation
The package can be installed using `pip` from a terminal:
```shell
$ pip install dpmhm
```
which will also install all dependencies, notably `tensorflow-datasets`.

## Workflow
Here is a basic workflow with `dpmhm`:

1. Installing a dataset
2. Preprocessing a dataset
3. Use with machine learning models

### Installing a dataset
This package does not provide the actual datasets, only the facilities for their preprocessing. A dataset must be installed before being loaded into memory. The general procedure consists of first downloading the original data then building the dataset of [tfrecord](https://www.tensorflow.org/tutorials/load_data/tfrecord) format using `tensorflow_datasets`. See the page [Datasets](datasets.md#Installation) for more details.


### Preprocessing a dataset
After installation, a built dataset may still needs extra setups before being fed to a ML model, including:

1. **File-level** preprocessing: data selection, label modification, signal resampling & truncation etc.
2. **Data-level** preprocessing: feature extraction, windowed view, data augmentation etc.
3. **Model-level** preprocessing: modification to meet the specification of a machine learning model.

Let's take example of a dataset of 3-channels acoustic records of 10 seconds each file. The corresponding steps would be:

1. Select only the first channel. Make finer labels.
<!-- Split the long signal into chunks of 1 second. -->
2. Compute the spectrogram of each record and split into patches of shape (64, 64).
3. Make batches of paired view of patches, for the training of a contrastive learning model.

See the page [Datasets](datasets.md#Preprocessing) for more details.

### Use with ML models
No extra installation is needed for the machine learning models, just load and configure a model then apply it on some dataset for training.

See the page [Models](models.md) for more details.

## MHM & Deep learning

### Few-shot & Transfer learning

### Self-supervised representation learning


## Tutorials
Tutorials and examples of usage are provided. Most of them come in form of Jupyter notebooks and can be modified by users for their own purpose. See the Tutorial section for more details.

## Reference
Documentation on the API can be found in the Reference section.