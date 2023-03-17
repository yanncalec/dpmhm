# DPMHM

**DPMHM** (prononced "Deep HM") stands for **Diagnosis and Prognosis in Machine Health Monitoring**. It is a library written in Python and Tensorflow for benchmark test of deep-learning based algorithms in the context of machine health monitoring (MHM), featuring

- A collection of open-access database of vibration signals, standardized and formatted using [`tensorflow-datasets`](https://www.tensorflow.org/datasets/overview)
- Pipelines for automatic preprocessing & transformation of data
- Baseline algorithms: Supervised/Unsupervised/Self-Supervised representation learning, Transfer learning/Few-shot learning for fault diagnosis, Remaining Useful Life (RUL) prediction etc.

#### About the name
> A diagnosis is an identification of a disease via examination. What follows is a prognosis, which is a prediction of the course of the disease as well as the treatment and results. A helpful trick is that a diagnosis comes before a prognosis, and diagnosis is before prognosis alphabetically. Additionally, diagnosis and detection both start with "d" whereas prognosis and prediction both start with "p."
> -- <cite>Merriam-Webster dictionary</cite>

As the name suggests, `dpmhm` intends to cover *diagnosis* and *prognosis*, two main aspects of machine condition monitoring. In `dpmhm`, we implement *diagnosis* as anomaly detection and *prognosis* as RUL prediction.

### Purpose
Data is of paramount importance for deep learning based model development. Many benchmark studies in MHM today are performed on several well-recognized datasets (such as [CWRU](https://engineering.case.edu/bearingdatacenter) and [NASA](https://www.nasa.gov/intelligent-systems-division#bearing)), despite the availability of alternative open access datasets. Possible reasons impeding the wide adoptation of alternative datasets by researchers include:

- Although freely available, they are much less well recognized hence less visible and accessible;
- In the absence of widely accepted guidelines, datasets of diverse sources in MHM often come in arbitrary format and convention. For each new dataset, the user has to pay extra overhead on apprehension & preparation.

<!-- On the other hand, methods developed for MHM are increasingly based on modern deep learning models, which are often vision oriented or general purposed (like CNN and auto-encoder). Although computer programs of these DL models can be found for both classical or SOTA architectures, we believe it benificial to reimplement them from scratch in a coherent manner by adapting these models to the context of MHM. -->

This package aims for offering a self-contained environment of benchmark test, including both open-access datasets and deep learning models (implemented from scratch), in order to facilitate the development of MHM models. In `dpmhm`, a large collection of open datasets is encapsulated in a common interface in order to alleviate efforts in preliminary operations like data loading and preparation. Moreover with the provided models, users can easily build transfer learning benchmarks of model/dataset cross-test.


## Installation
The package can be installed simply using `pip` from a terminal:
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
This package does not provide the actual datasets (the total of original files is ~ 158 Gb, see [Datasets](datasets.md)), only the facilities for their preprocessing. A dataset must be installed before being loaded into memory. The general procedure consists of first downloading the original data then building the dataset of [tfrecord](https://www.tensorflow.org/tutorials/load_data/tfrecord) format using `tensorflow_datasets`.


### Preprocessing a dataset
After installation, a built dataset may still needs extra setups before being fed to a ML model, including data selection, feature extraction and adaptation etc.

### Use with ML models
No extra installation is needed for the machine learning models in general, just load and configure a model then apply it on some dataset for training & test.

See the page [Datasets](datasets.md#Installation) and [Models](models.md) for more details.

## MHM & Deep learning
Machine Health Monitoring (MHM) embodies the ability to assess and react to the machine health and is well established in manufacturing environments. It is also called predictive maintenance, condition monitoring, or machine fault diagnosis.

> A McKinsey study estimated that the appropriate use of MHM techniques by process manufacturers “typically reduces machine downtime by 30 to 50 percent and increases machine life by 20 to 40 percent” (Dilda et al., 2017). Furthermore, with the increase in computational power and data availability over the past decade, new tools are now enabled (such as end-to-end deep learning) that diverge from how MHM was traditionally implemented. Thus, it is imperative that manufacturers, and MHM researchers, understand the benefits of these new tools, along with their potential drawbacks. [@von_hahn_self-supervised_2021]
> -- <cite> von Hahn, T., Mechefske, C., 2021. Self-supervised learning for tool wear monitoring with a disentangled-variational-autoencoder. </cite>

We refer the reader to the section [MHM & Deep Learning](topics/mhm.md) for a short account on the topic.

## Tutorials
Tutorials and examples of usage are provided and come in form of Jupyter notebooks. They can be modified and adapted by users for their own purpose. See the Tutorial section for more details.

## Reference
Documentation on the API can be found in the Reference section.