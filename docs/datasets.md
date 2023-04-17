# Datasets
Available datasets:

| Name | Type of experiment | Channels | Sampling rate | Original size |
| :--- | :----------------- | :------- | :------------ | :------------ |
| [CWRU](datasets/cwru.md) | Initiated faults | up to 3 vibrations | 12 kHz / 48 kHz | 656 Mb |
| [Dcase2020](datasets/dcase2020.md) | Labelled data | 1 acoustic | 16 kHz | 16.5 Gb |
| [Dcase2021](datasets/dcase2021.md) | Labelled data | 1 acoustic  | 16 kHz | 17.7 Gb |
| [Dcase2022](datasets/dcase2022.md) | Labelled data | 1 acoustic  | 16 kHz | 15 Gb |
| [DIRG](datasets/dirg.md) | Initiated faults & Run-to-failure | 6 vibrations | 51.2 kHz / 102.4 kHz | 3 Gb |
| [FEMTO](datasets/femto.md) | Run-to-failure | up to 3, 2 vibrations + 1 temperature | 25.6 kHz / 10 Hz | 3 Gb |
| [Fraunhofer151](datasets/fraunhofer151.md) | Labelled data | 5, 1 voltage + 1 rpm + 3 vibrations | 4096 Hz | 11 Gb |
| [Fraunhofer205](datasets/fraunhofer205.md) | Initiated faults | 2, 1 vibration + 1 acoustic | 8192 Hz / 390625 Hz | 9.4 Gb |
| [IMS](datasets/ims.md) | Run-to-failure | up to 8 vibrations | 20480 Hz | 6.1 Gb |
| [Mafaulda](datasets/mafaulda.md) | Initiated faults | 8, 1 rpm + 3 vibrations + 3 vibrations + 1 acoustic | 50 kHz | 31 Gb |
| [Ottawa](datasets/ottawa.md) | Initiated faults | 2, 1 vibration + 1 rpm | 200 kHz | 763 Mb |
| [Paderborn](datasets/paderborn.md) | Initiated faults & Run-to-failure | 8, 2 currents + 1 vibration + 3 mechanic + 1 temperature | 64 kHz / 4 kHz / 1 Hz | 20.8 Gb |
| [Phmap2021](datasets/phmap2021.md) | Initiated faults | 2 vibrations | 10544 Hz | 10.8 Gb |
| [SEUC](datasets/seuc.md) | Initiated faults | 8, 1 motor vibration + 3 planetary gearbox vibrations + 1 torque + 3 parallel gearbox vibrations | ? | 1.6 Gb |
| [XJTU](datasets/xjtu.md) | Run-to-failure | 2 vibrations | 25.6 kHz | 11.4 Gb |

Meaning of different types of experiment:

- Initiated faults: fault is precisely labelled, e.g. by the name of the faulty component.
- Labelled data: fault is roughly labelled, e.g. by 'normal' or 'faulty'.
- Run-to-failure: continuous experiment, no information about the faulty state is available in general.

## Installation & Preprocessing

!!! note "Nomenclature"
    In `dpmhm` we distinguish three concepts:

    - {==Original dataset==}: what is provided by the original source
    - {==Built dataset==}: original dataset formatted by `tensorflow-datasets`
    - {==Preprocessed dataset==}: built dataset after preprocessing steps

As mentionned in [Workflow](index.md#workflow) a dataset needs to be installed and preprocessed before being fed to ML models.

### Installation
Let's take example of the dataset CWRU. For installation simply use
```python

import dpmhm
dpmhm.datasets.install('CWRU')
```

See the tutorial [Installation of Datasets](notebooks/datasets/installation.ipynb) and [Preprocessing of Datasets](notebooks/datasets/preprocessing.ipynb) for a in-depth walkthrough.


### Preprocessing
A preprocessing pipeline consists of 3 levels of transformations:

1. **File-level** preprocessing: data selection, label modification, signal resampling & truncation etc.
2. **Data-level** preprocessing: feature extraction, windowed view, data augmentation etc.
3. **Model-level** preprocessing: adaptation to the specification of a machine learning model.

Let's take example of a dataset of 3-channels acoustic records of 10 seconds each file. The corresponding steps would be:

1. Select only the first channel. Make finer labels.
<!-- Split the long signal into chunks of 1 second. -->
2. Compute the spectrogram of each record and split into patches of shape (64, 64).
3. Make batches of paired view of patches, for the training of a contrastive learning model.

See the page [Datasets](datasets.md#Preprocessing) for more details.

## Convention for the data structure
Built datasets have standard interface that can be inspected using the property `.element_spec`. In `dpmhm` a built dataset contains the following fileds:

| Name        | Type   | Description     |
| :--------   | :----- |  :---------     |
| `signal` | dict | data of this record |
| `sampling_rate` | int or dict | sampling rate (Hz) of `signal` |
| `metadata`| dict | all other information about this record |
<!-- | `label`  | str or int| label of this record | -->

and the same structrue is followed by all items of the dataset. When `sampling_rate` is a dictionary, it has the same keys as `signal` and the values is the corresponding sampling rate of each channel in `signal`, otherwise it is a number representing the common sampling rate of all channels.

For example the structure of  the dataset [CWRU](datasets/cwru.md) looks like:

```python
>>> ds['train'].element_spec

{'metadata': {'FaultComponent': TensorSpec(shape=(), dtype=tf.string, name=None),
  'FaultLocation': TensorSpec(shape=(), dtype=tf.string, name=None),
  'FaultSize': TensorSpec(shape=(), dtype=tf.float32, name=None),
  'FileName': TensorSpec(shape=(), dtype=tf.string, name=None),
  'LoadForce': TensorSpec(shape=(), dtype=tf.uint32, name=None),
  'NominalRPM': TensorSpec(shape=(), dtype=tf.uint32, name=None),
  'RPM': TensorSpec(shape=(), dtype=tf.uint32, name=None)},
 'sampling_rate': TensorSpec(shape=(), dtype=tf.uint32, name=None),
 'signal': {'BA': TensorSpec(shape=(None,), dtype=tf.float32, name=None),
  'DE': TensorSpec(shape=(None,), dtype=tf.float32, name=None),
  'FE': TensorSpec(shape=(None,), dtype=tf.float32, name=None)}}
```
which contains 3 channels ['BA', 'DE', 'FE'] and a common sampling rate. The meaning of each field can be found in the description pages of the corresponding dataset.

!!! note "Data sturcture"
    We follow the principle that all original information has to be preserved in the built dataset. However some of them may be non-essential and can be dropped in the subsequent preprocessing steps, which may modify the element specification of the dataset.


## Performance
The subsequent preprocessing steps of a dataset actually define a pipeline of transformations, very often heavy-lifted by the method `.map()`. In the graph execution mode of Tensorflow these transformations are not effective until an element of the preprocessed dataset is loaded into memory. Some intermediate steps may be repetitive and hinder the performance. As a remedy one can first pre-compute the heavyweight intermediate transformations and export the transformed dataset to disk, then postpone the lightweight final transformations in memory.

<!-- See the tutorial [Performance](notebooks/datasets/performance.ipynb) for details. -->