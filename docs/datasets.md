# Datasets
Available datasets:

- [CWRU](datasets/cwru.md)
- DCASE: 2020, 2021, 2022
- DIRG
- Femto
- Fraunhofer

!!! note "Nomenclature"
    In `dpmhm` we distinguish of three concepts:

    - {==Original dataset==}: what is provided by the original source
    - {==Built dataset==}: original dataset formatted by `tensorflow-datasets`
    - {==Preprocessed dataset==}: built dataset after preprocessing steps

As mentionned in [Workflow](index.md#workflow) a dataset needs to be installed and preprocessed before being fed to ML models. See the tutorial [Installation of Datasets](notebooks/datasets/installation.ipynb) and [Preprocessing of Datasets](notebooks/datasets/preprocessing.ipynb) for a in-depth walkthrough.


## Convention for the data structure
Built datasets have standard interface that can be inspected using the property `.element_spec`. In `dpmhm` a built dataset contains the following fileds:

| Name        | Type   | Description     |
| :--------   | :----- |  :---------     |
| `signal` | dict | data of this record |
| `label`  | str or int| label of this record |
| `sampling_rate` | int or dict | sampling rate (Hz) of `signal` |
| `metadata`| dict | all other information about this record |

and the same structrue is followed by all items of the dataset. When `sampling_rate` is a dictionary, it has the same keys as `signal` and the values is the corresponding sampling rate of each channel in `signal`, otherwise it is a number representing the common sampling rate of all channels.

For example the structure of  the dataset [CWRU](datasets/cwru.md) looks like:

```python
>>> ds['train'].element_spec

{'label': TensorSpec(shape=(), dtype=tf.int64, name=None),
 'metadata': {'FaultComponent': TensorSpec(shape=(), dtype=tf.string, name=None),
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

See the tutorial [Performance](notebooks/datasets/performance.ipynb) for details.