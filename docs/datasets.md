# Datasets
Available datasets:

- [CWRU](datasets/cwru.md)
- DCASE: 2020, 2021, 2022
- DIRG
- Femto
- Fraunhofer

!!! note "Nomenclature"
    We distinguish of three concepts:

    - {==Original dataset==}: what is provided by the original source
    - {==Built dataset==}: original dataset formatted by `tensorflow-datasets`
    - {==Preprocessed dataset==}: built dataset after preprocessing steps


## Installing a dataset
As mentionned in [Workflow](index.md#workflow), actual datasets must be installed before they can be loaded into memory and used with ML models.

Installation consists of first downloading the original data then building the dataset of [tfrecord](https://www.tensorflow.org/tutorials/load_data/tfrecord) format using [`tensorflow-datasets`](https://www.tensorflow.org/datasets/overview). The procedure can be manual or fully automatic and eventually time-consuming depending on the specific dataset. However it needs to be run only once.

Specific instructions of installation can be found in the document page of each dataset.

!!! note "Preservation of original information"

    The building step involves parsing and converting the original dataset. In `dpmhm` we followed the principle that all information from the original dataset should be preserved in the built dataset, in particular there is one-to-one correspondance between original data files and items in the built dataset.

### Example of installation
Following is the example of installation for the [CWRU](datasets/cwru.md) bearing dataset.
<!-- The same procedure can be adapted to other datasets. -->

!!! info "Location of the tfds database"
    On Unix-like systems, the default location for the tensorflow datasets (including temporary files of download & extraction) is `~/tensorflow_datasets`. This can be changed by setting the environment variable `TFDS_DATA_DIR` or passing the argument in the python methods `tfds.builder()` or `tfds.load()` or via the CLI `tfds build`.

#### Method 1: Automatic installation
We use the method [`tfds.load()`](https://www.tensorflow.org/datasets/api_docs/python/tfds/load) to automatically download and prepare the dataset.

```python
import tensorflow_datasets as tfds
import dpmhm.datasets.cwru  # <- register the name 'cwru' in available builders.

assert 'cwru' in tfds.list_builders()  # True

ds = tfds.load('cwru')
```
This will install the dataset `cwru` in the place specified by the environment variable `TFDS_DATA_DIR` (default to `~/tensorflow_datasets`).

`tfds.load()` will save the download into the subfolder `downloads` and extract the data files into `downloads/extracted`. These folders can be modified by specifying the keyword arguments `download_and_prepare_kwargs` in `tfds.load()`, see also [`tfds.builder()`](https://www.tensorflow.org/datasets/api_docs/python/tfds/builder).


#### Method 2: Manual installation
In some cases the automatic installation cannot be completed, due to e.g. difficulties in downloading the orginal data files. We resort to manual installation by separating the steps of download and build.

First download and extract manually all data files, for example via the terminal commands
```sh
$ cd ~/tmp
$ curl file:///run/media/han/ExFAT/Database/dpmhm/cwru.zip -o cwru.zip
$ unzip cwru.zip  # extract all files into the folder `~/tmp/cwru`.
```
Then build the dataset with the command [`tfds build`](https://www.tensorflow.org/datasets/cli)
```sh
$ tfds build cwru --imports dpmhm.datasets.cwru --manual_dir ~/tmp/cwru
```
which will build the dataset into the folder of the tfds datasets (default to `~/tensorflow_datasets`).

#### Load & remove the dataset
After installation the dataset can be loaded into memory by
```python
ds, ds_info = tfds.load('cwru', with_info=True)
```
The variable `ds` may contain multiple fields of split, and `ds_info` contains information about the dataset.

To remove the installed dataset, simply delete the folder `~/tensorflow_datasets/cwru`.


### Convention for the data structure
Installed datasets have standard interface which can be inspected using the method `element_spec()`. All datasets in `dpmhm` contain the following fileds:

| Name        | Type   | Description     |
| :--------   | :----- |  :---------     |
| `signal` | dict | data of this record |
| `label`  | str or int| label of this record |
| `sampling_rate` | int or dict | sampling rate (Hz) of `signal` |
| `metadata`| dict | all other information about this record |

and the same structrue is followed by all items of the dataset. When `sampling_rate` is a dictionary, it has the same keys as `signal` and the values is the corresponding sampling rate of each channel in `signal`, otherwise it is a number which is the common sampling rate of all channels.

<!-- For example, a dataset may have channels `signal={'acc1':..., 'acc2':...}` and `sampling_rate={'acc1': 16000, 'acc2':24000}`. -->

For example the data structure of  `cwru` looks like:

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
which contains 3 channels ['BA', 'DE', 'FE'] and a common sampling rate. See the descriptions in [CWRU](datasets/cwru.md) for the meaning of each field.

## Preprocessing a dataset
As mentionned in [Preprocessing](index.md#preprocessing-a-dataset)