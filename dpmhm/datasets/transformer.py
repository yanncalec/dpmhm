"""Classes for dataset transformer.

Transformers allows to define a pipeline of preprocessing.

Convention
----------
We follow the convention of channel first: The original dataset as well as the transformed dataset has the channel as the first dimension and time as the last dimension.
"""

from typing import Union # List, Dict
from abc import ABC, abstractmethod
import itertools

import numpy as np
# import random
import itertools
import tensorflow as tf
from tensorflow.data import Dataset
from tensorflow.experimental.numpy import atleast_2d

# from tensorflow.python.data.ops.dataset_ops import DatasetV2

import skimage
# from skimage import filters

import librosa
# import scipy

from dpmhm.datasets import utils, _DTYPE, _ENCLEN
from dpmhm.datasets.augment import randomly, random_crop, fade

import logging
Logger = logging.getLogger(__name__)


class AbstractDatasetTransformer(ABC):
    @abstractmethod
    def build(self) -> Dataset:
        """Build the transformed dataset.
        """
        pass

    @property
    def dataset(self):
        """Transformed dataset.
        """
        try:
            return self._dataset
        except:
            self._dataset = self.build()
            self._dataset.__transformer__ = self.__class__
            return self._dataset

    @dataset.setter
    def dataset(self, df):
        self._dataset = df
        self._dataset.__transformer__ = self.__class__

    def serialize(self, outdir:str, *, compression:str=None):
        """Serialize the dataset to disk.

        Serialization consists of saving the dataset and reloading it from a file. This can boost the subsequent performance of a dataset at the cost of storage. Compression by 'GZIP' method can be used.
        """
        Dataset.save(self.dataset, outdir, compression=compression)
        self.dataset = Dataset.load(outdir, compression=compression)

    # @abstractproperty
    # def data_key(self) -> str:
    # 	"""Key name of the data field.
    # 	"""
    # 	pass

    # @abstractproperty
    # def data_dim(self):
    # 	pass

def get_number_of_channels(spec, channels):
    n = 0
    for c in channels:
        try:
            # case `TensorSpec(shape=(?,None),...)`
            n += spec[c].shape[0]
        except:
            # case `TensorSpec(shape=(None,),...)`
            n += 1
    return n


class DatasetCompactor(AbstractDatasetTransformer):
    """Class for dataset compactor.

    This class performs the following preprocessing steps on the raw signal:

    - resampling
    - filtration
    - extraction of new labels
    - sliding window view

    Note
    ----
    - The input dataset must contain the fields {'signal', 'sampling_rate', 'metadata'}.
    - Data of the subfield 'signal' must be either 1D tensor or 2D tensor of shape `(channel, time)`.
    """

    def __init__(self, dataset:Dataset, *, channels:list=[], keys:list=[], filters:dict={}, resampling_rate:int=None, window_size:int=None, hop_size:int=None, split_channel:bool=False):
        """
        Parameters
        ----------
        dataset
            original dataset
        channels
            channels for extraction of data, subset of 'signal', if not given all channels are extracted.
        keys
            keys for extraction of new labels, subset of 'metadata', if not given no label is extracted.
        filters
            filters on the field 'metadata', a dictionary of keys and admissible value(s). By default no filter is applied.
        resampling_rate
            rate for resampling, if None use the original sampling rate.
        window_size
            size of the sliding window on time axis, if None no window is applied.
        hop_size
            hop size for the sliding window. No hop if None or `hop_size=1` (no downsampling). Effective only when `window_size` is given.
        split_channel
            if True any multidimensional channel is splitted into 1d channels.
        """
        self._channels = channels if channels else list(dataset.element_spec['signal'].keys())
        self._channels_dim = get_number_of_channels(dataset.element_spec['signal'], self._channels)
        self._keys = keys
        # self._n_chunk = n_chunk
        self._resampling_rate = resampling_rate
        self._filters = filters

        self._window_size = window_size
        self._hop_size = hop_size
        self._split_channel = split_channel

        # dictionary for extracted labels, will be populated only after scanning the compacted dataset
        self._label_dict = {}
        # self._dataset_origin = dataset
        # filtered original dataset, of shape (channel, time)
        self._dataset_origin = self.filter_metadata(dataset, self._filters)

    @classmethod
    def filter_metadata(cls, ds:Dataset, fs:dict):
        """Filter a dataset by values of its field 'metadata'.
        """
        @tf.function
        def _filter(X, k, v):
            return tf.reduce_any(tf.equal(X['metadata'][k], v))

        for k,v in fs.items():
            ds = ds.filter(lambda X: _filter(X, k, v))
        return ds

    def build(self):
        ds = self.compact(self.resample(self._dataset_origin, self._resampling_rate))
        if self._window_size is not None:
            ds = Dataset.from_generator(
                utils.sliding_window_generator(ds, 'signal', self._window_size, self._hop_size),
                output_signature=ds.element_spec,
            )
        if self._split_channel:
            foo = ds.element_spec.copy()  # must use `.copy()`
            foo['signal'] = tf.TensorSpec((1,None,))
            ds = Dataset.from_generator(
                utils.split_dims_generator(ds, 'signal'),
                output_signature=foo,
            )

        return ds

    @property
    def label_dict(self) -> dict:
        """Dictionary of compacted labels.
        """
        try:
            self._label_dict_scanned
        except:
            self._label_dict = {}
            # make a full scan of the compacted dataset
            for x in self.dataset:
                pass
            self._label_dict_scanned = self._label_dict

        return self._label_dict_scanned

    # @property
    # def label_dict_index(self):
    #   # label index
    #   return {k: n for n, k in enumerate(self.label_dict.keys())}

    def encode_labels(self, *args) -> str:
        """MD5 encoding of a list of labels.

        From:
        https://stackoverflow.com/questions/5417949/computing-an-md5-hash-of-a-data-structure
        """
        dn = [d.numpy() for d in args]
        # v = [str(d.numpy()) for d in args]
        v = [a.decode('utf-8') if type(a) is (bytes or str) else str(a) for a in dn]

        lb = utils.md5_encoder(*v)[:_ENCLEN]
        # if lb in self._label_dict:
        #   assert self._label_dict[lb] == v
        # else:
        #   self._label_dict[lb] = v
        self._label_dict[lb] = v
        return lb

    @property
    def full_label_dict(self) -> dict:
        """Full dictionary of compacted labels.

        Unlike `label_dict` which depends on the choice of active channels, the full label dictionary aggregates the labels of all channels.
        """
        try:
            return self._full_label_dict
        except:
            channels = list(self._dataset_origin.element_spec['signal'].keys())

            for ch in channels:
                compactor = DatasetCompactor(self._dataset_origin, keys=self._keys, channels=[ch])
                try:
                    ld.update(compactor.label_dict)
                except:
                    ld = compactor.label_dict
            self._full_label_dict = ld
            return ld

    @classmethod
    def resample(cls, dataset:Dataset, rsr:int):
        """Resample the dataset to a common target rate.
        """
        @tf.function
        def _resample(X):
            Y = X.copy()
            if rsr is None:
                # try:
                #     # if the original sampling rate is a dict
                #     vsr = tf.stack(list(X['sampling_rate'].values()))
                #     tf.Assert(
                #         tf.reduce_all(tf.equal(vsr[0], vsr)),  #
                #         ['All channels must have the sampling rate:', vsr]
                #     )  # must be all constant
                #     Y['sampling_rate'] = vsr[0]
                # except:
                #     # if the original sampling rate is a number
                Y['sampling_rate'] = X['sampling_rate']
            else:
                xs = {}
                for k in X['signal'].keys():
                    if tf.size(X['signal'][k]) > 0:  # type: ignore
                        try:
                            # X['sampling_rate'] has nested structure
                            xs[k] = tf.py_function(
                                func=lambda x, sr:librosa.resample(x.numpy(), orig_sr=float(sr), target_sr=float(rsr)),
                                inp=[X['signal'][k], X['sampling_rate'][k]],
                                Tout=_DTYPE
                            )
                        except (KeyError, TypeError):
                            # X['sampling_rate'] is a number
                            xs[k] = tf.py_function(
                                func=lambda x, sr:librosa.resample(x.numpy(), orig_sr=float(sr), target_sr=float(rsr)),
                                inp=[X['signal'][k], X['sampling_rate']],
                                Tout=_DTYPE
                            )
                            # tf.print('Resampling...')
                        except Exception as msg:
                            # tf.print(X['sampling_rate'])
                            Logger.exception(msg)
                            xs[k] = X['signal'][k]
                    else:
                        xs[k] = X['signal'][k]
                Y['signal'] = xs
                Y['sampling_rate'] = rsr
            return Y

        return dataset.map(_resample, num_parallel_calls=tf.data.AUTOTUNE)

    def compact(self, dataset:Dataset):
        """Transform a dataset into a compact form.

        This method compacts the original dataset by
        - stacking the selected channels (must have the same length)
        - renaming the label using selected keys
        - dropping the redundant data field `metadata`

        The compacted dataset has a dictionary structure with the fields {'label', 'sampling_rate', 'signal'}.

        Note
        ----
        Stacking channels may filter out some samples.
        """
        @tf.function  # necessary for infering the size of tensor
        def _has_channels(X):
            """Test if not empty channels are present in data.
            """
            flag = True
            for ch in self._channels:
                # if tf.size(X['signal'][ch]) == 0:  # to infer the size in graph mode
                if tf.equal(tf.size(X['signal'][ch]), 0):
                # if X['signal'][ch].shape == 0:  # raise strange "shape mismatch error"
                # if len(X['signal'][ch]) == 0:  # raise TypeError
                    flag = False
                    # break or return False are not supported by tensorflow
            return flag

        @tf.function
        def _compact(X):
            # Check all channels have the same sampling rate
            try:
                # if the original sampling rate is a dict
                vsr = tf.stack([X['sampling_rate'][ch] for ch in self._channels])
                tf.Assert(
                    tf.reduce_all(tf.equal(vsr[0], vsr)),  #
                    ['All channels must have the same sampling rate:', vsr]
                )  # must be all constant
                rsr = vsr[0]
            except TypeError:
                # if the original sampling rate is a number
                rsr = X['sampling_rate']

            # d = [X['label']] + [X['metadata'][k] for k in self._keys]
            d = [X['metadata'][k] for k in self._keys]
            x = tf.concat([atleast_2d(X['signal'][ch]) for ch in self._channels], 0)
            # x = [atleast_2d(X['signal'][ch]) for ch in self._channels]  # <- This fails if the ranks of channel are different.
            x = tf.squeeze(x)

            return {
                # `self.encode_labels(d)` doesn't work
                # `ensure_shape()` recover the lost shape due to `py_function()`
                'label': tf.ensure_shape(tf.py_function(func=self.encode_labels, inp=d, Tout=tf.string), ()),
                'metadata': X['metadata'],
                'sampling_rate': rsr,
                # 'signal': tf.squeeze(x),
                'signal': tf.reshape(x, (self._channels_dim, -1))
                # 'signal': tf.reshape(x, (len(self._channels), -1))  # this works only for the case where each channel contains only one 1d signal
            }
        # ds = dataset.filter(_has_channels)  # <- doesn't work?
        ds = dataset.filter(lambda X:_has_channels(X))
        return ds.map(_compact, num_parallel_calls=tf.data.AUTOTUNE)


class FeatureExtractor(AbstractDatasetTransformer):
    """Class for feature extractor.

    This class performs the feature transform on a compacted dataset. A feature transform increases the dimension of data by 1, e.g. from 1D signal to 2D spectrogram. The feature transformed dataset has fields {'label', 'feature'}.
    """

    def __init__(self, dataset, extractor:callable):
        """
        Parameters
        ----------
        dataset
            compacted dataset.
        extractor
            a callable taking arguments `(signal, sampling_rate)` and returning extracted 2D features.
        """
        assert dataset.__transformer__ is DatasetCompactor
        self._dataset_origin = dataset
        self._extractor = extractor

    def build(self):
        return self.to_feature(self._dataset_origin, self._extractor)

    # @property
    # def full_label_dict(self) -> dict:
    # 	return self._dataset_origin.full_label_dict

    @classmethod
    def to_feature(cls, ds:Dataset, extractor:callable) -> Dataset:
        """Feature transform of a compacted dataset of signal.

        The transformed database has a dictionary structure which contains the fields {'label', 'feature'}
        """
        n_channels = ds.element_spec['signal'].shape[0]

        @tf.function
        def _feature_map(X):
            Xf = tf.py_function(
                    func=lambda x, sr: extractor(x.numpy(), sr),  # makes it a tf callable. x.numpy() must be used inside the method `extractor()`
                    inp=[X['signal'], X['sampling_rate']],
                    Tout=_DTYPE
                )
            Xf.set_shape((n_channels, None, None))
            return {
                'label': X['label'],  # string label
                'metadata': X['metadata'],
                # 'feature': tf.reshape(Xf, tf.shape(Xf))  # has no effect
                'feature': Xf
            }

        return ds.map(_feature_map, num_parallel_calls=tf.data.AUTOTUNE)


class WindowSlider(AbstractDatasetTransformer):
    """Sliding window view of a time-frequency feature dataset.

    This class performs the sliding window view with downsampling on a feature-transformed dataset. Window views are time-frequency patches of a complete spectral feature. It is obtained by sliding a small window along the time-frequency axes.
    """

    def __init__(self, dataset:Dataset, *, window_size:tuple|int, hop_size:tuple|int=None, no_meta:bool=True):
        """
        Parameters
        ----------
        dataset
            feature dataset, must have a dictionary structure with the field 'feature', which contains the spectral feature and has dimension (channel, freqeuncy, time).
        window_size
            size of sliding window
        hop_size
            size of hop between two positions
        no_meta
            do not contain the field `metadata` in the transformed dataset

        Returns
        -------
        Transformed dataset of tuple form (label, info, window).

        Notes
        -----
        We follow the convertion of channel first here for both the input and the output dataset.
        """
        assert dataset.__transformer__ is FeatureExtractor
        self._dataset_origin = dataset
        self._n_channels = dataset.element_spec['feature'].shape[0]
        self._window_size = window_size
        self._hop_size = hop_size
        self._no_meta = no_meta

    def build(self):
        @tf.function
        def _drop_meta(X):
            return {'feature': X['feature'], 'label': X['label']}

        ds = self.to_windows(self._dataset_origin, self._window_size, self._hop_size)
        # ds = utils.restore_shape(ds, 'feature', self.data_dim)

        if self._no_meta:
            ds = ds.map(_drop_meta, num_parallel_calls=tf.data.AUTOTUNE)

        # return utils.restore_shape(ds)
        return ds

    # @property
    # def full_label_dict(self) -> dict:
    # 	return self._dataset_origin.full_label_dict

    @classmethod
    def to_windows(cls, dataset, window_size, hop_size):
        # output signature, see:
        # https://www.tensorflow.org/api_docs/python/tf/data/Dataset#from_generator

        return Dataset.from_generator(
            utils.sliding_window_generator(dataset, 'feature', window_size, hop_size),
            output_signature=dataset.element_spec,
        )

    @property
    def data_dim(self):
        """Dimension of the data vector.
        """
        try:
            self._data_dim
        except:
            self._data_dim = tuple(list(self.dataset.take(1))[0]['feature'].shape)
        return self._data_dim


class SpecAugment(AbstractDatasetTransformer):
    """Spectrogram augmentation of a dataset.

    This class performs random augmentations on a feature dataset:

    1. crop a random rectangle patch of the spectrogram
    2. randomly flip the time axis
    3. randomly fade along the time axis
    4. randomly blur the spectrogram

    These augmentations are applied in order and independently with probability (for step 2,3,4, step 1 is always applied).

    The input and output dataset are both channel-first.
    """
    def __init__(self, dataset:Dataset, *,
                 output_shape:tuple=(64,64),
                 crop_kwargs:dict={},
                 flip_kwargs:dict={'axis':-1, 'prob':0.},
                 blur_kwargs:dict={'sigma':1., 'prob':0.},
                 fade_kwargs:dict={'ratio':0.5, 'prob':0.},
                 **kwargs):
        # Type check turned off in order to take dataset after processing like `restore_cardinality()`:
        # assert dataset.__transformer__ is FeatureExtractor

        self._dataset_origin = dataset
        self._n_channels = dataset.element_spec['feature'].shape[0]
        self._output_shape = output_shape

        def _crop(x):
            return random_crop(x, output_shape, channel_axis=0, **crop_kwargs)[0]

        @randomly(flip_kwargs['prob'])
        def _flip(x):
            return np.flip(x, axis=flip_kwargs['axis'])

        @randomly(blur_kwargs['prob'])
        def _blur(x):
            return skimage.filters.gaussian(x, sigma=blur_kwargs['sigma'], channel_axis=0)

        @randomly(fade_kwargs['prob'])
        def _fade(x):
            return fade(x, fade_kwargs['ratio'])

        # _crop = lambda x: random_crop(x, output_shape, channel_axis=0, **crop_kwargs)

        # _flip = lambda x: randomly(flip_kwargs['prob'])(np.flip)(x, axis=flip_kwargs['axis'])

        # _blur = lambda x: randomly(blur_kwargs['prob'])(skimage.filters.gaussian)(x, sigma=blur_kwargs['sigma'], channel_axis=0)

        # _fade = lambda x: randomly(fade_kwargs['prob'])(fade)(x, fade_kwargs['ratio'])

        self.spec_aug = lambda x: _blur(_fade(_flip(_crop(x))))

    def build(self):
        @tf.function
        def _mapper(X):
            Y = X.copy()
            Y['feature'] = tf.py_function(
                func=self.spec_aug,
                inp=[X['feature']],
                Tout=_DTYPE
            )
            tf.reshape(Y['feature'], (-1, *self._output_shape))

            return Y

        return utils.restore_shape(
            self._dataset_origin.map(_mapper, num_parallel_calls=tf.data.AUTOTUNE),
            key='feature',
            shape=(self._n_channels, *self._output_shape)
        )

"""
```python
compactor = transformer.DatasetCompactor(ds0, keys=keys, channels=channels)
labels = list(compactor.full_label_dict.keys())
N = utils.get_dataset_size(ds0)

extractor = transformer.FeatureExtractor(compactor.dataset, _extractor)

specaugment = transformer.SpecAugment(extractor.dataset)

product = transformer.Product(specaugment.dataset, positive=None)
```

SpecAugment is a randomly mapped dataset. We can verify that the randomness is preserved through `repeat()`.
```python
ds = specaugment.dataset
eles = list(ds.repeat(2).take(N*2))

X = eles[0]['feature'].numpy()
Y = eles[N]['feature'].numpy()  # are different
```

Similarly, the randomness is also preserved through epoch:
```
eles1 = list(specaugment.dataset)
eles2 = list(specaugment.dataset)
X = eles1[n]['feature'].numpy()
Y = eles2[n]['feature'].numpy()  # are different
```

and also preserved through product:
```
ds = product.dataset
eles = list(ds.repeat(2).take(2*N**2))

X = eles[0][0]['feature'].numpy()
Y = eles[N**2][1]['feature'].numpy()  # are different
```
"""

class Product(AbstractDatasetTransformer):
    """Cartesian product of a dataset.

    A product dataset has samples `(x,y,flag)` with `x,y` sampled from an original dataset, and `flag` is the indicator that `x` and `y` are positive samples.

    By construction of the product dataset, iteration on the dataset will show repeated `x` in the output `(x,y,flag)`, which is the expected behavior. To break this partial determinism, use `shuffle()`.
    """
    def __init__(self, dataset:Dataset, dataset2:Dataset=None, *, keys:list=[], positive:bool=None):
        """
        Parameters
        ----------
        keys
            keys of the field `metadata` for pair comparison along with `label`.
        positive
            if True/False the positive pair (same label and metadata) will be retained/rejected. By default all samples are kept.
        """
        # self._fold = fold
        self._keys = keys
        self._positive = positive
        self._dataset_origin = dataset
        if dataset2 is None:
            self._dataset_origin2 = dataset
        else:
            self._dataset_origin2 = dataset2

    def build(self):
        @tf.function
        def _filter(X, Y):
            v = tf.reduce_all(
                [tf.equal(X['label'], Y['label'])] +
                [tf.equal(X['metadata'][k], Y['metadata'][k]) for k in self._keys]
                )
            return v
            # if self._positive:
            # 	return v
            # else:
            # 	return tf.logical_not(v)

        def _generator():
            for eles in itertools.product(self._dataset_origin, self._dataset_origin2):
                yield (*eles, _filter(*eles))
            #
            # the following is equivalent but much slower
            # for X in self._dataset_origin:
            # 	for Y in self._dataset_origin2:
            # 		yield (X, Y, _filter(X, Y))

        ds = tf.data.Dataset.from_generator(
            _generator,
            # output_signature=(self._dataset_origin.element_spec,)*2
            output_signature=(
                self._dataset_origin.element_spec,
                self._dataset_origin.element_spec,
                tf.TensorSpec(shape=(), dtype=tf.bool, name=None)  # type: ignore
            )
        )
        if self._positive is None:
            return ds
        else:
            return ds.filter(lambda x,y,v: v if self._positive else tf.logical_not(v))

            # 	return ds.map(lambda X,Y: {
            # 		'label': X['label'],
            # 		'metadata': X['metadata'],
            # 		'feature': (X['feature'], Y['feature'])
            # 	})


# Problem of determinism in random `map()` and `shuffle()`:
# https://github.com/tensorflow/tensorflow/issues/35682

__all__ = ['AbstractDatasetTransformer', 'DatasetCompactor', 'FeatureExtractor', 'WindowSlider', 'SpecAugment', 'Product']


# 	def build(self):
# 		def _generator():
# 			# counter = 0
# 			for X in self._dataset_origin:
# 				Y = X.copy()
# 				for n in range(self._n_view):
# 					v = []
# 					for n in range(2):
# 						v.append(
# 							tf.py_function(
# 								func=self.spec_aug,
# 								inp=[X['feature']],
# 								Tout=_DTYPE
# 							)
# 						)
# 					Y['feature'] = tuple(v)
# 					yield Y


