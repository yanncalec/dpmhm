"""Dataset manipulation related.
"""

from typing import Union
# from sklearn.utils import shuffle
import tensorflow as tf
import hashlib
import json
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from numbers import Number
from tensorflow.data import Dataset


def md5_encoder(*args) -> str:
	"""Encode a list of arguments to a string.
	"""
	return hashlib.md5(json.dumps(args, sort_keys=True).encode('utf-8')).hexdigest()


def md5sum(fname:str) -> str:
	"""md5sum of a file.
	"""
	with open(fname, 'r') as fp:
		ms = hashlib.md5(fp.read().encode()).hexdigest()
	return ms


def get_dataset_size(ds:Dataset) -> int:
	"""Get the number of elements in a dataset.
	"""
	# count = ds.reduce(0, lambda x, _: x + 1)  # same efficiency
	count = 0
	for x in ds:
		count += 1
	return count


def extract_by_category(ds:Dataset, labels:list, *, key:int|str='label') -> dict:
	"""Extract from a dataset the sub-datasets corresponding to the given categories.

	Parameters
	----------
	ds
		dataset that can either have a dictionary structure which 	contains a label field, or a tuple structure `(data, label)`.
	labels
		categories (all distinct) to be extracted.
	key
		name or index of the label field for `ds` of nested or tuple structure

	Returns
	-------
	A dictionary containing sub-dataset of each category.
	"""
	dp = {}
	for l in labels:  # set(labels)
		if type(ds.element_spec) is dict:  # dictionary structure
			dp[l] = ds.filter(lambda x: tf.equal(x[key],l))
		else:  # otherwise must be a tuple structure
			dp[l] = ds.filter(lambda *x: tf.equal(x[key],l))

	return dp


def split_dims_generator(ds:Dataset, key:str, *, axis:int=0) -> callable:
    """Generator function for splitting dimensions.

    This generator create samples from a nd dataset by splitting a dimension. The new samples have the same rank as the original samples. For example, samples of shape `(3, None)` will be separated to 3 samples of shape `(1, None)`.

    Parameters
    ----------
    ds
        input dataset with a dictionary structure.
    key
        ds[key] is the signal to be separated.
	axis
		axis to be splitted
    """
    assert ds.element_spec[key].shape.ndims > 1

    def _get_generator():
        for X in ds:
            Y = X.copy()
            for n in range(tf.shape(X[key])[axis]):
                  Y[key] = tf.gather(X[key], [n], axis=axis)  # use `[n]` to preserve the rank
                  yield Y
			# For the 2D channel-first dataset, this is equivalent to:
            # for x in X[key]:
            #     Y[key] = tf.reshape(x, [1,-1])
            #     # Y[key] = x  # drop the channel dimension
            #     yield Y
    return _get_generator


def split_signal_generator(ds:Dataset, key:str, n_chunk:int, *, axis:int=-1) -> callable:
    """Generator function for splitting a signal into chunks.

    Parameters
    ----------
    ds
        input dataset with dictionary structure.
    key
        data field of `ds` to be divided.
    """
    def _get_generator():
        for X in ds:
            truncs = np.array_split(X[key], n_chunk, axis=axis)
            # truncs = tf.split(X[key], num_or_size_splits=n_chunk, axis=axis)
            Y = X.copy()
            for x in truncs:
                Y[key] = x
                yield Y
    return _get_generator


def sliding_window_generator(ds:Dataset, key:str, window_size:int|tuple, hop_size:int|tuple=None) -> callable:
    """Get the generator for sliding windows of view.
    """
    def _complete(s, w):
		# try:
		# 	return (*s[:-len(w)], *w)  # tuple or list
		# except:
		# 	return (*s[:-len(w)], w)  # int
        if isinstance(w, int):
            w = (*s[:-1], w)
        else:  # tuple or list
            w = (*s[:-len(w)], *w)
        return w

    # from numbers import Number
    ele = list(ds.take(1).as_numpy_iterator())[0][key]
    ndim = ele.ndim
    # ndim = len(ds.element_spec[key].shape)

    if ndim == 1:
        assert isinstance(window_size, int) or len(window_size)==1
        if hop_size is None:
            idx = (slice(None), slice(None))
        else:
            # assert isinstance(hop_size, int)
            idx = (slice(None,None,hop_size), slice(None))
    else:  # nd case
        window_size = _complete(ele.shape, window_size)

        assert ndim == len(window_size)
        if hop_size is None:
            idx = (slice(None),) * (ndim*2)
        else:
            hop_size = _complete(ele.shape, hop_size)
            # assert isinstance(hop_size, tuple)
            idx = tuple(slice(None,None,h) for h in hop_size) + (slice(None),) * ndim

    def _generator():
        for X in ds:
            W = sliding_window_view(X[key], window_size)[idx]
            Ws = W.reshape((-1,*W.shape[ndim:]))
            # print(Ws.shape, W.shape)
            Y = X.copy()
            for w in Ws:
                Y[key] = w
                yield Y

    return _generator


def random_split_dataset(ds:Dataset, splits:dict, *, shuffle_size:int=None, ds_size:int=None, **kwargs):
	"""Randomly split a dataset according to the specified ratio.

	Parameters
	----------
	ds
		input dataset.
	splits
		dictionary specifying the name and ratio of the splits.
	shuffle_size
		size of shuffle, 1 for no shuffle (deterministic), None for full shuffle.
	ds_size
		real size of `ds`.
	kwargs
		other keywords arguments to the method `shuffle()`, e.g. `reshuffle_each_iteration=False`, `seed=1234`.

	Returns
	-------
	A dictionary of datasets with the same keys as `split`.
	"""
	assert all([v>=0 for k,v in splits.items()])
	assert tf.reduce_sum(list(splits.values())) == 1.

	# Gotcha:
	# It may happen for a filtered dataset that `len(ds)` returns 1 and `ds.cardinality()` returns a negative number.
	# if necessary, scan the dataset to find out its real size
	if ds_size is None or ds_size < 0:
		ds_size = get_dataset_size(ds)

	if ds_size == 0: # empty dataset
		return {k: ds for k in splits.keys()}

	# Specify the random seed to always have the same split distribution
	# e.g. seed=1234
	if shuffle_size is None:
		shuffle_size = ds_size
	ds = ds.shuffle(shuffle_size, **kwargs)

	keys = list(splits.keys())
	sp_size = {k: int(splits[k]*ds_size) for k in keys[:-1]}
	sp_size[keys[-1]] = ds_size - int(np.sum(list(sp_size.values())))
	assert all([(splits[k]==0.) | (sp_size[k]>0) for k in keys]), "Empty split."

	dp = {}
	s = 0
	for k, v in sp_size.items():
		dp[k] = ds.skip(s).take(v)
		s += v
	return dp


def split_dataset(ds:Dataset, splits:dict={'train':0.7, 'val':0.2, 'test':0.1}, *, labels:list=None, key:str|int='label', **kwargs):
	"""Randomly split a dataset globally or per category.

	Parameters
	----------
	ds
		input dataset.
	splits
		dictionary specifying the name and ratio of the splits.
	labels
		list of categories. If given apply the few-shot style split (i.e. split per category) otherwise apply the normal split. This is incompatible with the keyword argument `ds_size`.
	kwargs
		arguments for `random_split_dataset()`

	Returns
	-------
	A dictionary of datasets with the same keys as `split`.

	Notes
	-----
	- In case of few-shot style split, the returned datasets are obtained by concatenating per-category split, hence need to be shuffled before use.
	- In case of normal split the dataset is first shuffled globally.
	"""
	if labels is None:
		dp = random_split_dataset(ds, splits, **kwargs)
	else:
		ds = extract_by_category(ds, labels, key=key)
		dp = {}
		for n, (k,v) in enumerate(ds.items()):
			try:
				dq = random_split_dataset(
					v, splits, ds_size=None, **kwargs
				)
			except:
				raise Exception("`ds_size` not supported in per category split")
			if n == 0:
				dp.update(dq)
			else:
				for s in splits.keys():
					dp[s] = dq[s].concatenate(dp[s])

	return dp


def restore_shape(ds:Dataset, *, key:str|int=None, shape:tuple[int]=None) -> Dataset:
	"""Restore the shape of a dataset.

	Parameters
	----------
	ds
		input dataset
	key
		data field name or index. For example, use `key=0` for a tuple dataset `(feature, label)`, and `key='feature'` for a nested dataset containing the field `'feature'`. If not set, `None` is used assuming a flat dataset `ds`.
	shape
		shape to be restored, if not provided it will be inferred from the dataset.

	Notes
	-----
	The shape information can be lost after transformations (an issue in Keras 3) and may cause the traning to fail. See:

	https://github.com/tensorflow/tensorflow/issues/64177
	"""
	if shape is None:
		try:
			shape = list(ds.take(1).as_numpy_iterator())[0].shape
		except:
			shape = list(ds.take(1).as_numpy_iterator())[0][key].shape
	# print(shape, key)

	@tf.function
	def _mapper(X):
		try:
			# flat dataset
			Y = tf.ensure_shape(X, shape)
		except:
			# nested dataset
			Y = X.copy()
			Y[key] = tf.ensure_shape(X[key], shape)
		return Y

	@tf.function
	def _mapper_tuple(*X):
		# tuple dataset
		# This code looks suspicious but actually works...
		Y = list(X)
		Y[key] = tf.ensure_shape(X[key], shape)
		return Y  # automatically converted back to tuple

	if type(ds.element_spec) is tuple:
		return ds.map(_mapper_tuple, num_parallel_calls=tf.data.AUTOTUNE)
	else:
		return ds.map(_mapper, num_parallel_calls=tf.data.AUTOTUNE)


def restore_cardinality(ds:Dataset, card:int=None) -> Dataset:
	"""Restore the cardinality of a dataset.
	"""
	if card is None:
		card = get_dataset_size(ds)

	return ds.apply(tf.data.experimental.assert_cardinality(card))


def constant_dataset(cst:float=None) -> Dataset:
	def _gen():
		while True:
			yield cst

	return Dataset.from_generator(_gen, output_types=tf.float32, output_shapes=())


def twins_dataset_ssl(ds0, key:str='feature', stack:bool=False, fake_label:bool=True):
	"""Make a twins dataset for self-supervised learning.

	Parameters
	----------
	ds0
		Input dataset.
	key, optional
		Name of the data field in `ds0`, by default 'feature'
	stack, optional
		Stack one copy on top of the other to make a single tensor, by default False
	fake_label, optional
		Add a fake label to the output dataset, by default True
	"""

	# To channel last format
	ds1 = restore_shape(
		ds0.map(lambda x: tf.transpose(x[key], [1,2,0]), num_parallel_calls=tf.data.AUTOTUNE),
	)
	input_shape = ds1.element_spec.shape

	# Paired dataset
	ds2 = tf.data.Dataset.zip(ds1, ds1)
	# ds2 = tf.data.Dataset.zip(ds1, ds1, ds1)  # or even more...

	if stack:
		ds2 = ds2.map(lambda x,y: tf.stack([x,y]))

	if fake_label:
		ds2 = tf.data.Dataset.zip(
			ds2,
			constant_dataset()
		)

	return ds2, input_shape
