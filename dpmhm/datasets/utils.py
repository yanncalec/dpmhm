"""Dataset manipulation related.
"""

# from sklearn.utils import shuffle
import tensorflow as tf
import hashlib
import json
import numpy as np


def md5_encoder(*args):
  """Encode a list of arguments to a string.
  """
  return hashlib.md5(json.dumps(args, sort_keys=True).encode('utf-8')).hexdigest()


def md5sum(fname:str):
  """md5sum of a file.
  """
  with open(fname, 'r') as fp:
    ms = hashlib.md5(fp.read().encode()).hexdigest()
  return ms


def get_dataset_size(ds) -> int:
  """Get the number of elements in a dataset.
  """
  # count = ds.reduce(0, lambda x, _: x + 1)  # same efficiency
  count = 0
  for x in ds:
    count += 1
  return count


def extract_by_category(ds, labels:list):
  """Extract from a dataset the sub-datasets corresponding to the given categories.

  Args
  ----
  ds: tf.data.Dataset
    input dataset of dictionary structure which contains the field 'label'.
  labels: list
    label of categories to be extracted.

  Returns
  -------
  a dictionary containing sub-dataset of each category.
  """
  dp = {}
  for l in set(labels):
    dp[l] = ds.filter(lambda x: tf.equal(x['label'],l))
  return dp


def split_signal_generator(ds, key:str, n_trunk:int):
	"""Generator function for splitting a signal into trunks.

	Args
	----
	ds:
		input dataset with dictionary structure.
	key: str
		ds[key] is the signal to be divided.
	"""
	def _get_generator():
		for X in ds:
			truncs = np.array_split(X[key], n_trunk, axis=-1)
			# truncs = tf.split(X[key], num_or_size_splits=n_trunk, axis=-1)
			Y = X.copy()
			for x in truncs:
				Y[key] = x
				yield Y
	return _get_generator


def random_split_dataset(ds, splits:dict, *, shuffle_size:int=None, **kwargs):
  """Randomly split a dataset according to the specified ratio.

  Args
  ----
  ds: tf.data.Dataset
    input dataset.
  splits: dict
    dictionary specifying the name and ratio of the splits.
  shuffle_size: int
    size of shuffle, 1 for no shuffle (deterministic), None for full shuffle.
  kwargs:
    other keywords arguments to the method `shuffle()`, e.g. `reshuffle_each_iteration=False`, `seed=1234`.

  Return
  ------
  A dictionary of datasets with the same keys as `split`.
  """
  assert all([v>=0 for k,v in splits.items()])
  assert tf.reduce_sum(list(splits.values())) == 1.

  # Gotcha:
  # It may happen for a filtered dataset that `len(ds)` returns 1 and `ds.cardinality()` returns a negative number.
  ds_size = get_dataset_size(ds)

  if ds_size == 0: # empty dataset
    return {k: ds for k in splits.keys()}

  # Specify seed to always have the same split distribution between runs
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


def split_dataset(ds, splits:dict={'train':0.7, 'val':0.2, 'test':0.1}, *, labels:list=None, **kwargs):
  """Randomly split a dataset either on either global or per category basis.

  Args
  ----
  ds: tf.data.Dataset
    input dataset with element of type (value, label).
  splits: dict
    dictionary specifying the name and ratio of the splits.
  labels: list
    list of categories. If given apply the few-shot style split (i.e. split per category) otherwise apply the normal split.
  *args, **kwargs: arguments for `split_dataset_random()`

  Return
  ------
  A dictionary of datasets with the same keys as `split`.

  Notes
  -----
  - In case of few-shot style split, the returned datasets are obtained by concatenating per-category split, hence need to be shuffled before use.
  - In case of normal split the dataset is first shuffled globally.
  """
  if labels is None:
    dp = random_split_dataset(ds, splits, **kwargs)
  else:
    ds = extract_by_category(ds, labels)
    dp = {}
    for n, (k,v) in enumerate(ds.items()):
      dq = random_split_dataset(v, splits, **kwargs)
      if n == 0:
        dp.update(dq)
      else:
        for s in splits.keys():
          dp[s] = dq[s].concatenate(dp[s])

  return dp

