"""Dataset related.
"""

import tensorflow as tf
import hashlib
import json
import numpy as np


def md5_encoder(*args):
  """Encode a list of arguments to a string.
  """
  return hashlib.md5(json.dumps(args, sort_keys=True).encode('utf-8')).hexdigest()


def get_dataset_size(ds) -> int:
  """Get the number of elements of a dataset.
  """
  # count = ds.reduce(0, lambda x, _: x + 1)  # same efficiency
  count = 0
  for x in ds:
    count += 1
  return count


def extract_by_category(ds, labels:set=None):
  """Extract from a dataset the sub-datasets corresponding to the given categories.

  Args
  ----
  ds: tf.data.Dataset
    input dataset of dictionary structure which contains the field 'label'.
  labels: list
    list of category to be extracted. If not given the labels will be extracted by scanning the dataset (can be time-consuming).

  Returns
  -------
  a dictionary containing sub-dataset of each category.
  """
  if labels is None:
    labels = set([l.numpy() for x,l in ds])

  dp = {}
  for l in labels:
    dp[l] = ds.filter(lambda x: tf.equal(x['label'],l))
  return dp


def random_split_dataset(ds, splits:dict, *, shuffle_size:int=1, **kwargs):
  """Randomly split a dataset according to the specified ratio.

  Args
  ----
  ds: tf.data.Dataset
    input dataset.
  splits: dict
    dictionary specifying the name and ratio of the splits.
  shuffle_size: int
    size of shuffle, 1 for no shuffle, None for full shuffle.
  kwargs:
    other keywords arguments to the method `shuffle()`, e.g. `reshuffle_each_iteration=False`, `seed=1234`.

  Return
  ------
  A dictionary of datasets with the same keys as `split`.
  """
  assert all([v>=0 for k,v in splits.items()])
  assert tf.reduce_sum(list(splits.values())) == 1.

  try:
    ds_size = len(ds)  # will raise error if length is unknown
  except:
    ds_size = get_dataset_size(ds)
  assert ds_size >= 0
  # assert (ds_size := ds.cardinality()) >= 0

  # Specify seed to always have the same split distribution between runs
  # e.g. seed=1234
  ds = ds.shuffle(ds_size if shuffle_size is None else shuffle_size, **kwargs)

  keys = list(splits.keys())
  sp_size = {k: int(splits[k]*ds_size) for k in keys[:-1]}
#   print(sp_size)
  sp_size[keys[-1]] = ds_size - int(np.sum(list(sp_size.values())))
#   print(ds_size, int(np.sum(list(sp_size.values()))), sp_size)
  assert all([(splits[k]==0.) | (sp_size[k]>0) for k in keys]), "Empty split."

#   keys = list(splits.keys())
#   sp_size = {k: tf.cast(splits[k]*ds_size, tf.int64) for k in keys[:-1]}
#   print(sp_size)
#   sp_size[keys[-1]] = ds_size - tf.reduce_sum(list(sp_size.values()))
#   print(ds_size, tf.reduce_sum(list(sp_size.values())), sp_size)
#   assert all([(splits[k]==0.) | (sp_size[k]>0) for k in keys]), "Empty split."

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
    list of categories. If given apply the split per category otherwise apply it globally on the whole dataset.
  *args, **kwargs: arguments for `split_dataset_random()`

  Return
  ------
  A dictionary of datasets with the same keys as `split`.
  """
  if labels is None:
    dp = random_split_dataset(ds, splits, **kwargs)
  else:
    ds = extract_by_category(ds, labels)
    dp = {}
    for n, (k,v) in enumerate(ds.items()):
      # dp[k] = random_split_dataset(v, splits, **kwargs)
      if n == 0:
        dp.update(random_split_dataset(v, splits, **kwargs))
      else:
        dq = random_split_dataset(v, splits, **kwargs)
        for kk in dp.keys():
          dp[kk] = dp[kk].concatenate(dq[kk])

  return dp

