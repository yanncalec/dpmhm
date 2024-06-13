"""Methods for preprocessing datasets before training.

Convention
----------
The original dataset is channel first and becomes channel last after preprocessing.
"""

import numpy as np
# import scipy
import tensorflow as tf
from keras import layers, models, ops
# from tensorflow.keras import layers, models, ops
from tensorflow.data import Dataset

import logging
Logger = logging.getLogger(__name__)


def get_mapping_supervised(labels:list, *, feature_field:str='feature', label_field:str='label') -> callable:
    """Get a preprocessing mapping for supervised training.

    This processing model performs the following transformations on a dataset:
    - conversion of label from string to integer
    - conversion from the format of channel-first to channel-last

    After transformation, the dataset is in the format `(data, label)` where `label` is now integer and `data` is in channel-last format.

    Usage
    -----
    ```python
    func = get_mapping_supervised(labels)
    ds = ds.map(func, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.map(lambda x,y: (tf.ensure_shape(x, dimx), y)
    ```

    Note
    ----
    The `ensure_shape()` step seems necessary in Keras 3 to recover the dimension information lost after `.map()`. Otherwise the training will be failed.

    See also:
    https://github.com/tensorflow/tensorflow/issues/64177
    """
    label_layer = layers.StringLookup(
        # num_oov_indices=0,   # force zero-based integer
        vocabulary=labels,
        # output_mode='one_hot'
    )
    func = lambda x: (
            tf.transpose(x[feature_field], [1,2,0]),
            tf.cast(label_layer(x[label_field]), tf.int32)
            )
    return func


def nested_type_spec(sp:dict) -> dict:
    tp = {}
    for k,v in sp.items():
        if isinstance(v, dict):
            tp[k] = nested_type_spec(v)
        # elif
        else:
            # tp[k] = layers.Input(type_spec=v)  # Keras 2 only
            tp[k] = layers.Input(shape=v.shape, dtype=v.dtype, name=k)
    return tp


# The following method doesn't work in Keras 3. Use `get_mapping_supervised()` instead.

def keras_model_supervised(ds:Dataset, labels:list=None, normalize:bool=False, *, shape:tuple=None, feature_field:str='feature', label_field:str='label') -> models.Model:
    """Initialize a Keras preprocessing model for supervised training.

    The processing model performs the following transformations on a dataset:
    - conversion of label from string to integer
    - normalization of data
    - conversion from the format of channel first to channel last

    Parameters
    ----------
    ds
        input dataset. The feature field is supposed channel-first.
    labels
        list of string labels for lookup. If not given the labels will be automatically determined from the dataset.
    normalize
        if True, estimate the mean and variance by channel and apply the normalization.
    shape
        restore shape information of the feature
    feature_field
        name of the field of feature
    label_field
        name of the field of label

    Returns
    -------
    A Keras preprocessing model that can be applied on a dataset as `ds.map(lambda x: model(x))`
    """
    # For nested dataset
    # Gotcha: in Keras 3, nested input type is no more supported by `keras.models.Model()`.
    inputs = nested_type_spec(
        ds.element_spec
        )
    # Make a flat dataset by dropping nested fields
    inputs = {feature_field:inputs[feature_field], label_field:inputs[label_field]}
    feature_input = inputs[feature_field]
    label_input = inputs[label_field]

    # # For dataset of type (feature, label)
    # inputs = tuple(layers.Input(type_spec=v) for v in ds.element_spec)
    # feature_input, label_input = inputs

    # Label conversion: string to int/one-hot
    # Gotcha: by default the converted integer label is zero-based and has zero as the first label which represents the off-class. This increases the number of class by 1 and must be taken into account in the model fitting.
    label_layer = layers.StringLookup(
        # num_oov_indices=0,   # force zero-based integer
        vocabulary=labels,
        # output_mode='one_hot'
    )
    if labels is None:
        # Adaptation depends on the given dataset
        # label_layer.adapt([x[-1].numpy() for x in ds])
        label_layer.adapt([x['label'].numpy() for x in ds])

    label_output = label_layer(label_input)

    # Normalization
    # https://keras.io/api/layers/preprocessing_layers/numerical/normalization/
    if normalize:
        raise NotImplementedError()

        # Manually compute the mean and variance of the data
        X = np.asarray([x['feature'].numpy() for x in ds]).transpose([1,0,2,3]); X = X.reshape((X.shape[0],-1))
        # X = np.hstack([x[data_name].numpy().reshape((n_channels,-1)) for x in ds])
        mean = X.mean(axis=-1)
        variance = X.var(axis=-1)

        feature_layer = layers.Normalization(axis=0, mean=mean, variance=variance)  # have to provide mean and variance if axis=0 is used
        feature_output = feature_layer(feature_input)

        # # The following generates extra dimensions when applied on data
        # X = np.asarray([x[data_name].numpy() for x in ds]).transpose([0,2,3,1])
        # layer = keras.layers.Normalization(axis=-1)
        # layer.adapt(X)
        # feature_output = layer(tf.transpose(inputs[data_name], [1,2,0]))
        #
        # # This won't work
        # layer = keras.layers.Normalization(axis=0)
        # layer.adapt(X)  # complain about the unknown shape at dimension 0
    else:
        feature_output = feature_input

    # to channel-last
    feature_output = ops.transpose(feature_output, [0,2,3,1])

    # Restore the shape information
    if shape is not None:
        feature_output = ops.reshape(feature_output, shape)
    label_output = ops.reshape(label_output, ())

    outputs = (feature_output, label_output)

    model = models.Model(inputs, outputs)

    # def model(X):
    #     # Restore the shape information.
    #     x, y = _model(X)
    #     if shape is not None:
    #         # assert tf.size(x) == np.prod(shape)  # doesn't work
    #         # x.set_shape(shape)
    #         x = tf.reshape(x, shape)
    #     y.set_shape(())
    #     return x, y

    return model


__all__ = ['nested_type_spec', 'get_mapping_supervised']