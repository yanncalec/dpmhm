import tensorflow as tf
from keras import layers, models


class TDense(layers.Layer):
    """Transposed dense layer.
    """
    def __init__(self, fwd, **kwargs):
        super(TDense, self).__init__(**kwargs)
        self.fwd = fwd
        # self.bwd = models.Sequential([
        #     layers.Input(fwd.output_shape),
        #     layers.Dense(fwd.input_shape, activation=fwd.activation, use_bias=False, **kwargs)
        # ])
        # # self.trainable = False  # will change also the state of fwd ?!

    def call(self, inputs):
        # if self.fwd.bias is None:
        try:
            return tf.matmul(inputs-self.fwd.bias, tf.transpose(self.fwd.kernel))
        except:
            return tf.matmul(inputs, tf.transpose(self.fwd.kernel))


class TConv1D(layers.Layer):
    """Not tested.
    """
    def __init__(self, fwd, **kwargs):
        super(TConv1D, self).__init__(**kwargs)
        self.fwd = fwd
        if fwd.data_format == 'channels_last':
            input_shape = (None, fwd.output_shape[-1])
            filters = fwd.input_shape[-1]
        else:
            input_shape = (fwd.output_shape[1], None)
            filters = fwd.input_shape[1]

        config = fwd.get_config()
        config['use_bias'] = False
        config['filters'] = filters
        # config['name'] += '_transpose'

        self.bwd = models.Sequential([
            layers.Input(input_shape),
            # layers.Conv1DTranspose(filters, kernel_size=fwd.kernel_size, strides=fwd.strides, padding=fwd.padding, activation=fwd.activation, dtype=fwd.dtype, data_format=fwd.data_format, use_bias=False, **kwargs)
            layers.Conv1DTranspose(**config)
        ])
        # self.bwd.build(in)
        # self.trainable = False

    def call(self, inputs):
        # backward op initiated with the same kernel as the forward op.
        self.bwd.layers[-1].kernel = self.fwd.kernel
        try:
            return self.bwd(inputs-self.fwd.bias)  # time reversal is automatically handelled by Conv2DTransfpose.
        except:
            return self.bwd(inputs)


class TConv2D(layers.Layer):
    """Not tested.
    """
    def __init__(self, fwd, **kwargs):
        super(TConv2D, self).__init__(**kwargs)
        self.fwd = fwd
        if fwd.data_format == 'channels_last':
            input_shape = (None, None, fwd.output_shape[-1])
            filters = fwd.input_shape[-1]
        else:
            input_shape = (fwd.output_shape[1], None, None)
            filters = fwd.input_shape[1]

        config = fwd.get_config()
        config['use_bias'] = False
        config['filters'] = filters

        self.bwd = models.Sequential([
            layers.Input(input_shape),
            layers.Conv2DTranspose(**config)
        ])
        # self.bwd.build(in)
        # self.trainable = False

    def call(self, inputs):
        # backward op initiated with the same kernel as the forward op.
        self.bwd.layers[-1].kernel = self.fwd.kernel
        try:
            return self.bwd(inputs-self.fwd.bias)  # time reversal is automatically handelled by Conv2DTransfpose.
        except:
            return self.bwd(inputs)
