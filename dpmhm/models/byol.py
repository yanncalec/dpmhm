"""BYOL.

References
----------
- Grill, J.-B., Strub, F., Altché, F., Tallec, C., Richemond, P., Buchatskaya, E., Doersch, C., Avila Pires, B., Guo, Z., Gheshlaghi Azar, M., Piot, B., kavukcuoglu,  koray, Munos, R., Valko, M., 2020. Bootstrap Your Own Latent - A New Approach to Self-Supervised Learning, in: Advances in Neural Information Processing Systems. Curran Associates, Inc., pp. 21271–21284.
- Niizumi, D., Takeuchi, D., Ohishi, Y., Harada, N., Kashino, K., 2021. Byol for audio: Self-supervised learning for general-purpose audio representation, in: 2021 International Joint Conference on Neural Networks (IJCNN). IEEE, pp. 1–8.
- Niizumi, D., Takeuchi, D., Ohishi, Y., Harada, N., Kashino, K., 2022. BYOL for Audio: Exploring Pre-trained General-purpose Audio Representations. https://doi.org/10.48550/arXiv.2204.07402
- Elbanna, G., Scheidwasser-Clow, N., Kegler, M., Beckmann, P., Hajal, K.E., Cernak, M., 2022. BYOL-S: Learning Self-supervised Speech Representations by Bootstrapping. https://doi.org/10.48550/arXiv.2206.12038
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import models, layers, regularizers, callbacks, losses
from keras.applications import resnet
# import numpy as np
from math import cos, pi

# https://keras.io/guides/writing_your_own_callbacks/#writing-your-own-callbacks
# https://keras.io/api/callbacks/base_callback/

"""
rate = tf.Variable(0.90)
ema = tf.train.ExponentialMovingAverage(rate)
rate.assign(0.5)
ema._decay # 0.5
"""

class BYOL_Callback(callbacks.Callback):
    """
    https://keras.io/api/callbacks/base_callback/
    """
    def on_train_batch_end(self, batch, logs=None):
        # `self.params` has keys ['verbose', 'epochs', 'steps']
        try:
            t = cos((1-(batch+1)/self.params['steps'])*pi/2)
            # t = (batch+1)/self.params['steps']
            # assert 0 <= t <= 1
            rate = t + (1-t)*self.model._ema_rate_base
        except:
            rate = self.model._ema_rate_base
        self.model._ema_rate.assign(rate)
        # Call `set_weights` in graph mode would raise `NotImplementedError`
        self.model._target.set_weights([self.model._ema.average(v)
                                        for v in self.model._encoder_projector.variables])


# https://keras.io/guides/customizing_what_happens_in_fit
# https://keras.io/guides/making_new_layers_and_models_via_subclassing/#the-model-class

"""
def similarity(X, Y, axis=-1):
    res = []
    for x,y in zip(X,Y):
        res.append(-np.dot(x,y)/np.sqrt(np.dot(x,x)*np.dot(y,y)))
    return np.asarray(res)

X = np.random.randn(10,2)
Y = np.random.randn(10,2)

Then `similarity(X,Y)` has the same value as `keras.losses.cosine_similarity(X,Y)`
"""

class BYOL(models.Model):
    def __init__(self, input_shape, train_encoder:bool=True, **kwargs):
        super().__init__()
        self.loss_tracker = keras.metrics.Mean(name='loss')
        # self.mae_metric = keras.metrics.MeanAbsoluteError(name="mae")

        # config for the network
        # weights can also be loaded from a path, which takes the same amount of time ~1s
        self._encoder = resnet.ResNet50(include_top=False, weights='imagenet', input_shape=input_shape, pooling='avg')
        self._encoder.trainable = train_encoder

        self._projector = models.Sequential([
            layers.Flatten(name='flatten'),
            layers.Dense(4096, activation='relu', name='fc1'),
            layers.BatchNormalization(),
            layers.Dense(256, activation=None, name='fc2'),
        ], name='projector')

        self._predictor = models.Sequential([
            layers.Flatten(name='flatten'),
            layers.Dense(4096, activation='relu', name='fc1'),
            layers.BatchNormalization(),
            layers.Dense(256, activation=None, name='fc2'),
        ], name='predictor')

        self._encoder_projector = models.Sequential([
            self._encoder,
            self._projector
        ], name='encoder_projector')

        self._online = models.Sequential([
            self._encoder_projector,
            self._predictor
        ], name='online')

        self._target = models.clone_model(self._encoder_projector)
        self._target.trainable = False

        self._ema_rate_base = 0.996
        self._ema_rate = tf.Variable(self._ema_rate_base, trainable=False)
        # assigning new rate will change automatically ema._decay
        self._ema = tf.train.ExponentialMovingAverage(self._ema_rate)
        self._ema.apply(self._encoder_projector.variables)  # create shadow copy of weights

    def call(self, inputs):
        # `call()` can take only one argument:
        # `def call(self, x1, x2)` will raise error
        x1, x2 = inputs
        # # https://keras.io/guides/making_new_layers_and_models_via_subclassing/#the-addloss-method
        # loss1 = 2 + 2 * losses.cosine_similarity(self._online(x1), self._target(x2))
        # loss2 = 2 + 2 * losses.cosine_similarity(self._online(x2), self._target(x1))
        # self.add_loss(loss1 + loss2)
        return self._online(x1), self._target(x2)

    def train_step(self, inputs):
        # https://keras.io/guides/customizing_what_happens_in_fit
        x1, x2 = inputs
        with tf.GradientTape() as tape:
            # https://keras.io/api/losses/regression_losses/#cosinesimilarity-function
            # `cosine_similarity` already has the sign inverted, hence `+` in place of `-`
            # loss1 = 2 + 2 * losses.cosine_similarity(self._online(x1), self._target(x2))
            # loss2 = 2 + 2 * losses.cosine_similarity(self._online(x2), self._target(x1))
            # or equivalently
            loss1 = 2 + 2 * losses.cosine_similarity(*self.call((x1, x2)))
            loss2 = 2 + 2 * losses.cosine_similarity(*self.call((x2, x1)))

            # loss1 = self.compiled_loss(*self.call((x1, x2), training=True), regularization_losses=self.losses)
            # loss2 = self.compiled_loss(*self.call((x2, x1), training=True), regularization_losses=self.losses)

            loss = loss1 + loss2

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # # Update metrics (includes the metric that tracks the loss)
        # self.compiled_metrics.update_state(y, y_pred)
        # # Return a dict mapping metric names to current value
        # return {m.name: m.result() for m in self.metrics}

        # Compute our own metrics
        self.loss_tracker.update_state(loss)
        # self.mae_metric.update_state(y, y_pred)
        # return {'loss': self.loss_tracker.result(), "mae": self.mae_metric.result()}
        return {'loss': self.loss_tracker.result()}

#     def test_step(self, data):
#         pass



#         self.strategy = tf.distribute.MirroredStrategy()

#         # EMA:
#         # https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
#         self.ema = tfm.optimization.ExponentialMovingAverage(self.optimizer, average_decay=0.996,
#                                                              dynamic_decay=False)
#         self.ema.shadow_copy(self)
# #         self.ema = tf.train.ExponentialMovingAverage(0.996)
# #         self.step_var = tf.Variable(0, dtype=tf.int64)  # tracking the iteration (i.e. the batch number) inside an epoch

#     def compile(self, *args, **kwargs):
#         super().compile(self, *args, **kwargs)
#         # EMA:
#         # https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
#         try:
#         self.ema = tfm.optimization.ExponentialMovingAverage(self.optimizer, average_decay=0.996,
#                                                              dynamic_decay=False)
#         self.ema.shadow_copy(self)
# #         self.ema = tf.train.ExponentialMovingAverage(0.996)
# #         self.step_var = tf.Variable(0, dtype=tf.int64)  # tracking the iteration (i.e. the batch number) inside an epoch

#     @property
#     def ema(self):
#         try:
#             return self._ema
#         except:
# #             self._ema = tfa.optimizers.MovingAverage(self.optimizer, average_decay=0.996,
# #                                                                  dynamic_decay=False)
# #             self._ema.shadow_copy(self.weights)
# #             self._ema = tfm.optimization.ExponentialMovingAverage(self.optimizer, average_decay=0.996,
# #                                                                  dynamic_decay=False)
# #             self._ema.shadow_copy()
#             self._ema = tf.train.ExponentialMovingAverage(0.996)
#             self._ema.apply(self.encoder_projector.variables)
#             return self._ema

#     @tf.function
#     def _call_online(self, x):
#         return self.predictor(self.projector(self.encoder(x)))

#     @tf.function
#     def _call_target(self, x):
# #         with self.strategy.scope():
# #             self.ema.swap_weights()
# #             y = self.projector(self.encoder(x))  # with EMA
# #             self.ema.swap_weights()

# #         self._target.set_weights([self.ema.average(v) for v in self.encoder_projector.variables])
# #         target_network.build((None, 10)) # replace 10 with number of variables in input layer
# #         model_copy.compile(optimizer='rmsprop', loss='categorical_crossentropy')
# #         tf.train.Checkpoint()
#         y = self._target(x)  # with EMA
#         return y

#     def _call_target(self, x):
#         self.ema.swap_weights()
#         y = self.projector(self.encoder(x))  # with EMA
#         self.ema.swap_weights()
#         return y
# #         raise NotImplementedError()


__all__ = ['BYOL_Callback', 'BYOL']