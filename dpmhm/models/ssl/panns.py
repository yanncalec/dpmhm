""" Deep InfoMax.

References
----------
- Hjelm, R.D., Fedorov, A., Lavoie-Marchildon, S., Grewal, K., Bachman, P., Trischler, A., Bengio, Y., 2019. Learning deep representations by mutual information estimation and maximization. https://doi.org/10.48550/arXiv.1808.06670
- Belghazi, M.I., Baratin, A., Rajeswar, S., Ozair, S., Bengio, Y., Courville, A., Hjelm, R.D., 2021. MINE: Mutual Information Neural Estimation. https://doi.org/10.48550/arXiv.1801.04062
- Nowozin, S., Cseke, B., Tomioka, R., 2016. f-GAN: Training Generative Neural Samplers using Variational Divergence Minimization, in: Advances in Neural Information Processing Systems. Curran Associates, Inc.


Code
----
https://github.com/rdevon/DIM

https://github.com/rcalland/deep-INFOMAX

https://github.com/DuaneNielsen/DeepInfomaxPytorch

Baselines to be implemented:

- VAE: Kingma, D.P., Welling, M., 2013. Auto-Encoding Variational Bayes. arXiv:1312.6114 [cs, stat].
- beta-VAE: Higgins, I., Matthey, L., Pal, A., Burgess, C., Glorot, X., Botvinick, M., Mohamed, S., Lerchner, A., 2017. beta-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework. ICLR 2, 6.
Alemi, A.A., Fischer, I., Dillon, J.V., Murphy, K., 2019. Deep Variational Information Bottleneck. https://doi.org/10.48550/arXiv.1612.00410
- Adversarial AE (AAE): Makhzani, A., Shlens, J., Jaitly, N., Goodfellow, I., Frey, B., 2016. Adversarial Autoencoders. https://doi.org/10.48550/arXiv.1511.05644
- BiGAN: Donahue, J., Krähenbühl, P., Darrell, T., 2017. Adversarial Feature Learning. https://doi.org/10.48550/arXiv.1605.09782
Dumoulin, V., Belghazi, I., Poole, B., Mastropietro, O., Lamb, A., Arjovsky, M., Courville, A., 2017. Adversarially Learned Inference. https://doi.org/10.48550/arXiv.1606.00704
- Noise As Targets (NAT): Bojanowski, P., Joulin, A., 2017. Unsupervised Learning by Predicting Noise, in: Proceedings of the 34th International Conference on Machine Learning. Presented at the International Conference on Machine Learning, PMLR, pp. 517–526.
"""

import sys
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras, linalg
from tensorflow.keras import models, layers, regularizers, callbacks, losses
from tensorflow.keras.applications import resnet

from dataclasses import dataclass, field

# from tensorflow.keras.losses import cosine_similarity

# import numpy as np


@dataclass
class Config:
    # AUTO = tf.data.AUTOTUNE
    batch_size:int = 512
    epochs:int = 5
    # n_batch:int = ?  # data_size/batch_size
    training_steps:int = 10**4  # epochs * n_batch

    # encoder:dict = field(default_factory=lambda: dict(include_top=False, weights='imagenet', pooling='avg'))
    train_encoder:bool = True

    projector_units:int = 2048
    predictor_units:int = 512
    use_bias:bool = False
    weight_decay:float = 0.0005

    # projector:dict = field(default_factory=lambda: dict(units=2048, use_bias=False))

    @classmethod
    def from_dict(cls, obj: dict):
        return cls(**obj)

    def optimizer(self, learning_rate=0.05, weight_decay=1e-4, momentum=0.9):
        lr = keras.optimizers.schedules.CosineDecay(
            learning_rate * self.batch_size / 256,
            self.training_steps
        )
        return tfa.optimizers.SGDW(weight_decay=weight_decay, learning_rate=lr, momentum=momentum)


class SimSiam(models.Model):
    def __init__(self, input_shape, c:Config):
        # input_shape, train_encoder:bool=True, tau:float=1e-1, **kwargs):
        super().__init__()
        self._config = c
        self.loss_tracker = keras.metrics.Mean(name='loss')
        # self.mae_metric = keras.metrics.MeanAbsoluteError(name="mae")

        # config for the network
        self._encoder = resnet.ResNet50(input_shape=input_shape, include_top=False, weights='imagenet', pooling='avg')
        self._encoder.trainable = c.train_encoder

        self._projector = models.Sequential([
            layers.Flatten(name='flatten'),
            layers.Dense(c.projector_units, use_bias=c.use_bias, activation='relu', kernel_regularizer=regularizers.l2(c.weight_decay), name='proj_fc1'),
            layers.BatchNormalization(),
            # layers.Dense(c.projector_units, use_bias=c.use_bias, activation='relu', kernel_regularizer=regularizers.l2(c.weight_decay), name='proj_fc2'),
            # layers.BatchNormalization(),
            layers.Dense(c.projector_units, use_bias=c.use_bias, activation=None, kernel_regularizer=regularizers.l2(c.weight_decay), name='proj_fc2'),
            ], name='projector')

        # self._encoder_projector = models.Sequential([
        #     self._encoder,
        #     self._projector
        # ], name='encoder_projector')

        self._predictor = models.Sequential([
            # layers.Flatten(name='flatten'),
            layers.Dense(c.predictor_units, use_bias=c.use_bias, activation='relu', kernel_regularizer=regularizers.l2(c.weight_decay), name='pred_fc1'),
            layers.BatchNormalization(),
            layers.Dense(c.projector_units, activation=None, name='pred_fc2'),
        ], name='predictor')

        # self.trainable_variables = self._encoder.trainable_variables + self._projector.trainable_variables + self.predictor.trainable_variables

    # @property
    # def metrics(self):
    #     return [self.loss_tracker]

    @tf.function
    def call(self, inputs):
        x1, x2 = inputs  # treated as an iterator, not allowed in graph mode
        y1, y2 = self._projector(self._encoder(x1)), self._projector(self._encoder(x2))
        return y1, y2

    def _loss_func(self, p, z):
        return tf.reduce_mean(losses.cosine_similarity(p, tf.stop_gradient(z)), axis=0)

    def train_step(self, inputs):
        x1, x2 = inputs
        with tf.GradientTape() as tape:
            # z1, z2 = *self.call(inputs)
            z1, z2 = self._projector(self._encoder(x1)), self._projector(self._encoder(x2))
            p1, p2 = self._predictor(z1), self._predictor(z2)
            loss = (self._loss_func(p1, z2) + self._loss_func(p2, z1)) / 2

        # Compute gradients
        gradients = tape.gradient(loss, self.trainable_variables)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        # Update metrics (includes the metric that tracks the loss)
        # self.compiled_metrics.update_state(y, y_pred)
        # Return a dict mapping metric names to current value
        self.loss_tracker.update_state(loss)
        # return {m.name: m.result() for m in self.metrics}
        return {'loss': self.loss_tracker.result()}
