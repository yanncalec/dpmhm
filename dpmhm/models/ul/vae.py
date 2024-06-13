"""Classes of Variational Auto-Encoder.
"""

import tensorflow as tf
from keras import layers, models, backend as K
from math import pi
from dataclasses import dataclass
from .. import AbstractConfig


@dataclass
class Config(AbstractConfig):
    """Global parameters for AE.
    """
    # Dimension of hidden representation
    n_embedding:int = 2
    n_mc:int = 1  # Number of Monte-Carlo samples
    activation:str = 'sigmoid'
    use_bias:bool = False

    def optimizer(self):
        return tf.keras.optimizers.Adam()


class Gaussian_VAE(models.Model):
    """Gaussian VAE.

    Notes
    -----
    """
    def log_likelihood(self):
        """
        """
        pass

    def _ELBO_terms(self, X, Y, n_mc:int=1):
        n_batch = K.shape(X)[0]
        # n_dim = K.shape(X)[1:]  # dimension of data axes

        # z_mean, z_logvar = self.encoder(X)  # n_batch x n_latent
        z_mean, z_logvar = self.encoder_mean(X), self.encoder_logvar(X)  # n_batch x n_latent
        kl = - 0.5 * K.sum(1+z_logvar-K.square(z_mean)-K.exp(z_logvar), axis=-1)  # n_batch x 1

        # Z = Gaussian_VAE.sampling(z_mean, z_logvar, n_mc)  # n_batch x n_latent
        # x_mean, x_logvar = self.decoder_mean(Z), self.decoder_logvar(Z)  # n_batch x n_dim
        # d_axes = K.arange(K.ndim(x_mean))[1:]  # data axes, with 0,1 being the axis of mc samples and batch
        # vll = -K.sum(K.log(2*pi)+x_logvar+K.square(Y-x_mean)*K.exp(-x_logvar), axis=d_axes)/2  # n_batch x 1

        Z = K.reshape(Gaussian_VAE.sampling(z_mean, z_logvar, n_mc), [n_batch*n_mc,-1])  # (n_mc x n_batch) x n_latent
        x_mean, x_logvar = self.decoder_mean(Z), self.decoder_logvar(Z)  # (n_mc x n_batch) x n_dim
        xdim = K.concatenate([[n_mc, n_batch], K.shape(x_mean)[1:]], axis=0)
        x_mean = K.reshape(x_mean, xdim)
        x_logvar = K.reshape(x_logvar, xdim)
        # tf.print(x_mean, x_logvar)
        d_axes = K.arange(K.ndim(x_mean))[2:]  # data axes, with 0,1 being the axis of mc samples and batch
        # vll = K.mean(-K.sum(x_logvar+K.square(Y-x_mean)*K.exp(-x_logvar), axis=d_axes)/2, axis=0)  # n_batch x 1
        vll = K.mean(-K.sum(K.log(2*pi)+x_logvar+K.square(Y-x_mean)*K.exp(-x_logvar), axis=d_axes)/2, axis=0)  # n_batch x 1

        # # tf.print(tf.shape(kl), tf.shape(vll))
        return vll, kl  # variational likelihood and KL

    def ELBO(self, *args):
        vll, kl = self._ELBO_terms(*args)
        tf.print(K.sum(vll), K.sum(kl), K.sum(vll-kl))
        return K.sum(vll - kl, axis=0)

    @staticmethod
    def sampling(μ, logσ2, n:int=1):
        # epsilon = K.random_normal(shape=K.shape(μ), mean=0., stddev=1.)  # one sample only
        epsilon = K.random_normal(shape=K.concatenate([[n], K.shape(μ)],axis=0), mean=0., stddev=1.)
        return μ + K.exp(0.5*logσ2) * epsilon

    def __init__(self, c:Config, ae:models.Model):
        assert c.input_shape == ae._config.input_shape
        # assert c.n_embedding == ae._config.n_embedding
        self._config = c
        super().__init__()
        # self._ae = ae

        self.encoder_mean = models.Sequential([
            ae.encoder,
            layers.Flatten(name='flatten'),
            layers.Dense(c.n_embedding, activation=None, name='fc_enc_mean')
        ], name='encoder_mean')
        self.encoder_logvar = models.Sequential([
            ae.encoder,
            layers.Flatten(name='flatten'),
            # layers.Dense(c.n_embedding, activation=c.activation, name='fc_enc_logvar1'),
            layers.Dense(c.n_embedding, activation=None, use_bias=c.use_bias, name='fc_enc_logvar')
        ], name='encoder_logvar')

        self.decoder_mean = models.Sequential([
            layers.Input(shape=c.n_embedding, name='input_dec'),
            layers.Dense(ae.decoder.layers[0].input_shape[-1], activation=None, name='fc_dec'),
            ae.decoder,
        ], name='decoder_mean')
        self.decoder_logvar = models.Sequential([
            layers.Input(shape=c.n_embedding, name='input_dec'),
            # layers.Dense(ae.decoder.layers[0].input_shape[-1], activation=c.activation, name='fc_dec1'),
            layers.Dense(ae.decoder.layers[0].input_shape[-1], activation=c.activation, use_bias=c.use_bias, name='fc_dec'),
            ae.decoder,
        ], name='decoder_logvar')

        self.encoder = layers.Lambda(lambda x:(self.encoder_mean(x), self.encoder_logvar(x)))
        self.decoder = layers.Lambda(lambda z:(self.decoder_mean(z), self.decoder_logvar(z)))

        self.loss_func = lambda x, y: -self.ELBO(x, y, c.n_mc)
        self.loss_tracker = tf.keras.metrics.Mean(name='loss')

    # @tf.function
    def call(self, x):
        z_mean = self.encoder_mean(x)
        z_logvar = self.encoder_logvar(x)
        # sample from the encoder output
        # z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_logvar])

        # epsilon = K.random_normal(shape=(self._config.n_samples, *K.shape(z_mean)), mean=0., stddev=1.)
        # z = K.reduce_mean(z_mean + K.exp(0.5 * z_logvar) * epsilon, axis=0)
        epsilon = K.random_normal(shape=K.shape(z_mean), mean=0., stddev=1.)
        z = z_mean + K.exp(0.5 * z_logvar) * epsilon

        x_mean = self.decoder_mean(z)
        x_logvar = self.decoder_logvar(z)
        epsilon = K.random_normal(shape=K.shape(x_mean), mean=0., stddev=1.)
        return x_mean + K.exp(0.5 * x_logvar) * epsilon

    def train_step(self, inputs):
        with tf.GradientTape() as tape:
            loss = self.loss_func(*inputs)

        gradients = tape.gradient(loss, self.trainable_variables)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        self.loss_tracker.update_state(loss)
        return {'loss': self.loss_tracker.result()}
