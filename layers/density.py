import tensorflow as tf
import tensorflow_probability as tfp
from .loss import normal_kld
tfd = tfp.distributions

class BaseDensity:
    def __call__(self, inputs):
        return self.call(inputs)

    def call(self, inputs):
        raise NotImplementedError('missing implementation for call')

    def reparameterize(self, parameters):
        """
        Applies reparameterization trick to an arbitrary density, given an array of
        tensors as inputs. Returns a reparameterized TFP Distribution.
        """
        raise NotImplementedError('missing implementation of reparameterize')

    def kld(self, z_0, z_k, ldj, parameters):
        """
        Computes the KL divergence for this density, given base distribution samples z_0,
        transformed samples z_k, the log det jacobian, and distribution parameters.
        """
        raise NotImplementedError('missing implementation of kld')


class GaussianDensity(BaseDensity):
    def __init__(self):
        super().__init__()
        self.dist_layer = tfp.layers.DistributionLambda(self.reparameterize)

    def call(self, inputs):
        return self.dist_layer(inputs)

    def reparameterize(self, parameters):
        z_mu, z_log_var = parameters
        z_stdev = tf.sqrt(tf.exp(z_log_var))
        dist = tfd.MultivariateNormalDiag(loc=z_mu, scale_diag=z_stdev)
        return dist

    def kld(self, z_0, z_k, ldj, parameters):
        z_mu, z_log_var = parameters
        return normal_kld(z_mu, z_log_var, z_0, z_k, ldj)
