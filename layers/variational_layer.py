import tensorflow as tf
import tensorflow.keras.layers as layers
from .density import BaseDensity, GaussianDensity

class VariationalLayer(layers.Layer):
    def __init__(self, flow, density=GaussianDensity(), min_beta=0.01, max_beta=1):
        """
        Creates a new VariationalLayer with the given base density. If 'flow' is None, no transform
        is applied to the base density, thus making reparameterization equivalent to
        a canonical VAE.

        flow : a Flow that will be applied to the base density; None for vanilla VAE
        density : base density to reparameterize; defaults to spherical Gaussian
        """
        super().__init__()
        self.flow = flow
        self.density = density
        self.min_beta = min_beta
        self.max_beta = max_beta
        self.beta = tf.Variable(min_beta, dtype=tf.float32, trainable=False)

    def call(self, inputs):
        """
        Requires parameters: *density parameters, flow parameters (if flow is present)
        Returns reparameterized z_0, transformed z_k, summed log det jacobian
        """
        if self.flow is not None:
            density_params, flow_params = inputs[:-1], inputs[-1]
        else:
            density_params = inputs
        # reparameterize z_mu, z_log_var
        z_0 = self.density(density_params)
        # compute forward flow
        z_k, ldj = self.flow.forward(z_0, flow_params) if self.flow is not None else (z_0, tf.constant(0.))
        # compute KL divergence
        kld = self.density.kld(z_0, z_k, ldj, density_params)
        self.add_loss(self.beta*kld)
        return z_0, z_k, ldj, kld

    def set_beta(value):
        self.beta.assign(tf.maximum(tf.minimum(value, self.max_beta), self.min_beta))
