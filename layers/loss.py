import tensorflow as tf

@tf.function
def normal_kld(z_mu, z_log_var, z_0, z_k, ldj):
    """
    Computes the Gaussian KL divergence for the given embedding parameters, initial density z_0,
    latent density z_k, and log det jacobian.
    """
    log_qz0 = tf.reduce_sum(-0.5*(z_log_var + (z_0 - z_mu)**2 / tf.exp(z_log_var)), axis=1)
    log_pzk = tf.reduce_sum(-0.5*z_k**2, axis=1)
    kld = tf.reduce_mean(log_qz0 - log_pzk - ldj)
    return kld
