import tensorflow_probability as tfp
tfp = tfp.experimental.substrates.jax
tfd = tfp.distributions
import jax.numpy as np
import collections
loss_metrices_name = ['bayes_loss', 'gibbs_loss']
loss_metrics = collections.namedtuple('loss_metrices',loss_metrices_name)


def get_bayes_loss(loss_train, tlbs):
    mu, sig = loss_train
    mu = mu.squeeze()
    return tfd.MultivariateNormalFullCovariance(mu, np.eye(mu.shape[0]) + sig).log_prob(tlbs).sum()

def get_gibbs_loss(loss_train, tlbs, tcut):
        mu, sig = loss_train
        mu = mu.squeeze()
        return -0.5 * np.sum((tlbs - mu) ** 2) - 0.5 * np.trace(sig) - tcut / 2.0 * np.log(
            2 * np.pi)

def get_losses(loss_train, tlbs, tcut):
    gibbs_loss = get_gibbs_loss(loss_train, tlbs, tcut)
    bayes_loss = get_bayes_loss(loss_train, tlbs)
    return loss_metrics(bayes_loss = bayes_loss, gibbs_loss =gibbs_loss )
