import tensorflow_probability as tfp
from jax import jit
tfp = tfp.experimental.substrates.jax
tfd = tfp.distributions
import neural_tangents as nt

from typing import Union, Tuple, Callable, Iterable, Dict, Any

import jax.numpy as np
import collections
loss_metrices_name = ['bayes_loss', 'gibbs_loss']
ixt_metrices_name = ['ixt_lower', 'ixt_upper']
dkl_metrices_name = ['dkl_output']
dist_metrices_name = ['parameter_distance']
itd_metrices_name = ['i_theta_data']
LOSS_MET = collections.namedtuple('loss_metrices',loss_metrices_name)
IXT_MET = collections.namedtuple('ixt_metrics',ixt_metrices_name)
DKL_MET = collections.namedtuple('dkl_output',dkl_metrices_name)
DIST_MET = collections.namedtuple('parameter_distance',dist_metrices_name)
ITD_MET = collections.namedtuple('i_theta_data',itd_metrices_name)

from jax import vmap
from functools import partial
import jax
def get_bayes_loss(gauss, tlbs):
    mu, sig = gauss
    mu = mu.squeeze()
    #TODO - check for multi dimenstions output - right now we look on each dimension separately and sum up
    gausian_ind = tfd.Independent(
        distribution=tfd.MultivariateNormalFullCovariance(mu.T, np  .tile(sig+np.eye(sig.shape[0]), (mu.T.shape[0], 1, 1))))
    return gausian_ind.log_prob(tlbs.T).sum()

def get_gibbs_loss(loss_train, tlbs, tcut):
        mu, sig = loss_train
        mu = mu.squeeze()
        loss = -0.5 * (np.linalg.norm(tlbs- mu, axis=0)**2) - 0.5 * np.trace(sig) - tcut / 2.0 * np.log(2 * np.pi)
        return np.sum(loss)

def get_losses(gauss, tlbs, init_pred_train=None):
    tcut = tlbs.shape[0]
    gibbs_loss = get_gibbs_loss(gauss, tlbs, tcut)
    bayes_loss = get_bayes_loss(gauss, tlbs)
    return LOSS_MET(bayes_loss = bayes_loss, gibbs_loss =gibbs_loss )

@jit
def get_info_nec(gauss, tlbs,init_pred_train=None, num_of_samples=2, th = 1e-10):
    rng = jax.random.PRNGKey(0)
    mu, sig = gauss
    mu = mu.squeeze()
    #pzx = tfd.Independent(distribution=tfd.Normal(loc=mu, scale=np.diag(sig+th)))
    # todo - Check the multivariate gaussian
    pzx = tfd.Independent(distribution=tfd.MultivariateNormalDiag(mu, np.tile(np.diag(sig) + th, (mu.shape[1], 1)).T))
    samples = pzx.sample(seed = rng, sample_shape = 1).squeeze()
    logprob_pxt = []
    for i in range(len(samples)):
        logprob_pxt_inner = pzx.log_prob(samples[i]).squeeze()
        logprob_pxt.append(logprob_pxt_inner)
    logprob_pxt = np.stack(logprob_pxt)
    im_lower = inner_info_nec_lower(logprob_pxt)
    im_upper = inner_info_nec_upper(logprob_pxt)
    return IXT_MET(ixt_lower = im_lower, ixt_upper =im_upper )

def remove_dig(A):
    return A[~np.eye(A.shape[0], dtype=bool)].reshape(A.shape[0], -1)

@jit
def inner_info_nec_lower(scores):
    """InfoNCE estimator for I(diX;T)  - van den Oord et al. (2018):"""
    nll = np.mean(np.diag(scores) - jax.scipy.special.logsumexp(scores, axis=1))
    K = scores.shape[1]
    mi = np.log(K) + nll
    return mi

@jit
def inner_info_nec_upper(scores):
    """InfoNCE estimator for I(diX;T)  - van den Oord et al. (2018):"""
    scores_no_diagonal = remove_dig(scores)
    nll = np.mean(np.diag(scores) - jax.scipy.special.logsumexp(scores_no_diagonal, axis=1))
    K = scores.shape[1]
    mi = np.log(K-1) + nll
    return mi

@jit
def get_kl_posterior_prior(gauss, tlbs, initial_gauss):
    """The dkl between the posterior and the prior distribution of the output -upper bound on I(Z;D|X)"""
    mu, sig = gauss
    mu = mu.squeeze()
    init_mu, init_sig = initial_gauss
    init_mu = init_mu.squeeze()
    #Todo check about multi dimmensional gaussian
    pz_xd = tfd.Independent(
        distribution=tfd.MultivariateNormalFullCovariance(mu.T,
                                                          np.tile(sig, (mu.T.shape[0], 1, 1))))
    pz_x_init = tfd.Independent(
        distribution=tfd.MultivariateNormalFullCovariance(init_mu.T,
                                                          np.tile(init_sig, (init_mu.T.shape[0], 1, 1))))
    kl = tfp.distributions.kl_divergence(pz_xd, pz_x_init)
    return DKL_MET(dkl_output = np.mean(kl))

@jit
def get_fisher(theta):
    """The fisher infomration is constant along the training - trace(Theta)"""
    return np.trace(theta)

@jit
def get_parameter_distance(y_train, kdd=None,  eigenspace=None):
    normalization = y_train.size

    expm1_fn =_make_times_expm1_fn(normalization)

    @nt.predict._jit_cpu(kdd)
    def predict(t=None):
        evals, evecs = eigenspace
        #fl, ufl = nt.predict._make_flatten_uflatten(g_td, y_train)

        op_evals = -expm1_fn(evals, 2*t)
        yexpm1y =np.einsum(
            'j,ji,i,ki,k',
            y_train.T, evecs, op_evals, evecs, y_train, optimize=True)
        kdd_exp1m = np.einsum('kj, ji,i,li->kl', kdd.nngp, evecs, op_evals, evecs, optimize=True)
        dist =  0.5*(np.trace(kdd_exp1m)+yexpm1y)
        return DIST_MET(parameter_distance = dist)
    return predict


def _make_times_expm1_fn(normalization):
  expm1_fn = nt.predict._make_expm1_fn(normalization)

  def _inv_expm1_fn(evals, dt):
    return expm1_fn(evals, dt) * np.abs(evals)

  return _inv_expm1_fn

@jit
def get_idt(y_train, eigenspace, kdd=None):
    """lower bound on I(\theta;D)"""
    normalization = y_train.size
    op_fn = _make_inv_expm1_fn_double(normalization)
    trace_ntk = np.trace(kdd.ntk)

    def predict(t):
        evals, evecs = eigenspace
        fl, ufl = nt.predict._make_flatten_uflatten(kdd.ntk, y_train)

        op_evals = -op_fn(evals, t)
        yexpm1y =np.einsum(
            'j...,ji,i,ki,k...',
            fl(y_train), evecs, op_evals, evecs, fl(y_train), optimize=True)
        trace_ntk_t = t*trace_ntk
        kdd_exp1m = np.einsum('kj, ji,i,li->kl', kdd.nngp, evecs, op_evals, evecs, optimize=True)
        idt = np.trace(kdd_exp1m) +np.mean(yexpm1y)+trace_ntk_t
        return idt
    return predict

def _make_flatten_uflatten(g_td, y_train):
  """Create the flatten and unflatten utilities."""
  output_dimension = y_train.shape[-1]

  def fl(fx):
    """Flatten outputs."""
    return np.reshape(fx, (-1,))

  def ufl(fx):
    """Unflatten outputs."""
    return np.reshape(fx, (-1, output_dimension))

  return fl, ufl

def _make_inv_expm1_fn_double(normalization):
  expm1_fn = nt.predict._make_expm1_fn(normalization)

  def _inv_expm1_fn(evals, dt):
    return (expm1_fn(evals, dt)**2) / np.abs(evals)

  return _inv_expm1_fn