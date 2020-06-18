import sys
import os
sys.path.append("/home/aa-user/ravidziv/neural-tangents")
from absl import app
from absl import flags
import jax.numpy as np
import jax
import jax.numpy as np
from jax import vmap, grad, jit, random
import collections
from functools import partial
from models import WideResnet
flags.DEFINE_string('datast', 'MNIST',
                   'The dataset to load.')
flags.DEFINE_integer('train_size', 10,
                     'Dataset size to use for training.')
flags.DEFINE_integer('test_size', 10,
                     'Dataset size to use for testing.')
flags.DEFINE_string('backend', 'gpu',  'The backend for computation')
flags.DEFINE_string('net_type', 'FC',  'The network to train')
flags.DEFINE_float('ts_min', -2, 'Min val for t (log)')
flags.DEFINE_float('ts_max', 8, 'max val for t (log)')
flags.DEFINE_integer('num_ts', 100, 'Number of ts')

flags.DEFINE_float('sigs_min', -2, 'Min val for t (log)')
flags.DEFINE_float('sigs_max', 0.3, 'max val for t (log)')
flags.DEFINE_integer('num_sigs', 100, 'Number of sigmas')
flags.DEFINE_bool('double_vmap', False, 'If do vmap on both sigs and ts')
FLAGS = flags.FLAGS
Result = collections.namedtuple('Result', 'train_bayes train_gibbs test_bayes test_gibbs analytical train_analytical')

import neural_tangents as nt
import tensorflow_probability as tfp
tfp = tfp.experimental.substrates.jax
tfd = tfp.distributions


def get_netwrok(net_type = 'FC', w_std=1., b_std=0.):
    if net_type == 'FC':
        init_fn, apply_fn, ker_fn = nt.stax.serial(
            nt.stax.Dense(1000, W_std=w_std, b_std=b_std),
            nt.stax.Erf(),
            nt.stax.Dense(1000, W_std=w_std, b_std=b_std),
            nt.stax.Erf(),
            nt.stax.Dense(1, W_std=w_std, b_std=b_std))
    if net_type == 'wide_resent':
        init_fn, apply_fn, ker_fn = WideResnet(block_size=2, k=1, num_classes=1, W_std=w_std, b_std=b_std)
    return init_fn, apply_fn, ker_fn


def maker(t=1e2, W_std=1.0, b_std=0., train_images=None, test_images=None, train_labels=None, test_labels=None):
    init_fn, apply_fn, ker_fn = get_netwrok(net_type=FLAGS.net_type, w_std=W_std, b_std=b_std)
    cut = train_images.shape[0]
    tcut = test_images.shape[0]
    # ker_fn = jit(ker_fn, static_argnums=(2,), backend=backend)
    ker_fn = nt.batch(ker_fn, batch_size=5, device_count=1, store_on_device=True)
    ker_fn = jit(ker_fn, static_argnums=(2,), backend=FLAGS.backend)

    predict_fn = nt.predict.gradient_descent_mse_gp(kernel_fn=ker_fn, x_train=train_images, y_train=train_labels[:cut],
                                                    x_test=test_images, get='ntk', diag_reg=0.0,
                                                    compute_cov=True)
    predict_fn = jit(predict_fn, backend=FLAGS.backend)
    train_predict_fn = nt.predict.gradient_descent_mse_gp(ker_fn, train_images, train_labels,
                                                          train_images, "ntk", 0.0,
                                                          compute_cov=True)
    train_predict_fn = jit(train_predict_fn, backend=FLAGS.backend)

    def make_losses(loss_train, tlbs, tcut):
        def bayes_loss():
            mu, sig = loss_train
            mu = mu.squeeze()
            return tfd.MultivariateNormalFullCovariance(mu, np.eye(mu.shape[0]) + sig).log_prob(
                tlbs).sum()

        def gibbs_loss():
            mu, sig = loss_train
            mu = mu.squeeze()
            return -0.5 * np.sum((tlbs - mu) ** 2) - 0.5 * np.trace(sig) - tcut / 2.0 * np.log(
                2 * np.pi)
        return bayes_loss, gibbs_loss

    pred_train = train_predict_fn(t)
    pred_test = predict_fn(t)
    bayes_loss, gibbs_loss = make_losses(pred_train, test_labels, tcut)
    train_bayes_loss, train_gibbs_loss = make_losses(pred_test, train_labels, cut)

    return Result(test_bayes=bayes_loss(), test_gibbs=gibbs_loss(),
                  train_bayes=train_bayes_loss(), train_gibbs=train_gibbs_loss())



def vmap_inner(sig, ts, backend, maker_part):
    """Vmap inside for loop due to memory limitations"""
    results = []
    for i in range(len(sig)):
        jmaker = partial(maker_part, W_std = sig[i])
        jmaker = jit(jmaker, backend=backend)
        print (i, len(sig))
        res = vmap(jmaker, (0))(ts)
        results.append(res)
    return results

def main(unused_argv):
    train_images, train_labels, test_images, test_labels = load_data()
    ts = np.logspace(FLAGS.ts_min, FLAGS.ts_max, FLAGS.num_ts)
    sigs = np.logspace(FLAGS.sigs_min, FLAGS.sigs_max, FLAGS.num_sigs)
    print (maker(t=1e2, W_std=1.0, train_images=train_images, test_images=test_images,
                     train_labels=train_labels, test_labels=test_labels))
    jmaker = partial(maker, train_images=train_images, test_images=test_images,
                     train_labels=train_labels, test_labels=test_labels)
    jmaker = jit(jmaker, backend=FLAGS.backend)
    #if we have enough memory
    if FLAGS.double_vmap:
        func = jit( jax.vmap(vmap(jmaker, in_axes = (0, None) ), in_axes= (None,0)))
    else:
        func = partial(vmap_inner, backend = FLAGS.backend, maker_part = jmaker)
    result = func(sigs, ts)
    print (1)
if __name__ == '__main__':
    app.run(main)