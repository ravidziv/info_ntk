"""
This file is an example of how to run and calculate different information measures."""
import sys
from jax.config import config
config.update('jax_disable_jit', True)
sys.path.append("/home/aa-user/ravidziv/neural-tangents")
from absl import app
from absl import flags
import jax
import pandas as pd
import jax.numpy as np
from jax import vmap, jit
import collections
from info_ntk.datasets import load_data
from functools import partial
from info_ntk.models import WideResnet
from info_ntk.utils import create_df, reorder_dict
flags.DEFINE_string('dataset', 'MNIST',
                   'The dataset to load.')
flags.DEFINE_integer('train_size', 20,
                     'Dataset size to use for training.')
flags.DEFINE_integer('test_size', 20,
                     'Dataset size to use for testing.')
flags.DEFINE_string('backend', 'gpu',  'The backend for computation')
flags.DEFINE_string('net_type', 'FC',  ' The network to train')
flags.DEFINE_string('save_path', './results.csv',  'The path to save the results')
flags.DEFINE_multi_string('metrices', ['i_theta_data', 'losses','ixt', 'dkl_output', 'fisher'],  'which metrices to calculate  '
                                                         '- i_theta_data, losses,ixt, dkl_output, fisher')
flags.DEFINE_float('ts_min', 0.1, 'Min val for t (log)')
flags.DEFINE_float('ts_max', 1, 'max val for t (log)')
flags.DEFINE_float('b_std', 0.0, 'initial b std')
flags.DEFINE_integer('num_ts', 3, 'Number of ts')

flags.DEFINE_float('sigs_min', 0, 'Min val for t (log)')
flags.DEFINE_float('sigs_max', 0.3, 'max val for t (log)')
flags.DEFINE_integer('num_sigs', 2, 'Number of sigmas')
flags.DEFINE_bool('double_vmap',True, 'If do vmap on both sigs and ts')
flags.DEFINE_integer('batch_size', 5, 'The size of the minibatch for the kernels')

FLAGS = flags.FLAGS
from info_ntk.metrics import  loss_metrices_name, ixt_metrices_name, dkl_metrices_name, dist_metrices_name
from info_ntk.metrics import get_losses, get_info_nec, get_kl_posterior_prior, get_fisher, get_parameter_distance, get_idt
import neural_tangents as nt
import tensorflow_probability as tfp
tfp = tfp.experimental.substrates.jax
tfd = tfp.distributions

MetricesTuple = collections.namedtuple('metrices',['names', 'func', 'args'])
Results = collections.namedtuple('Results',['name', 'value', 'mode'] )
metrices = {'losses':MetricesTuple(loss_metrices_name, get_losses, args = {}),
            'ixt':MetricesTuple(ixt_metrices_name, get_info_nec, args= {'num_of_samples':2}),
            'dkl_output':MetricesTuple(dkl_metrices_name, get_kl_posterior_prior, args= {})
            }
metrices_names = loss_metrices_name +ixt_metrices_name +dkl_metrices_name
Metrics_vals = collections.namedtuple('metrics_vals',
                                      ['{}_train'.format(loss) for  loss in metrices_names]
                                      + ['{}_val'.format(loss) for  loss in metrices_names],
                                      defaults=(None,) * 2*len(metrices_names))


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

@partial(jax.jit, static_argnums=(3,))
def maker(t=1e2, W_std=1.0, b_std=0., metrics_func= [], train_images=None, test_images=None, train_labels=None,
          test_labels=None):
    init_fn, apply_fn, ker_fn = get_netwrok(net_type=FLAGS.net_type, w_std=W_std, b_std=b_std)
    ker_fn = nt.batch(ker_fn, batch_size=FLAGS.batch_size, device_count=-1, store_on_device=True)
    ker_fn = jit(ker_fn, static_argnums=(2,), backend=FLAGS.backend)

    predict_fn = nt.predict.gradient_descent_mse_gp(kernel_fn=ker_fn, x_train=train_images, y_train=train_labels,
                                                    x_test=test_images, get='ntk', diag_reg=0.0,
                                                    compute_cov=True)
    predict_fn = jit(predict_fn, backend=FLAGS.backend)
    train_predict_fn = nt.predict.gradient_descent_mse_gp(ker_fn, train_images, train_labels,
                                                          train_images, "ntk", 0.0,
                                                          compute_cov=True)
    train_predict_fn = jit(train_predict_fn, backend=FLAGS.backend)
    pred_train = train_predict_fn(t)
    pred_test = predict_fn(t)
    dict_val = {}

    init_pred_train = train_predict_fn(0)
    init_pred_test = predict_fn(0)
    kdd, ktd, nngp_tt = nt.predict._get_matrices(ker_fn, train_images, test_images, 'ntk',
                                                 compute_cov=True)
    if 'fisher' in FLAGS.metrices:
        train_fisher = get_fisher(kdd.ntk)
        test_fisher = get_fisher(ktd.ntk)
        dict_val['fisher_train'] = train_fisher
        dict_val['fisher_val'] = test_fisher
    if 'parameter_distance' in FLAGS.metrices or 'i_theta_data' in FLAGS.metrices:
        k_dd_plus_reg = nt.predict._add_diagonal_regularizer(kdd.ntk, diag_reg=0.)
        eigenspace = np.linalg.eigh(k_dd_plus_reg)
        if 'parameter_distance' in FLAGS.metrices:
            pred_parameter_distance = get_parameter_distance(y_train =train_labels , kdd=kdd, eigenspace=eigenspace)
            dict_val['parameter_distance'] = pred_parameter_distance(t)
        if 'i_theta_data' in FLAGS.metrices:
            pred_idt = get_idt(y_train =train_labels , kdd=kdd, eigenspace=eigenspace)
            dict_val['i_theta_data'] = pred_idt(t)
    #fallten the metric func to keys with train and test
    for metrice_dict in metrics_func:
        metrices_name = metrice_dict.names
        metric_func_inner = metrice_dict.func
        args =  metrice_dict.args
        train_metric_vals = metric_func_inner(pred_train, test_labels, init_pred_train,  **args)
        test_metric_vals = metric_func_inner(pred_test, test_labels, init_pred_test, **args)
        for metric_name in metrices_name:
            train_val =getattr(train_metric_vals, metric_name)
            test_val =  getattr(test_metric_vals, metric_name)
            dict_val['{}_train'.format(metric_name)] =train_val
            dict_val['{}_val'.format(metric_name)] =test_val
    return dict_val

@partial(jax.jit, static_argnums=(3,4))
def vmap_inner(ts, sig, b_std,
               metrics_func, maker_part, backend='gpu' ):
    """Vmap inside for loop due to memory limitations"""
    results = []
    for i in range(len(sig)):
        print (i, len(sig))
        #res = maker_part(ts[0], sig[i],b_std, metrics_func)
        res = vmap(maker_part, (0, None, None, None))(ts, sig[i],b_std, metrics_func)
        results.append(res)
    return results


def main(unused_argv):
    train_images, train_labels, test_images, test_labels = load_data(datast=FLAGS.dataset,
                                                                     train_size=FLAGS.train_size,
                                                                     test_size=FLAGS.test_size)
    ts = np.logspace(FLAGS.ts_min, FLAGS.ts_max, FLAGS.num_ts)
    sigs = np.logspace(FLAGS.sigs_min, FLAGS.sigs_max, FLAGS.num_sigs)
    metrics_func =[ metrices[key] for key in FLAGS.metrices if key in metrices]
    b_std = 0
    jmaker = partial(maker, train_images=train_images, test_images=test_images,
                     train_labels=train_labels, test_labels=test_labels)
    jmaker = jit(jmaker, static_argnums =(3,), backend=FLAGS.backend)
    #if we have enough memory
    if FLAGS.double_vmap:
        func = jax.vmap(vmap(jmaker, in_axes = (0, None, None, None) ), in_axes= (None,0, None, None))
        func = jit(func, static_argnums =(3,), backend=FLAGS.backend)
        result = func(ts, sigs, b_std, metrics_func)
    else:
        result = vmap_inner(ts, sigs, b_std, metrics_func, jmaker, backend=FLAGS.backend)
        result = reorder_dict(result)
    new_df = create_df(result, ts, sigs)
    new_df.to_csv(FLAGS.save_path)
if __name__ == '__main__':
    app.run(main)