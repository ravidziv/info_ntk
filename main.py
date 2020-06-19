import sys

sys.path.append("/home/aa-user/ravidziv/neural-tangents")
from absl import app
from absl import flags
import jax
import pandas as pd
import jax.numpy as np
from jax import vmap, jit
from info_ntk.metrics import get_losses
import collections
from info_ntk.datasets import load_data
from functools import partial
from info_ntk.models import WideResnet
flags.DEFINE_string('dataset', 'MNIST',
                   'The dataset to load.')
flags.DEFINE_integer('train_size', 10,
                     'Dataset size to use for training.')
flags.DEFINE_integer('test_size', 10,
                     'Dataset size to use for testing.')
flags.DEFINE_string('backend', 'gpu',  'The backend for computation')
flags.DEFINE_string('net_type', 'FC',  'The network to train')
flags.DEFINE_multi_string('metrices', ['losses'],  'which metrices to calculate')
flags.DEFINE_float('ts_min', -2, 'Min val for t (log)')
flags.DEFINE_float('ts_max', 8, 'max val for t (log)')
flags.DEFINE_float('b_std', 0.0, 'initial b std')
flags.DEFINE_integer('num_ts', 100, 'Number of ts')

flags.DEFINE_float('sigs_min', -2, 'Min val for t (log)')
flags.DEFINE_float('sigs_max', 0.3, 'max val for t (log)')
flags.DEFINE_integer('num_sigs', 100, 'Number of sigmas')
flags.DEFINE_bool('double_vmap', True, 'If do vmap on both sigs and ts')
FLAGS = flags.FLAGS
from info_ntk.metrics import  loss_metrices_name

MetricesTuple = collections.namedtuple('metrices',['names', 'func'] )
Results = collections.namedtuple('Results',['name', 'value', 'mode'] )
metrices = {'losses':MetricesTuple(loss_metrices_name, get_losses)}
Metrics_vals = collections.namedtuple('metrics_vals',
                                      ['{}_train'.format(loss) for  loss in loss_metrices_name]
                                      + ['{}_val'.format(loss) for  loss in loss_metrices_name],
                                      defaults=(None,) * 2*len(loss_metrices_name))

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

@partial(jax.jit, static_argnums=(3,))
def maker(t=1e2, W_std=1.0, b_std=0., metrics_func= [], train_images=None, test_images=None, train_labels=None,
          test_labels=None):
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
    pred_train = train_predict_fn(t)
    pred_test = predict_fn(t)
    dict_val = {}
    #fallten the metric func to key with train and test
    for metrice_dict in metrics_func:
        metrices_name = metrice_dict.names
        metric_func_inner = metrice_dict.func
        train_metric_vals = metric_func_inner(pred_train, test_labels, tcut)
        test_metric_vals = metric_func_inner(pred_test, test_labels, tcut)
        for metric_name in metrices_name:
            train_val =getattr(train_metric_vals, metric_name)
            test_val =  getattr(test_metric_vals, metric_name)
            dict_val['{}_train'.format(metric_name)] =train_val
            dict_val['{}_val'.format(metric_name)] =test_val
    return Metrics_vals(**dict_val)



def vmap_inner(sig, ts, backend, maker_part, b_std,
               metrics_func ):
    """Vmap inside for loop due to memory limitations"""
    results = []
    for i in range(len(sig)):
        #jmaker = jit(jmaker, backend=backend)
        print (i, len(sig))
        res = vmap(maker_part, (0, None, None, None))(ts, sig[i],b_std, metrics_func)
        results.append(res)
    return results

def main(unused_argv):
    train_images, train_labels, test_images, test_labels = load_data(datast=FLAGS.dataset,
                                                                     train_size=FLAGS.train_size,
                                                                     test_size=FLAGS.test_size)
    ts = np.logspace(FLAGS.ts_min, FLAGS.ts_max, FLAGS.num_ts)
    sigs = np.logspace(FLAGS.sigs_min, FLAGS.sigs_max, FLAGS.num_sigs)
    metrics_func =[ metrices[key] for key in FLAGS.metrices]
    b_std = 0
    jmaker = partial(maker, train_images=train_images, test_images=test_images,
                     train_labels=train_labels, test_labels=test_labels)
    jmaker = jit(jmaker, static_argnums =(3,), backend=FLAGS.backend)
    #if we have enough memory
    if FLAGS.double_vmap:
        func = jax.vmap(vmap(jmaker, in_axes = (0, None, None, None) ), in_axes= (None,0, None, None))
    else:
        func = partial(vmap_inner, backend = FLAGS.backend, maker_part = jmaker)
    func = jit(func, backend=FLAGS.backend)
    result = func(ts, sigs, b_std, metrics_func)
    print (1)
if __name__ == '__main__':
    app.run(main)