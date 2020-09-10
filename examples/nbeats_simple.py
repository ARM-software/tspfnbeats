import sys
sys.path.insert(0, '../')

import os
import yaml
import tensorflow as tf
from collections import defaultdict
from functools import partial

from data.M4.m4dataset import M4Dataset, M4SourcesLite, Subset
from models.nbeats.NBeatsTF import NBeatsTF
from utils.model_params import NBeatsParams
from utils.train_params import TrainParams
from utils.plot import plot_past_future, plot_stack
from utils.tools import mk_clear_dir, load_config, smape_simple


def get_results_dir(config: dict):
    return os.path.join(wd, 'results', 'simple', config['model_type'], config['load_subset'])


def get_plot_dir(results_dir: str):
    return os.path.join(results_dir, 'plots')


def get_model_dir(results_dir: str):
    return os.path.join(results_dir, 'model')


def train(config: dict, save_model: bool, delete_existing: bool = False) -> tf.keras.models.Model:
    results_dir = mk_clear_dir(get_results_dir(config), delete_existing)
    model_dir = mk_clear_dir(get_model_dir(results_dir), delete_existing)

    m4 = M4Dataset(sources=M4SourcesLite()).update_param(**config).read_source(config['test'])

    # Update backast and forecast lengths
    config.update({'backcast_length': m4.H[Subset(config['load_subset'])] * m4.h_mult,
                   'forecast_length': m4.H[Subset(config['load_subset'])]})

    trainparms = TrainParams().update_param(**config)
    netparams = NBeatsParams().update_param(**config)

    train_data_fn = partial(m4.dataset, trainparms.epoch_sample_size, lh=config['lh'])
    model = NBeatsTF.create_model(netparams, trainparms)
    model_file_name = os.path.join(model_dir, f'model.mdl')
    model.train(train_data_fn, trainparms.epochs, trainparms.batch_size, save_model, model_file_name)

    with open(os.path.join(results_dir, 'config.yaml'), 'w') as f:
        yaml.dump(config, f)

    return model


def test(config: dict, gen_plots: bool = False):
    results_dir = get_results_dir(config)
    plot_dir = mk_clear_dir(get_plot_dir(results_dir), True) if gen_plots else None
    model_dir = get_model_dir(results_dir)
    model_file_name = os.path.join(model_dir, f'model.mdl')

    m4 = M4Dataset(sources=M4SourcesLite()).update_param(**config).read_source(True)
    model = NBeatsTF.create_model(NBeatsParams().update_param(**config), TrainParams().update_param(**config))
    model.load(model_file_name)

    test_dset = m4.test_dataset()
    x, y, w = next(test_dset.batch(len(test_dset)).as_numpy_iterator())

    stack_coll = defaultdict(lambda: 0)
    yhat = model.call(x, False, stack_out=stack_coll).numpy()

    if gen_plots:
        samples2print = list(range(8))
        try:
            if config['model_type'] == 'generic':
                labels = ['Stack1-Generic', 'Stack2-Generic']
            elif config['model_type'] == 'interpretable':
                labels = ['Stack1-Interp', 'Stack2-Interp']
            else:
                raise Exception()

            plot_stack(y, yhat, dict(stack_coll), samples2print, labels=labels, block_plot_cnt=2, plot_dir=plot_dir,
                       show=False)
            plot_past_future(x[..., 0], yhat[..., 0], y[..., 0], plot_dir, n=samples2print, show=False)
        except:
            pass

    test_result = 'test smape: {:.3f}'.format(smape_simple(y, yhat, w))
    print(test_result)

    with open(os.path.join(results_dir, 'results.txt'), 'w') as f:
        f.write(test_result)


import argparse

wd = os.getcwd()

if __name__ == "__main__":
    do_train = False
    do_test = False

    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Specify configuration file (.yaml). If not in running directory, " +
                                       "../config will be searched", type=str)
    parser.add_argument("--train", help="Train a model", action="store_true")
    parser.add_argument("--test", help="Test a model", action="store_true")
    args = parser.parse_args()

    if args.train:
        do_train = args.train
    if args.test:
        do_test = args.test

    config_name = args.config
    config = load_config(config_name)
    print('N-BEATS: {} - {} - {}'.format(config['load_subset'], config['model_type'], config_name))

    if do_train:
        print('Training ...')
        train(config, True)

    if do_test:
        print('Testing ...')
        test(config, True)

    exit(0)
