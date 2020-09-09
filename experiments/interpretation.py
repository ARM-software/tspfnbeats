"""
Interpretation experiment:
Train and test individual generic or interpretable N-BEAT models.
"""

import os, yaml
from pathlib import Path
from collections import defaultdict
from functools import partial

from utils.tools import silence_tensorflow
silence_tensorflow()  # Needs to be called before importing tf
import tensorflow as tf

from data.M4.m4dataset import M4Dataset, M4Sources, Subset
from models.nbeats.NBeatsTF import NBeatsTF
from utils.model_params import NBeatsParams, BlockType
from utils.train_params import TrainParams
from utils.plot import plot_past_future, plot_stack
from utils.tools import mk_clear_dir, load_config, smape_simple


def get_results_dir(config: dict):
    return os.path.join(wd, 'results', 'interpretation', config['model_type'], config['load_subset'])


def get_plot_dir(results_dir: str):
    return os.path.join(results_dir, 'plots')


def get_model_dir(results_dir: str):
    return os.path.join(results_dir, 'model')


subset2lh = {Subset.hourly: 10,
             Subset.daily: 10,
             Subset.weekly: 10,
             Subset.monthly: 1.5,
             Subset.quarterly: 1.5,
             Subset.yearly: 1.5}


def train(config: dict, save_model: bool, delete_existing: bool = False) -> tf.keras.models.Model:
    results_dir = mk_clear_dir(get_results_dir(config), delete_existing)
    model_dir = mk_clear_dir(get_model_dir(results_dir), delete_existing)

    m4 = M4Dataset(sources=M4Sources()).update_param(**config).read_source(config['test'])

    # Update backast and forecast lengths
    config.update({'backcast_length': m4.H[Subset(config['load_subset'])] * m4.h_mult,
                   'forecast_length': m4.H[Subset(config['load_subset'])]})

    # For interpretable model, set seasonality theta dim to H
    if config['model_type'] == 'interpretable':
        th_dim = eval(config['thetas_dim'])
        assert eval(config['stack_types']) == (BlockType.TREND_BLOCK, BlockType.SEASONALITY_BLOCK)
        config['thetas_dim'] = str((th_dim[0], config['forecast_length']))

    trainparms = TrainParams().update_param(**config)
    netparams = NBeatsParams().update_param(**config)

    train_data_fn = partial(m4.dataset, trainparms.epoch_sample_size, lh=config['lh'])
    model = NBeatsTF.create_model(netparams, trainparms)
    model_file_name = os.path.join(model_dir, f'model.mdl')
    model.train(train_data_fn, trainparms.epochs, trainparms.batch_size, save_model, model_file_name)

    with open(os.path.join(results_dir, 'config.yaml'), 'w') as f:
        yaml.dump(config, f)

    return model


def test(config: dict, gen_plots: bool = False, show_plots: bool = False):
    results_dir = get_results_dir(config)
    plot_dir = mk_clear_dir(get_plot_dir(results_dir), True) if gen_plots else None
    model_dir = get_model_dir(results_dir)
    model_file_name = os.path.join(model_dir, f'model.mdl')

    m4 = M4Dataset(sources=M4Sources()).update_param(**config).read_source(True)
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

            plot_stack(y, yhat, dict(stack_coll), samples2print, labels=labels,
                       block_plot_cnt=2, plot_dir=plot_dir, show=show_plots)
            plot_past_future(x[..., 0], yhat[..., 0], y[..., 0], plot_dir, n=samples2print, show=show_plots)
        except:
            pass

    test_result = 'test smape: {:.3f}'.format(smape_simple(y, yhat, w))
    print(test_result)

    with open(os.path.join(results_dir, 'results.txt'), 'w') as f:
        f.write(test_result)


def main(config_names, subsets, train, test, gen_plots):
    if train:
        for subset in subsets:
            for config_name in config_names:
                print(f'==> {subset.value} - {config_name} <==')
                config = load_config(config_name)
                config['load_subset'] = subset.value
                config['lh'] = subset2lh[subset]
                train(config, save_model=True, delete_existing=True)

    if test:
        # Load models files and
        for cfg in Path(os.path.join(wd, 'results', 'interpretation')).rglob('*.yaml'):
            # Load saved model config
            with open(cfg) as f:
                config = yaml.safe_load(f)
            test(config, gen_plots=gen_plots, show_plots=False)


import argparse
wd = os.getcwd()

if __name__ == "__main__":
    do_train = False
    do_test = False
    gen_plots = False
    config_names = ["generic.yaml", "interpretable.yaml"]
    subsets = list(Subset)
    parser = argparse.ArgumentParser()

    parser.add_argument("-c", "--config", action="append", required=False,
                        help="Configuration file/s (.yaml). Default: generic.yaml, interpretable.yaml" +
                             "If file not in running directory, ../config will be searched")

    parser.add_argument("-s", "--subsets", action="append", required=False,
                        help="M4 subsets (at least one). Default all subsets",)

    parser.add_argument("--train", help="Train models", action="store_true", required=False)
    parser.add_argument("--test", help="Test models", action="store_true", required=False)
    parser.add_argument("--plot", help="Generate test plots", action="store_true", required=False)
    args = parser.parse_args()

    if args.config:
        config_names = args.config
    if args.subsets:
        subsets = [Subset(sstr) for sstr in args.subsets]
    if args.train:
        do_train = args.train
    if args.test:
        do_test = args.test
    if args.plot:
        gen_plots = args.plot

    if do_train or do_test:
        main(config_names, subsets, do_train, do_test, gen_plots)
    else:
        print("Nothing to do. Set at least one train/test flag. Bye!")

exit(0)
