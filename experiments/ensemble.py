"""
 Bagged ensemble experiment.
 Individual ensemble memebers (i.e. generic, interpretable) trained independently.
 Final ensemble using bagging.
 Training parameters contained within each member's config (see: generic.yaml and interpretable.yaml)
 Ensemble parameters in ensemble.yaml

 Experiment outputs are stored in ./results/ensemble/<M4 subset>/

"""
import os
import yaml
from pathlib import Path
from functools import partial
from typing import Optional, List

from utils.tools import silence_tensorflow
silence_tensorflow()  # Needs to be called before importing tf
import tensorflow as tf

from data.M4.m4dataset import M4Dataset, M4Sources, M4SourcesLite, Subset
from models.nbeats.NBeatsTF import NBeatsTF
from models.nbeats.NBeatsEnsemble import NBeatsEnsemble
from utils.model_params import NBeatsParams, EnsembleParams, BlockType
from utils.train_params import TrainParams
from utils.plot import plot_past_future
from utils.tools import mk_clear_dir, load_config, smape_simple

subset2lh = {Subset.hourly: 10,
             Subset.daily: 10,
             Subset.weekly: 10,
             Subset.monthly: 1.5,
             Subset.quarterly: 1.5,
             Subset.yearly: 1.5}


def get_results_dir(load_subset: str) -> str:
    return os.path.join(wd, 'results', 'ensemble', load_subset)


def get_model_dir(results_dir: str, model_name: str) -> str:
    return os.path.join(results_dir, 'mdl_' + model_name)


def get_plot_dir(results_dir: str, aggregation_method: str) -> str:
    return os.path.join(results_dir, 'plots', aggregation_method)


def interpretable_mod(config: dict):
    # For interpretable model, set seasonality theta dim to H
    if config['model_type'] == 'interpretable':
        th_dim = eval(config['thetas_dim'])
        assert eval(config['stack_types']) == (BlockType.TREND_BLOCK, BlockType.SEASONALITY_BLOCK)
        config['thetas_dim'] = str((th_dim[0], config['forecast_length']))


def train(config: dict, save_model: bool):
    name = '{}_{}H_{}'.format(config['model_type'], config['h_mult'], config['loss'])
    results_dir = mk_clear_dir(get_results_dir(config['load_subset']), False)
    model_dir = mk_clear_dir(get_model_dir(results_dir, name), False)

    m4 = M4Dataset(sources=M4Sources()).update_param(**config).read_source()

    config.update({'backcast_length': m4.H[Subset(config['load_subset'])] * m4.h_mult,
                   'forecast_length': m4.H[Subset(config['load_subset'])], 'name': name})
    interpretable_mod(config)

    trainparms = TrainParams().update_param(**config)
    netparams = NBeatsParams().update_param(**config)

    train_data_fn = partial(m4.dataset, trainparms.epoch_sample_size, lh=config['lh'])
    model = NBeatsTF.create_model(netparams, trainparms)
    model_file_name = os.path.join(model_dir, f'model.mdl')
    model.train(train_data_fn, trainparms.epochs, trainparms.batch_size, save_model, model_file_name)

    with open(os.path.join(model_dir, 'config.yaml'), 'w') as f:
        yaml.dump(config, f)


def train_population(config_names, losses, hx_min, hx_max, subsets: Optional[List[Subset]] = None):
    h_mults = list(range(hx_min, hx_max))
    subsets = list(Subset) if subsets is None else subsets
    for loss in losses:
        for subset in subsets:
            for h_mult in h_mults:
                for config_name in config_names:
                    print(f'==> {loss} - {subset.value} - {config_name} - {h_mult}H <==')
                    config = load_config(config_name)
                    config['loss'] = loss
                    config['load_subset'] = subset.value
                    config['lh'] = subset2lh[subset]
                    config['h_mult'] = h_mult
                    train(config, save_model=True)


def load_ensemble(load_subset: Subset) -> tf.keras.Model:
    results_dir = get_results_dir(load_subset.value)
    ensemble_model = NBeatsEnsemble(EnsembleParams().update_param(**load_config('ensemble.yaml')))
    params = []
    for d in Path(results_dir).rglob('mdl_*'):
        with open(os.path.join(d, 'config.yaml')) as f:
            params.append((NBeatsParams().update_param(**yaml.safe_load(f)), os.path.join(d, f'model.mdl')))
    ensemble_model.load(params)
    return ensemble_model


def test_ensemble(subset: Subset, gen_plots: bool = False, show_plots: bool = False):
    config = load_config('ensemble.yaml')
    config['load_subset'] = subset.value
    results_dir = get_results_dir(subset.value)

    # Load model
    model = NBeatsEnsemble(EnsembleParams().update_param(**config))
    params = []
    for d in Path(results_dir).rglob(f'mdl_*'):
        with open(os.path.join(d, 'config.yaml')) as f:
            params.append((NBeatsParams().update_param(**yaml.safe_load(f)), os.path.join(d, f'model.mdl')))
    model.load(params)

    m4 = M4Dataset(sources=M4Sources()).update_param(**config).read_source(True)
    test_dset = m4.test_dataset()
    x, y, w = next(test_dset.batch(len(test_dset)).as_numpy_iterator())
    model_outs = dict()
    yhat = model.call(x, False, model_outputs=model_outs).numpy()

    test_result = 'test smape: {:.3f}'.format(smape_simple(y, yhat, w))
    print(test_result)

    with open(os.path.join(results_dir, f'{model.aggregation_method}_results.txt'), 'w') as f:
        f.write(test_result)

    if len(model_outs) > 0:
        import pickle
        with open(os.path.join(results_dir, f'{model.aggregation_method}_model_outputs.pkl'), 'wb') as f:
            pickle.dump({k: v.numpy() for k, v in model_outs.items()}, f)

    if gen_plots:
        plot_dir = mk_clear_dir(get_plot_dir(results_dir, model.aggregation_method), True)
        samples2print = list(range(64))
        plot_past_future(x[..., 0], yhat[..., 0], y[..., 0], plot_dir, n=samples2print, show=show_plots)


def main(config_names, losses, train, test, subsets=None, gen_plots=False):
    if train:
        train_population(config_names, losses, 2, 8, subsets)
    if test:
        for subset in subsets:
            test_ensemble(subset, gen_plots=gen_plots, show_plots=False)


import argparse

wd = os.getcwd()

if __name__ == "__main__":
    do_train = False
    do_test = False
    gen_plots = False
    config_names = ["generic.yaml", "interpretable.yaml"]
    losses = ['SymmetricMAPELoss', 'MAPELoss']
    subsets = list(Subset)

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", action="append", required=False,
                        help="Configuration file/s (.yaml). Default: [generic.yaml, interpretable.yaml]" +
                             "If file not in running directory, ../config will be searched")

    parser.add_argument("-l", "--losses", action="append", required=False,
                        help="Losses. Default: ['SymmetricMAPELoss', 'MAPELoss']")

    parser.add_argument("-s", "--subsets", action="append", required=False,
                        help="M4 subsets (at least one). Default all subsets", )

    parser.add_argument("--train", help="Train models", action="store_true", required=False)
    parser.add_argument("--test", help="Test models", action="store_true", required=False)
    parser.add_argument("--plot", help="Generate test plots", action="store_true", required=False)
    args = parser.parse_args()

    if args.config:
        config_names = args.config
    if args.losses:
        losses = args.losses
    if args.subsets:
        subsets = [Subset(sstr) for sstr in args.subsets]
    if args.train:
        do_train = args.train
    if args.test:
        do_test = args.test
    if args.plot:
        gen_plots = args.plot

    if do_train or do_test:
        main(config_names, losses, do_train, do_test, subsets, gen_plots)
    else:
        print("Nothing to do. Set at least one train/test flag. Bye!")

    exit(0)
