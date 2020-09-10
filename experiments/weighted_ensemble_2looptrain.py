import sys
sys.path.insert(0, '../')

import os
import yaml
from pathlib import Path
from functools import partial
from itertools import product
from typing import Tuple, Iterable, Dict

from utils.tools import silence_tensorflow
silence_tensorflow()  # Needs to be called before importing tf
import tensorflow as tf

from data.M4.m4dataset import M4Dataset, M4Sources, Subset
from models.nbeats.NBeatsWeightedEnsemble2TrainLoop import NBeatsWeightedEnsemble
from utils.model_params import NBeatsParams, WeightedEnsembleParams, BlockType
from utils.train_params import TrainParams
from utils.plot import plot_past_future
from utils.tools import mk_clear_dir, load_config, smape_simple

subset2lh = {Subset.hourly: 10, Subset.daily: 10, Subset.weekly: 10,
             Subset.monthly: 1.5, Subset.quarterly: 1.5, Subset.yearly: 1.5}

subset2epoch = {Subset.hourly: 300, Subset.daily: 300, Subset.weekly: 300,
                Subset.monthly: 500, Subset.quarterly: 500, Subset.yearly: 500}


def get_results_dir(ensemble_config: dict) -> str:
    return os.path.join(wd, 'results', 'weighted_ensemble_2looptrain', ensemble_config['load_subset'])


def get_model_dir(results_dir: str, ensemble_config: dict) -> str:
    ensemble_name = '{}_{}'.format(ensemble_config['name'], ensemble_config['weighted_aggregation_method'])
    return os.path.join(results_dir, 'mdl_' + ensemble_name)


def get_plot_dir(results_dir: str) -> str:
    return os.path.join(results_dir, 'plots')


def interpretable_mod(config: dict):
    # For interpretable model, set seasonality theta dim to H
    if config['model_type'] == 'interpretable':
        th_dim = eval(config['thetas_dim'])
        stack_types = eval(config['stack_types'])
        thetas = []
        for i, stack in enumerate(stack_types):
            if stack == BlockType.TREND_BLOCK:
                thetas.append(th_dim[i])
            elif stack == BlockType.SEASONALITY_BLOCK:
                thetas.append(config['forecast_length'])
            else:
                raise Exception('Unexpected block')
        config['thetas_dim'] = str(tuple(thetas))


def configs(ensemble_config_name: str, member_config_names: str, hx_min, hx_max, subset: Subset, step: int = 1) -> Tuple[Dict, Iterable[Dict]]:
    submodel_configs = []
    h_mults = list(range(hx_min, hx_max, step))
    for h_mult, config_name in product(h_mults, member_config_names):
        config = load_config(config_name)
        config['load_subset'] = subset.value
        config['lh'] = subset2lh[subset]
        config['h_mult'] = h_mult
        submodel_configs.append(config)
    ensemble_config = load_config(ensemble_config_name)
    ensemble_config['lh'] = subset2lh[subset]
    ensemble_config['h_mult'] = h_mult
    ensemble_config['epochs'] = subset2epoch[subset]
    return ensemble_config, submodel_configs


def pretrained_configs(dir: str) -> Iterable[Dict]:
    configs = []
    for d in Path(dir).rglob('mdl_*'):
        with open(os.path.join(d, 'config.yaml')) as f:
            configs.append(yaml.safe_load(f))
    return configs


def train(ensemble_config: Dict, submodel_configs: Iterable[Dict], save_model=True, resume_training=False)-> tf.keras.models.Model:
    results_dir = mk_clear_dir(get_results_dir(ensemble_config), False)
    model_dir = mk_clear_dir(get_model_dir(results_dir, ensemble_config), False)
    trainparms = TrainParams().update_param(**ensemble_config)
    m4 = M4Dataset(sources=M4Sources()).update_param(**ensemble_config).read_source()

    if resume_training:
        model = NBeatsWeightedEnsemble.load_ensemble(model_dir, trainparms)
    else:
        # Instantiate new ensemble
        ensemble_params = WeightedEnsembleParams().update_param(**ensemble_config)
        submodel_params = []
        for cfg in submodel_configs:
            name = '{}_{}H'.format(cfg['model_type'], cfg['h_mult'])
            cfg.update({'backcast_length': m4.H[m4.load_subset] * cfg['h_mult'],
                        'forecast_length': m4.H[m4.load_subset], 'name': name})
            interpretable_mod(cfg)
            submodel_params.append(NBeatsParams().update_param(**cfg))
        model = NBeatsWeightedEnsemble.create_model(ensemble_params, submodel_params)

    save_fun = partial(NBeatsWeightedEnsemble.save_ensemble, ensemble_config=ensemble_config,
                       model_configs=submodel_configs, ensemble_root_dir=model_dir, delete_root_dir=False)

    train_data_fn = partial(m4.dataset, trainparms.epoch_sample_size, lh=ensemble_config['lh'])
    model.train(trainparms=trainparms,
                train_data_fn=train_data_fn,
                epochs=trainparms.epochs,
                batch_size=trainparms.batch_size,
                save_model=save_model,
                save_callback=save_fun,
                run_eagerly=False)

    return model


def test(ensemble_config: Dict, show_plots: bool = False, gen_plots=False):
    results_dir = mk_clear_dir(get_results_dir(ensemble_config), False)
    model_dir = mk_clear_dir(get_model_dir(results_dir, ensemble_config), False)
    plot_dir = mk_clear_dir(get_plot_dir(model_dir), True) if gen_plots else None
    trainparms = TrainParams().update_param(**ensemble_config)
    m4 = M4Dataset(sources=M4Sources()).update_param(**ensemble_config).read_source(True)

    model = NBeatsWeightedEnsemble.load_ensemble(model_dir, trainparms)
    test_dset = m4.test_dataset()
    x, y, w = next(test_dset.batch(len(test_dset)).as_numpy_iterator())
    yhat = model.call(x, False).numpy()

    test_result = 'test smape: {:.3f}'.format(smape_simple(y, yhat, w))
    print(test_result)

    with open(os.path.join(model_dir, 'results.txt'), 'w') as f:
        f.write(test_result)

    if gen_plots:
        samples2print = list(range(8))
        plot_past_future(x[..., 0], yhat[..., 0], y[..., 0], plot_dir, n=samples2print, show=show_plots)


def main():
    do_train = True
    do_test = False

    subsets = list(Subset)
    for subset in subsets:
        for agg_method in aggregation_methods:
            ensemble_config, submodel_configs = configs(2, 3, subset, agg_method)
            if do_train:
                train(ensemble_config, submodel_configs, resume_training=True)
            if do_test:
                test(ensemble_config, gen_plots=True)


import argparse
wd = os.getcwd()
if __name__ == "__main__":

    do_train = False
    do_test = False
    gen_plots = False
    member_config_names = ["generic.yaml", "interpretable.yaml"]
    ensemble_config_name = "weighted_ensemble_2looptrain.yaml"
    subsets = list(Subset)

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--member_configs", action="append", required=False,
                        help="Member configuration files (.yaml). Default: generic.yaml, interpretable.yaml" +
                             "If file not in running directory, ../config will be searched")

    parser.add_argument("-e", "--ensemble_config", action="store_const", required=False,
                        help="Ensemble configuration file (.yaml). Default: weighted_ensemble_2looptrain.yaml" +
                             "If file not in running directory, ../config will be searched")

    parser.add_argument("-s", "--subsets", action="append", required=False,
                        help="M4 subsets (at least one). Default all subsets", )

    parser.add_argument("--train", help="Train models", action="store_true", required=False)
    parser.add_argument("--test", help="Test models", action="store_true", required=False)
    parser.add_argument("--plot", help="Generate test plots", action="store_true", required=False)
    args = parser.parse_args()

    if args.member_configs:
        member_config_names = args.member_configs
    if args.ensemble_config:
        ensemble_config = args.ensemble_config
    if args.subsets:
        subsets = [Subset(sstr) for sstr in args.subsets]
    if args.train:
        do_train = args.train
    if args.test:
        do_test = args.test
    if args.plot:
        gen_plots = args.plot

    if do_train or do_test:
        main(config_names, subsets, train, test, gen_plots)
    else:
        print("Nothing to do. Set at least one train/test flag. Bye!")
    exit(0)
