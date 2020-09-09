import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import yaml
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from pathlib import Path
from typing import Iterable, Dict, Callable, Optional
from tensorflow_probability.python.distributions import Multinomial

from models.nbeats.NBeatsTF import NBeatsTF
from utils.model_params import NBeatsParams, WeightedEnsembleParams
from utils.train_params import TrainParams
from utils.tools import mk_clear_dir


def load_config(config_name: str) -> dict:
    with open(config_name) as file:
        config = yaml.safe_load(file)
    return config


def save_config(config, config_name: str):
    with open(config_name, 'w') as f:
        yaml.dump(config, f)


class Mixer(keras.Model):
    def __init__(self, input_size: int, output_size: int, hidden_units: int, method: str):
        super(Mixer, self).__init__(name='Mixer')
        assert method is not None, 'Specify method'
        self.method = method.lower()
        self.input_size = input_size
        self.out_size = output_size
        self.fc1 = keras.layers.Dense(hidden_units, input_shape=(input_size,), activation='relu')
        self.fc2 = keras.layers.Dense(hidden_units, activation='relu')
        self.thetas = keras.layers.Dense(output_size, use_bias=False, activation='relu')
        self.dropout = keras.layers.Dropout(0.5)

    def call(self, x, training=None, mask=None):
        x = self.thetas(self.fc2(self.fc1(x)))
        if 'dropout' in self.method:
            x = self.dropout(x, training=training)
        if ('argmax' in self.method) and training:
            x *= tf.one_hot(tf.argmax(x, axis=1), on_value=1.0, off_value=0.0, depth=self.out_size)
        elif 'weighted' in self.method:
            x = keras.activations.softmax(x, axis=1) # < 1
        elif 'multinomial' in self.method:
            x = keras.activations.softmax(x, axis=1)  # probability / positive weights
            if training:
                dist = Multinomial(total_count=1, probs=x)
                x *= dist.sample()
            # else:
            #     x *= tf.one_hot(tf.argmax(x, axis=1), on_value=1.0, off_value=0.0, depth=self.out_size)

        return x

    def save(self, file_name: str):
        self.save_weights(file_name, save_format='tf')

    def load(self, dir: str):
        '''
            dir: directory of the saved model
        '''
        x = np.zeros((1, self.input_size))
        super(Mixer, self).predict(x)
        super(Mixer, self).load_weights(dir)


class NBeatsWeightedEnsemble(keras.Model):

    def __init__(self, ensemble_params: WeightedEnsembleParams):
        assert isinstance(ensemble_params.name, str)
        super(NBeatsWeightedEnsemble, self).__init__(name=ensemble_params.name)
        print('Aggregation method: {}'.format(ensemble_params.weighted_aggregation_method))
        self.weighted_aggregation_method = ensemble_params.weighted_aggregation_method
        self.hidden_layer_units = ensemble_params.hidden_layer_units
        self.submodel_names = sorted(
            ensemble_params.submodel_names) if ensemble_params.submodel_names is not None else None
        self.backcast_length = -1
        self.sub_models = dict()
        self.mixer = None
        self.m_history = {'loss': [], 'val_loss': []}

    def create_components(self, submodel_params: Iterable[NBeatsParams]) -> keras.Model:
        """
            Create ensemble components (submodels and mixer)
            submodel_params: Sequence of sub-model params
        """
        # 1 - create sub-models
        self.backcast_length = -1
        for param in submodel_params:
            mdl = NBeatsTF(param)
            if mdl.backcast_length > self.backcast_length:
                self.backcast_length = mdl.backcast_length
            self.sub_models[mdl.name] = mdl

        if self.submodel_names is None:
            self.submodel_names = sorted(self.sub_models.keys())
        else:  # Are these models what we were expecting ?
            assert self.submodel_names == sorted(self.sub_models.keys())

        # 2 - create mixer model. backast length the largest backast of submodels
        self.mixer = Mixer(self.backcast_length, len(self.sub_models), self.hidden_layer_units,
                           self.weighted_aggregation_method)

        return self

    def call(self, backcast, training=None, mask=None):
        assert len(backcast.shape) == 3
        assert self.backcast_length <= backcast.shape[1]  # T

        # Get learner outputs
        y_submod = [self.sub_models[n](tf.identity(backcast[:, -self.sub_models[n].backcast_length:, :]),
                                       training=training, mask=mask) for n in sorted(self.submodel_names)]
        y_submod = tf.stack(y_submod, axis=3)  # <batch x T x chan x ensemble>

        # Mix submodel outputs
        y = []
        for c in range(backcast.shape[2]):
            th = self.mixer(backcast[:, -self.backcast_length:, c], training=training, mask=mask)

            num = th[:, None, :] * y_submod[:, :, c, :]
            den = (tf.reduce_sum(th[:, None, :], axis=-1, keepdims=True) + 1.0e-5)
            y.append(tf.reduce_sum(num / den, axis=-1))
        y = tf.stack(y, axis=2)
        return y

    def m_compile(self, trainparms: TrainParams, metric: str = 'smape', run_eagerly: bool = False):
        opt = trainparms.optimizer if self.optimizer is None else self.optimizer
        self.compile(opt, trainparms.loss, metrics=trainparms.metrics[metric], run_eagerly=run_eagerly)

    def train(self, trainparms: TrainParams, train_data_fn: Callable, epochs: int, batch_size: int,
              save_model: bool = False, save_callback: Callable = None, metric: str = 'smape',
              run_eagerly: bool = False, epochs_patience: Optional[int] = None) -> float:
        """
            Train routine
            save_callback: save callback with signature: 'save_callback(model)'
        """
        best_val_loss = np.inf
        mdl_sel = -5
        e_offset = len(self.m_history['loss'])
        epoch = e_offset
        no_improvement = 0
        while epoch < epochs + e_offset:
            if -6 < mdl_sel < 0:
                print('-> Frozen Mixer')
                for l in self.mixer.layers:
                    l.trainable = False
                for _, mdl in self.sub_models.items():
                    for l in mdl.layers:
                        l.trainable = True
                mdl_sel += 1
            elif 0 <= mdl_sel < 5:
                print('-> Frozen Submodels')
                for l in self.mixer.layers:
                    l.trainable = True
                for _, mdl in self.sub_models.items():
                    for l in mdl.layers:
                        l.trainable = False
                mdl_sel += 1
            else:
                mdl_sel = -5 if mdl_sel > 0 else 0
                continue

            if (mdl_sel == -4) or (mdl_sel == 1):
                self.m_compile(trainparms, metric, run_eagerly)
                print('-> Model compiled ')

            train, val = train_data_fn(gen_validation=True)
            train = train.shuffle(len(train)).batch(batch_size)
            val = val.batch(batch_size)

            print(f'=> Epoch: {epoch + 1}/{epochs}')
            logs = self.fit(x=train, epochs=1, validation_data=val)
            self.m_history['loss'].append(logs.history['loss'][0])
            self.m_history['val_loss'].append(logs.history['val_loss'][0])
            epoch += 1
            if logs.history['val_loss'][0] < best_val_loss:
                best_val_loss = logs.history['val_loss'][0]
                no_improvement = 0
                if save_model and (save_callback is not None):
                    save_callback(self)
            else:
                no_improvement += 1

            if (epochs_patience is not None) and (no_improvement > epochs_patience):
                print('Early termination')
                break

        return best_val_loss

    @staticmethod
    def create_model(ensemble_params: WeightedEnsembleParams, submodel_params: Iterable[NBeatsParams]) -> keras.Model:
        model = NBeatsWeightedEnsemble(ensemble_params).create_components(submodel_params)
        return model

    @staticmethod
    def load_ensemble(ensemble_root_dir: str, trainparms: TrainParams, metric: str = 'smape',
                      run_eagerly: bool = False) -> keras.Model:
        """
            Load an already generated ensemble saved with 'NBeatsWeightedEnsemble.save_ensemble()'
        """

        # A: Instantiate ensemble
        # 1 - ensemble params
        cfg = load_config(os.path.join(ensemble_root_dir, 'weighted_ensemble.yaml'))
        ensemble_params = WeightedEnsembleParams().update_param(**cfg)

        # 2 - submodel params
        submodel_params = dict()
        submodel_files = dict()
        for d in Path(ensemble_root_dir).rglob('mdl_*'):
            cfg_file = os.path.join(d, 'config.yaml')
            mdl_file = os.path.join(d, 'model.mdl')
            parms = NBeatsParams().update_param(**load_config(cfg_file))
            if parms.name in ensemble_params.submodel_names:
                submodel_params[parms.name] = parms
                submodel_files[parms.name] = mdl_file
        assert len(submodel_files) > 0, 'Ensemble submodels could not be loaded'

        # 3 - instantiate ensemble
        ensemble_model = NBeatsWeightedEnsemble.create_model(ensemble_params, list(submodel_params.values()))
        ensemble_model.m_compile(trainparms, metric, run_eagerly)

        # B: Load weights
        # 1 - weights of sub-models
        [mdl.load(submodel_files[nm]) for nm, mdl in ensemble_model.sub_models.items()]

        # 2 - weights of mixer
        mixer_file = str(list(Path(ensemble_root_dir).rglob('mixer_*'))[0])
        mixer_file = os.path.join(mixer_file, 'model.mdl')
        ensemble_model.mixer.load(mixer_file)

        import pickle
        with open(os.path.join(ensemble_root_dir, 'meta.pkl'), 'rb') as f:
            meta = pickle.load(f)
            ensemble_model.m_history = meta[0]
            ensemble_model.optimizer = meta[1]
        return ensemble_model

    @staticmethod
    def save_ensemble(ensemble_model, ensemble_config: Dict, model_configs: Iterable[Dict],
                      ensemble_root_dir: str, delete_root_dir: bool = False):
        """
            Save ensemble. Model to be loaded with 'NBeatsWeightedEnsemble.load_ensemble()'

            ensemble_model: model to be saved
            ensemble_config: ensemble config object
            model_configs: sequence of (model name, model configs)
            ensemble_root_dir: ensemble root directory
            delete_root_dir: whether to delete existing ensemble root directory.
                         If false, model directories will still be deleted.
        """

        assert isinstance(ensemble_model, NBeatsWeightedEnsemble)
        ensemble_root_dir = mk_clear_dir(ensemble_root_dir, delete_root_dir)
        ensemble_config['submodel_names'] = list(ensemble_model.submodel_names)
        save_config(ensemble_config, os.path.join(ensemble_root_dir, 'weighted_ensemble.yaml'))  # Save ensemble config

        # Save submodel configs and weights
        model_configs = {cfg['name']: cfg for cfg in model_configs}
        for mdl_name, mdl in ensemble_model.sub_models.items():
            cfg = model_configs[mdl_name]
            d = mk_clear_dir(os.path.join(ensemble_root_dir, 'mdl_' + mdl_name), True)
            mdl.save(os.path.join(d, 'model.mdl'))
            save_config(cfg, os.path.join(d, 'config.yaml'))

        # Save mixer weights
        mixer_dir = mk_clear_dir(os.path.join(ensemble_root_dir, 'mixer_' + ensemble_model.mixer.name), True)
        ensemble_model.mixer.save(os.path.join(mixer_dir, 'model.mdl'))

        import pickle
        with open(os.path.join(ensemble_root_dir, 'meta.pkl'), 'wb') as f:
            meta = (ensemble_model.m_history, ensemble_model.optimizer)
            pickle.dump(meta, f)
