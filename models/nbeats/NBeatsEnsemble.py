import numpy as np
from copy import deepcopy
import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow.keras as keras
from typing import Iterable, Tuple, Optional, Dict

from models.nbeats.NBeatsTF import NBeatsTF
from utils.model_params import NBeatsParams, EnsembleParams


class NBeatsEnsemble(keras.Model):
    def __init__(self, params: EnsembleParams):
        super(NBeatsEnsemble, self).__init__(name=params.name)
        self.aggregation_method = params.aggregation_method
        self.sub_models = list()
        self.backcast_length = -1

    def call(self, backcast: tf.Tensor, training: Optional[bool] = None, mask: Optional[tf.Tensor] = None,
             model_outputs: Optional[Dict] = None) -> tf.Tensor:
        assert len(backcast.shape) == 3
        assert self.backcast_length <= backcast.shape[1]
        chans = backcast.shape[2]

        y_submod = []
        for mdl in self.sub_models:
            _y = mdl(tf.identity(backcast[:, -mdl.backcast_length:, :]), training=training, mask=mask)
            if model_outputs is not None:
                model_outputs[mdl.name] = _y
            y_submod.append(_y)
        y_submod = tf.stack(y_submod, axis=3) # <batch x T x chan x ensemble>

        y = []
        for c in range(chans):
            if self.aggregation_method == 'mean':
                _y = tf.reduce_mean(y_submod[:, :, c, :], axis=-1)  # Reduce across ensemble
            elif self.aggregation_method == 'median':
                _y = tfp.stats.percentile(y_submod[:, :, c, :], 50.0, axis=-1, interpolation='midpoint')
            else:
                raise Exception('Method not recognized')
            y.append(_y)

        y = tf.stack(y, axis=2)

        if model_outputs is not None:
            model_outputs[self.name] = y
        return y

    def load(self, params: Iterable[Tuple[NBeatsParams, str]]):
        """
            params: List((NBeatsParams, weight_path))
        """
        for p, d in params:
            mdl = NBeatsTF(p)
            if mdl.backcast_length > self.backcast_length:
                self.backcast_length = mdl.backcast_length
            mdl.load(d)
            self.sub_models.append(mdl)
        assert self.backcast_length > 0, 'Backast length not determined'
