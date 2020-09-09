import tensorflow as tf
from tensorflow.python.keras.metrics import Mean, MeanMetricWrapper, Metric


class MAE(MeanMetricWrapper):
    def __init__(self, name='mae', **kwargs):
        super(MAE, self).__init__(MAE.mae, name=name, **kwargs)

    @staticmethod
    def mae(y_true, y_pred):
        time_axis = 1
        _metric = tf.reduce_mean(tf.abs(y_true - y_pred), axis=time_axis)
        return _metric


class MSE(MeanMetricWrapper):
    def __init__(self, name='mse', **kwargs):
        super(MSE, self).__init__(MSE.mse, name=name, **kwargs)

    @staticmethod
    def mse(y_true, y_pred):
        time_axis = 1
        _metric = tf.reduce_mean((y_true - y_pred) ** 2, axis=time_axis)
        return _metric


class SymmetricMAPE(MeanMetricWrapper):
    def __init__(self, name='smape', **kwargs):
        super(SymmetricMAPE, self).__init__(SymmetricMAPE.symmetric_mape, name=name, **kwargs)

    @staticmethod
    def symmetric_mape(y_true, y_pred):
        time_axis = 1
        values = 200.0 * tf.math.abs(y_true - y_pred) / (tf.math.abs(y_true) + tf.math.abs(y_pred) + 1.0e-5)
        _metric = tf.reduce_mean(values, axis=time_axis)
        return _metric


class MAPE(MeanMetricWrapper):
    def __init__(self, name='mape', **kwargs):
        super(MAPE, self).__init__(MAPE.mape, name=name, **kwargs)

    @staticmethod
    def mape(y_true, y_pred):
        time_axis = 1
        values = 100.0 * tf.math.abs(y_true - y_pred) / (tf.math.abs(y_true) + 1.0e-5)
        _metric = tf.reduce_mean(values, axis=time_axis)
        return _metric


class MASE(MeanMetricWrapper):
    def __init__(self, name='mase', **kwargs):
        super(MASE, self).__init__(MAPE.mape, name=name, **kwargs)
        self.periodicity = kwargs['periodicity']

    @staticmethod
    def mape(y_true, y_pred):
        time_axis = 1
        values = 100.0 * tf.math.abs(y_true - y_pred) / (tf.math.abs(y_true) + 1.0e-5)
        _metric = tf.reduce_mean(values, axis=time_axis)
        return _metric
