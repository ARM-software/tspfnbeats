import tensorflow as tf


class SymmetricMAPELoss(tf.keras.losses.Loss):
    def __init__(self, time_axis=1):
        super(SymmetricMAPELoss, self).__init__()
        self.time_axis = time_axis

    def __call__(self, y_true, y_pred, sample_weight=None):
        '''
            Weights are 0/1 for each time sample
        '''
        return super(SymmetricMAPELoss, self).__call__(sample_weight * y_true, sample_weight * y_pred)

    def call(self, y_true, y_pred):
        """Invokes the `Loss` instance.

            Args:
              y_true: Ground truth values. shape = `[batch_size, d0, .. dN]`, except
                sparse loss functions such as sparse categorical crossentropy where
                shape = `[batch_size, d0, .. dN-1]`
              y_pred: The predicted values. shape = `[batch_size, d0, .. dN]`

            Returns:
              Loss values with the shape `[batch_size, d0, .. dN-1]`.
            """
        values = 200.0 * tf.math.abs(y_true - y_pred) / (tf.math.abs(y_true) + tf.math.abs(y_pred) + 1.0e-5)
        loss = tf.reduce_mean(values, axis=self.time_axis)
        return loss


class MAPELoss(tf.keras.losses.Loss):
    def __init__(self, time_axis=1):
        super(MAPELoss, self).__init__()
        self.time_axis = time_axis

    def __call__(self, y_true, y_pred, sample_weight=None):
        '''
            Weights are 0/1 for each time sample
        '''
        return super(MAPELoss, self).__call__(sample_weight * y_true, sample_weight * y_pred)

    def call(self, y_true, y_pred):
        """Invokes the `Loss` instance.

            Args:
              y_true: Ground truth values. shape = `[batch_size, d0, .. dN]`, except
                sparse loss functions such as sparse categorical crossentropy where
                shape = `[batch_size, d0, .. dN-1]`
              y_pred: The predicted values. shape = `[batch_size, d0, .. dN]`

            Returns:
              Loss values with the shape `[batch_size, d0, .. dN-1]`.
            """
        values = 100.0 * tf.math.abs(y_true - y_pred) / (tf.math.abs(y_true) + 1.0e-5)
        loss = tf.reduce_mean(values, axis=self.time_axis)
        return loss


class SymmetricMAPE_Sq_Loss(tf.keras.losses.Loss):
    def __init__(self, time_axis=1):
        super(SymmetricMAPE_Sq_Loss, self).__init__()
        self.time_axis = time_axis

    def __call__(self, y_true, y_pred, sample_weight=None):
        '''
            Weights are 0/1 for each time sample
        '''
        return super(SymmetricMAPE_Sq_Loss, self).__call__(sample_weight * y_true, sample_weight * y_pred)

    def call(self, y_true, y_pred):
        """Invokes the `Loss` instance.

            Args:
              y_true: Ground truth values. shape = `[batch_size, d0, .. dN]`, except
                sparse loss functions such as sparse categorical crossentropy where
                shape = `[batch_size, d0, .. dN-1]`
              y_pred: The predicted values. shape = `[batch_size, d0, .. dN]`

            Returns:
              Loss values with the shape `[batch_size, d0, .. dN-1]`.
            """
        smape_vals = 200.0 * tf.math.abs(y_true - y_pred) / (tf.math.abs(y_true) + tf.math.abs(y_pred) + 1.0e-5)
        sq_err_vals = smape_vals ** 2
        loss = tf.reduce_mean(smape_vals + 0.1*sq_err_vals, axis=self.time_axis)
        return loss
