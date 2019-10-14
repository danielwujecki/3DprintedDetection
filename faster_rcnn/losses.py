import keras.backend as K


def rpn_loss_cls(y_true, y_pred):
    nb_a = y_pred.shape[-1]
    cross = K.binary_crossentropy(y_pred[:, :, :, :], y_true[:, :, :, nb_a:])
    res = K.sum(y_true[:, :, :, :nb_a] * cross)
    res /= K.sum(y_true[:, :, :, :nb_a]) + 1e-5
    return res


def get_loss_regr(weight):
    def loss_regr(y_true, y_pred):
        # x is the difference between true value and predicted vaue
        x = y_true - y_pred
        x_abs = K.abs(x)
        # If x_abs <= 1.0, x_bool = 1
        x_small = K.cast(K.less_equal(x_abs, 1.), 'float32')
        x_obj = K.cast(K.not_equal(y_true, -100.), 'float32')

        # 0.5*x*x (if x_abs < 1)
        res = x_small * (.5 * x * x) + (1. - x_small) * (x_abs - .5)
        res *= x_obj
        res = K.sum(res) / K.sum(x_obj) + 1e-5
        return res * float(weight)
    return loss_regr
