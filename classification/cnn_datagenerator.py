from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import warnings


class MyImageDataGen(ImageDataGenerator):
    """ My own ImageDataGenerator which inherit from the original
    and applies precalculated std and mean"""

    def __init__(self, mean=None, std=None, **kwargs):
        super().__init__(**kwargs)
        self._mean = mean
        self._std = std

    def standardize(self, x):
        if self._mean is not None and self._std is not None:
            assert x.shape[-3:] == self._mean.shape
            assert x.shape[-3:] == self._std.shape

        if self.preprocessing_function:
            x = self.preprocessing_function(x)
        if self.rescale:
            x *= self.rescale
        if self.samplewise_center:
            x -= np.mean(x, keepdims=True)
        if self.samplewise_std_normalization:
            x /= (np.std(x, keepdims=True) + 1e-6)

        if self.featurewise_center:
            if self._mean is not None:
                x -= self._mean
            else:
                warnings.warn('This MyImageDataGen specifies '
                              '`featurewise_center`, but the mean isn\'t given.')
        if self.featurewise_std_normalization:
            if self._std is not None:
                x /= (self._std + 1e-6)
            else:
                warnings.warn('This MyImageDataGen specifies '
                              '`featurewise_std_normalization`, '
                              'but the std isn\'t given.')
        if self.zca_whitening:
            warnings.warn('This MyImageDataGen specifies '
                          '`zca_whitening`, but isn\'t implemented.')
        return x


def get_train_gen(mean_path='./model/featurewise_mean.npy', std_path='./model/featurewise_std.npy'):
    mean = np.load(mean_path, allow_pickle=True)
    std = np.load(std_path, allow_pickle=True)
    kwargs = dict(
        # set rescaling factor (applied before any other transformation)
        rescale=1. / 255,
        # set input mean to 0 over the dataset
        featurewise_center=True,
        featurewise_std_normalization=True,
        # samplewise_center=True,
        # samplewise_std_normalization=True,

        # randomly rotate images in the range (degrees, 0 to 180)
        # rotation_range=10,
        # randomly shift images horizontally (fraction of total width)
        width_shift_range=0.02,
        # randomly shift images vertically (fraction of total height)
        height_shift_range=0.02,
        # horizontal_flip=True,  # randomly flip images
        # vertical_flip=True,  # randomly flip images
    )
    datagen = MyImageDataGen(mean, std, **kwargs)
    return datagen


def get_test_gen(mean_path='./model/featurewise_mean.npy', std_path='./model/featurewise_std.npy'):
    mean = np.load(mean_path, allow_pickle=True)
    std = np.load(std_path, allow_pickle=True)
    kwargs = dict(
        # set rescaling factor (applied before any other transformation)
        rescale=1. / 255,
        # set input mean to 0 over the dataset
        featurewise_center=True,
        featurewise_std_normalization=True,
        # samplewise_center=True,
        # samplewise_std_normalization=True,
    )
    datagen = MyImageDataGen(mean, std, **kwargs)
    return datagen
