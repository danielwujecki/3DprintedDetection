import numpy as np
from tensorflow import keras

CNN_JSON = '../classification/model/cnn_vgg16.json'
CNN_WEIGHTS = '../classification/model/cnn_vgg16.hdf5'
CNN_STD = '../classification/model/featurewise_std.npy'
CNN_MEAN = '../classification/model/featurewise_mean.npy'


class CNN(object):
    def __init__(self):
        self.__model = self.__load_model()
        self.__std = np.load(CNN_STD, allow_pickle=True)
        self.__mean = np.load(CNN_MEAN, allow_pickle=True)
        assert self.__mean.shape == self.__std.shape

    def __load_model(self):
        json_string = None
        with open(CNN_JSON, 'r') as file:
            json_string = file.readline()
        model = keras.models.model_from_json(json_string)
        model.load_weights(CNN_WEIGHTS)

        model.compile(optimizer='sgd', loss='mse')
        return model

    def __preprocess_img(self, img):
        img = img.astype(np.float64)
        img /= 255.
        img -= self.__mean
        img /= self.__std
        return np.asarray([img])

    def predict(self, img):
        assert self.__mean.shape == img.shape
        img = self.__preprocess_img(img)
        prediction = self.__model.predict(img)
        prediction = prediction[0]
        classnum = np.argmax(prediction)
        confidence = prediction[classnum]
        return confidence, classnum
