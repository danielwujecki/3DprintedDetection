from tensorflow import keras
import numpy as np

import cnn_loaddata
from cnn_datagenerator import get_test_gen

### --- open model --- ###

json_string = None
with open('./model/cnn_vgg16.json', 'r') as file:
    json_string = file.readline()
model = keras.models.model_from_json(json_string)
model.load_weights('./model/cnn_vgg16.hdf5')
model.compile(optimizer='sgd', loss='mse', metrics=['accuracy'])

### --- end open --- ###

def predict(img_name: str):
    img = cnn_loaddata.test_image(img_name)
    prediction = model.predict(img)
    prediction = np.argmax(prediction.flatten())
    print("\n=====\nPrediction: %d\n=====" % prediction)
    return prediction


def score_model():
    datagen = get_test_gen()
    X, Y = cnn_loaddata.test()
    Y = keras.utils.to_categorical(Y, num_classes=45)
    X, Y = datagen.flow(X, Y, batch_size=X.shape[0]).next()

    # Score trained model.
    scores = model.evaluate(X, Y, verbose=1)
    print('%d samples tested' % Y.shape[0])
    print('Test loss:', scores[0])
    print('Test accuracy:', scores[1])
    return


if __name__ == "__main__":
    # predict('00_pic6.png')
    score_model()
