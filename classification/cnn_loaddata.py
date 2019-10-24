import os
import cv2
import numpy as np


BASE_PATH = '/'.join(os.getcwd().split('/'))
TEST_PATH = os.path.join(BASE_PATH, 'data/test_croped')

def test(dtype='float32'):
    test_images = os.listdir(TEST_PATH)

    testi = []
    testl = []

    for img_name in test_images:
        img = cv2.imread(os.path.join(TEST_PATH, img_name))
        label = int(img_name.split('_')[0])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        testi.append(img)
        testl.append(label)

    testi = np.asfarray(testi, dtype=dtype)
    testl = np.asfarray(testl)

    return testi, testl


def test_image(img_name: str,
               meanf='model/featurewise_mean.npy',
               stdf='model/featurewise_std.npy', dtype='float32'):

    img = cv2.imread(os.path.join(TEST_PATH, img_name))
    img = np.asfarray(img, dtype=dtype)

    mean = np.load(meanf, allow_pickle=True)
    std = np.load(stdf, allow_pickle=True)
    img /= np.max(img)
    img -= mean
    img /= std
    return img


if __name__ == "__main__":
    pass
