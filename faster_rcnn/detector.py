import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from config import Config
from model import build_model
from roipooling import rpn_to_roi
from non_max_suppression import nm_suppress


class Detector(object):
    __color_list = [[255, 0, 0], [0, 255, 0],
                    [0, 0, 255], [150, 150, 150],
                    [200, 200, 0], [0, 200, 200],
                    [200, 0, 200]]


    def __init__(self):
        self.__nbc = len(self.__color_list)
        self.__conf = Config()

        modelparts = build_model(self.__conf)
        self.__rpn, self.__classifier = modelparts

        self.__rpn.load_weights(self.__conf.weights_path, by_name=True)
        self.__classifier.load_weights(self.__conf.weights_path, by_name=True)

        self.__rpn.compile(optimizer='sgd', loss='mse')
        self.__classifier.compile(optimizer='sgd', loss='mse')


    def __resize_img(self, img):
        im_size = self.__conf.im_size
        height, width, _ = img.shape
        if width <= height:
            ratio = im_size / width
            new_height = int(ratio * height)
            new_width = int(im_size)
        else:
            ratio = im_size / height
            new_width = int(ratio * width)
            new_height = int(im_size)
        img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        factor = width / new_width
        return img, factor


    def __preprocess_img(self, img):
        """return:
            * image preprocessed for FRCNN
            * ratio: size difference between preprocessed and original image
        """
        img_prepro, factor = self.__resize_img(img)
        img_prepro = img_prepro.astype(np.float32)
        img_prepro -= np.array(self.__conf.img_channel_mean)
        img_prepro /= self.__conf.img_scaling_factor
        img_prepro = np.expand_dims(img_prepro, axis=0)
        return img_prepro, factor


    def __run_frcnn(self, img):
        y_class, y_regr, features = self.__rpn.predict(img)

        w, h = img.shape[1:-1]
        rois = rpn_to_roi(y_class, y_regr, w, h, self.__conf)
        rois[:, 2] -= rois[:, 0]    # (x, y, w, h) format
        rois[:, 3] -= rois[:, 1]
        rois_inp = np.expand_dims(rois / self.__conf.base_network_downscale, axis=0)

        y_class, y_regr = self.__classifier.predict([features, rois_inp])

        return y_class, y_regr, rois


    def __apply_regr(self, y_regr, rpn_rois):
        assert y_regr.shape == rpn_rois.shape

        y_regr = y_regr / self.__conf.classifier_regr_std

        y_regr = y_regr.transpose()
        rpn_rois = rpn_rois.transpose()

        c_rois = np.zeros((2, rpn_rois.shape[1]))
        c_rois[0] = rpn_rois[0] + rpn_rois[2] / 2
        c_rois[1] = rpn_rois[1] + rpn_rois[3] / 2

        cx = y_regr[0] * rpn_rois[2] + c_rois[0]
        cy = y_regr[1] * rpn_rois[3] + c_rois[1]
        w = np.exp(y_regr[2]) * rpn_rois[2]
        h = np.exp(y_regr[3]) * rpn_rois[3]
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2

        rois = np.stack((x1, y1, x2, y2)).transpose()
        return rois


    def __choose_best(self, y_class, y_regr, rpn_rois, thresh=0.25, nm_thresh=0.4):
        if thresh is None:
            thresh = 0.25
        assert 0. <= thresh <= 1.
        assert 0. <= nm_thresh <= 1.

        print("Take all objects with more than {}% confidence".format(int(thresh * 100)))
        best_idx = np.argmax(y_class, axis=1)
        best_prob = y_class[np.arange(y_class.shape[0]), best_idx]

        max_class = self.__conf.nb_classes - 1
        idx = np.where(np.bitwise_or(best_prob < thresh,
                                     best_idx == max_class))[0]
        y_regr = np.delete(y_regr, idx, axis=0)
        rpn_rois = np.delete(rpn_rois, idx, axis=0)
        best_idx = np.delete(best_idx, idx, axis=0)
        best_prob = np.delete(best_prob, idx, axis=0)

        if not y_regr.shape[0]:
            return None, None, None

        y_regr = y_regr.reshape(y_regr.shape[0], -1, 4)
        y_regr = y_regr[np.arange(y_regr.shape[0]), best_idx]

        rois = self.__apply_regr(y_regr, rpn_rois)
        idx = nm_suppress(rois, best_prob, overlap_thresh=nm_thresh)

        return rois[idx], best_prob[idx], best_idx[idx]


    def __assignment(self, y_class, y_regr, rpn_rois, objects, thresh=0.1, nm_thresh=0.4):
        if thresh is None:
            thresh = 0.1
        assert 0. <= thresh <= 1.
        assert 0. <= nm_thresh <= 1
        assert objects is not None

        print('Assignment solver - search for the following objects:', objects)
        while True:
            roi_idx, col_idx = linear_sum_assignment(y_class[:, objects], maximize=True)
            roi_idx = roi_idx[np.argsort(col_idx)]

            probs = y_class[roi_idx, objects]

            regr = y_regr.reshape(y_regr.shape[0], -1, 4)
            regr = regr[roi_idx, objects]

            rois = self.__apply_regr(regr, rpn_rois[roi_idx])

            # find rois, which overleap with other bbox
            idx = nm_suppress(rois, probs, overlap_thresh=nm_thresh)
            idx_out = np.delete(np.arange(objects.shape[0]), idx, 0)
            idx_out = np.concatenate((idx_out, np.where(probs[idx] < thresh)[0]))

            if not idx_out.size:
                return rois, probs, objects

            best_prob = np.max(probs[idx_out])

            # delete rois, that overleap
            y_regr = np.delete(y_regr, roi_idx[idx_out], 0)
            y_class = np.delete(y_class, roi_idx[idx_out], 0)
            rpn_rois = np.delete(rpn_rois, roi_idx[idx_out], 0)

            # if there isn't a possible roi with prob > thresh or not enough rois, exit
            if best_prob < thresh or y_class.shape[0] < objects.shape[0]:
                print("Didn't found the following objects in the image:", objects[idx_out])
                return rois[idx], probs[idx], objects[idx]


    def predict(self, img, objects=None, thresh=None):
        img_prepro, factor = self.__preprocess_img(img)
        y_class, y_regr, rpn_rois = self.__run_frcnn(img_prepro)
        y_class, y_regr = y_class[0], y_regr[0]

        if not objects:
            rois, probs, cls_idx = self.__choose_best(y_class, y_regr, rpn_rois,
                                                      thresh=thresh)
        else:
            objects = np.array(objects, dtype='int')
            rois, probs, cls_idx = self.__assignment(y_class, y_regr, rpn_rois,
                                                     objects, thresh=thresh)

        if rois is None:
            return None, None

        res = np.zeros((rois.shape[0], 5), dtype='int')
        res[:, 0] = cls_idx.astype('int')
        res[:, 1:] = (rois * factor).astype('int')
        return res, probs


    def detect(self, img, objects=None, thresh=None):
        """Take a BGR Image and returns a RGB image
        predicted bboxes and classes are drawn into the image
        """
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        rois, probs = self.predict(img, objects, thresh)
        if rois is None:
            return img
        for i, r in enumerate(rois):
            x1, y1, x2, y2 = r[1:]
            col_idx = np.random.randint(0, self.__nbc)
            color = self.__color_list[col_idx]
            img = cv2.rectangle(img, (x1, y1), (x2, y2), color, 10)
            text = "Class {}: {}%".format(r[0], int(probs[i] * 100))
            img[y1-100:y1, x1-6:x1+600] = color
            img = cv2.putText(img, text, (x1, y1 - 20), cv2.FONT_HERSHEY_SIMPLEX,
                              2.5, (255, 255, 255), 7)
        return img
