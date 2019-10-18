import cv2
import numpy as np


class Detector(object):
    __color_list = [[255, 0, 0], [0, 255, 0],
                    [0, 0, 255], [150, 150, 150],
                    [200, 200, 0], [0, 200, 200],
                    [200, 0, 200]]

    def __init__(self, cnn, tresh=0.01):
        self.__color_count = len(self.__color_list)
        self.__cnn = cnn
        self.__tresh = tresh
        self.__mask = None
        self.__blurred = None

    def __normalize_img(self, img):
        # min value 0, max value 255
        img -= np.min(img)
        img = img.astype('uint64') * 255 // np.max(img)
        return img.astype('uint8')

    def __resize_img(self, img, shape=(128, 128), debug=True):
        if debug and (img.shape[0] < shape[0] or img.shape[1] < shape[1]):
            print("Image shape is small: ", img.shape)
        return cv2.resize(img, (shape[0], shape[1]))

    def __square(self, x, y, w, h):
        if w < h:
            x -= (h - w) // 2
            w = h
        elif h < w:
            y -= (w - h) // 2
            h = w
        return x, y, w, h

    def __find_contours(self, frame):
        self.__blurred = cv2.GaussianBlur(frame, (21, 21), 5)

        hsv = cv2.cvtColor(self.__blurred, cv2.COLOR_RGB2HSV)

        low1 = (0, 0, 85)
        high1 = (80, 110, 255)

        low2 = (120, 0, 110)
        high2 = (179, 100, 255)

        mask1 = cv2.inRange(hsv, low1, high1)
        mask2 = cv2.inRange(hsv, low2, high2)
        t_img = cv2.bitwise_or(mask1, mask2)

        t_img = cv2.dilate(t_img, np.ones((11, 11), np.uint8), iterations=1)
        self.__mask = t_img

        contours, _ = cv2.findContours(
            t_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours

    def __cnn_prepare(self, frame, x, y, w, h):
        if self.__mask is None or self.__blurred is None:
            print("__cnn_prepare: execute __find_contours before.")
            return None

        img = frame.copy()

        mask = self.__mask
        mask = np.dstack((mask, mask, mask))
        mask_inverse = np.ones(mask.shape).astype(np.uint8) * 255 - mask

        img = cv2.bitwise_and(img, mask)
        img += cv2.bitwise_and(self.__blurred, mask_inverse)

        img = img[y:y + h, x:x + w]
        img = self.__normalize_img(self.__resize_img(img))
        return img

    def predict(self, frame):
        min_size = frame.shape[0] * frame.shape[1] * self.__tresh

        frame = self.__normalize_img(frame)
        contours = self.__find_contours(frame)

        object_cnt = 0

        boxes = []
        for cont in contours:
            xo, yo, wo, ho = cv2.boundingRect(cont)
            if wo * ho >= min_size:
                object_cnt += 1

                x, y, w, h = self.__square(xo, yo, wo, ho)
                case1 = x <= 0 or y <= 0 or w <= 16 or h <= 16
                case2 = y + h >= frame.shape[0] or x + w >= frame.shape[1]
                if case1 or case2:
                    continue

                cnn_img = self.__cnn_prepare(frame, x, y, w, h)
                if cnn_img is None:
                    continue

                confidence, classnum = self.__cnn.predict(cnn_img)
                boxes.append((classnum, confidence, xo, yo, wo, ho))

        # print("Found {} objects in {} contours.".format(object_cnt, len(contours)))
        return boxes

    def detect(self, frame):
        boxes = self.predict(frame)
        for i, bbox in enumerate(boxes):
            classnum, confidence = bbox[:2]
            x, y, w, h = bbox[2:]
            color = self.__color_list[i % self.__color_count]

            # frame = cv2.drawContours(frame, contours, i, color, 2)
            frame = cv2.rectangle(frame, (x, y), (x + w, y + h), color, 5)

            text = "Class {}: {}%".format(classnum, int(confidence * 100))
            frame[y-100:y, x-6:x+600] = color
            frame = cv2.putText(frame, text, (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX,
                                2.5, (255, 255, 255), 5)
        return frame
