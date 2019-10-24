# import matplotlib.pyplot as plt
import os
import cv2
import numpy as np


def process(in_path: str, out_path: str):
    img = cv2.imread(in_path, 1)
    img = normalize_img(img)
    img = crop_img(img, debug=False)
    img = resize_img(img, shape=(128, 128))
    img = normalize_img(img)
    cv2.imwrite(out_path, img)
    return


def normalize_img(img):
    # min value 0, max value 255
    img -= np.min(img)
    img = img.astype('uint64') * 255 // np.max(img)
    return img.astype('uint8')


def crop_img(img, blowout=True, blur=False, blackout=False, debug=False):
    t_img = cv2.GaussianBlur(img, (21, 21), 5)

    hsv = cv2.cvtColor(t_img, cv2.COLOR_BGR2HSV)

    low1 = (0, 0, 85)
    high1 = (80, 110, 255)

    low2 = (120, 0, 110)
    high2 = (179, 100, 255)

    mask1 = cv2.inRange(hsv, low1, high1)
    mask2 = cv2.inRange(hsv, low2, high2)

    t_img = cv2.bitwise_or(mask1, mask2)

    if blowout:
        t_img = cv2.dilate(t_img, np.ones((15, 15), np.uint8), iterations=1)

    contours, _ = cv2.findContours(t_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find object with the biggest bounding box
    mx = (0, 0, 0, 0)      # biggest bounding box so far
    mx_area = 0
    mx_contourIdx = 0
    org_area = img.shape[0] * img.shape[1]  # area if original image

    for i, cont in enumerate(contours):
        x, y, w, h = cv2.boundingRect(cont)
        area = w * h
        if area > mx_area and area < org_area:
            mx = x, y, w, h
            mx_area = area
            mx_contourIdx = i

    # blur outside of contour
    if blur:
        blurred_image = cv2.GaussianBlur(img, (31, 31), 10)

        mask = np.zeros(img.shape, dtype=np.uint8)
        cv2.fillPoly(mask, [contours[mx_contourIdx]], (255, 255, 255))
        mask_inverse = np.ones(mask.shape).astype(np.uint8) * 255 - mask

        img = cv2.bitwise_and(img, mask)
        img += cv2.bitwise_and(blurred_image, mask_inverse)

    # black out outside of contour
    if blackout:
        black = np.zeros(img.shape).astype(img.dtype)
        cont = [contours[mx_contourIdx]]
        black_white = cv2.fillPoly(black, cont, 255)
        img = cv2.bitwise_and(img, black_white)

    # make bounding box quadratic
    x, y, w, h = mx
    if w < h:
        x -= (h - w) // 2
        w = h
    elif h < w:
        y -= (w - h) // 2
        h = w

    assert x > 0
    assert y > 0
    assert x + w < img.shape[1]
    assert y + h < img.shape[0]

    if debug:
        # draw contour and box
        img_cont = img
        img_cont = cv2.drawContours(img_cont, contours, mx_contourIdx, (0, 255, 255), 5)
        img_cont = cv2.rectangle(img_cont, (x, y), (x + w, y + h), (255, 0, 255), 5)
        return img_cont

    return img[y:y + h, x:x + w]


def resize_img(img, shape=(128, 128), debug=True):
    # resize image
    if debug and (img.shape[0] < shape[0] or img.shape[1] < shape[1]):
        print("Image shape is small: ", img.shape)
    return cv2.resize(img, (shape[0], shape[1]))


if __name__ == "__main__":
    save_path = '../data/test_croped/'
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    path = '../data/classification/'
    subfolders = os.listdir(path)
    for j, folder in enumerate(subfolders):
        msg = 'Fortschritt - Ordner: {} von {}\r'.format(j + 1, len(subfolders))
        print(msg, end='', flush=True)
        content = os.listdir(path + folder)
        for i, img_name in enumerate(content):
            input_path = '%s%s/%s' % (path, folder, img_name)
            output_path = '%s%s_pic%d.png' % (save_path, folder, i)
            # output_path = '%s%s_%s.png' % (save_path, folder, img_name)
            try:
                process(input_path, output_path)
            except AssertionError:
                print("Assertion Error for:\n%s" % input_path)
