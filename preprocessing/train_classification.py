import os
import cv2
import numpy as np


def process(in_path: str, out_path: str, colors=None):
    img = cv2.imread(in_path, cv2.IMREAD_UNCHANGED)
    assert img.shape == (512, 512, 4)

    img, mask = crop_img(img)
    save_imgs(img, mask, out_path, colors)


def crop_img(img, blowout=True):
    mask = np.zeros((512,) * 2, dtype='uint8')
    mask[np.where(img[:, :, 3] != 0)] = 255

    if blowout:
        t_img = cv2.dilate(mask.copy(), np.ones((4, 4), np.uint8), iterations=1)

    contours, _ = cv2.findContours(t_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find object with the biggest bounding box
    mx = (0, 0, 0, 0)      # biggest bounding box so far
    mx_area = 0
    org_area = img.shape[0] * img.shape[1]  # area if original image

    for i, cont in enumerate(contours):
        x, y, w, h = cv2.boundingRect(cont)
        area = w * h
        if area > mx_area and area < org_area:
            mx = x, y, w, h
            mx_area = area

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

    return img[y:y + h, x:x + w], mask[y:y + h, x:x + w]


def resize_img(img, shape=(256, 256), debug=False):
    # resize image
    if debug and (img.shape[0] < shape[0] or img.shape[1] < shape[1]):
        print("Image shape is small: ", img.shape)
    return cv2.resize(img, shape)


def save_imgs(img, mask, out_path: str, colors=None):
    if not colors:
        img = resize_img(img[:, :, :3], shape=(128, 128))
        cv2.imwrite(out_path, img)
        return

    img = img[:, :, :3]
    img = (img * np.random.uniform(low=0.85, high=1.0)).astype('uint8')
    yellow_off = np.random.randint(0, 17)
    id0, id1 = np.where(img[:, :, 0] > yellow_off)
    img[id0, id1, 0] -= yellow_off

    h, w = img.shape[:2]
    mask = np.dstack((mask, mask, mask))
    mask_inverse = np.ones(mask.shape, dtype='uint8') * 255 - mask

    for j, col in enumerate(colors):
        backg = np.zeros((h, w, 3), dtype=np.uint8)
        backg[:, :] = col
        max_diffuse = np.random.randint(10, 41, dtype='uint8')
        backg += np.random.randint(0, max_diffuse,
                                   size=backg.shape,
                                   dtype='uint8')

        img_col = cv2.bitwise_and(img, mask)
        img_col += cv2.bitwise_and(backg, mask_inverse)
        img_col = resize_img(img_col, shape=(128, 128))

        out_p = "{}_col{}{}".format(out_path[:-4], j, out_path[-4:])
        cv2.imwrite(out_p, img_col)


if __name__ == "__main__":
    color_list = [[230, 10, 10], [10, 230, 10], [10, 10, 230], [150, 150, 150]]
    color_list += [[0, 0, 0], [200, 200, 10], [10, 200, 200], [200, 10, 200]]

    save_path = './images_croped'
    if not os.path.isdir(save_path):
        os.mkdir(save_path)

    path = './images'
    subfolders = os.listdir(path)
    failed_objects = []
    for i, folder in enumerate(subfolders):
        msg = 'Fortschritt - Ordner: {} von {}\r'.format(i, len(subfolders))
        print(msg, end='', flush=True)

        if not os.path.isdir('{}/{}'.format(path, folder)):
            continue

        if not os.path.isdir('%s/%s' % (save_path, folder)):
            os.mkdir('%s/%s' % (save_path, folder))

        content = os.listdir('{}/{}'.format(path, folder))
        for img_name in content:
            input_path = '%s/%s/%s' % (path, folder, img_name)
            output_path = '%s/%s/%s' % (save_path, folder, img_name)
            try:
                process(input_path, output_path, color_list)
            except AssertionError:
                if folder not in failed_objects:
                    failed_objects.append(folder)
    print('\n', failed_objects)
