import os
import cv2
import copy
import numpy as np
from frcnn_rpn_gt import calc_gt_rpn


def get_new_img_size(width, height, img_min_side=450):
    if width <= height:
        f = img_min_side / width
        resized_height = int(f * height)
        resized_width = img_min_side
    else:
        f = img_min_side / height
        resized_width = int(f * width)
        resized_height = img_min_side

    return resized_width, resized_height, f


def get_data(input_path, folder_path='../detection_images'):
    """Parse the data from annotation file

    Args:
        input_path: annotation file path
        folder_path: path to place before filepaths in annotation file
    """

    all_imgs = {}
    classes_count = {}

    file = open(input_path, 'r')

    #  0     1      2       3  4  5  6  7
    # (path, width, height, x, y, w, h, class_ident)
    for line in file:
        splitted = line.split(',')

        class_ident = int(splitted[7])
        if class_ident in classes_count:
            classes_count[class_ident] += 1
        else:
            classes_count[class_ident] = 1

        filename = splitted[0]
        if filename not in all_imgs:
            all_imgs[filename] = {'filepath': os.path.join(folder_path, filename),
                                  'width': int(splitted[1]),
                                  'height': int(splitted[2]),
                                  'bboxes': []}

        bbox = {'class': class_ident,
                'x': int(splitted[3]),
                'y': int(splitted[4]),
                'w': int(splitted[5]),
                'h': int(splitted[6])}

        all_imgs[filename]['bboxes'].append(bbox)

    all_data = list(all_imgs.values())
    classes_count['bg'] = 0

    file.close()
    return all_data, classes_count


def data_generator(all_data, config, debug=False):
    """ Yield image and the ground-truth anchors

    Args:
        all_data: list(filepath, width, height, list(bboxes))
        config
        debug: if true, don't normalize img for processing

    Returns:
        img: image after resizing and scaling (smallest size = 450px)
        Y: [y_class, y_regr]
        img_data: resized img_data
        num_pos: number of positive anchors
    """

    while True:
        for img_data in all_data:
            # copy img_data, so we doesn't change the original
            img_data = copy.deepcopy(img_data)

            # check if data is valid
            assert 'filepath' in img_data
            assert 'bboxes' in img_data
            assert 'width' in img_data
            assert 'height' in img_data

            # read image
            img = cv2.imread(img_data['filepath'])

            width, height = img_data['width'], img_data['height']
            rows, cols, _ = img.shape

            assert cols == width
            assert rows == height

            # resize img and bboxes
            img_data['width'], img_data['height'], fac = get_new_img_size(width, height, config.im_size)
            img = cv2.resize(img, (img_data['width'], img_data['height']), interpolation=cv2.INTER_CUBIC)
            for bbox in img_data['bboxes']:
                bbox['x'] = int(fac * bbox['x'])
                bbox['y'] = int(fac * bbox['y'])
                bbox['w'] = int(fac * bbox['w'])
                bbox['h'] = int(fac * bbox['h'])

            y_class, y_regr, num_pos = calc_gt_rpn(img_data, config)

            # TODO: ist das wirklich hilfreich?
            # preprocess image
            if not debug:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
                img -= np.array(config.img_channel_mean)
                img /= config.img_scaling_factor
            img = np.expand_dims(img, axis=0)

            yield img, y_class, y_regr, img_data, num_pos
