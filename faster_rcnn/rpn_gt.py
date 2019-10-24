import random
import numpy as np
from functools import reduce
from utils import calc_iou_fast


def calc_gt_rpn(img_data, config):
    """ calc ground truth labels for alle possible anchors
    Args:
        config
        img_data
    Returns:
        y_class: array which marks objects for anchors
            shape = (1, feature_height, feature_width, num_anchors, 2)
        y_regr: normalized regrission scores for positive anchors
            shape = (1, feature_height, feature_width, num_anchors, 4)
    """

    downscale = config.base_network_downscale
    num_anchors = config.num_anchors
    anchor_sizes = config.anchor_box_scales
    anchor_ratios = config.anchor_box_ratios

    # to calculate offset value
    num_bboxes = len(img_data['bboxes'])

    # calculate the output map size based on the network architecture
    feature_width, feature_height = img_data['width'] // downscale, img_data['height'] // downscale

    # calculated all anchor coordinates
    anchor_cord = np.zeros((4, feature_height, feature_width, num_anchors), dtype='float')

    curr_layer = -1
    for a_size in anchor_sizes:
        for a_ratio in anchor_ratios:
            curr_layer += 1

            anchor_w = a_size * a_ratio[0]
            anchor_h = a_size * a_ratio[1]

            # For every point in x, there are all the y points and vice versa
            X, Y = np.meshgrid(np.arange(feature_width), np.arange(feature_height))
            anchor_cord[0, :, :, curr_layer] = X + .5
            anchor_cord[1, :, :, curr_layer] = Y + .5
            anchor_cord[0, :, :, curr_layer] *= downscale
            anchor_cord[1, :, :, curr_layer] *= downscale
            anchor_cord[0, :, :, curr_layer] -= anchor_w / 2
            anchor_cord[1, :, :, curr_layer] -= anchor_h / 2
            anchor_cord[2, :, :, curr_layer] = anchor_w
            anchor_cord[3, :, :, curr_layer] = anchor_h

    # set unallowed anchors to 0
    ai, bi = anchor_cord[0] < 0, anchor_cord[1] < 0
    ci = anchor_cord[0] + anchor_cord[2] > img_data['width']
    di = anchor_cord[1] + anchor_cord[3] > img_data['height']
    idx = np.where(reduce(np.logical_or, (ai, bi, ci, di)))
    anchor_cord[:, idx[0], idx[1], idx[2]] = 1e-5

    anchor_center = anchor_cord[:2] + anchor_cord[2:] / 2

    # initialise empty output objectives
    y_class = np.zeros((feature_height, feature_width, num_anchors, 2), dtype='uint8')
    y_regr = np.zeros((feature_height, feature_width, num_anchors, 4))

    # Initialize with 'negative'
    y_class[:, :, :] = [1, 0]

    best_iou_for_loc = np.zeros((feature_height, feature_width, num_anchors), dtype='float')

    for bbox_num in range(num_bboxes):
        # get IOU of the current GT box and the current anchor box
        bbox = img_data['bboxes'][bbox_num]
        a = (bbox['x'], bbox['y'], bbox['w'], bbox['h'])
        ious = calc_iou_fast(a, anchor_cord)

        x_center = bbox['x'] + .5 * bbox['w']
        y_center = bbox['y'] + .5 * bbox['h']
        # normalisierte abweichungen/ziele berechnen
        tx = (x_center - anchor_center[0]) / (anchor_cord[2])
        ty = (y_center - anchor_center[1]) / (anchor_cord[3])
        tw = np.log(bbox['w'] / anchor_cord[2])
        th = np.log(bbox['h'] / anchor_cord[3])
        loc_regr = np.stack((tx, ty, tw, th)).transpose((1, 2, 3, 0))

        # set neutral, where not already positive and iou < rpn_max_overlap
        neutral = np.logical_and(config.rpn_max_overlap > ious, ious >= config.rpn_min_overlap)
        negative = (y_class[:, :, :, 1] == 0)
        idx = np.where(np.logical_and(neutral, negative))
        y_class[idx] = [0, 0]

        idx = np.logical_and(ious >= config.rpn_max_overlap, ious > best_iou_for_loc)
        idx = np.where(idx)
        y_class[idx] = [1, 1]
        y_regr[idx] = loc_regr[idx]
        best_iou_for_loc[idx] = ious[idx]

        if idx[0].shape[0] == 0 and np.max(ious) > 0.:
            best = np.where(ious == np.max(ious))
            y_class[best] = [1, 1]
            y_regr[best] = loc_regr[best]
            # take this box for sure
            best_iou_for_loc[best] = 1.1

    pos_locs = np.where(np.logical_and(y_class[:, :, :, 0] == 1, y_class[:, :, :, 1] == 1))
    neg_locs = np.where(np.logical_and(y_class[:, :, :, 0] == 1, y_class[:, :, :, 1] == 0))
    num_pos = len(pos_locs[0])
    num_neg = len(neg_locs[0])

    # one issue is that the RPN has many more negative than positive regions,
    # so we turn off some of the negative regions
    num_regions = config.max_num_regions

    if num_pos > num_regions / 2:
        val_locs = random.sample(range(num_pos), num_pos - num_regions // 2)
        y_class[pos_locs[0][val_locs], pos_locs[1][val_locs], pos_locs[2][val_locs]] = [0, 0]
        num_pos = num_regions // 2

    # only so many negatives as positives
    if num_neg + num_pos > num_regions:
        val_locs = random.sample(range(num_neg), num_neg - num_pos)
        y_class[neg_locs[0][val_locs], neg_locs[1][val_locs], neg_locs[2][val_locs]] = [0, 0]

    # scale regression values
    y_regr *= config.std_scaling
    # set regression to -1, when there isn't a object
    y_regr[np.where(y_class[:, :, :, 1] == 0)] = -100.

    # adapt shapes to match RPN output
    y_class = np.concatenate((y_class[:, :, :, 0], y_class[:, :, :, 1]), axis=2)
    y_regr = np.reshape(y_regr, y_regr.shape[:-2] + (-1,))
    y_class = np.expand_dims(y_class, axis=0)
    y_regr = np.expand_dims(y_regr, axis=0)

    return y_class, y_regr, num_pos
