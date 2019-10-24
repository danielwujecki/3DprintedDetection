import numpy as np
from keras.utils import to_categorical
from utils import calc_iou_fast


def calc_gt_class(rois, img_data, config):
    # prepare bboxes for iou calculation
    bboxes = img_data['bboxes']
    bbox_classes = np.zeros(len(bboxes))
    bboxes_iou = np.zeros((4, rois.shape[0], len(bboxes)))
    for bbox_num, bbox in enumerate(bboxes):
        bboxes_iou[0, :, bbox_num] = bbox['x']
        bboxes_iou[1, :, bbox_num] = bbox['y']
        bboxes_iou[2, :, bbox_num] = bbox['w']
        bboxes_iou[3, :, bbox_num] = bbox['h']
        bbox_classes[bbox_num] = bbox['class']

    # calc rois in (x, y, w, h) style
    rois_formatted = rois.copy().transpose()
    rois_formatted[2] -= rois_formatted[0]
    rois_formatted[3] -= rois_formatted[1]

    # prepare rois for iou calculation
    rois_iou = np.zeros((len(bboxes), 4, rois.shape[0]))
    rois_iou[:] = rois_formatted
    rois_iou = rois_iou.transpose((1, 2, 0))

    # calculate all ious - shape: (num_rois, num_bboxes)
    ious = calc_iou_fast(rois_iou, bboxes_iou)
    best_ious = np.max(ious, axis=1)
    best_args = np.argmax(ious, axis=1)

    # get the indices of valid rois
    bg_idx = np.where(np.logical_and(best_ious >= config.classifier_min_overlap,
                                     best_ious < config.classifier_max_overlap))[0]
    fg_idx = np.where(best_ious >= config.classifier_max_overlap)[0]
    all_idx = np.concatenate((bg_idx, fg_idx))

    # choose valid rois and their best IoUs
    x_roi = rois_formatted.transpose()[all_idx]
    x_roi = x_roi / config.base_network_downscale
    x_roi = np.expand_dims(x_roi, axis=0)
    IoUs = best_ious[all_idx]

    y_class_all = np.zeros(rois.shape[0])
    y_class_all[bg_idx] = config.nb_classes - 1
    y_class_all[fg_idx] = bbox_classes[best_args[fg_idx]]
    y_class = to_categorical(y_class_all[all_idx], num_classes=config.nb_classes)
    y_class = np.expand_dims(y_class, axis=0)
    y_class_all = y_class_all.astype('int')     # needed later ;-)

    # offset calculation
    bboxes_center = np.zeros((2, len(bboxes)))
    bboxes_center[0] = bboxes_iou[0, 0, :] + bboxes_iou[2, 0, :] / 2.
    bboxes_center[1] = bboxes_iou[1, 0, :] + bboxes_iou[3, 0, :] / 2.
    bboxes_center = bboxes_center.transpose()

    rois_center = np.zeros((2, rois.shape[0]))
    rois_center[0] = rois_formatted[0] + rois_formatted[2] / 2.
    rois_center[1] = rois_formatted[1] + rois_formatted[3] / 2.

    tx = (bboxes_center[best_args, 0] - rois_center[0]) / rois_formatted[2]
    ty = (bboxes_center[best_args, 1] - rois_center[1]) / rois_formatted[3]
    tw = bboxes_iou[2, 0, best_args] / rois_formatted[2]
    th = bboxes_iou[3, 0, best_args] / rois_formatted[3]

    tx = tx[fg_idx]
    ty = ty[fg_idx]
    tw = np.log(tw[fg_idx])
    th = np.log(th[fg_idx])

    sx, sy, sw, sh = config.classifier_regr_std
    y_regr_cord = np.stack((tx * sx, ty * sy, tw * sw, th * sh)).transpose()

    y_regr = -100. * np.ones((rois.shape[0], config.nb_classes - 1, 4))
    y_regr[fg_idx, y_class_all[fg_idx]] = y_regr_cord
    y_regr = np.reshape(y_regr, (y_regr.shape[0], -1))
    y_regr = np.expand_dims(y_regr[all_idx], axis=0)

    return x_roi, y_class, y_regr, IoUs
