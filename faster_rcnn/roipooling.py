import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.engine import Layer

from utils import apply_regr
from non_max_suppression import nm_suppress


class ROIPoolingLayer(Layer):
    """ Implements Region Of Interest Max Pooling
        for channel-first images and relative bounding box coordinates

        # Constructor parameters
            pooled_height, pooled_width (int) --
              specify height and width of layer outputs

        Shape of inputs
            [(batch_size, pooled_height, pooled_width, n_channels),
             (batch_size, num_rois, 4)]

        Shape of output
            (batch_size, num_rois, pooled_height, pooled_width, n_channels)

    """

    def __init__(self, pooled_height=9, pooled_width=9, **kwargs):
        self.pooled_height = pooled_height
        self.pooled_width = pooled_width

        super(ROIPoolingLayer, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        """ Returns the shape of the ROI Layer output
        """
        feature_map_shape, rois_shape = input_shape
        assert feature_map_shape[0] == rois_shape[0]
        batch_size = feature_map_shape[0]
        n_rois = rois_shape[1]
        n_channels = feature_map_shape[3]
        return (batch_size, n_rois, self.pooled_height,
                self.pooled_width, n_channels)

    def call(self, x):
        """ Maps the input tensor of the ROI layer to its output

            # Parameters
                x[0] -- Convolutional feature map tensor,
                        shape (batch_size, pooled_height, pooled_width, n_channels)
                x[1] -- Tensor of region of interests from candidate bounding boxes,
                        shape (batch_size, num_rois, 4)

            # Output
                pooled_areas -- Tensor with the pooled region of interest, shape
                    (batch_size, num_rois, pooled_height, pooled_width, n_channels)
        """
        def curried_pool_rois(x):
            return ROIPoolingLayer._pool_rois(x[0], x[1],
                                              self.pooled_height,
                                              self.pooled_width)

        return tf.map_fn(curried_pool_rois, x, dtype=tf.float32)

    @staticmethod
    def _pool_rois(feature_map, rois, pooled_height, pooled_width):
        """ Applies ROI pooling for a single image and varios ROIs
        """
        def curried_pool_roi(roi):
            return ROIPoolingLayer._pool_roi(feature_map, roi,
                                             pooled_height, pooled_width)

        return tf.map_fn(curried_pool_roi, rois, dtype=tf.float32)

    @staticmethod
    def _pool_roi(feature_map, roi, pooled_height, pooled_width):
        """ Applies ROI pooling to a single image and a single region of interest
        """

        x = K.cast(roi[0], 'int32')
        y = K.cast(roi[1], 'int32')
        w = K.cast(roi[2], 'int32')
        h = K.cast(roi[3], 'int32')

        # Resized roi of the image to pooling size (7x7)
        with tf.device('/cpu:0'):
            resized = tf.image.resize(feature_map[y:y+h, x:x+w, :],
                                      (pooled_height, pooled_width))

        return resized


def rpn_to_roi(y_class, y_regr, w, h, config):
    """Convert rpn layer to roi bboxes

    Args:
        y_class: output layer rpn classification
            shape: (1, feature_height, feature_width, num_anchors)
        y_regr: output layer rpn regression
            shape: (1, feature_height, feature_width, num_anchors * 4)
        config

    Returns:
        result: boxes from non-max-suppression (shape=(300, 4))
            boxes: coordinates for bboxes (on the feature map)
    """

    assert y_class.shape[0] == y_regr.shape[0] == 1

    y_regr /= config.std_scaling

    # cord.shape = (4, feature_map.height, feature_map.width, num_anchors)
    rows, cols = y_class.shape[1:3]
    cord = np.zeros((4, rows, cols, config.num_anchors))

    anchor_sizes = config.anchor_box_scales   # (4 in here)
    anchor_ratios = config.anchor_box_ratios  # (3 in here)
    curr_layer = -1
    for a_size in anchor_sizes:
        for a_ratio in anchor_ratios:
            curr_layer += 1

            # For every point in x, there are all the y points and vice versa
            X, Y = np.meshgrid(np.arange(cols), np.arange(rows))
            cord[0, :, :, curr_layer] = X + .5
            cord[1, :, :, curr_layer] = Y + .5
            cord[0, :, :, curr_layer] *= config.base_network_downscale
            cord[1, :, :, curr_layer] *= config.base_network_downscale
            cord[2, :, :, curr_layer] = a_size * a_ratio[0]
            cord[3, :, :, curr_layer] = a_size * a_ratio[1]

            # curr_layer: 0~8 (9 anchors)
            regr = y_regr[0, :, :, 4 * curr_layer:4 * curr_layer + 4]  # shape => (18, 25, 4)
            regr = np.transpose(regr, (2, 0, 1))                       # shape => (4, 18, 25)

            # Apply regression to x, y, w and h if there is rpn regression layer
            cord[:, :, :, curr_layer] = apply_regr(cord[:, :, :, curr_layer], regr)

            # Convert (x, y , w, h) to (x1, y1, x2, y2)
            cord[2, :, :, curr_layer] += cord[0, :, :, curr_layer]
            cord[3, :, :, curr_layer] += cord[1, :, :, curr_layer]

            # Avoid bboxes drawn outside of the image
            cord[0, :, :, curr_layer] = np.maximum(0, cord[0, :, :, curr_layer])
            cord[1, :, :, curr_layer] = np.maximum(0, cord[1, :, :, curr_layer])
            cord[2, :, :, curr_layer] = np.minimum(w, cord[2, :, :, curr_layer])
            cord[3, :, :, curr_layer] = np.minimum(h, cord[3, :, :, curr_layer])

    # flatten to (all_dims, 4)
    boxes = np.reshape(cord, (4, -1)).transpose()
    probs = y_class.flatten()

    # Find out the bboxes which are illegal or very small and delete them from bboxes list
    idxs = np.where(np.logical_or(boxes[:, 0] - boxes[:, 2] > -1 * config.base_network_downscale,
                                  boxes[:, 1] - boxes[:, 3] > -1 * config.base_network_downscale))
    boxes = np.delete(boxes, idxs, 0)
    probs = np.delete(probs, idxs, 0)

    idx = nm_suppress(boxes, probs,
                      overlap_thresh=config.overlap_threshold,
                      max_boxes=config.num_boxes)

    return boxes[idx].astype('int')
