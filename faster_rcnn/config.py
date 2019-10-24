from math import sqrt


class Config(object):
    """
    Configuration class to set hyperparameters etc.
    """

    def __init__(self):
        self.train_path = '../data/train_detection/bboxes_detect_imgs.csv'
        self.weights_path = 'model/frcnn_weights.hdf5'
        self.history_path = 'model/history.csv'
        self.plot_path = 'model/history_plot.png'

        # image channel-wise mean to subtract
        self.img_channel_mean = [111.476, 114.427, 111.402]
        self.img_scaling_factor = 255.

        # desired size of the smallest side of the image
        # Original setting in paper is 600
        self.im_size = 600

        # Anchor box scales
        # Note that if im_size is smaller, anchor_box_scales should be scaled
        # Original anchor_box_scales in the paper: [128, 256, 512]
        self.anchor_box_scales = [64, 114, 184, 256]

        # Anchor box ratios (width, height)
        self.anchor_box_ratios = [(1., 1.),
                                  (1. / sqrt(2), 2. / sqrt(2)),
                                  (2. / sqrt(2), 1. / sqrt(2))]

        self.num_anchors = len(self.anchor_box_scales) * len(self.anchor_box_ratios)

        self.max_num_regions = 256

        # stride at the RPN - depends on the network architecture (VGG16 here)
        self.base_network_downscale = 16
        self.base_network_trainable = False
        self.learning_rate = 1e-5

        # original paper: 512
        self.rpn_conv_size = 512
        # original paper: 4096
        self.classifier_neurons = 4096

        # number of classes (have to include a background class)
        self.nb_classes = 45 + 1

        # number of ROIs processed at once in ROIPolling Layer
        self.num_rois = 8
        self.pool_size = 9

        # values for non max suppression
        self.num_boxes = 300
        self.overlap_threshold = 0.7

        # scaling the stdev
        self.std_scaling = 4.0
        self.classifier_regr_std = [8.0, 8.0, 4.0, 4.0]

        # overlaps for RPN
        self.rpn_min_overlap = 0.3
        self.rpn_max_overlap = 0.7

        # overlaps for classifier ROIs
        self.classifier_min_overlap = 0.1
        self.classifier_max_overlap = 0.5

        # training settings
        self.epoch_length = 1000 # 8210 # batches
        self.num_epochs = 16 # epochs to train before terminating
