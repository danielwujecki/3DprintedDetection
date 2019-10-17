from keras.layers import Flatten, Dense, Conv2D, TimeDistributed
from keras.layers import Dropout, Input
from keras.models import Model
from keras.applications import VGG16
from roipooling import ROIPoolingLayer


def rpn_layer(X_img, config):
    """
    Create a rpn layer
        Step1: Pass input to a 3x3 512 channels convolutional layer, padding 'same' to preserve input size
        Step2: two (1,1) convolutional layers:
                classification layer: num_anchors for binary classification: object / no object
                regression layer: num_anchors * 4 for regression of bboxes with linear activation
    Args:
        X_img: (1, rows, cols, channels) output of ConvNet (eg. VGG16)
        config: configuration for hyperparameter

    Returns:
        tuple of:
            x_class: classification for whether it's an object
            x_regr: bboxes regression
    """

    num_anchors = config.num_anchors

    x = Conv2D(config.rpn_conv_size, (3, 3),
               padding='same',
               activation='relu',
               kernel_initializer='normal',
               name='rpn_conv1')(X_img)

    x_class = Conv2D(num_anchors, (1, 1),
                     activation='sigmoid',
                     kernel_initializer='uniform',
                     name='rpn_out_class')(x)

    x_regr = Conv2D(num_anchors * 4, (1, 1),
                    activation='linear',
                    kernel_initializer='zero',
                    name='rpn_out_regress')(x)

    return x_class, x_regr


def classification_layer(X_img, X_roi, config):
    """
    Create a classifier layer

    Args:
        X_img: (1, rows, cols, channels) output of ConvNet (eg. VGG16)
        X_roi: (1, num_rois, 4) list of rois (x, y, w, h)
        config

    Returns:
        tuple off
            x_class: classifier layer output
            x_regr: regression layer output
    """

    neurons = config.classifier_neurons

    # 7x7 roi pooling with 4 rois processed at once
    out_roi_pool = ROIPoolingLayer(config.pool_size, config.pool_size)([X_img, X_roi])

    # Flatten the convlutional layer and connected to 2 FC with dropout
    x = TimeDistributed(Flatten(name='flatten'))(out_roi_pool)
    x = TimeDistributed(Dense(neurons, activation='relu', name='fc1'))(x)
    x = TimeDistributed(Dropout(0.5))(x)
    x = TimeDistributed(Dense(neurons, activation='relu', name='fc2'))(x)
    x = TimeDistributed(Dropout(0.5))(x)

    # There are two output layers
    # out_class: softmax acivation function for classify the class name of the object
    # out_regr: linear activation function for bboxes coordinates regression

    # shape: (1, num_rois, 46)
    x_class = TimeDistributed(Dense(config.nb_classes,
                                    activation='softmax',
                                    kernel_initializer='zero'),
                              name='fc_class')(x)
    # note: no regression target for bg class
    # shape: (1, num_rois, 180)
    x_regr = TimeDistributed(Dense(4 * config.nb_classes - 4,
                                   activation='linear',
                                   kernel_initializer='zero'),
                             name='fc_regress')(x)

    return x_class, x_regr


def build_model_training(config):
    """
    build the Faster-RCNN Keras Model based on VGG16 for training
    """

    img_input = Input(shape=(None, None, 3))
    roi_input = Input(shape=(None, 4))

    vgg16 = VGG16(include_top=False, input_tensor=img_input)
    for layer in vgg16.layers:
        layer.trainable = config.base_network_trainable

    shared_layers = vgg16.get_layer('block5_conv2').output

    rpn = rpn_layer(shared_layers, config)
    classifier = classification_layer(shared_layers, roi_input, config)

    model_rpn = Model(img_input, rpn)
    model_classifier = Model((img_input, roi_input), classifier)
    model_all = Model([img_input, roi_input], rpn + classifier)

    return model_rpn, model_classifier, model_all


def build_model(config):
    """
    build the Faster-RCNN Keras Model based on VGG16
    """

    img_input = Input(shape=(None, None, 3))
    roi_input = Input(shape=(None, 4))
    feature_map_input = Input(shape=(None, None, 512))

    vgg16 = VGG16(include_top=False, input_tensor=img_input)
    shared_layers = vgg16.get_layer('block5_conv2').output

    rpn = rpn_layer(shared_layers, config)
    classifier = classification_layer(feature_map_input, roi_input, config)

    model_rpn = Model(img_input, rpn + (shared_layers,))
    model_classifier = Model((feature_map_input, roi_input), classifier)

    return model_rpn, model_classifier
