import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.optimizers import Adam
from keras.utils import generic_utils
from keras.metrics import categorical_crossentropy

from frcnn_config import Config
from frcnn_model import build_model_training
from frcnn_roipooling import rpn_to_roi
from frcnn_class_gt import calc_gt_class
from frcnn_data import get_data, data_generator
from frcnn_losses import rpn_loss_cls, get_loss_regr

assert os.path.isdir('model')

# get config and data
conf = Config()
train_imgs, classes_count = get_data(conf.train_path)
data_gen_train = data_generator(train_imgs, conf)

# build the model
model_rpn, model_classifier, model_all = build_model_training(conf)

# check if model is already existing
if not os.path.isfile(conf.weights_path):
    # start a new history
    record_df = pd.DataFrame(columns=['mean_overlapping_bboxes',
                                      'class_acc',
                                      'loss_rpn_cls',
                                      'loss_rpn_regr',
                                      'loss_class_cls',
                                      'loss_class_regr',
                                      'curr_loss',
                                      'elapsed_time'])
else:
    # load weights for whole model (maybe rpn, class separately?)
    model_all.load_weights(conf.weights_path, by_name=True)
    # load history
    record_df = pd.read_csv(conf.history_path)
    print('Continue training based on previous trained model.')
    print('Already trained {} epochs.'.format(len(record_df)))


# prepare training
model_rpn.compile(optimizer=Adam(lr=conf.learning_rate),
                  loss=[rpn_loss_cls, get_loss_regr(10)],
                  loss_weights=[1., 1.])

model_classifier.compile(optimizer=Adam(lr=conf.learning_rate),
                         loss=[categorical_crossentropy, get_loss_regr(4)],
                         loss_weights=[1., 1.],
                         metrics={'fc_class': 'accuracy'})


epoch_length, num_epochs = conf.epoch_length, conf.num_epochs
total_epochs, r_epochs = (len(record_df),) * 2
total_epochs += num_epochs

metrics = np.zeros((epoch_length, 6))

if record_df.empty:
    best_loss = np.Inf
else:
    best_loss = np.min(record_df['curr_loss'])


for epoch in range(num_epochs):
    r_epochs += 1
    progbar = generic_utils.Progbar(epoch_length)
    print('Epoch {}/{}'.format(r_epochs, total_epochs))
    start_time = time.time()

    i = 0
    while i < epoch_length:
        # fetch data from generator
        X, y_class, y_regr, image_data, _ = next(data_gen_train)

        # Train rpn model and get loss value
        metrics_rpn = model_rpn.train_on_batch(X, [y_class, y_regr])

        # Get predicted rpn from rpn model [rpn_cls, rpn_regr]
        y_class, y_regr = model_rpn.predict_on_batch(X)

        # Convert rpn output to roi bboxes
        w, h = image_data['width'], image_data['height']
        rois = rpn_to_roi(y_class, y_regr, w, h, conf)

        # X2: bboxes
        X2, y_class, y_regr, IouS = calc_gt_class(rois, image_data, conf)

        # If X2 is empty, continue
        if not X2.shape[1]:
            continue

        # Find out the positive anchors and negative anchors
        pos_samples = np.where(y_class[0, :, -1] == 0)[0]
        neg_samples = np.where(y_class[0, :, -1] == 1)[0]

        metrics[i, 5] = pos_samples.shape[0]

        num_pos = conf.num_rois // 2
        if conf.num_rois % 2 == 1:
            num_pos += np.random.randint(0, 2)

        if pos_samples.shape[0] <= num_pos:
            sel_pos_samples = pos_samples
        else:
            sel_pos_samples = np.random.choice(pos_samples, num_pos, replace=False)

        num_neg = conf.num_rois - sel_pos_samples.shape[0]
        if neg_samples.shape[0] <= num_neg:
            sel_neg_samples = neg_samples
        else:
            sel_neg_samples = np.random.choice(neg_samples, num_neg, replace=False)

        sel_samples = sel_pos_samples.tolist() + sel_neg_samples.tolist()

        X = [X, X2[:, sel_samples]]
        Y = [y_class[:, sel_samples], y_regr[:, sel_samples]]

        metrics_class = model_classifier.train_on_batch(X, Y)

        metrics[i, 0] = metrics_rpn[1]      # rpn class loss
        metrics[i, 1] = metrics_rpn[2]      # rpn regr loss
        metrics[i, 2] = metrics_class[1]    # final class loss
        metrics[i, 3] = metrics_class[2]    # final regr loss
        metrics[i, 4] = metrics_class[3]    # final class accurancy

        i += 1
        progbar.update(i, [('rpn_cls', np.mean(metrics[:i+1, 0])),
                           ('rpn_regr', np.mean(metrics[:i+1, 1])),
                           ('final_cls', np.mean(metrics[:i+1, 2])),
                           ('final_regr', np.mean(metrics[:i+1, 3]))])


    loss_rpn_cls = np.mean(metrics[:, 0])
    loss_rpn_regr = np.mean(metrics[:, 1])
    loss_class_cls = np.mean(metrics[:, 2])
    loss_class_regr = np.mean(metrics[:, 3])
    class_acc = np.mean(metrics[:, 4])
    mean_overlapping_bboxes = np.mean(metrics[:, 5])

    curr_loss = loss_rpn_cls + loss_rpn_regr + loss_class_cls + loss_class_regr
    elapsed_time = int(time.time() - start_time)

    print('\nMean #bboxes from RPN overlapping with GT-Boxes: {}'.format(mean_overlapping_bboxes))
    print('Classifier accuracy for RPN boxes: {}'.format(class_acc))
    print('Loss RPN classifier: {}'.format(loss_rpn_cls))
    print('Loss RPN regression: {}'.format(loss_rpn_regr))
    print('Loss final classifier: {}'.format(loss_class_cls))
    print('Loss final regression: {}'.format(loss_class_regr))
    print('Total loss: {}'.format(curr_loss))
    print('Elapsed time: {}\n'.format(elapsed_time))

    if curr_loss < best_loss:
        print('Total loss decreased from {} to {}\n'.format(best_loss, curr_loss))
        best_loss = curr_loss

    new_row = {'mean_overlapping_bboxes': round(mean_overlapping_bboxes, 3),
               'class_acc': round(class_acc, 3),
               'loss_rpn_cls': round(loss_rpn_cls, 3),
               'loss_rpn_regr': round(loss_rpn_regr, 3),
               'loss_class_cls': round(loss_class_cls, 3),
               'loss_class_regr': round(loss_class_regr, 3),
               'curr_loss': round(curr_loss, 3),
               'elapsed_time': elapsed_time}

    record_df = record_df.append(new_row, ignore_index=True)
    record_df.to_csv(conf.history_path, index=0)
    model_all.save_weights(conf.weights_path)

print('Training complete, exiting.')
