import cv2
import time
import numpy as np
import matplotlib.pyplot as plt

from data import *
from config import Config


# read data and create config
train_path = '../data/train_detection/bboxes_detect_imgs.csv'
img_number = np.random.randint(0, 1000)
print("Img Nr.: {}".format(img_number))

conf = Config()
train_imgs, classes_count = get_data(train_path)
org_image_data = train_imgs[img_number]

# create data generator
data_gen_train = data_generator([train_imgs[img_number]], conf, debug=True)

start = time.perf_counter()
X, y_class, y_regr, image_data, debug_num_pos = next(data_gen_train)
end = time.perf_counter()
print('Fast datagen needed {}s'.format(end - start))

# infos about output of datagen
downscale = conf.base_network_downscale
msg = 'Original image:\t\theight=%d width=%d\n'
msg = msg % (org_image_data['height'], org_image_data['width'])
msg += 'Resized image:\t\theight=%d width=%d\n'
msg = msg % (image_data['height'], image_data['width'])
msg += 'downscale:\t\t%d\n' % downscale
msg += 'Feature map size:\theight = %d width = %d\n'
msg = msg % (y_class.shape[1], y_class.shape[2])
print(msg)

msg = 'Shape of y_cls:\t\t{}\nShape of y_regr:\t{}'
msg = msg.format(y_class.shape, y_regr.shape)
print(msg)

nb_anch = y_class.shape[-1] // 2
assert nb_anch == conf.num_anchors

pos_locs = np.where(np.logical_and(y_class[0, :, :, :nb_anch] == 1,
                                   y_class[0, :, :, nb_anch:] == 1))
neg_locs = np.where(np.logical_and(y_class[0, :, :, :nb_anch] == 1,
                                   y_class[0, :, :, nb_anch:] == 0))
num_pos = len(pos_locs[0])
num_neg = len(neg_locs[0])
assert num_pos == debug_num_pos

msg = 'Positive anchors:\t{}\nNegative anchors:\t{}'
msg = msg.format(num_pos, num_neg)
print(msg)

assert num_pos > 0

# visualize different anchors from the config
img = X[0].copy()
color = [255, 0, 255]
center = (img.shape[1] // 2, img.shape[0] // 2)
img = cv2.circle(img, center, 2, color, -1)

colors = [color, [255, 0, 0], [0, 0, 255], [0, 255, 255]]
for i, a_size in enumerate(conf.anchor_box_scales):
    for a_ratio in conf.anchor_box_ratios:
        anchor_w = a_size * a_ratio[0] // 2
        anchor_h = a_size * a_ratio[1] // 2
        x1, y1 = int(center[0] - anchor_w), int(center[1] - anchor_h)
        x2, y2 = int(center[0] + anchor_w), int(center[1] + anchor_h)
        img = cv2.rectangle(img, (x1, y1), (x2, y2), colors[i], 2)

fig = plt.figure(figsize=(24, 9))
ax1 = fig.add_subplot(1, 2, 1)
ax1.imshow(img)

# visualize some positive anchors with ground truth boxes
img = X[0].copy()
for idx in range(num_pos):
    size_idx = pos_locs[2][idx] // len(conf.anchor_box_ratios)
    ratio_idx = pos_locs[2][idx] % len(conf.anchor_box_ratios)
    a_size = conf.anchor_box_scales[size_idx]
    a_ratio = conf.anchor_box_ratios[ratio_idx]
    w = a_size * a_ratio[0]
    h = a_size * a_ratio[1]
    x1 = int(downscale * (pos_locs[1][idx] + .5) - w / 2)
    x2 = int(downscale * (pos_locs[1][idx] + .5) + w / 2)
    y1 = int(downscale * (pos_locs[0][idx] + .5) - h / 2)
    y2 = int(downscale * (pos_locs[0][idx] + .5) + h / 2)
    img = cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    center = (x1 + int(w / 2), y1 + int(h / 2))
    img = cv2.circle(img, center, 2, [227, 16, 16], -1)

for bbox in image_data['bboxes']:
    x, y, w, h = bbox['x'], bbox['y'], bbox['w'], bbox['h']
    img = cv2.rectangle(img, (x, y), (x+w, y+h), [0, 255, 0], 2)

ax2 = fig.add_subplot(1, 2, 2)
ax2.imshow(img)
# plt.figure(figsize=(24, 18))
# plt.imshow(img)
plt.show()
# plt.savefig('fast.png')
