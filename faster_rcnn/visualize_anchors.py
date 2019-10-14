import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from frcnn_config import Config

# base_path = '../detection_images/detect_imgs'
# train_images = os.listdir(base_path)
# image_path = os.path.join(base_path, np.random.choice(train_images))
image_path = '/home/daniel/Dropbox/bachelorarbeit_box/bilder/detection_test2.jpg'

conf = Config()

img = cv2.imread(image_path)
if img is None:
    print("Can't read image.")

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
height, width = img.shape[:2]

if width <= height:
    f = conf.im_size / width
    resized_height = int(f * height)
    resized_width = conf.im_size
else:
    f = conf.im_size / height
    resized_width = int(f * width)
    resized_height = conf.im_size

img = cv2.resize(img, (resized_width, resized_height),
                 interpolation=cv2.INTER_CUBIC)

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

cv2.imwrite('anchors.png', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
plt.imshow(img)
plt.show()
