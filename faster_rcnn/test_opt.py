import os
import cv2
from detector import Detector

SAVE_PATH = '../data/results'
PATH = '../data/detection_synth'
CLASS_LIST = '../data/class_list.txt'

assert os.path.isfile(CLASS_LIST)

classes = {}
classes_inv = {}
with open(CLASS_LIST, 'r') as clsfile:
    for clsi, clsline in enumerate(clsfile):
        classes[clsi] = clsline.strip()
        classes_inv[clsline.strip()] = clsi

if not os.path.isdir(SAVE_PATH):
    os.mkdir(SAVE_PATH)

d = Detector()

def image_folder(pfad):
    files = os.listdir(pfad)
    files = list(filter(lambda x: '.jpg' in x or '.png' in x, files))

    for i, fname in enumerate(files):
        print(i)

        objects = []
        with open(os.path.join(pfad, fname[:-4] + '.txt'), 'r') as file:
            for line in file:
                obj_n = line.split(' ')[0].strip()
                objects.append(classes_inv[obj_n])

        img = cv2.imread(os.path.join(pfad, fname))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        rois, probs = d.predict(img, objects=objects, thresh=0.)

        with open(os.path.join(SAVE_PATH, fname[:-4] + '.txt'), 'w') as file:
            if rois is None:
                continue
            for j, r in enumerate(rois):
                line = '{} {} {} {} {} {}\n'
                line = line.format(classes[r[0]], probs[j], r[1], r[2], r[3], r[4])
                file.write(line)

# folders = os.listdir(PATH)
# for fld in folders:
#     print(fld)
#     pfado = os.path.join(PATH, fld)
#     image_folder(pfado)

image_folder(PATH)
