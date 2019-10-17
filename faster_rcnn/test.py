import os
import cv2
import time
from detector import Detector

SAVE_PATH = '../data/results'
PATH = '../data/detection_synth'
CLASS_LIST = '../data/class_list.txt'

assert os.path.isfile(CLASS_LIST)

classes = {}
with open(CLASS_LIST, 'r') as clsfile:
    for clsi, clsline in enumerate(clsfile):
        classes[clsi] = clsline.strip()

if not os.path.isdir(SAVE_PATH):
    os.mkdir(SAVE_PATH)

d = Detector()

detection_times = []

def image_folder(pfad):
    files = os.listdir(pfad)
    files = list(filter(lambda x: '.jpg' in x or '.png' in x, files))

    for i, fname in enumerate(files):
        print(i)
        img = cv2.imread(os.path.join(pfad, fname))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        start = time.time()
        rois, probs = d.predict(img, thresh=0.)
        detection_times.append(time.time() - start)

        with open(os.path.join(SAVE_PATH, fname[:-4] + '.txt'), 'w') as file:
            if rois is None:
                continue
            for j, r in enumerate(rois):
                line = '{} {} {} {} {} {}\n'
                line = line.format(classes[r[0]], probs[j], r[1], r[2], r[3], r[4])
                file.write(line)

    print("Durchschnittliche Zeit: ", sum(detection_times) / len(detection_times))

# folders = os.listdir(PATH)
# for fld in folders:
#     print(fld)
#     pfado = os.path.join(PATH, fld)
#     image_folder(pfado)

image_folder(PATH)
