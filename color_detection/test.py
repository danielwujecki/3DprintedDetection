import os
import cv2
import time
from CNN import CNN
from Detector import Detector

SAVE_PATH = '../data/results'
PATH = '../data/detection_black'
CLASS_LIST = '../data/class_list.txt'

assert os.path.isfile(CLASS_LIST)

classes = {}
with open(CLASS_LIST, 'r') as clsfile:
    for clsi, clsline in enumerate(clsfile):
        classes[clsi] = clsline.strip()

if not os.path.isdir(SAVE_PATH):
    os.mkdir(SAVE_PATH)

d = Detector(CNN())

detection_times = []

def image_folder(pfad):
    files = os.listdir(pfad)
    files = list(filter(lambda x: '.jpg' in x or '.png' in x, files))

    for i, fname in enumerate(files):
        print(i)
        img = cv2.imread(os.path.join(pfad, fname))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        start = time.time()
        boxes = d.predict(img)
        detection_times.append(time.time() - start)

        with open(os.path.join(SAVE_PATH, fname[:-4] + '.txt'), 'w') as file:
            if boxes is None:
                continue
            for bbox in boxes:
                line = '{} {} {} {} {} {}\n'
                line = line.format(
                    classes[bbox[0]], bbox[1], bbox[2], bbox[3], bbox[4] + bbox[2], bbox[5] + bbox[3])
                file.write(line)

    print("Durchschnittliche Zeit: ", sum(detection_times) / len(detection_times))

# folders = os.listdir(PATH)
# for fld in folders:
#     print(fld)
#     pfado = os.path.join(PATH, fld)
#     image_folder(pfado)

image_folder(PATH)
