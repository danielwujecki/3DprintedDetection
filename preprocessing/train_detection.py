import os
import cv2
import pickle
import numpy as np
from uuid import uuid4

BASE_PATH = '/home/daniel/Schreibtisch/bachelorarbeit/src'
IMG_SAVE_PATH = 'detect_imgs'
TRAINING_IMG_PATH = os.path.join(BASE_PATH, 'train_images/images')
BBOX_FILE = os.path.join(IMG_SAVE_PATH, 'bboxes_detect_imgs.csv')
CLASS_FILE = os.path.join(IMG_SAVE_PATH, 'class_mapping.pickle')

CAM_ROT = 16
CAM_ANGLES = 8
CAM_DIST = 1

COLOR_LIST = [[200, 10, 10], [10, 200, 10],
              [10, 10, 200], [150, 150, 150],
              [200, 200, 10], [10, 200, 200],
              [200, 10, 200], [0, 0, 0]]


def calc_iou(A, B):
    x, y, w, h = A
    xo, yo, wo, ho = B

    xA, yA = max(x, xo), max(y, yo)
    xB, yB = min(x+w, xo+wo), min(y+h, yo+ho)
    wI, hI = xB - xA, yB - yA

    if wI < 0 or hI < 0:
        inter = 0.
    else:
        inter = wI * hI

    union = w * h + wo * ho - inter
    try:
        res = inter / union
    except ZeroDivisionError:
        print(A)
        print(B)
        res = inter

    return res


# klassennamen und dateien einlesen
def class_summary():
    classes = {}
    with open(os.path.join(TRAINING_IMG_PATH, 'names.txt'), 'r') as file:
        for line in file:
            splitted = line.split(':')
            ident = int(splitted[0])
            name = splitted[1].split('.stl')[0]
            classes[ident] = name

    files = {}
    for ident in classes:
        files[ident] = os.listdir(os.path.join(TRAINING_IMG_PATH, '%02d' % ident))

    return classes, files


# überprüfen ob alle bilder vorhanden sind
def check_rendered_all(files):
    for ident in files:
        check = []
        pathlist = files[ident]
        for file in pathlist:
            splitted = file.split('_')
            cmp = int(splitted[1][3:])
            pose = int(splitted[2][4:])

            while len(check) <= cmp:
                check.append([])
            while len(check[cmp]) <= pose:
                check[cmp].append(
                    np.zeros((CAM_ROT, CAM_ANGLES, CAM_DIST), dtype='uint8'))

            rot = int(splitted[3][3:])
            dist = int(splitted[4][4:])
            cam = int(splitted[5][3:])

            check[cmp][pose][rot, cam, dist] = 1
        for i in check:
            for arr in i:
                if not np.all(arr == 1):
                    print('missing some files')
                    return False
    return True


def cut_img(img):
    assert img.shape == (512, 512, 4)
    mask = np.zeros((512,) * 2, dtype='uint8')
    mask[np.where(img[:, :, 3] != 0)] = 255

    contours, _ = cv2.findContours(
        mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find object with the biggest bounding box
    mx = (0, 0, 0, 0)                       # biggest bounding box so far
    mx_area = 0
    org_area = img.shape[0] * img.shape[1]  # area if original image

    for cont in contours:
        x, y, w, h = cv2.boundingRect(cont)
        area = w * h
        if mx_area < area < org_area:
            mx = x, y, w, h
            mx_area = area

    x, y, w, h = mx
    return img[y:y + h, x:x + w, :3], mask[y:y + h, x:x + w]


def create_imgs(files, maxiou=0.1, size=(1280, 720), scale_low=0.3, debug=False):
    file = open(BBOX_FILE, 'a')

    width = size[0]
    height = size[1]

    while files:
        img_path = os.path.join(IMG_SAVE_PATH, uuid4().hex + '.png')

        img = np.zeros((height, width, 3), dtype='uint8')
        bg_color = COLOR_LIST[np.random.randint(0, len(COLOR_LIST))]
        img[:, :] = bg_color
        max_diffuse = np.random.randint(10, 41, dtype='uint8')
        img += np.random.randint(0, max_diffuse, size=img.shape, dtype='uint8')

        bboxes = []

        for _ in range(np.random.randint(3, 9)):
            if not files:
                break

            class_ident = np.random.choice(list(files))
            obj_name = np.random.choice(files[class_ident])

            files[class_ident].remove(obj_name)
            if not files[class_ident]:
                files.pop(class_ident, None)

            obj_path = os.path.join(TRAINING_IMG_PATH, ('%02d/' + obj_name) % class_ident)
            obj = cv2.imread(obj_path, cv2.IMREAD_UNCHANGED)
            obj, mask = cut_img(obj)

            ready = False
            for _ in range(50):
                scale = np.random.uniform(low=scale_low, high=1.)
                w, h = int(obj.shape[1] * scale), int(obj.shape[0] * scale)
                x = np.random.randint(0, img.shape[1] - w)
                y = np.random.randint(0, img.shape[0] - h)

                iou_sum = 0.
                ready = True
                for xo, yo, wo, ho, _ in bboxes:
                    iou_sum += calc_iou((x, y, w, h), (xo, yo, wo, ho))
                    if iou_sum > maxiou:
                        ready = False
                        break

            if not ready:
                continue

            bboxes.append((x, y, w, h, class_ident))
            file.write(('{},'*7+'{}\n').format(img_path, width, height, x, y, w, h, class_ident))

            if debug:
                img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 2)

            obj = cv2.resize(obj, (w, h))
            mask = cv2.resize(mask, (w, h))

            obj = (obj * np.random.uniform(low=0.85, high=1.0)).astype('uint8')

            yellow_off = np.random.randint(0, 20)
            id0, id1 = np.where(obj[:, :, 0] > yellow_off)
            obj[id0, id1, 0] -= yellow_off

            bigobj = np.zeros(img.shape, dtype='uint8')
            bigobj[y:y+h, x:x+w] = obj

            bigmask = np.zeros((height, width), dtype='uint8')
            bigmask[y:y+h, x:x+w] = mask
            bigmask = np.dstack((bigmask, bigmask, bigmask))
            bigmask_inv = np.ones(bigmask.shape, dtype='uint8') * 255 - bigmask

            img = cv2.bitwise_and(img, bigmask_inv)
            img += cv2.bitwise_and(bigobj, bigmask)

        cv2.imwrite(img_path, img)
    file.close()


if __name__ == "__main__":
    if not os.path.isdir(IMG_SAVE_PATH):
        os.mkdir(IMG_SAVE_PATH)

    class_list, file_list = class_summary()
    pickle.dump(class_list, open(CLASS_FILE, "wb"))

    if check_rendered_all(file_list):
        create_imgs(file_list, debug=False, maxiou=0.1)

        print("Generate 2nd image set")
        _, file_list = class_summary()
        create_imgs(file_list, debug=False, maxiou=0.01, scale_low=0.4)
