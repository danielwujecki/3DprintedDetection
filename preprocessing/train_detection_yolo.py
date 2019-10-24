import os
import cv2
import numpy as np
from uuid import uuid4

BASE_PATH = '/'.join(os.getcwd().split('/'))
IMG_SAVE_PATH = os.path.join(BASE_PATH, 'data/train_detection_yolo')
TRAINING_IMG_PATH = os.path.join(BASE_PATH, 'rendering/images')
TRAIN_FILE = os.path.join('data/3DObjects_train.txt')

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
    file = open(TRAIN_FILE, 'a')

    width = size[0]
    height = size[1]

    while files:
        img_path = os.path.join(IMG_SAVE_PATH, uuid4().hex)
        file.write("{}\n".format(img_path + '.png'))

        img = np.zeros((height, width, 3), dtype='uint8')
        bg_color = COLOR_LIST[np.random.randint(0, len(COLOR_LIST))]
        img[:, :] = bg_color
        max_diffuse = np.random.randint(10, 41, dtype='uint8')
        img += np.random.randint(0, max_diffuse, size=img.shape, dtype='uint8')

        bboxes = []
        img_box_file = open(img_path + '.txt', 'w')

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

            line = '{} {} {} {} {}\n'
            line = line.format(class_ident,
                               x / size[0] + w / (2 * size[0]),
                               y / size[1] + h / (2 * size[1]),
                               w / size[0], h / size[1])
            img_box_file.write(line)

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

        cv2.imwrite(img_path + '.png', img)
        img_box_file.close()
    file.close()


if __name__ == "__main__":
    if not os.path.isdir(IMG_SAVE_PATH):
        os.mkdir(IMG_SAVE_PATH)

    class_list, file_list = class_summary()

    create_imgs(file_list, debug=False, maxiou=0.1)

    print("Generate 2nd image set")
    _, file_list = class_summary()
    create_imgs(file_list, debug=False, maxiou=0.01, scale_low=0.4)
