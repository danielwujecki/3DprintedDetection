import numpy as np


def nm_suppress(boxes, probs, overlap_thresh=0.9, max_boxes=300):
    # code used from here: http://www.pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/

    # Process explanation:
    #   Step 1: Sort the probs list
    #   Step 2: pick the last bounding box (with biggest probability) and mark as picked
    #   Step 3: Calculate the IoU with 'last' box and other boxes in the list. If the IoU is larger than overlap_threshold, delete the box from list
    #   Step 4: Repeat step 2 and step 3 until there is no item in the probs list
    if not isinstance(boxes, np.ndarray) or boxes.shape[0] == 0:
        return []

    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    np.testing.assert_array_less(x1, x2)
    np.testing.assert_array_less(y1, y2)

    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    # initialize the list of picked indexes
    pick = []

    # calculate the areas
    area = (x2 - x1) * (y2 - y1)

    # sort the bounding boxes
    idx = np.argsort(probs)

    # keep looping while some indexes still remain in the indexes list
    while idx.shape[0]:
        # grab the last index and add to the picked indices
        i = idx[-1]
        idx = idx[:-1]
        pick.append(i)

        # find the intersection
        x1_int = np.maximum(x1[i], x1[idx])
        y1_int = np.maximum(y1[i], y1[idx])
        x2_int = np.minimum(x2[i], x2[idx])
        y2_int = np.minimum(y2[i], y2[idx])

        ww_int = np.maximum(0, x2_int - x1_int)
        hh_int = np.maximum(0, y2_int - y1_int)

        area_int = ww_int * hh_int

        # find the union
        area_union = area[i] + area[idx] - area_int

        # compute the ratio of overlap
        overlap = area_int / (area_union + 1e-6)

        # delete all indexes from the index list that have
        idx = np.delete(idx, np.where(overlap > overlap_thresh)[0])

        if len(pick) >= max_boxes:
            break

    return np.array(pick, dtype='int')
