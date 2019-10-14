import numpy as np


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


def calc_iou_fast(A, B):
    x1, y1, w1, h1 = A[0], A[1], A[2], A[3]
    x2, y2, w2, h2 = B[0], B[1], B[2], B[3]

    a1, a2 = w1 * h1, w2 * h2

    x1_int = np.maximum(x1, x2)
    y1_int = np.maximum(y1, y2)
    x2_int = np.minimum(x1 + w1, x2 + w2)
    y2_int = np.minimum(y1 + h1, y2 + h2)

    w_int = np.maximum(0, x2_int - x1_int)
    h_int = np.maximum(0, y2_int - y1_int)
    a_int = w_int * h_int

    a_union = a1 + a2 - a_int

    return a_int / a_union


def apply_regr(X, T):
    """Apply regression layer to all anchors in one feature map
    Args:
        X: shape=(4, 18, 25) the current anchor type for all points in the feature map
        T: regression layer shape=(4, 18, 25)
    Returns:
        X: regressed position and size for current anchor
    """
    xa = X[0, :, :]
    ya = X[1, :, :]
    wa = X[2, :, :]
    ha = X[3, :, :]

    tx = T[0, :, :]
    ty = T[1, :, :]
    tw = T[2, :, :]
    th = T[3, :, :]

    cx = tx * wa + xa
    cy = ty * ha + ya

    w = np.exp(tw.astype(np.float64)) * wa
    h = np.exp(th.astype(np.float64)) * ha
    x = cx - w / 2.
    y = cy - h / 2.

    x = np.round(x)
    y = np.round(y)
    w = np.round(w)
    h = np.round(h)
    return np.stack([x, y, w, h])
