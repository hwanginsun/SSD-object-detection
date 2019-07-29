"""
Copyright 2019, SangJae Kang, All rights reserved.
Mail : rocketgrowthsj@gmail.com
"""
import numpy as np


def non_maximum_suppression(boxes, confs, overlap_threshold=0.5):
    pick = []
    cx, cy, w, h = boxes.T
    x1, x2 = cx - w / 2, cx + w / 2
    y1, y2 = cy - h / 2, cy + h / 2

    indices = np.argsort(confs)[::-1]
    area = (x2 - x1) * (y2 - y1)

    while len(indices) > 1:
        idx, indices = indices[0], indices[1:]
        pick.append(idx)

        xx1 = np.maximum(x1[idx], x1[indices])
        yy1 = np.maximum(y1[idx], y1[indices])
        xx2 = np.minimum(x2[idx], x2[indices])
        yy2 = np.minimum(y2[idx], y2[indices])

        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)

        overlap = w * h / (area[indices] + 1e-8)
        indices = indices[overlap <= overlap_threshold]

    return pick