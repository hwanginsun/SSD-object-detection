"""
Copyright 2019, SangJae Kang, All rights reserved.
Mail : rocketgrowthsj@gmail.com
"""
import numpy as np
import pandas as pd
import cv2


def draw_rectangle(image, digits, color=(255,0,0), thickness=1):
    """ 주어진 좌표값 Dataframe에 따라, image에 사각형을 그리는 메소드
    """
    if isinstance(digits, np.ndarray):
        if digits.shape[1] == 4:
            digits = pd.DataFrame(digits, columns=['cx','cy','w','h'])
        elif digits.shape[2] == 5:
            digits = pd.DataFrame(digits, columns=['cx', 'cy', 'w', 'h','label'])

    elif isinstance(digits, pd.DataFrame):
        pass
    else:
        raise TypeError("digits은 numpy.ndarray 혹은 pandas.Dataframe으로 이루어져 있어야 합니다.")

    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)
    else:
        image = image.copy()
    for idx, row in digits.iterrows():
        xmin = row.cx - row.w / 2
        xmax = row.cx + row.w / 2
        ymin = row.cy - row.h / 2
        ymax = row.cy + row.h / 2

        start = tuple(np.array((xmin, ymin), dtype=np.int32))
        end = tuple(np.array((xmax, ymax), dtype=np.int32))
        image = cv2.rectangle(image, start, end, color, thickness)
        if "label" in row:
            cv2.putText(image, str(int(row.label)), start,
                        cv2.FONT_HERSHEY_DUPLEX, 0.3, color)
    return image
