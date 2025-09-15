"""
@author: zjf
@create time: 2022/7/8 20:22
@desc: 用来调整范围
"""

import numpy as np


def adjust(img, min_c, max_c):
    min_img = np.min(img)
    max_img = np.max(img)

    img = (img - min_img) / (max_img - min_img)
    img = min_c + img * (max_c - min_c)
    return img
