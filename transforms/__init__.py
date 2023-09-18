# -*- coding: utf-8 -*-
# @Time    : 2023/5/8 14:23
# @Author  : Pan
# @Software: PyCharm
# @Project : VisualFramework
# @FileName: __init__.py

import cv2
import paddle
from paddle.vision import transforms
from transforms.basetransforms import *


class Compose:
    def __init__(self, config):
        self.ops_list = []
        for item in config:
            self.ops_list.append(eval(item["type"])(item))

    def __call__(self, data):
        for ops in self.ops_list:
            data = ops(data)
        return data


if __name__ == "__main__":
    config = [
        {
            "type": "LoadData",
            "keys": ["img"],
            "func": "cv2"
        },
        {
            "type": "ResizeByShort",
            "keys": ["img"],
            "short": [128, 256, 512],
            "inter": ["NEAREST", "LINEAR", "AREA", "CUBIC", "LANCZOS4"]
        },
        {
            "type": "RandPaddingCrop",
            "keys": ["img"],
            "pad_size": (128, 128),
            "crop_size": (128, 128)
        }
    ]
    trans = Compose(config)
    data = {"img": "test_image\\generate_cat.png"}
    image = trans(data)["img"]
    cv2.imshow("test", image)
    cv2.waitKey()
