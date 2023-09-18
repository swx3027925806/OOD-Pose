# -*- coding: utf-8 -*-
# @Time    : 2023/5/8 14:09
# @Author  : Pan
# @Software: PyCharm
# @Project : VisualFramework
# @FileName: __init__.py

import paddle
from networks.Pose import pose


def make_model(config):
    model = eval(config["type"])(config["network"])
    if "print" in config.keys():
        input_shape = config["print"]["input_shape"]
        print_detail = config["print"]["print_detail"] if "print_detail" in config.keys() else True
        paddle.flops(model, input_shape, print_detail=print_detail)
    return model


if __name__ == "__main__":
    config = {
        "type": "mae",
        "network": {
            "type": "SwinTransformer_tiny_patch4_window7_224"
        }
    }
    model = make_model(config)
