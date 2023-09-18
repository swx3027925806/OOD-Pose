# -*- coding: utf-8 -*-
# @Time    : 2023/9/1 17:17
# @Author  : Pan
# @Software: PyCharm
# @Project : VisualFramework
# @FileName: __init__.py
import paddle
from .csp_resnet import csp_resnet
from .vit import BEiTv2_vit_base_patch16_224, BEiTv2_vit_large_patch16_224, CAE_vit_base_patch16_224, EVA_vit_giant_patch14

def pose(config):
    model = eval(config["type"])(config)
    if "pretrained" in config and config["pretrained"] is not None:
        model.set_state_dict(paddle.load(config["pretrained"]))
    return model
