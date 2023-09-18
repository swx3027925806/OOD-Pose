# -*- coding: utf-8 -*-
# @Time    : 2023/7/7 19:15
# @Author  : Pan
# @Software: PyCharm
# @Project : VisualFramework
# @FileName: to_static

import os
import paddle
import networks
from core import base
from paddle import jit


def to_static(config):
    model = jit.to_static(networks.make_model(config["network"]))
    static_model, _, _, _ = base.load_model(model, pretrained=config["base_info"]["pretrained"])
    save_path = os.path.join(config["base_info"]["save_path"], "static_model")
    x = paddle.rand((1, 3, 512, 512))
    paddle.jit.save(static_model, save_path, [x])

