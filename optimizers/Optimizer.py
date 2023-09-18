# -*- coding: utf-8 -*-
# @Time    : 2023/5/9 12:21
# @Author  : Pan
# @Software: PyCharm
# @Project : VisualFramework
# @FileName: Optimizer

import paddle
from paddle import optimizer


def adam(config, model):
    beta1 = config["beta1"] if "beta1" in config.keys() else 0.9
    beta2 = config["beta2"] if "beta2" in config.keys() else 0.999
    epsilon = config["epsilon"] if "epsilon" in config.keys() else 1e-08
    optim = optimizer.Adam(config["lr_scheduler"],
                   beta1,
                   beta2,
                   epsilon,
                   model.parameters(),
                   weight_decay=config["decay"])
    return optim


def sgd(config, model):
    optim = optimizer.SGD(config["lr_scheduler"],
                   model.parameters(),
                   weight_decay=config["decay"])
    return optim
