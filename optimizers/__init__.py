# -*- coding: utf-8 -*-
# @Time    : 2023/5/8 14:09
# @Author  : Pan
# @Software: PyCharm
# @Project : VisualFramework
# @FileName: __init__.py

import paddle
from paddle import regularizer
from optimizers.LRScheduler import *
from optimizers.Optimizer import *


def make_optim(config, model):
    config["decay"] = make_decay(config["decay"])
    config["lr_scheduler"] = eval(config["lr_scheduler"]["type"])(config["lr_scheduler"])
    optim = eval(config["type"])(config, model)
    return model, optim, config["lr_scheduler"]


def make_decay(config):
    if config is not None:
        decay = config["type"] if "type" in config.keys() else "nums"
        coeff = config["coeff"] if "coeff" in config.keys() else 0
        config = eval(decay)(coeff)
    return config


def l1_decay(coeff):
    return regularizer.L1Decay(coeff)


def l2_decay(coeff):
    return regularizer.L2Decay(coeff)


def nums(coeff):
    return coeff


if __name__ == "__main__":
    config = {
        "type": "adam",
        "lr_scheduler": {
            "type": "cosine",
            "learning_rate": 0.01,
            "max_steps": 1000
        },
        "decay": {
            "type": "nums",
            "coeff": 0.001
        }
    }
    model = paddle.nn.Linear(100, 100)
    print(make_optim(config, model))

