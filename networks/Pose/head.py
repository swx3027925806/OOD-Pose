# -*- coding: utf-8 -*-
# @Time    : 2023/9/1 18:33
# @Author  : Pan
# @Software: PyCharm
# @Project : VisualFramework
# @FileName: head.py
import math
import paddle
import numpy as np
from paddle import nn
from paddle import ParamAttr
import paddle.nn.functional as F

def make_head(config):
    return eval(config["type"])(config)


class MutilHeadDecoder(nn.Layer):
    def __init__(self, parameters):
        super(MutilHeadDecoder, self).__init__()
        self.decoders = nn.LayerList([eval(config["type"])(config) for config in parameters["heads"]])

    def forward(self, x, **kwargs):
        outs = []
        for decoder in self.decoders:
            outs.append(decoder(x, **kwargs))
        return outs


class Classify(nn.Layer):
    def __init__(self, parameters):
        super(Classify, self).__init__()
        self.num_class = parameters["num_class"]
        self.features = parameters["features"]
        layers = parameters["layers"] if "layers" in parameters.keys() else 0
        stdv = parameters["stdv"] if "stdv" in parameters.keys() else 1
        stdv = 1.0 / math.sqrt(self.features) * stdv
        self.head = [nn.Linear(self.features, self.features, weight_attr=ParamAttr(initializer=nn.initializer.Uniform(-stdv, stdv))), nn.BatchNorm1D(self.features)] * layers
        self.head.append(nn.Linear(self.features, self.num_class, weight_attr=ParamAttr(initializer=nn.initializer.Uniform(-stdv, stdv))))
        self.head = nn.Sequential(*self.head)

    def forward(self, x, **kwargs):
        x = self.head(x)
        return x


class Regression(nn.Layer):
    def __init__(self, parameters):
        super(Regression, self).__init__()
        self.features = parameters["features"]
        layers = parameters["layers"] if "layers" in parameters.keys() else 0
        stdv = parameters["stdv"] if "stdv" in parameters.keys() else 1
        stdv = 1.0 / math.sqrt(self.features) * stdv
        self.head = [nn.Linear(self.features, self.features, weight_attr=ParamAttr(initializer=nn.initializer.Uniform(-stdv, stdv))), nn.BatchNorm1D(self.features)] * layers
        self.head.append(nn.Linear(self.features, 1, weight_attr=ParamAttr(initializer=nn.initializer.Uniform(-stdv, stdv))))
        self.head = nn.Sequential(*self.head)

    def forward(self, x, **kwargs):
        x = self.head(x)
        return x


if __name__ == "__main__":
    configs = [
        {
            "type": "Classify",
            "num_class": 10,
            "features": 1280,
        },
        {
            "type": "Classify",
            "num_class": 7,
            "features": 1280,
        },
        {
            "type": "Regression",
            "features": 1280,
        },
        {
            "type": "Regression",
            "features": 1280,
        }
    ]
    model = MutilHeadDecoder(configs)
