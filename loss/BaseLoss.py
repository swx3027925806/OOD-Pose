# -*- coding: utf-8 -*-
# @Time    : 2023/5/10 10:56
# @Author  : Pan
# @Software: PyCharm
# @Project : VisualFramework
# @FileName: BaseLoss

import paddle
from paddle import nn


class L1Loss(nn.Layer):
    def __init__(self, config, coef, idx):
        super(L1Loss, self).__init__()
        self.weights = paddle.to_tensor(config['weights']) if 'weights' in config.keys() else None
        self.coef = coef
        self.idx = idx
        reduction = config['reduction'] if 'reduction' in config.keys() else "mean"
        self.loss_statis = []
        self.loss = nn.L1Loss(reduction)

    def forward(self, predict, label, **kwargs):
        if self.weights:
            predict = self.weights * predict
            label = self.weights * label
        value = self.loss(predict, label) * self.coef
        self.loss_statis.append(value.tolist())
        return value

    def get_loss_info(self):
        info = {
            "name": "L1Loss" + str(self.idx),
            "value": paddle.to_tensor(self.loss_statis).mean().numpy()[0]
        }
        self.loss_statis.clear()
        return [info]


class L2Loss(nn.Layer):
    def __init__(self, config, coef, idx):
        super(L2Loss, self).__init__()
        self.weights = paddle.to_tensor(config['weights']) if 'weights' in config.keys() else None
        self.coef = coef
        self.idx = idx
        reduction = config['reduction'] if 'reduction' in config.keys() else "mean"
        self.loss_statis = []
        self.loss = nn.MSELoss(reduction)

    def forward(self, predict, label, **kwargs):
        if self.weights:
            predict = self.weights * predict
            label = self.weights * label
        value = self.loss(predict, label) * self.coef
        self.loss_statis.append(value.tolist())
        return value

    def get_loss_info(self):
        info = {
            "name": "L2Loss" + str(self.idx),
            "value": paddle.to_tensor(self.loss_statis).mean().numpy()[0]
        }
        self.loss_statis.clear()
        return [info]


class CrossEntropyLoss(nn.Layer):
    def __init__(self, config, coef, idx):
        super(CrossEntropyLoss, self).__init__()
        self.weights = paddle.to_tensor(config['weights']) if 'weights' in config.keys() else None
        self.coef = coef
        self.idx = idx
        self.reduction = config['reduction'] if 'reduction' in config.keys() else "mean"
        self.loss_statis = []

    def forward(self, predict, label, **kwargs):
        if self.weights:
            predict = self.weights * predict
            label = self.weights * label
        value = nn.functional.cross_entropy(predict, label) * self.coef
        self.loss_statis.append(value.tolist())
        return value

    def get_loss_info(self):
        info = {
            "name": "CrossEntropyLoss" + str(self.idx),
            "value": paddle.to_tensor(self.loss_statis).mean().numpy()[0]
        }
        self.loss_statis.clear()
        return [info]
