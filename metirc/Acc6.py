# -*- coding: utf-8 -*-
# @Time    : 2023/5/30 10:24
# @Author  : Pan
# @Software: PyCharm
# @Project : VisualFramework
# @FileName: Accuracy

import paddle
from paddle.metric import Metric


class AccPi(Metric):
    def __init__(self, config):
        super(AccPi, self).__init__()
        self.acc_pi = {}

    def compute(self, predict, target):
        predict = target[:, -3:]
        target = target[:, -3:]
        psnr = 20 * paddle.log10(255 / paddle.sqrt(((predict - target) ** 2).mean()))
        return psnr

    def update(self, psnr):
        self.psnr.append(psnr)

    def accumulate(self):
        return paddle.to_tensor(self.psnr).mean()

    def reset(self):
        self.psnr.clear()

    def get_info(self):
        return [{"name": "PSNR", "value": self.accumulate()}]

    def name(self):
        return "PSNR"

