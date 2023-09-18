# -*- coding: utf-8 -*-
# @Time    : 2023/7/4 12:37
# @Author  : Pan
# @Software: PyCharm
# @Project : VisualFramework
# @FileName: PSNR

import paddle
from paddle.metric import Metric


class PSNR(Metric):
    def __init__(self, config):
        super(PSNR, self).__init__()
        self.psnr = []

    def compute(self, predict, target):
        predict = (paddle.clip(predict * 255, 0, 255) + 0.5) / 1
        target = (paddle.clip(target * 255, 0, 255) + 0.5) / 1
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

