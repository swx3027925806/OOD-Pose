# -*- coding: utf-8 -*-
# @Time    : 2023/5/8 14:22
# @Author  : Pan
# @Software: PyCharm
# @Project : VisualFramework
# @FileName: __init__.py

import paddle
from metirc.Accuracy import ACC
from metirc.PSNR import PSNR
from metirc.SSIM import SSIM, MSSSIM


class MetricsCompose:
    def __init__(self, configs):
        self.metrics_combine = [eval(cfg["type"])(cfg) for cfg in configs]

    def calculate(self, predict, label, **kwargs):
        for item in range(len(self.metrics_combine)):
            correct = self.metrics_combine[item].compute(predict, label, **kwargs)
            self.metrics_combine[item].update(correct)

    def get_metrics_info(self):
        infos = []
        for item in range(len(self.metrics_combine)):
            infos += self.metrics_combine[item].get_info()
        return infos

    def reset_metrics(self):
        for item in range(len(self.metrics_combine)):
            self.metrics_combine[item].reset()


if __name__ == "__main__":
    configs = [
        {
            "type": "SSIM",
        },
        {
            "type": "MSSSIM",
        }
    ]
    metric = MetricsCompose(configs)
    for i in range(10):
        pred = paddle.rand((16, 3, 224, 224), dtype=paddle.float32) * 2 - 1
        targ = paddle.rand((16, 3, 224, 224), dtype=paddle.float32) * 2 - 1
        metric.calculate(pred, targ)
    infos = metric.get_metrics_info()
    metric.reset_metrics()
    print(infos)
