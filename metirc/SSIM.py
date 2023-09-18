# -*- coding: utf-8 -*-
# @Time    : 2023/7/4 12:37
# @Author  : Pan
# @Software: PyCharm
# @Project : VisualFramework
# @FileName: MSSSIM

import paddle
from paddle.metric import Metric
import paddle.nn.functional as F


def gaussian1d(window_size, sigma):
    x = paddle.arange(window_size, dtype='float32')
    x = x - window_size//2
    gauss = paddle.exp(-x ** 2 / float(2 * sigma ** 2))
    return gauss / gauss.sum()


def create_window(window_size, sigma, channel):
    _1D_window = gaussian1d(window_size, sigma).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).unsqueeze(0).unsqueeze(0)
    return _2D_window.expand([channel,1,window_size,window_size])


def _ssim(predict, target, window, padding, channel, C, avg=True):
    mu1 = F.conv2d(predict, window, padding=padding, groups=channel)
    mu2 = F.conv2d(target, window, padding=padding, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    sigma1_sq = F.conv2d(predict * predict, window, padding=padding, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(target * target, window, padding=padding, groups=channel) - mu2_sq
    sigma12 = F.conv2d(predict * target, window, padding=padding, groups=channel) - mu1_mu2
    C1 = (C[0] * 255.) ** 2
    C2 = (C[1] * 255.) ** 2
    sc = (2 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)
    lsc = ((2 * mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1)) * sc
    if avg:
        return lsc.mean()
    else:
        return lsc.flatten(2).mean(-1), sc.flatten(2).mean(-1)


class SSIM(Metric):
    def __init__(self, config):
        super(SSIM, self).__init__()
        self.window_size = config["window_size"] if "window_size" in config.keys() else 11
        self.sigma = config["sigma"] if "sigma" in config.keys() else 1.5
        self.channel = config["channel"] if "channel" in config.keys() else 3
        self.C = config["C"] if "C" in config.keys() else [0.01, 0.03]
        self.window = create_window(self.window_size, self.sigma, self.channel)
        self.ssim = []

    @paddle.no_grad()
    def compute(self, predict, target):
        padding = self.window_size // 2

        predict = paddle.clip((predict * 255 + 127.5), 0, 255)
        target = paddle.clip((target * 255 + 127.5), 0, 255)

        return _ssim(predict, target, self.window, padding, self.channel, self.C)

    def update(self, ssim):
        self.ssim.append(ssim)

    def accumulate(self):
        return paddle.to_tensor(self.ssim).mean()

    def reset(self):
        self.ssim.clear()

    def get_info(self):
        return [{"name": "SSIM", "value": self.accumulate()}]

    def name(self):
        return "SIIM"


class MSSSIM(Metric):
    def __init__(self, config):
        super(MSSSIM, self).__init__()
        self.window_size = config["window_size"] if "window_size" in config.keys() else 11
        self.sigma = config["sigma"] if "sigma" in config.keys() else 1.5
        self.channel = config["channel"] if "channel" in config.keys() else 3
        self.C = config["C"] if "C" in config.keys() else [0.01, 0.03]
        self.weight = config["weights"] if "weights" in config.keys() else [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]
        self.window = create_window(self.window_size, self.sigma, self.channel)
        self.ms_ssim = []

    @paddle.no_grad()
    def compute(self, predict, target):
        padding = self.window_size // 2

        predict = paddle.clip((predict * 255 + 127.5), 0, 255)
        target = paddle.clip((target * 255 + 127.5), 0, 255)

        if predict.shape != target.shape:
            raise ValueError("Input images should have the same dimensions.")

        if predict.dtype != target.dtype:
            raise ValueError("Input images should have the same dtype.")

        if len(predict.shape) == 4:
            avg_pool = F.avg_pool2d
        elif len(predict.shape) == 5:
            avg_pool = F.avg_pool3d
        else:
            raise ValueError(f"Input images should be 4-d or 5-d tensors, but got {predict.shape}")

        smaller_side = min(predict.shape[-2:])

        assert smaller_side > (self.window_size - 1) * (2 ** 4), "Image size should be larger than %d due to the 4 downsamplings " \
                                                            "with window_size %d in ms-ssim" % ((self.window_size - 1) * (2 ** 4), self.window_size)

        self.weight = paddle.to_tensor(self.weight)

        assert self.window.shape == [self.channel, 1, self.window_size, self.window_size], " window.shape error"

        levels = self.weight.shape[0]  # 5
        mcs = []
        for i in range(levels):
            ssim_per_channel, cs = _ssim(predict, target, self.window, padding, self.channel, self.C, avg=False)
            if i < levels - 1:
                mcs.append(F.relu(cs))
                padding = [s % 2 for s in predict.shape[2:]]
                predict = avg_pool(predict, kernel_size=2, padding=padding)
                target = avg_pool(target, kernel_size=2, padding=padding)

        ssim_per_channel = F.relu(ssim_per_channel)  # (batch, channel)
        mcs_and_ssim = paddle.stack(mcs + [ssim_per_channel], axis=0)  # (level, batch, channel) 按照等级堆叠
        ms_ssim_val = paddle.prod(mcs_and_ssim ** self.weight.reshape([-1, 1, 1]), axis=0)  # level 相乘
        return ms_ssim_val.mean()

    def update(self, ssim):
        self.ms_ssim.append(ssim)

    def accumulate(self):
        return paddle.to_tensor(self.ms_ssim).mean()

    def reset(self):
        self.ms_ssim.clear()

    def get_info(self):
        return [{"name": "MS-SSIM", "value": self.accumulate()}]

    def name(self):
        return "MS-SIIM"

