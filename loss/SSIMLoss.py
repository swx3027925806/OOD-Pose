# -*- coding: utf-8 -*-
# @Time    : 2023/7/6 15:40
# @Author  : Pan
# @Software: PyCharm
# @Project : VisualFramework
# @FileName: PSNRLoss

import paddle
import paddle.nn.functional as F


def gaussian1d(window_size, sigma):
    x = paddle.arange(window_size,dtype='float32')
    x = x - window_size//2
    gauss = paddle.exp(-x ** 2 / float(2 * sigma ** 2))
    return gauss / gauss.sum()

def create_window(window_size, sigma, channel):
    _1D_window = gaussian1d(window_size, sigma).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).unsqueeze(0).unsqueeze(0)
    return _2D_window.expand([channel,1,window_size,window_size])

def _ssim(img1, img2, window, window_size, channel=3 ,data_range = 255.,size_average=True,C=None):

    padding = window_size // 2

    mu1 = F.conv2d(img1, window, padding=padding, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padding, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    sigma1_sq = F.conv2d(img1 * img1, window, padding=padding, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padding, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padding, groups=channel) - mu1_mu2
    if C ==None:
        C1 = (0.01*data_range) ** 2
        C2 = (0.03*data_range) ** 2
    else:
        C1 = (C[0]*data_range) ** 2
        C2 = (C[1]*data_range) ** 2
    sc = (2 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)
    lsc = ((2 * mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1))*sc

    if size_average:
        return lsc.mean()
    else:
        return lsc.flatten(2).mean(-1),sc.flatten(2).mean(-1)


def ms_ssim(
    img1, img2,window, data_range=255, size_average=True, window_size=11, channel=3, sigma=1.5, weights=None, C=(0.01, 0.03)
):

    if img1.shape != img2.shape:
        raise ValueError("Input images should have the same dimensions.")

    if img1.dtype != img2.dtype:
        raise ValueError("Input images should have the same dtype.")

    if len(img1.shape) == 4:
        avg_pool = F.avg_pool2d
    elif len(img1.shape) == 5:
        avg_pool = F.avg_pool3d
    else:
        raise ValueError(f"Input images should be 4-d or 5-d tensors, but got {img1.shape}")

    smaller_side = min(img1.shape[-2:])

    assert smaller_side > (window_size - 1) * (2 ** 4), "Image size should be larger than %d due to the 4 downsamplings " \
                                                        "with window_size %d in ms-ssim" % ((window_size - 1) * (2 ** 4),window_size)

    if weights is None:
        weights = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]
    weights = paddle.to_tensor(weights)

    if window is None:
        window = create_window(window_size, sigma, channel)
    assert window.shape == [channel, 1, window_size, window_size], " window.shape error"

    levels = weights.shape[0] # 5
    mcs = []
    for i in range(levels):
        ssim_per_channel, cs =  _ssim(img1, img2, window=window, window_size=window_size,
                                       channel=3, data_range=data_range,C=C, size_average=False)
        if i < levels - 1:
            mcs.append(F.relu(cs))
            padding = [s % 2 for s in img1.shape[2:]]
            img1 = avg_pool(img1, kernel_size=2, padding=padding)
            img2 = avg_pool(img2, kernel_size=2, padding=padding)

    ssim_per_channel = F.relu(ssim_per_channel)  # (batch, channel)
    mcs_and_ssim = paddle.stack(mcs + [ssim_per_channel], axis=0)  # (level, batch, channel) 按照等级堆叠
    ms_ssim_val = paddle.prod(mcs_and_ssim ** weights.reshape([-1, 1, 1]), axis=0) # level 相乘
    if size_average:
        return ms_ssim_val.mean()
    else:
        # 返回各个channel的值
        return ms_ssim_val.flatten(2).mean(1)


class SSIMLoss(paddle.nn.Layer):
    def __init__(self, config, coef):
        """
        2. 构造函数根据自己的实际算法需求和使用需求进行参数定义即可
        """
        super(SSIMLoss, self).__init__()
        self.coef = coef
        self.data_range = config["data_range"] if "data_range" in config.keys() else 255.
        self.C = [0.01, 0.03]
        self.window_size = config["window_size"] if "window_size" in config.keys() else 11
        self.channel = config["channel"] if "channel" in config.keys() else 3
        self.sigma = config["sigma"] if "sigma" in config.keys() else 1.5
        self.window = create_window(self.window_size, self.sigma, self.channel)
        self.loss_statis = []

    def forward(self, input, label, **kwargs):
        input = paddle.clip((input * 255 + 127.5), 0, 255)
        label = paddle.clip((label * 255 + 127.5), 0, 255)
        value = 1-_ssim(input, label,data_range = self.data_range,
                      window = self.window, window_size=self.window_size, channel=3,
                      size_average=True,C=self.C) * self.coef
        self.loss_statis.append(value)
        return value

    def get_loss_info(self):
        info = {
            "name": "SSIMLoss",
            "value": paddle.to_tensor(self.loss_statis).mean().numpy()[0]
        }
        self.loss_statis.clear()
        return [info]


class MSSSIMLoss(paddle.nn.Layer):
    def __init__(self, config, coef):
        super(MSSSIMLoss, self).__init__()
        self.coef = coef
        self.data_range = config["data_range"] if "data_range" in config.keys() else 255.
        self.C = [0.01, 0.03]
        self.window_size = config["window_size"] if "window_size" in config.keys() else 11
        self.channel = config["channel"] if "channel" in config.keys() else 3
        self.sigma = config["sigma"] if "sigma" in config.keys() else 1.5
        self.window = create_window(self.window_size, self.sigma, self.channel)
        self.loss_statis = []

    def forward(self, input, label, **kwargs):
        input = paddle.clip((input * 255 + 127.5), 0, 255)
        label = paddle.clip((label * 255 + 127.5), 0, 255)
        value = 1-ms_ssim(input, label, data_range=self.data_range,
                      window = self.window, window_size=self.window_size, channel=self.channel,
                      size_average=True,  sigma=self.sigma,
                      weights=None, C=self.C) * self.coef
        self.loss_statis.append(value)
        return value

    def get_loss_info(self):
        info = {
            "name": "MS-SSIMLoss",
            "value": paddle.to_tensor(self.loss_statis).mean().numpy()[0]
        }
        self.loss_statis.clear()
        return [info]