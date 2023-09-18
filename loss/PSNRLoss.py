# -*- coding: utf-8 -*-
# @Time    : 2023/7/6 15:53
# @Author  : Pan
# @Software: PyCharm
# @Project : VisualFramework
# @FileName: PSNRLoss

import paddle
from paddle import nn


class PSNRLoss(nn.Layer):
    def __init__(self, config, coef, idx):
        """
        2. 构造函数根据自己的实际算法需求和使用需求进行参数定义即可
        """
        super(PSNRLoss, self).__init__()
        self.coef = coef
        self.idx = idx
        self.loss_statis = []

    def forward(self, image1, image2, **kwargs):
        mse = ((image1 - image2) ** 2).mean()
        value = ((100 - 20 * paddle.log10(1 / paddle.sqrt(mse))) / 100) ** 2 * self.coef
        self.loss_statis.append(value)
        return value

    def get_loss_info(self):
        info = {
            "name": "PSNRLoss" + str(self.idx),
            "value": paddle.to_tensor(self.loss_statis).mean().numpy()[0]
        }
        self.loss_statis.clear()
        return [info]
