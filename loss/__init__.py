# -*- coding: utf-8 -*-
# @Time    : 2023/5/9 10:03
# @Author  : Pan
# @Software: PyCharm
# @Project : VisualFramework
# @FileName: __init__.py

import paddle
from paddle import nn
from loss.SimMIMLoss import SimMIMLoss
from loss.BaseLoss import L1Loss, L2Loss, CrossEntropyLoss
from loss.PSNRLoss import PSNRLoss
from loss.SSIMLoss import SSIMLoss, MSSSIMLoss


class LossCompose:
    def __init__(self, configs):
        assert len(configs["loss_list"]) == len(configs["loss_coef"]), "loss_list 与 loss_coef 尺度不匹配"
        self.loss_coef = configs["loss_coef"]
        self.loss_combine = [eval(cfg["type"])(cfg, self.loss_coef[idx], idx) for idx, cfg in enumerate(configs["loss_list"])]

    def __call__(self, predict, label, **kwargs):
        loss_sum = 0
        if isinstance(predict, list):
            for item in range(len(self.loss_combine)):
                loss_value = self.loss_combine[item](predict[item], label[item], **kwargs)
                loss_sum = loss_value if item == 0 else loss_sum + loss_value
        else:
            loss_sum = self.loss_combine[0](predict, label, **kwargs)
        return loss_sum

    def get_loss_info(self):
        infos = []
        sum_info = {"name": "Loss", "value": 0}
        for item in range(len(self.loss_combine)):
            info = self.loss_combine[item].get_loss_info()
            for i in info:
                if i["name"].count("/") == 0:
                    sum_info["value"] += i["value"]
                i["name"] = "Loss/" + i["name"]
                infos.append(i)
        infos.append(sum_info)
        return infos


class MixLoss(nn.Layer):
    def __init__(self, configs, coef, idx):
        super(MixLoss, self).__init__()
        assert len(configs["loss_list"]) == len(configs["loss_coef"]), "loss_list 与 loss_coef 尺度不匹配"
        self.coef = coef
        self.idx = idx
        self.loss_coef = configs["loss_coef"]
        self.loss_combine = [eval(cfg["type"])(cfg, self.loss_coef[idx] * self.coef, idx) for idx, cfg in enumerate(configs["loss_list"])]

    def __call__(self, predict, label, **kwargs):
        loss_sum = 0
        for item in range(len(self.loss_combine)):
            loss_value = self.loss_combine[item](predict, label, **kwargs)
            loss_sum = loss_value if item == 0 else loss_sum + loss_value
        return loss_sum

    def get_loss_info(self):
        infos = []
        mix_info = {"name": "MixLoss"+str(self.idx), "value": 0}
        for item in range(len(self.loss_combine)):
            info = self.loss_combine[item].get_loss_info()
            for i in info:
                mix_info["value"] += i["value"]
                i["name"] = "MixLoss%s/" % (str(self.idx)) + i["name"]
                infos.append(i)
        infos.append(mix_info)
        return infos


if __name__ == "__main__":
    config = {
        "loss_list": [
            {
                "type": "L1Loss"
            },
            {
                "type": "MixLoss",
                "loss_list": [
                    {
                        "type": "L1Loss"
                    },
                    {
                        "type": "L2Loss"
                    }
                ],
                "loss_coef": [0.8, 0.5]
            }
        ],
        "loss_coef": [1, 0.5]
    }
    loss = LossCompose(config)
    for i in range(100):
        predict = paddle.randn((2, 100))
        label = paddle.randn((2, 100))
        loss(predict, label)
        if i % 10 == 9:
            print(loss.get_loss_info())
            print("*"*100)
