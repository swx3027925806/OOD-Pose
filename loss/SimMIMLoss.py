# -*- coding: utf-8 -*-
# @Time    : 2023/5/20 15:36
# @Author  : Pan
# @Software: PyCharm
# @Project : VisualFramework
# @FileName: SimMIMLoss
import paddle
from paddle import nn
import paddle.nn.functional as F


class SimMIMLoss(nn.Layer):
    def __init__(self, parameters):
        super(SimMIMLoss, self).__init__()
        self.patch_size = parameters["patch_size"] if "patch_size" in parameters.keys() else 4
        self.norm_target = parameters["norm_target"] if "norm_target" in parameters.keys() else False
        self.in_channels = parameters["in_channels"] if "in_channels" in parameters.keys() else 3
        self.loss_statis = []

    def forward(self, predict, label, **kwargs):
        mask = kwargs["mask"]
        mask = mask.repeat_interleave(self.patch_size, 1).repeat_interleave(self.patch_size, 2).unsqueeze(1)

        # norm target as prompted
        if self.norm_target:
            label = self.norm_targets(label, self.norm_target)

        loss_recon = F.l1_loss(label, predict, reduction='none')
        loss = (loss_recon * mask).sum() / (mask.sum() + 1e-6) / self.in_channels
        self.loss_statis.append(loss.tolist())
        return loss

    def norm_targets(self, targets, patch_size):
        assert patch_size % 2 == 1

        targets_ = targets
        targets_count = paddle.ones_like(targets)

        targets_square = targets ** 2.

        targets_mean = F.avg_pool2d(targets, kernel_size=patch_size, stride=1, padding=patch_size // 2)
        targets_square_mean = F.avg_pool2d(targets_square, kernel_size=patch_size, stride=1, padding=patch_size // 2)
        targets_count = F.avg_pool2d(targets_count, kernel_size=patch_size, stride=1, padding=patch_size // 2) * (patch_size ** 2)

        targets_var = (targets_square_mean - targets_mean ** 2.) * (targets_count / (targets_count - 1))
        targets_var = paddle.clip(targets_var, min=0.)

        targets_ = (targets_ - targets_mean) / (targets_var + 1.e-6) ** 0.5

        return targets_

    def get_loss_info(self):
        info = {
            "name": "SimMIMLoss",
            "value": paddle.to_tensor(self.loss_statis).mean().numpy()[0]
        }
        self.loss_statis.clear()
        return [info]
