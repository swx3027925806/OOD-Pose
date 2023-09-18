# -*- coding: utf-8 -*-
# @Time    : 2023/5/9 12:21
# @Author  : Pan
# @Software: PyCharm
# @Project : VisualFramework
# @FileName: LRScheduler

import math
import paddle
from paddle.optimizer import lr
from paddle.optimizer.lr import LRScheduler


def cosine(config):
    learning_rate = config["learning_rate"]
    max_steps = config["max_steps"]
    eta_min = config["eta_min"] if "eta_min" in config.keys() else 0
    last_epoch = config["last_epoch"] if "last_epoch" in config.keys() else -1
    return lr.CosineAnnealingDecay(learning_rate, max_steps, eta_min, last_epoch)


class WarmupCosineLR(LRScheduler):
    def __init__(self, parameters):
        self.warmup_epochs = parameters["warmup_steps"]
        self.total_epochs = parameters["total_steps"]
        self.warmup_start_lr = parameters["warmup_start_lr"]
        self.end_lr = parameters["end_lr"]

        super().__init__(parameters["learning_rate"])

    def get_lr(self):
        # linear warmup
        if self.last_epoch < self.warmup_epochs:
            lr = (self.base_lr - self.warmup_start_lr) * float(self.last_epoch) / float(self.warmup_epochs) + self.warmup_start_lr
            return lr

        # cosine annealing decay
        progress = float(self.last_epoch - self.warmup_epochs) / float(max(1, self.total_epochs - self.warmup_epochs))
        cosine_lr = max(0.0, 0.5 * (1. + math.cos(math.pi * progress)))
        lr = max(0.0, cosine_lr * (self.base_lr - self.end_lr) + self.end_lr)
        return lr
