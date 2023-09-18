# -*- coding: utf-8 -*-
# @Time    : 2023/5/16 16:40
# @Author  : Pan
# @Software: PyCharm
# @Project : VisualFramework
# @FileName: engine.py
import core
import config
from tools.to_static import to_static


def train(config):
    engine = core.choose_core(config)
    engine.train()


def eval(config):
    engine = core.choose_core(config)
    engine.eval()


def predict(config):
    engine = core.choose_core(config)
    engine.predict()


def to_statis(config):
    to_static(config)


if __name__ == "__main__":
    train(config.config)
