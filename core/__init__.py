# -*- coding: utf-8 -*-
# @Time    : 2023/5/8 14:08
# @Author  : Pan
# @Software: PyCharm
# @Project : VisualFramework
# @FileName: __init__.py

from core.Pose import PoseEngine


def choose_core(config):
    return eval(config["type"]+"Engine")(config)
