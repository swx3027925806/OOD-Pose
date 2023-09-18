# -*- coding: utf-8 -*-
# @Time    : 2023/5/10 16:33
# @Author  : Pan
# @Software: PyCharm
# @Project : VisualFramework
# @FileName: base

import os
import paddle
import shutil


def load_model(model, optim=None, lr=None, pretrained=None):
    step = 0
    if pretrained is not None:
        # model_dict = model.state_dict()
        # pre_model_dict = paddle.load(os.path.join(pretrained, "model.pdparams"))
        # keys1 = list(model_dict.keys())
        # keys2 = list(pre_model_dict.keys())
        # for i in range(len(min(keys1, keys2))):
        #     pre_model_dict[keys1[i]] = pre_model_dict[keys2[i]]
        #     # print("%50s %50s" % (keys1[i], keys2[i]))
        # model.set_dict(pre_model_dict)
        model.set_state_dict(paddle.load(os.path.join(pretrained, "model.pdparams"))) if model is not None and os.path.exists(os.path.join(pretrained, "model.pdparams")) else None
        optim.set_state_dict(paddle.load(os.path.join(pretrained, "model.pdopt"))) if optim is not None and os.path.exists(os.path.join(pretrained, "model.pdopt")) else None
        step = int(os.path.basename(pretrained)) if os.path.basename(pretrained).isdigit() else 0
        if lr is not None:
            lr.step(step)
    return model, optim, lr, step


def save_model(model, optim, step, save_path, best=False, only_last=False):
    if best:
        paddle.save(model.state_dict(), os.path.join(save_path, "best", 'model.pdparams'))
        paddle.save(optim.state_dict(), os.path.join(save_path, "best", 'model.pdopt'))
    paddle.save(model.state_dict(), os.path.join(save_path, str(step), 'model.pdparams'))
    paddle.save(optim.state_dict(), os.path.join(save_path, str(step), 'model.pdopt'))
    if only_last:
        for item in os.listdir(save_path):
            if item.isdigit() and int(item) != step:
                shutil.rmtree(os.path.join(save_path, item))


def time_std(time):
    sec = time % 60
    mins = (time // 60) % 60
    hour = (time // 3600) % 24
    days = time // (3600 * 24)
    return "%2dd%02dh%02dm%02ds" % (days, hour, mins, sec)
