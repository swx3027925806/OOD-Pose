# -*- coding: utf-8 -*-
# @Time    : 2023/5/20 15:52
# @Author  : Pan
# @Software: PyCharm
# @Project : VisualFramework
# @FileName: utils


import os
import paddle
import paddle.nn as nn
import paddle.nn.initializer as paddle_init

__all__ = [
    'to_2tuple', 'DropPath', 'Identity', 'trunc_normal_', 'zeros_', 'ones_',
    'init_weights', 'load_pretrained_model'
]


def load_pretrained_model(model, pretrained_model):
    if pretrained_model is not None:
        if os.path.exists(pretrained_model):
            para_state_dict = paddle.load(pretrained_model)
            model_state_dict = model.state_dict()
            keys = model_state_dict.keys()
            num_params_loaded = 0
            for k in keys:
                if k not in para_state_dict:
                    print(k, "not found")
                elif list(para_state_dict[k].shape) != list(model_state_dict[k].shape):
                    print(k, "error")
                else:
                    model_state_dict[k] = para_state_dict[k]
                    num_params_loaded += 1
            model.set_dict(model_state_dict)



def to_2tuple(x):
    return tuple([x] * 2)


def drop_path(x, drop_prob=0., training=False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ...
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = paddle.to_tensor(1 - drop_prob)
    shape = (paddle.shape(x)[0], ) + (1, ) * (x.ndim - 1)
    random_tensor = keep_prob + paddle.rand(shape).astype(x.dtype)
    random_tensor = paddle.floor(random_tensor)  # binarize
    output = x.divide(keep_prob) * random_tensor
    return output


class DropPath(nn.Layer):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Identity(nn.Layer):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, input):
        return input


trunc_normal_ = paddle_init.TruncatedNormal(std=.02)
zeros_ = paddle_init.Constant(value=0.)
ones_ = paddle_init.Constant(value=1.)


def init_weights(layer):
    """
    Init the weights of transformer.
    Args:
        layer(nn.Layer): The layer to init weights.
    Returns:
        None
    """
    if isinstance(layer, nn.Linear):
        trunc_normal_(layer.weight)
        if layer.bias is not None:
            zeros_(layer.bias)
    elif isinstance(layer, nn.LayerNorm):
        zeros_(layer.bias)
        ones_(layer.weight)

