# -*- coding: utf-8 -*-
# @Time    : 2023/6/2 14:51
# @Author  : Pan
# @Software: PyCharm
# @Project : VisualFramework
# @FileName: operator

import paddle
from paddle import nn


class SparseConv2D(nn.Conv2D):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 groups=1):
        super(SparseConv2D, self).__init__()
