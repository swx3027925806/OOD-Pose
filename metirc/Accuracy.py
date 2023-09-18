# -*- coding: utf-8 -*-
# @Time    : 2023/5/30 10:24
# @Author  : Pan
# @Software: PyCharm
# @Project : VisualFramework
# @FileName: Accuracy

import paddle
from paddle.metric import Accuracy


class ACC(Accuracy):
    def __init__(self, config):
        super(ACC, self).__init__(config["topk"])

    def get_info(self):
        topk_value = self.accumulate()
        return [{"name": "ACC/top"+str(self.topk[i]), "value": topk_value[i]} for i in range(len(self.topk))]
