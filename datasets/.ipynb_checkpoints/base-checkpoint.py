# -*- coding: utf-8 -*-
# @Time    : 2023/6/2 14:35
# @Author  : Pan
# @Software: PyCharm
# @Project : VisualFramework
# @FileName: base

import os


def recursion(path, recursion_identifier):
    path_list = []
    if os.path.isdir(path):
        name_list = os.listdir(path)
        for name in name_list:
            path_list += recursion(os.path.join(path, name), recursion_identifier)
    else:
        if recursion_identifier is None:
            path_list.append(path)
        elif path.split(".")[-1] in recursion_identifier:
            path_list.append(path)
    return path_list
