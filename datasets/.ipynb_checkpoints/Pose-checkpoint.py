# -*- coding: utf-8 -*-
# @Time    : 2023/9/1 17:16
# @Author  : Pan
# @Software: PyCharm
# @Project : VisualFramework
# @FileName: Pose.py

import os
import paddle
import numpy as np
from paddle import io
from datasets import base
from transforms import Compose

labels = ['aeroplane', 'bicycle', 'boat', 'bus', 'car', 'chair', 'diningtable', 'motorbike', 'sofa', 'train']
nuisance = ['context', 'iid_test', 'nuisance', 'occlusion', 'pose', 'shape', 'texture', 'weather']


class OODPoseDataset(io.Dataset):
    def __init__(self, config):
        super(OODPoseDataset, self).__init__()
        self.mode = config["mode"] if "mode" in config.keys() else "standard"
        self.data_root = config["data_root"] if "data_root" in config.keys() else None
        self.data_list_file = config["data_list"] if "data_list" in config.keys() else None
        self.split_sign = config["split_sign"] if "split_sign" in config.keys() else None
        self.recursion_identifier = config["recursion_identifier"] if "recursion_identifier" in config.keys() else None
        self.trans = Compose(config["transforms"])
        self.deviation = config["deviation"] if "deviation" in config.keys() else True
        self.azimuth_min = config["azimuth_min"] if "azimuth_min" in config.keys() else 0
        self.elevation_min = config["elevation_min"] if "elevation_min" in config.keys() else 0
        self.theta_min = config["theta_min"] if "theta_min" in config.keys() else 0

        self.azimuth_range = config["azimuth_range"] if "azimuth_range" in config.keys() else 0
        self.elevation_range = config["elevation_range"] if "elevation_range" in config.keys() else 0
        self.theta_range = config["theta_range"] if "theta_range" in config.keys() else 0

        self.data_list = self._make_list()
        if self.deviation:
            self.mean_std(self.data_list)

        print("="*50 + self.mode +"="*50)
        print("%7.5f %7.5f" % (self.azimuth_range, self.azimuth_min))
        print("%7.5f %7.5f" % (self.elevation_range, self.elevation_min))
        print("%7.5f %7.5f" % (self.theta_range, self.theta_min))

    def __getitem__(self, item):
        data = eval("self._" + self.mode)(item)
        return data

    def __len__(self):
        return len(self.data_list)

    def _standard(self, item):
        path = self.data_list[item]["img"] + ".JPEG" if self.data_list[item]["img"][-5:] != ".JPEG" else self.data_list[item]["img"]
        data = {"img": os.path.join(self.data_root, path),
                "path": self.data_list[item]["img"],
                "nuis": self.data_list[item]["nuisance"],
                "distance": eval(self.data_list[item]["distance"]),
                "label": [
                    paddle.to_tensor(labels.index(self.data_list[item]["labels"])),
                    paddle.to_tensor(314 * (eval(self.data_list[item]["azimuth"]) - self.azimuth_min) / self.azimuth_range - 0.5, dtype=paddle.int64),
                    paddle.to_tensor(314 * (eval(self.data_list[item]["elevation"]) - self.elevation_min) / self.elevation_range - 0.5,
                                     dtype=paddle.int64),
                    paddle.to_tensor(314 * (eval(self.data_list[item]["theta"]) - self.theta_min) / self.theta_range - 0.5, dtype=paddle.int64)
                ]}
        data = self.trans(data)
        return data

    def _predict(self, item):
        path = self.data_list[item]["img"] + ".JPEG" if self.data_list[item]["img"][-5:] != ".JPEG" else self.data_list[item]["img"]
        data = {
            "img": os.path.join(self.data_root, path),
            "nuis": self.data_list[item]["nuisance"],
            "distance": eval(self.data_list[item]["distance"]),
            "path": self.data_list[item]["img"]
        }
        data = self.trans(data)
        return data

    def _make_list(self):
        f = open(self.data_list_file, "r")
        lines = f.read().rstrip("\n").split("\n")
        f.close()
        dict_name = lines[0].split(",")
        lines[1:].sort()
        lines = [{dict_name[i]: value for i, value in enumerate(line.split(","))} for line in lines[1:]]
        return lines

    def mean_std(self, lines):
        lines_value = np.zeros((4, len(lines)), dtype="float32")
        for idx, line in enumerate(lines):
            lines_value[0, idx] = eval(line["azimuth"])
            lines_value[1, idx] = eval(line["elevation"])
            lines_value[2, idx] = eval(line["theta"])
            lines_value[3, idx] = eval(line["distance"])

        self.azimuth_min = np.min(lines_value[0])
        self.elevation_min = np.min(lines_value[1])
        self.theta_min = np.min(lines_value[2])
        self.distance_min = np.min(lines_value[3])

        self.azimuth_range = np.max(lines_value[0]) - self.azimuth_min
        self.elevation_range = np.max(lines_value[1]) - self.elevation_min
        self.theta_range = np.max(lines_value[2]) - self.theta_min
        self.distance_range = np.max(lines_value[3]) - self.distance_min
