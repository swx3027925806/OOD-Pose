# -*- coding: utf-8 -*-
# @Time    : 2023/5/8 14:10
# @Author  : Pan
# @Software: PyCharm
# @Project : VisualFramework
# @FileName: __init__.py

import paddle
from paddle import io
from datasets.Pose import OODPoseDataset


def make_dataloader(config):
    dataset = eval(config["type"])(config)
    dataloader = io.DataLoader(dataset,
                               batch_size=config["batch_size"],
                               shuffle=config["shuffle"],
                               num_workers=config["num_workers"])
    return dataloader


if __name__ == "__main__":
    config = {
        "type": "MAEDataset",
        "batch_size": 1,
        "shuffle": True,
        "num_workers": 2,
        "data_root": "D:\\code\\VisualFramework\\transforms\\test_image",
        "transforms": [
            {
                "type": "LoadData",
                "keys": ["img"],
                "func": "cv2"
            },
            {
                "type": "ResizeByShort",
                "keys": ["img"],
                "short": [128, 256, 512],
                "inter": ["NEAREST", "LINEAR", "AREA", "CUBIC", "LANCZOS4"]
            },
            {
                "type": "RandPaddingCrop",
                "keys": ["img"],
                "pad_size": (128, 128),
                "crop_size": (128, 128)
            }
        ]
    }
    dataloader = make_dataloader(config)
    print(dataloader)
