# -*- coding: utf-8 -*-
# @Time    : 2023/9/1 21:24
# @Author  : Pan
# @Software: PyCharm
# @Project : VisualFramework
# @FileName: CSP_ResNet_XX.py


image_size = (224, 224)
max_steps = 50000

config = {
    "type": "Pose",
    "base_info": {
        "step": max_steps,
        "dot": 20,
        "save_iters": 1000,
        "pretrained": "model/BEiTv2_vit_base_patch16_224",
        "save_path": "output/",
        "log_dir": "log_dir/",
    },
    "train_dataset": {
        "type": "OODPoseDataset",
        "batch_size": 32,
        "shuffle": True,
        "num_workers": 4,
        "data_root": "data/images",
        "data_list": "data/gt.csv",
        "transforms": [
            {
                "type": "LoadData",
                "keys": ["img"],
                "func": "cv2"
            },
            {
                "type": "ResizeByShort",
                "keys": ["img"],
                "short": [i for i in range(64, 256, 1)],
                "inter": ["bilinear"]
            },
            {
                "type": "RandPaddingCrop",
                "keys": ["img"],
                "pad_size": image_size,
                "crop_size": image_size
            },
            {
                "type": "ToTensor",
                "keys": ["img"]
            },
            {
                "type": "Normalize",
                "keys": ["img"],
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225]
            }
        ]
    },
    "val_dataset": {
        "type": "OODPoseDataset",
        "batch_size": 32,
        "shuffle": True,
        "num_workers": 4,
        "data_root": "data/images",
        "data_list": "data/gt.csv",
        "transforms": [
            {
                "type": "LoadData",
                "keys": ["img"],
                "func": "cv2"
            },
            {
                "type": "ResizeByShort",
                "keys": ["img"],
                "short": [224],
                "inter": ["bilinear"]
            },
            {
                "type": "RandPaddingCrop",
                "keys": ["img"],
                "pad_size": image_size,
                "crop_size": image_size
            },
            {
                "type": "ToTensor",
                "keys": ["img"]
            },
            {
                "type": "Normalize",
                "keys": ["img"],
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225]
            }
        ]
    },
    "test_dataset": {
        "type": "OODPoseDataset",
        "batch_size": 64,
        "shuffle": False,
        "num_workers": 2,
        "save_path": "pre.csv",
        "data_root": "data/images",
        "data_list": "data/gt.csv",
        "transforms": [
            {
                "type": "LoadData",
                "keys": ["img"],
                "func": "cv2"
            },
            {
                "type": "ResizeByShort",
                "keys": ["img"],
                "short": [224],
                "inter": ["bilinear"]
            },
            {
                "type": "RandPaddingCrop",
                "keys": ["img"],
                "pad_size": image_size,
                "crop_size": image_size
            },
            {
                "type": "ToTensor",
                "keys": ["img"]
            },
            {
                "type": "Normalize",
                "keys": ["img"],
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225]
            }
        ]
    },
    "optimizer": {
        "type": "adam",
        "lr_scheduler": {
            "type": "WarmupCosineLR",    # Warm up 学习率刚开始是由小变大，Cosine
            "learning_rate": 0.0001,
            "total_steps": max_steps,
            "warmup_steps": 1000,
            "warmup_start_lr": 1e-7,
            "end_lr": 1e-7
        },
        "decay": None
    },
    "network": {
        "type": "pose",
        "network": {
            "type": " ",
            "width_mult": 1.5,
            "depth_mult": 1.67,
            "decoder": {
                "type": "MutilHeadDecoder",
                "heads": [
                    {
                        "type": "Classify",
                        "features": 768,
                        "num_class": 10,
                    },
                    {
                        "type": "Classify",
                        "features": 768,
                        "num_class": 8,
                    },
                    {
                        "type": "Regression",
                        "features": 768
                    },
                    {
                        "type": "Regression",
                        "features": 768
                    },
                    {
                        "type": "Regression",
                        "features": 768,
                    },
                    {
                        "type": "Regression",
                        "features": 768,
                    }
                ]
            }
        }
    },
    "loss": {
        "loss_list": [
            {
                "type": "CrossEntropyLoss"
            },
            {
                "type": "CrossEntropyLoss"
            },
            {
                "type": "L1Loss"
            },
            {
                "type": "L1Loss"
            },
            {
                "type": "L1Loss"
            },
            {
                "type": "L1Loss"
            }
        ],
        "loss_coef": [1, 1, 0.1, 0.1, 0.1, 0.05]
    },
    # "amp": {
    #     "scale": 1024
    # }
}
