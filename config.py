# -*- coding: utf-8 -*-
# @Time    : 2023/9/1 21:24
# @Author  : Pan
# @Software: PyCharm
# @Project : VisualFramework
# @FileName: CSP_ResNet_XX.py


image_size = (224, 224)
max_steps = 40000

config = {
    "type": "Pose",
    "base_info": {
        "step": max_steps,
        "dot": 50,
        "save_iters": 1000,
        "pretrained": "model/BEvitv2",
        "save_path": "output/model/",
        "log_dir": "output/log_dir/",
    },
    "train_dataset": {
        "type": "OODPoseDataset",
        "batch_size": 128,
        "shuffle": True,
        "num_workers": 8,
        "data_root": "data/train/ood",
        "data_list": "data/train/gt.csv",
        "transforms": [
            {
                "type": "LoadData",
                "keys": ["img"],
                "func": "cv2"
            },
            {
                "type": "ResizeByShort",
                "keys": ["img"],
                "short": [i for i in range(48, 320, 1)],
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
        "batch_size": 128,
        "shuffle": False,
        "num_workers": 8,
        "deviation": False,
        "azimuth_min": 0.000,
        "azimuth_range": 6.28319,
        "elevation_min": -1.57080,
        "elevation_range": 3.14159,
        "theta_min": -3.14159,
        "theta_range": 6.28319,
        "data_root": "data/test/images",
        "data_list": "data/test/gt.csv",
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
        "mode": "predict",
        "save_path": "output/pred.csv",
        "batch_size": 32,
        "shuffle": False,
        "num_workers": 8,
        "deviation": False,
        "azimuth_min": 0.000,
        "azimuth_range": 6.28319,
        "elevation_min": -1.57080,
        "elevation_range": 3.14159,
        "theta_min": -3.14159,
        "theta_range": 6.28319,
        "data_root": "data/test/images",
        "data_list": "data/test/gt.csv",
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
            "warmup_steps": 2000,
            "warmup_start_lr": 1e-7,
            "end_lr": 1e-7
        },
        "decay": {
            "type": "l1_decay",
            "coeff": 0.00001,
        }
    },
    "network": {
        "type": "pose",
        "network": {
            "type": "BEiTv2_vit_base_patch16_224",
            "decoder": {
                "type": "MutilHeadDecoder",
                "heads": [
                    {
                        "type": "Classify",
                        "features": 768,
                        "num_class": 10
                    },
                    {
                        "type": "Classify",
                        "features": 768,
                        "num_class": 314
                    },
                    {
                        "type": "Classify",
                        "features": 768,
                        "num_class": 314
                    },
                    {
                        "type": "Classify",
                        "features": 768,
                        "num_class": 314
                    },
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
                "type": "CrossEntropyLoss"
            },
            {
                "type": "CrossEntropyLoss"
            },
        ],
        "loss_coef": [1, 1, 1, 1]
    },
    "amp": {
        "scale": 1024
    }
}
