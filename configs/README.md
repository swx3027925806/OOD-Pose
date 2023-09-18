# 配置文件

## 配置文件的组成

在该文件夹下，将包含不同任务的配置文件，通常我们通过文件夹的方式将这些配置文件归类。
其中，我们以图像分类任务为例，在图像分类任务下可能会出现以文件夹形式包含的不同网络在不同数据集上的配置情况。
但也可能存在单个配置文件。

## 配置文件的修改策略

配置文件包含了训练和预测模型的超参数，通过修改配置文件即可对配置文件做出简单调整。
以下以分类任务中ResNet50为例来讲解配置文件的不同部分：

配置文件以字典形式配置，通常为：
```python
# -*- coding: utf-8 -*-
# @Time    : 2023/6/2 19:27
# @Author  : Pan
# @Software: PyCharm
# @Project : VisualFramework
# @FileName: ResNet50.py

image_size = (224, 224)
max_steps = 20000

config = {
    "type": "Clas",
    "base_info": {},
    "train_dataset": {},
    "val_dataset": {},
    "optimizer": {},
    "network": {},
    "loss": {},
    "metric": [],
    "amp": {}
}

```

其中有两个统一定义的参数：图像尺寸和训练步长。暂时不支持以epoch的方式训练。

### type配置
配置文件首先需要定义任务类型，以此来统一不同任务的训练。即`"type": "Clas"`

```python
    "type": "Clas"                # 配置模型的任务类型
```


### base_info配置
```python
    "base_info": {                # 基础信息配置
        "step": max_steps,        # 模型训练的总步长
        "dot": 20,                # 打印信息的周期
        "save_iters": 200,        # 保存模型的周期
        "pretrained": None,       # 是否采用预训练
        "save_path": "output/",   # 模型保存的位置
        "log_dir": "log_dir/",    # 保存日志文件的位置
    }
```


### train_dataset配置
这部分是训练数据集的配置信息，其中根据不同的数据集的规范可能不相同，详情需要查看`datasets`下不同数据集加载的文件中看相关介绍。

```python
    "train_dataset": {
        "type": "ClasBaseDataset",   # 数据集加载的名称，一般是datasets下的数据集文件下的具体类名
        "batch_size": 256,           # batch_size
        "shuffle": True,
        "num_workers": 2,
        "data_root": "data",
        "data_list": "data/train.txt",
        "transforms": [              # 数据增强策略，其中keys是数据级代码中定义的字典关键字
            {
                "type": "LoadData",
                "keys": ["img"],
                "func": "cv2"
            },
            {
                "type": "ResizeByShort",
                "keys": ["img"],
                "short": [i for i in range(128, 512, 1)],
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
                "mean": 0.5,
                "std": 0.5
            }
        ]
    }
```


### optimizer 配置
这其中包含了具体的优化器的配置
```python
    "optimizer": {
        "type": "adam",                  # 优化器的名称
        "lr_scheduler": {                # 学习率的衰减策略，目前仅包含了这个【其他懒得添加了】
            "type": "WarmupCosineLR",    # Warm up 学习率刚开始是由小变大，Cosine
            "learning_rate": 0.001,
            "total_steps": max_steps,
            "warmup_steps": 500,
            "warmup_start_lr": 1e-7,
            "end_lr": 1e-7
        },
        "decay": None
    }
```


### network 配置
网络相关配置
```python
    "network": {
        "type": "clas",                 # type表示任务，规范仅用小写对照networks目录下的任务
        "network": {
            "type": "ResNet",
            "structure": 50,
            "num_classes": 257
        }
    }
```

### loss 配置

```python
"loss": {
        "loss_list": [       # loss_list是损失的列表，这里注意，列表对于的是网络不同的输出结果和不同的标签，如果想一个输出结果算不同损失可以用MixLoss
            {
                "type": "CrossEntropyLoss"
            }
        ],
        "loss_coef": [1]     # loss_coef是不同损失的权重，需要和loss_list长度相同
    }
```

### metric 配置
metric是在验证的时候对模型做评估的方法
```python
    "metric": [              # 这里同样是列表，表示可以加入多个评估方式
        {
            "type": "ACC",
            "topk": (1, 5)
        }
    ]
```

### amp 配置
混合精度训练，可以用更少的显存更快的时间来训练，目前仅支持O1模式。部分任务不适配
```python
    "amp": {
        "scale": 1024
    }
```


以上是简单的配置方式，还有一些其他的配置模式日后会一步一步完善。