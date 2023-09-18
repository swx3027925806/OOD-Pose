# -*- coding: utf-8 -*-
# @Time    : 2023/5/10 16:02
# @Author  : Pan
# @Software: PyCharm
# @Project : VisualFramework
# @FileName: basetransforms
import numbers
import random

import cv2
import paddle
import numpy as np
from PIL import Image
from paddle.vision import transforms as F


class Template:
    def __init__(self, config):
        self.keys = config["keys"] if "keys" in config.keys() else ["img"]

    def __call__(self, data):
        for key in self.keys:
            key_index = key.split("_")[0]
            data[key] = eval("self._" + key_index)(data[key_index])
        return data

    def _img(self, img):
        return img


class LoadData:
    def __init__(self, config):
        self.keys = config["keys"] if "keys" in config.keys() else ["img"]
        self.func = config["func"] if "func" in config.keys() else "cv2"

    def __call__(self, data):
        for key in self.keys:
            key_index = key.split("_")[0]
            data[key] = eval("self._" + key_index)(data[key])
        return data

    def _img(self, img):
        if self.func == "cv2":
            img = cv2.imread(img)
        elif self.func == "PIL":
            img = Image.open(img)
            img = np.asarray(img)
        return img

    def _seg(self, img):
        if self.func == "cv2":
            img = cv2.imread(img)
        elif self.func == "PIL":
            img = Image.open(img)
            img = np.asarray(img)
        return img


class ToTensor:
    def __init__(self, config):
        self.keys = config["keys"] if "keys" in config.keys() else ["img"]
        self.format = config["format"] if "format" in config.keys() else "CHW"

    def __call__(self, data):
        for key in self.keys:
            key_index = key.split("_")[0]
            data[key] = eval("self._" + key_index)(data[key])
        return data

    def _img(self, img):
        img = F.to_tensor(img, self.format)
        return img

    def _seg(self, img):
        img = paddle.to_tensor(img[:, :, 0], dtype=paddle.int64)
        return img


class AddGauss:
    def __init__(self, config):
        self.keys = config["keys"] if "keys" in config.keys() else ["img"]
        self.prob = config["prob"] if "prob" in config.keys() else 0.25
        self.radius = config["radius"] if "radius" in config.keys() else (0., 0.5)

    def __call__(self, data):
        for key in self.keys:
            key_index = key.split("_")[0]
            data[key] = eval("self._" + key_index)(data[key])
        return data

    def _img(self, img):
        if random.random() < self.prob:
            gauss = paddle.randn(shape=img.shape) * 2
            coef = random.random() * (self.radius[1] - self.radius[0]) + self.radius[0]
            img = paddle.clip(img + gauss * coef, -1, 1)
        return img


class Resize:
    def __init__(self, config):
        self.keys = config["keys"] if "keys" in config.keys() else ["img"]
        self.inter = config["inter"] if "inter" in config.keys() else "bilinear"
        # func 支持 NEAREST LINEAR AREA CUBIC LANCZOS4
        self.aim_size = config["aim_size"]
        if isinstance(self.aim_size, int):
            self.aim_size = (self.aim_size, self.aim_size)

    def __call__(self, data):
        for key in self.keys:
            key_index = key.split("_")[0]
            data[key] = eval("self._" + key_index)(data[key])
        return data

    def _img(self, img):
        img = F.resize(img, self.aim_size, interpolation=self.inter)
        return img

    def _seg(self, img):
        img = F.resize(img, self.aim_size, interpolation=self.inter)
        return img


class ResizeByShort:
    def __init__(self, config):
        self.keys = config["keys"] if "keys" in config.keys() else ["img"]
        self.short = config["short"] if "short" in config.keys() else [512]
        self.inter = config["inter"] if "inter" in config.keys() else ["bilinear"]
        # func 支持 bilinear LINEAR AREA CUBIC LANCZOS4

    def __call__(self, data):
        short_size = random.choice(self.short)
        for key in self.keys:
            key_index = key.split("_")[0]
            data[key] = eval("self._" + key_index)(data[key], short_size)
        return data

    def _img(self, img, short_size):
        h, w = img.shape[:2]
        inter = random.choice(self.inter)
        aim_h = short_size if h < w else h / (w / short_size)
        aim_w = short_size if w < h else w / (h / short_size)
        img = F.resize(img, (int(aim_w), int(aim_h)), interpolation=inter)
        return img

    def _seg(self, img, short_size):
        h, w = img.shape[:2]
        aim_h = short_size if h < w else h / (w / short_size)
        aim_w = short_size if w < h else w / (h / short_size)
        img = F.resize(img, (int(aim_w), int(aim_h)), interpolation="bilinear")
        return img


class Padding:
    def __init__(self, config):
        self.keys = config["keys"] if "keys" in config.keys() else ["img"]
        self.pad_value = config["pad_value"] if "pad_value" in config.keys() else [0, 0, 0]
        # pad_value 支持 [0, 0, 0] 或 random
        self.pad_size = config["pad_size"]
        if isinstance(self.pad_size, int):
            self.pad_size = (self.pad_size, self.pad_size)

    def __call__(self, data):
        for key in self.keys:
            key_index = key.split("_")[0]
            data[key] = eval("self._" + key_index)(data[key])
        return data

    def _img(self, img):
        top = (self.pad_size[0] - img.shape[0]) // 2 if self.pad_size[0] - img.shape[0] > 0 else 0
        bottom = self.pad_size[0] - img.shape[0] - top if self.pad_size[0] - img.shape[0] > 0 else 0
        left = (self.pad_size[1] - img.shape[1]) // 2 if self.pad_size[1] - img.shape[1] > 0 else 0
        right = self.pad_size[1] - img.shape[1] - left if self.pad_size[1] - img.shape[1] > 0 else 0
        if self.pad_value == "random":
            pad = np.random.randint(0, 256, 3).tolist()
        else:
            pad = self.pad_value
        img = F.pad(img, (left, top, right, bottom), fill=pad, padding_mode='constant')
        return img

    def _seg(self, img):
        top = (self.pad_size[0] - img.shape[0]) // 2 if self.pad_size[0] - img.shape[0] > 0 else 0
        bottom = self.pad_size[0] - img.shape[0] - top if self.pad_size[0] - img.shape[0] > 0 else 0
        left = (self.pad_size[1] - img.shape[1]) // 2 if self.pad_size[1] - img.shape[1] > 0 else 0
        right = self.pad_size[1] - img.shape[1] - left if self.pad_size[1] - img.shape[1] > 0 else 0
        img = F.pad(img, (left, top, right, bottom), fill=255, padding_mode='constant')
        return img


class Crop:
    def __init__(self, config):
        self.keys = config["keys"] if "keys" in config.keys() else ["img"]
        self.crop_size = config["crop_size"] if "crop_size" in config.keys() else (512, 512)

    def __call__(self, data):
        img = data[self.keys[0]]
        assert img.shape[0] >= self.crop_size[0] and img.shape[1] >= self.crop_size[1], "裁剪尺度小于图像尺度"
        y = np.random.randint(0, img.shape[0] - self.crop_size[0]+1)
        x = np.random.randint(0, img.shape[1] - self.crop_size[1]+1)
        for key in self.keys:
            key_index = key.split("_")[0]
            data[key] = eval("self._" + key_index)(data[key], y, x)
        return data

    def _img(self, img, x, y):
        img = F.crop(img, x, y, self.crop_size[0], self.crop_size[1])
        return img

    def _seg(self, img, x, y):
        img = F.crop(img, x, y, self.crop_size[0], self.crop_size[1])
        return img


class RandPaddingCrop:
    def __init__(self, config):
        self.crop = Crop(config)
        self.pad = Padding(config)

        self.keys = config["keys"] if "keys" in config.keys() else ["img"]

    def __call__(self, data):
        data = self.crop(self.pad(data))
        return data


class BalancePaddingCrop:
    def __init__(self, config):
        self.crop = BalanceCrop(config)
        self.pad = Padding(config)

        self.keys = config["keys"] if "keys" in config.keys() else ["img"]

    def __call__(self, data):
        data = self.crop(self.pad(data))
        return data


class BalanceCrop:
    def __init__(self, config):
        self.keys = config["keys"] if "keys" in config.keys() else ["img"]
        self.crop_size = config["crop_size"] if "crop_size" in config.keys() else (512, 512)
        self.weights = np.array(config["weights"]) if "weights" in config.keys() else None

    def __call__(self, data):
        flag = False
        img = data[self.keys[0]]
        assert img.shape[0] >= self.crop_size[0] and img.shape[1] >= self.crop_size[1], "裁剪尺度小于图像尺度"
        datas = {}
        while not flag:
            y = np.random.randint(0, img.shape[0] - self.crop_size[0] + 1)
            x = np.random.randint(0, img.shape[1] - self.crop_size[1] + 1)
            for key in self.keys:
                key_index = key.split("_")[0]
                datas[key], flag = eval("self._" + key_index)(data[key], y, x, flag)
        for key in self.keys:
            data[key] = datas[key]
        return data

    def _img(self, img, x, y, flag):
        img = F.crop(img, x, y, self.crop_size[0], self.crop_size[1])
        return img, flag

    def _seg(self, img, x, y, flag):
        img = F.crop(img, x, y, self.crop_size[0], self.crop_size[1])
        weight = np.zeros_like(self.weights)
        binc = np.bincount(img)
        weight[:len(binc)] = binc
        prob = np.sqrt(weight * self.weights / weight.sum())
        if np.random.rand() < prob:
            flag = True
        return img, flag


class Normalize:
    def __init__(self, config):
        self.keys = config["keys"] if "keys" in config.keys() else ["img"]
        self.mean = config["mean"] if "mean" in config.keys() else 0.5
        self.std = config["std"] if "std" in config.keys() else 0.5
        self.format = config["format"] if "format" in config.keys() else "CHW"
        if isinstance(self.mean, numbers.Number):
            self.mean = [self.mean, self.mean, self.mean]

        if isinstance(self.std, numbers.Number):
            self.std = [self.std, self.std, self.std]

    def __call__(self, data):
        for key in self.keys:
            key_index = key.split("_")[0]
            data[key] = eval("self._" + key_index)(data[key])
        return data

    def _img(self, img):
        img = F.normalize(img, self.mean, self.std, self.format)
        return img


class Transpose:
    def __init__(self, config):
        self.keys = config["keys"] if "keys" in config.keys() else ["img"]
        self.transpose = config["transpose"] if "transpose" in config.keys() else [2, 0, 1]

    def __call__(self, data):
        for key in self.keys:
            key_index = key.split("_")[0]
            data[key] = eval("self._" + key_index)(data[key])
        return data

    def _img(self, img):
        img = np.transpose(img, self.transpose)
        return img
