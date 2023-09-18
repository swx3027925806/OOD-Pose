# -*- coding: utf-8 -*-
# @Time    : 2023/5/30 9:52
# @Author  : Pan
# @Software: PyCharm
# @Project : VisualFramework
# @FileName: classification


import os
import cv2
import time
import loss
import paddle
import networks
import optimizers
import datasets
import numpy as np
from core import base
from tqdm import tqdm
from metirc import MetricsCompose
from visualdl import LogWriter


labels = ['aeroplane', 'bicycle', 'boat', 'bus', 'car', 'chair', 'diningtable', 'motorbike', 'sofa', 'train']
nuisance = ['context', 'iid_test', 'nuisance', 'occlusion', 'pose', 'shape', 'texture', 'weather']


class PoseEngine:
    def __init__(self, config):
        self.start_time = time.time()
        self.next_time = time.time()
        self.now_time = time.time()
        self.base = 0

        self.base_info_config = config["base_info"]
        self.train_dataset_config = config["train_dataset"]
        self.val_dataset_config = config["val_dataset"]
        self.test_dataset_config = config["test_dataset"] if "test_dataset" in config.keys() else None
        self.optimizer_config = config["optimizer"]
        self.network_config = config["network"]
        self.metrics_config = config["metric"] if "metric" in config.keys() else []
        self.loss_config = config["loss"]
        self.amp = config["amp"] if "amp" in config.keys() else None

        self.writer = LogWriter(logdir=self.base_info_config["log_dir"])

        self.model = networks.make_model(self.network_config)
        self.model, self.optimizer, self.lr = optimizers.make_optim(self.optimizer_config, self.model)
        self.train_dataloader = datasets.make_dataloader(self.train_dataset_config)
        self.val_dataloader = datasets.make_dataloader(self.val_dataset_config)
        if self.test_dataset_config is not None:
            self.test_dataloader = datasets.make_dataloader(self.test_dataset_config)
        self.metrics = MetricsCompose(self.metrics_config) if len(self.metrics_config) != 0 else None
        self.loss = loss.LossCompose(self.loss_config)

        if self.base_info_config["pretrained"] is None:
            self.step = 0
        else:
            self.model, self.optimizer, self.lr, self.step = base.load_model(self.model, self.optimizer, self.lr, self.base_info_config["pretrained"])
        if self.amp:
            self.amp = paddle.amp.GradScaler(init_loss_scaling=self.amp["scale"])

    def train(self):
        if self.amp is not None:
            self.fp16_train()
        else:
            while self.step < self.base_info_config["step"]:
                for idx, data in enumerate(self.train_dataloader):
                    self.step += 1
                    # img, label, c1, c2, c3 = data["img"], data["label"], data["c1"], data["c2"], data["c3"]
                    self.optimizer.clear_grad()
                    # predict = self.model(img, c1, c2, c3)
                    img, label = data["img"], data["label"]
                    predict = self.model(img)
                    loss_value = self.loss(predict, label)
                    loss_value.backward()
                    self.optimizer.step()
                    self.lr.step()
                    if self.step % self.base_info_config["dot"] == 0:
                        self.train_display()
                    if self.step % self.base_info_config["save_iters"] == 0:
                        self.eval()
                        print("save iters: %8d in %s" % (self.step, self.base_info_config["save_path"]))
                        base.save_model(self.model, self.optimizer, self.step, self.base_info_config["save_path"], only_last=True)

    @paddle.no_grad()
    def eval(self):
        eval_loss = loss.LossCompose(self.loss_config)
        for idx, data in enumerate(tqdm(self.val_dataloader)):
            # img, label, c1, c2, c3 = data["img"], data["label"], data["c1"], data["c2"], data["c3"]
            # predict = self.model(img, c1, c2, c3)
            img, label = data["img"], data["label"]
            predict = self.model(img)
            eval_loss(predict, label)
            if self.metrics is not None:
                self.metrics.calculate(predict[0], label)
        if self.metrics is not None:
            info = self.metrics.get_metrics_info() + eval_loss.get_loss_info()
            self.metrics.reset_metrics()
        else:
            info = eval_loss.get_loss_info()
        self.eval_display(info)

    def train_display(self):
        self.now_time = time.time()
        epoch = int(self.step / len(self.train_dataloader))
        process = self.step/self.base_info_config["step"]
        speed_time = self.now_time - self.start_time
        remain_time = (self.now_time - self.next_time) / (self.base_info_config["dot"] / self.base_info_config["step"]) * (1 - process)
        self.next_time = self.now_time
        info_list = [
            {
                "name": "learning_rate",
                "value": self.lr.get_lr()
            }
        ] + self.loss.get_loss_info()
        base_info = "\033[5;31;47m[Train]\033[0m %s epochs:%4d steps:%9d/%9d process:%5.2f%% speed_time:%s remain_time:%s" % (
            time.ctime(), epoch, self.step, self.base_info_config["step"], process*100, base.time_std(speed_time), base.time_std(remain_time)
        )
        for item in info_list:
            self.writer.add_scalar(tag="train/" + item["name"], step=self.step, value=item["value"])
            base_info += " %s:%f" % (item["name"], item["value"])
        print(base_info)

    def eval_display(self, infos):
        base_info = "\033[5;31;47m[Eval]\033[0m %s steps:%9d/%9d" % (time.ctime(), self.step, self.base_info_config["step"])
        for item in infos:
            self.writer.add_scalar(tag="eval/" + item["name"], step=self.step, value=item["value"])
            base_info += " %s:%f" % (item["name"], item["value"])
        print(base_info)

    def fp16_train(self):
        while self.step < self.base_info_config["step"]:
            for idx, data in enumerate(self.train_dataloader):
                self.step += 1
                # img, label, c1, c2, c3 = data["img"], data["label"], data["c1"], data["c2"], data["c3"]
                img, label = data["img"], data["label"]
                self.optimizer.clear_grad()
                with paddle.amp.auto_cast(custom_white_list={'elementwise_add'}, level='O1'):
                    # predict = self.model(img, c1, c2, c3)
                    predict = self.model(img)
                    loss_value = self.loss(predict, label)
                scaled = self.amp.scale(loss_value)
                scaled.backward()
                self.amp.step(self.optimizer)
                self.amp.update()
                self.lr.step()
                if self.step % self.base_info_config["dot"] == 0:
                    self.train_display()
                if self.step % self.base_info_config["save_iters"] == 0:
                    self.fp16_eval()
                    print("save iters: %8d in %s" % (self.step, self.base_info_config["save_path"]))
                    base.save_model(self.model, self.optimizer, self.step, self.base_info_config["save_path"], only_last=True)

    @paddle.no_grad()
    def fp16_eval(self):
        lines = ["img,labels,azimuth,elevation,theta,distance,nuisance"]
        eval_loss = loss.LossCompose(self.loss_config)
        for idx, data in tqdm(enumerate(self.val_dataloader)):
            # img, label, c1, c2, c3 = data["img"], data["label"], data["c1"], data["c2"], data["c3"]
            img, label = data["img"], data["label"]
            with paddle.amp.auto_cast(custom_white_list={'elementwise_add'}, level='O1'):
                # predict = self.model(img, c1, c2, c3)
                predict = self.model(img)
                eval_loss(predict, label)
            if self.metrics is not None:
                self.metrics.calculate(predict[0], label)
            for i in range(len(predict[0])):
                name = data["path"][i]
                nuis = data["nuis"][i]
                label = labels[paddle.argmax(predict[0][i]).tolist()[0]]
                azimuth = paddle.argmax(predict[1][i]).tolist()[0] / 314 * 6.283185 + 0
                elevation = paddle.argmax(predict[2][i]).tolist()[0] / 314 * 3.141592 + -1.570796
                theta = paddle.argmax(predict[3][i]).tolist()[0] / 314 * 6.283186 + -3.141593
                distance = data["distance"][i]
                line = "%s, %s, %f, %f, %f, %f, %s" % (name, label, azimuth, elevation, theta, distance, nuis)
                lines.append(line)
        file = open("val.csv", "w")
        file.write("\n".join(lines))
        file.close()
        if self.metrics is not None:
            info = self.metrics.get_metrics_info() + eval_loss.get_loss_info()
            self.metrics.reset_metrics()
        else:
            info = eval_loss.get_loss_info()
        self.eval_display(info)

    @paddle.no_grad()
    def predict(self):
        save_path = self.test_dataset_config["save_path"]
        lines = ["img,labels,azimuth,elevation,theta,distance,nuisance"]
        for idx, data in enumerate(tqdm(self.test_dataloader)):
            # img, c1, c2, c3 = data["img"], data["c1"], data["c2"], data["c3"]
            # predict = self.model(img, c1, c2, c3)
            img = data["img"]
            predict = self.model(img)
            for i in range(len(predict[0])):
                name = data["path"][i]
                nuis = data["nuis"][i]
                label = labels[paddle.argmax(predict[0][i]).tolist()[0]]
                azimuth = paddle.argmax(predict[1][i]).tolist()[0] / 314 * 6.283185 + 0
                elevation = paddle.argmax(predict[2][i]).tolist()[0] / 314 * 3.141592 + -1.570796
                theta = paddle.argmax(predict[3][i]).tolist()[0] / 314 * 6.283186 + -3.141593
                distance = data["distance"][i]
                line = "%s, %s, %f, %f, %f, %f, %s" % (name, label, azimuth, elevation, theta, distance, nuis)
                lines.append(line)
        file = open(save_path, "w")
        file.write("\n".join(lines))
        file.close()
