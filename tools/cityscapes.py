# -*- coding: utf-8 -*-
# @Time    : 2023/5/29 10:03
# @Author  : Pan
# @Software: PyCharm
# @Project : VisualFramework
# @FileName: cityscapes


import os
import cv2
import numpy as np
from tqdm import tqdm


def read_file(filename):
    file = open(filename, 'r')
    txt = file.read().split()
    file.close()
    return txt


def split_image(image, size, stride, save_path, flag=1):
    y = (image.shape[0] - (size[0] - stride[0])) // stride[0]
    x = (image.shape[1] - (size[1] - stride[1])) // stride[1]
    for i in range(y):
        for j in range(x):
            sub_image = image[i*stride[0]:i*stride[0]+size[0], j*stride[1]:j*stride[1]+size[1], :flag]
            cv2.imwrite(save_path + "_%d_%d.png" % (i, j), sub_image)


def traversing_image(root, list_file, save_path, size, scales, stride, flag):
    print("Dataset: %s start!" % list_file)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    image_list = read_file(list_file)
    image_list.sort()
    for idx in range(len(image_list)):
        image = cv2.imread(os.path.join(root, image_list[idx]))
        for scale in scales:
            split_image(cv2.resize(image, dsize=None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST),
                        size,
                        stride,
                        os.path.join(save_path, "%d_%d" % (idx, int(scale * 100))),
                        1 if flag else 3)

def make_list(root, image_dir, label_dir, save_path):
    image_list = os.listdir(os.path.join(root, image_dir))
    list_txt = ["%s %s" % (os.path.join(image_dir, name), os.path.join(label_dir, name)) for name in image_list]
    file = open(save_path, "w")
    file.write("\n".join(list_txt))
    file.close()


if __name__ == "__main__":
    root = "data/cityscapes/"
    list_file = [
                 "data/cityscapes/trainImages.txt",
                 "data/cityscapes/valImages.txt",
                 "data/cityscapes/testImages.txt",
                 "data/cityscapes/trainLabels.txt",
                 "data/cityscapes/valLabels.txt",
                 "data/cityscapes/testLabels.txt"]
    save_path = [
                 "cityscapes/Images/train",
                 "cityscapes/Images/train",
                 "cityscapes/Images/val",
                 "cityscapes/Labels/train",
                 "cityscapes/Labels/train",
                 "cityscapes/Labels/val"]

    # list_file = ["data/valImages.txt",
    #              "data/valLabels.txt"]
    # save_path = ["cityscapes/Images/val",
    #              "cityscapes/Labels/val"]
    size = (512, 512)
    scales = [0.5, 1.0]
    stride = (256, 256)
    for i in range(len(list_file)):
        flag = "Labels" in list_file[i]
        traversing_image(root, list_file[i], save_path[i], size, scales, stride, flag)

    make_list("cityscapes", "Images/train", "Labels/train", "cityscapes/train.txt")
    make_list("cityscapes", "Images/val", "Labels/val", "cityscapes/val.txt")
