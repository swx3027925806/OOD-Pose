# Paddle-OOD-Pose

在这个项目中，是针对ICCV2023 OOD Pose赛道设计的姿态估计代码。
项目基于`PaddlePaddle2.4.0`设计，在使用前请安装好对应版本。

## 数据集准备

将数据集和测试集全部转换成csv格式：
```
img,labels,azimuth,elevation,theta,distance,nuisance
PASCAL3D_aeroplane_n02690373_10001_00.JPEG,aeroplane,0.944742,-0.092142,-0.062302,4.089074,None
PASCAL3D_aeroplane_n02690373_10032_00.JPEG,aeroplane,1.208163,-0.323897,0.068477,4.676474,None
PASCAL3D_aeroplane_n02690373_10061_00.JPEG,aeroplane,3.509642,0.249522,0.071916,4.649304,None
PASCAL3D_aeroplane_n02690373_101_00.JPEG,aeroplane,1.505750,-0.785398,0.064802,5.131357,None
PASCAL3D_aeroplane_n02690373_10203_00.JPEG,aeroplane,4.319690,0.016337,-0.051467,13.972306,None
```
数据集未引用，故这里不做展开。

在测试时，5000iters可以达到90+的性能。
## 代码运行
配置文件位置在`config.py`中，通过`engine.py`选择train或者predict开启任务。

## 团队介绍

西安电子科技大学 `IPIU` 实验室

团队成员：

| 姓名      | 年级 | 邮箱 |
| ----------- | ----------- | ----------- |
| 佘文轩 | 2022级硕士研究生 | swx_sxpp@qq.com |
| 刘雨   | 2022级硕士研究生 | ly2865818487@163.com |
