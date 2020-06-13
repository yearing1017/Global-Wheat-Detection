# Global-Wheat-Detection

## 比赛简介

### 比赛描述

- 为了获得有关全世界麦田的大量准确数据，植物科学家使用“小麦头”（包含谷物的植物上的穗）的图像检测。这些图像用于估计不同品种的小麦头的密度和大小。但是，在室外野外图像中进行准确的小麦头检测可能在视觉上具有挑战性。密集的小麦植株经常重叠，并且风会使照片模糊。两者都使得难以识别单头。此外，外观会因成熟度，颜色，基因型和头部方向而异。最后，由于小麦在世界范围内种植，因此必须考虑不同的品种，种植密度，样式和田间条件。为小麦表型开发的模型需要在不同的生长环境之间进行概括。当前的检测方法涉及一阶段和两阶段的检测器（Yolo-V3和Faster-RCNN），但是即使在使用大型数据集进行训练时，仍然存在对训练区域的偏倚。
- The [Global Wheat Head Dataset](http://www.global-wheat.com/2020-challenge/) is led by nine research institutes from seven countries: the University of Tokyo, Institut national de recherche pour l’agriculture, l’alimentation et l’environnement, Arvalis, ETHZ, University of Saskatchewan, University of Queensland, Nanjing Agricultural University, and Rothamsted Research. These institutions are joined by many in their pursuit of accurate wheat head detection, including the Global Institute for Food Security, DigitAg, Kubota, and Hiphen.

### 比赛数据

- train.zip文件（3422张训练集图像，部分无小麦头），如下截图所示：

![train.png](https://github.com/yearing1017/Global-Wheat-Detection/blob/master/image/train.png)

- test.zip文件（10张测试图像)，如下截图所示：

![test.png](https://github.com/yearing1017/Global-Wheat-Detection/blob/master/image/test.png)
