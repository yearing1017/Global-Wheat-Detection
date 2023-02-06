# Global-Wheat-Detection

## 比赛简介

### 比赛描述

- 为了获得有关全世界麦田的大量准确数据，植物科学家使用“小麦头”（包含谷物的植物上的穗）的图像检测。这些图像用于估计不同品种的小麦头的密度和大小。但是，在室外野外图像中进行准确的小麦头检测可能在视觉上具有挑战性。密集的小麦植株经常重叠，并且风会使照片模糊。两者都使得难以识别单头。此外，外观会因成熟度，颜色，基因型和头部方向而异。最后，由于小麦在世界范围内种植，因此必须考虑不同的品种，种植密度，样式和田间条件。为小麦表型开发的模型需要在不同的生长环境之间进行概括。当前的检测方法涉及一阶段和两阶段的检测器（Yolo-V3和Faster-RCNN），但是即使在使用大型数据集进行训练时，仍然存在对训练区域的偏倚。
- The [Global Wheat Head Dataset](http://www.global-wheat.com/2020-challenge/) is led by nine research institutes from seven countries: the University of Tokyo, Institut national de recherche pour l’agriculture, l’alimentation et l’environnement, Arvalis, ETHZ, University of Saskatchewan, University of Queensland, Nanjing Agricultural University, and Rothamsted Research. These institutions are joined by many in their pursuit of accurate wheat head detection, including the Global Institute for Food Security, DigitAg, Kubota, and Hiphen.

### 比赛数据
- train.csv文件：每张训练图像中各个目标框的位置（bbox_xmin, bbox_ymin, bbox_width, bbox_height） 
- train.zip文件（3422张训练集图像，部分无小麦头），如下图所示：

![train.png](https://github.com/yearing1017/Global-Wheat-Detection/blob/master/image/train.png)

- test.zip文件（10张测试图像)，如下图所示：

![test.png](https://github.com/yearing1017/Global-Wheat-Detection/blob/master/image/test.png)


## 解决方案

### 方案1：[基于Faster-RCNN的识别](https://github.com/yearing1017/Global-Wheat-Detection/tree/master/Faster-RCNN-%231)

#### 1. 代码简介
- csv_remake.py：根据csv文件并返回train_df和valid_df
- WheatData.py：通过df文件载入image和targets，目标检测的数据集载入可参考issue
- evaluate.py：评估计算Iou和Map等指标
- train.py：训练代码
- WheaTesttData.py：载入测试数据
- predict.py：加载训练模型进行预测，并保存画出矩形框的图像

#### 2. 预测结果及问题
- 选取三种进行可视化描画矩形box，如下截图：

![](https://github.com/yearing1017/Global-Wheat-Detection/blob/master/Faster-RCNN-%231/predict_frc_0629/pre_1.jpg)

- 发现问题：在服务器上预测结果如上图所示，图像颜色与原图不一致，且预测结果较差；
- 在kaggle上预测的图像颜色正常，且相同模型预测结果较好，如下图所示：

![](https://github.com/yearing1017/Global-Wheat-Detection/blob/master/Faster-RCNN-%231/predict_frc_0629/pre-2.png)

- 问题：在自己服务器上运行相同程序，提示`Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).`
- 目前未解决该问题，但在kaggle上运行正确
- 得分：LB=0.6687

### 方案2：[基于Pseudo-Labeling策略训练Faster-RCNN](https://github.com/yearing1017/Global-Wheat-Detection/tree/master/Faster-RCNN-PL-%232)

#### 1. 思路
- 在方案一训练得到的模型基础上继续训练
- Pseudo-Labeling策略：先利用已有模型得出测试集的predict结果
- 筛选出置信度较高的pre结果和对应的image数据
- 将上步骤得到的数据加入到训练数据集中，进行再度训练

#### 2. 修改代码
- csv_remake.py：concat新数据和之前的train数据，并返回train_df和valid_df文件
- make_pl.py：根据已有模型得到Pseudo-Labeling数据
- WheaTesttData_df.py：以samplesubmission.csv格式载入测试数据

#### 3. 结果
- 在方案一的模型基础上训练6个轮次
- LB从0.6687提升到0.6914

### 方案3：[基于EfficientDet的识别](https://github.com/yearing1017/Global-Wheat-Detection/tree/master/EfficientDet-%233)

- 论文地址：[EfﬁcientDet: Scalable and Efﬁcient Object Detection](https://arxiv.org/pdf/1911.09070.pdf)

- Public LeaderBoard Score: 0.7506  |  106/1815  |  top 6%
![pb_score.png](https://github.com/yearing1017/Global-Wheat-Detection/blob/master/image/pb.png)

- Private LeaderBoard Score: 0.6681 |   22/2245  |  top 1%
![prb_score.png](https://github.com/yearing1017/Global-Wheat-Detection/blob/master/image/prb.png)

 
- 进行了探索性数据分析，发现图片存在下面三个问题：
    - 训练集与测试集图片分布差异大。训练集是来自欧洲和澳洲的小麦图片，而测试集是来自于日本的小麦图片，小麦的颜色，形状，大小存在很大的差异。
    - 小麦存在很多的重叠区域，这就导致检测框过程中位于重叠部分后方的小麦很难被检测出来。
    - 由于拍摄时风的扰动，小麦整体成像质量不高。
-  针对分布差异大的问题，想到了两个解决方法：
    - 第一让模型在训练过程中每一个batch能够训练多个来自不同地区的小麦，这样模型学到的更具有泛化性
    - 第二个是在提交阶段使用伪标签的方式，将置信度高于0.9的预测框作为真实框与测试集一起放入训练集中重新进行训练，这样模型就可以学习到来自于日本的图片。
-  针对重叠部分，我使用的是mixup数据增强，将来自于同一区域小麦图片融合，同时将预测框堆叠，这样能产生更多接近于真实的重叠图片，同时在后处理时使用的是soft-nms进行处理，能够减少重叠预测框的丢失。针对扰动问题，使用albumentation数据增强方式进行数据增强。
-  最后确定在yolov5和efficientdet模型，但是由于licence的问题，yolov5在比赛中不能使用，于是选择了efficientdetd6作为最终模型
-  在产生数据集阶段利用albumentation库进行heavy 数据增强，并使用cutmix融合更多来自不同地区的小麦照片，并用mixup产生更多重叠照片。
-  在训练阶段使用交叉验证的方法将数据集分成五折，在adms优化器在余弦退火的学习率更新方式下进行训练后，得到的模型使用soft-nms进行后处理。
-  在提交阶段，使用模型对测试集进行测试，将预测框大于0.9的添加上伪标签放到训练集中重新进行训练，训练后得到最终的模型，将模型对原图像，水平翻转，竖直翻转，旋转90度的图片分别进行检测，最后将多个检测框利用weighted box fusion的方式进行融合得到最终的预测框。