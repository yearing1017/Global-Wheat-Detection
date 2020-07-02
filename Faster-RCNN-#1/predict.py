import os 
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import PIL.Image
import cv2
import numpy as np

import torchvision

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator

from WheatTestData import test_dataloader

from WheatData import valid_data_loader

import matplotlib.pyplot as plt
from matplotlib.pyplot import plot,savefig

def predict():
    # 准备网络
    # load a model; pre-trained on COCO（以下4句为pytorch官方教程例子）
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    num_classes = 2  # 1 class (wheat) + background
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # 指定gpu
    device = torch.device('cuda:1') if torch.cuda.is_available() else torch.device('cpu')

    model.load_state_dict(torch.load("models_frc_0629/frc_0629_55.pth"))

    model = model.to(device)
    print('模型载入成功')
    save_dir = 'predict_frc_0629/'
    
    
    model.eval()
    with torch.no_grad():
        for image_name_batch, test_images in test_dataloader:
            images = list(image.to(device) for image in test_images)
            outputs = model(images)
            #print(image_name)
            for i, image in enumerate(images):
                detection_threshold = 0.5
                # (3,800,800)
                #sample = image.cpu().numpy()
                sample = image.permute(1,2,0).cpu().numpy()
                #image = image.cpu().numpy()
                boxes = outputs[i]['boxes'].data.cpu().numpy()
                scores = outputs[i]['scores'].data.cpu().numpy()

                boxes = boxes[scores >= detection_threshold].astype(np.int32)


                fig, ax = plt.subplots(1, 1, figsize=(8, 8))

                for box in boxes:
                    cv2.rectangle(sample,
                                (box[0], box[1]),
                                (box[2], box[3]),
                                (220, 0, 0), 2)
                ax.set_axis_off()
                ax.imshow(sample)
                plt.savefig(save_dir + image_name_batch[i])
                #cv2.imwrite(save_dir + image_name_batch[i], image)
                print(image_name_batch[i] + ': 已保存')
    
if __name__ == "__main__":
    predict()
