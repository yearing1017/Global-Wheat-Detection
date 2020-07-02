import os 
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import PIL.Image
import cv2
import numpy as np

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2


# 数据操作
def get_test_transform():
    return A.Compose([
        ToTensorV2(p=1.0)  #Convert image and mask to torch.Tensor.
    ])

class TestDataset(Dataset):

    def __init__(self, images_dir, transforms = None):
        self.transforms = transforms

    def __len__(self):
        return len(os.listdir('data/test'))

    def __getitem__(self, idx):
        image_name = os.listdir('data/test')[idx]
        image = cv2.imread('data/test/' + image_name, cv2.IMREAD_COLOR)
        #image = image.astype(np.float32)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0  # ToTensorV2：仅仅转为torch.tensor,所以手动/255


        if self.transforms:
            sample = {
                'image': image,
            }
            sample = self.transforms(**sample)
            image = sample['image']

    
        return  image_name, image

def collate_fn(batch):
    return tuple(zip(*batch))

DIR_TEST = 'data/test'
test_dataset = TestDataset(DIR_TEST, get_test_transform())

test_dataloader = DataLoader(test_dataset, batch_size=2,shuffle=False, num_workers=2, collate_fn=collate_fn)

if __name__ == "__main__":
    for index, (image_name,image) in enumerate(test_dataloader):
        print(image.shape)
        print(image_name)
        print('===========')
    
