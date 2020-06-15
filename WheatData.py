import os 
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import cv2

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

from csv_remake import train_df,valid_df

class WheatDataset(Dataset):

    def __init__(self, dataframe, image_dir, transforms=None):
        super.__init__()
        self.image_ids = dataframe['image_id'].unique()
        self.df = dataframe
        self.image_dir = image_dir
        self.transforms = transforms
    
    def __getitem__(self, index):
        # 根据index在不重复的ids中找到相应图片的id
        image_id = self.image_ids[index]
        # 查出该图片的信息，records是符合条件的df
        records = self.df[self.df['image_id'] == image_id]
        # cv2读取图像 转为RGB，/255进行归一化
        image = cv2.imread(f'{self.image_dir}/{image_id}.jpg', cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0  # ToTensorV2：仅仅转为torch.tensor,所以手动/255

        # boxes是df提取的四列的值
        '''
        [[834. 222.  56.  36.]
         [226. 548. 130.  58.]]
        '''
        boxes = records[['bbox_xmin', 'bbox_ymin', 'bbox_width', 'bbox_height']].values
        area = boxes[:, 3] * boxes[:, 2]
        area = torch.as_tensor(area, dtype=torch.float32)
        # 将2、3列变为x_max、y_max坐标值
        boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
        boxes[:, 3] = boxes[:, 1] + boxes[:, 3]

        # 维数为1  有records.shape[0]个元素  猜测这里的全1的labels指的是1代表小麦头类别
        labels = torch.ones((records.shape[0],), dtype=torch.int64)
        # 
        iscrowd = torch.zeros((records.shape[0],), dtype=torch.int64)

        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        target['image_id'] = torch.tensor([index])
        target['area'] = area
        target['iscrowd'] = iscrowd

        if self.transforms:
            sample = {
                'image': image,
                'bboxes': target['boxes'],
                'labels': labels
            }
        # test(**kwargs)：** 的作用则是把字典 kwargs 变成关键字参数传递。
        sample = self.transforms(**sample)
        image = sample['image']
        # 对于target['boxes']也许可以使用下句实现
        target['boxes'] = torch.tensor(sample['bboxes'])

        # zip：将bboxes中的四个坐标值各自合并[(所有的x_min),(所有的y_min),(),()]
        # map: 将函数应用到后面的坐标值上
        # tuple: 将列表转换为元组  ((所有的x_min),(所有的y_min),(),())
        # torch.stack: 
        # permute(1,0)：将1维和0维置换
        #target['boxes'] = torch.stack(tuple(map(torch.tensor, zip(*sample['bboxes'])))).permute(1, 0)

        return image, target, image_id
    
    def  __len__(self):
        return self.image_ids.shape[0]


# Albumentations 开源的数据增强库
# bbox_params若为pascal_voc数据集格式，则要求boxes信息为[x_min, y_min, x_max, y_max]
# label_fields ：指的是该图中含有的类别，多少框 对应 多少类别
def get_train_transform():
    return A.Compose([
        A.Flip(0.5),
        ToTensorV2(p=1.0)
    ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})

def get_valid_transform():
    return A.Compose([
        ToTensorV2(p=1.0)  #Convert image and mask to torch.Tensor.
    ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})


def collate_fn(batch):
    return tuple(zip(*batch))

train_dataset = WheatDataset(train_df, DIR_TRAIN, get_train_transform())
valid_dataset = WheatDataset(valid_df, DIR_TRAIN, get_valid_transform())


# split the dataset in train and test set
indices = torch.randperm(len(train_dataset)).tolist()

train_data_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=4,
    collate_fn=collate_fn # 自定义函数来设置取样本的方式
)

valid_data_loader = DataLoader(
    valid_dataset,
    batch_size=8,
    shuffle=False,
    num_workers=4,
    collate_fn=collate_fn
)
