# 此文件用来操作给的train.csv，将bbox列的四个坐标分别扩展到csv的4️列
import pandas as pd
import numpy as np
import cv2
import os
import re

from make_pl import test_df_pseudo


def new_df():
    train_df = pd.read_csv('data/train.csv')
    #print(train_df.shape) #(147793, 5)

    #test_df = pd.read_csv('data/sample_submission.csv')

    # split方法划分bbox列
    bbox_items = train_df.bbox.str.split(',', expand=True)
    #print(bbox_items) # 0       [834.0   222.0    56.0    36.0]
    # strip函数 用来删除开头或结尾的字符
    train_df['bbox_xmin'] = bbox_items[0].str.strip('[ ').astype(float)
    train_df['bbox_ymin'] = bbox_items[1].str.strip(' ').astype(float)
    train_df['bbox_width'] = bbox_items[2].str.strip(' ').astype(float)
    train_df['bbox_height'] = bbox_items[3].str.strip(' ]').astype(float)

    # 删除bbox列
    train_df = train_df.drop(['bbox'],axis=1)
    #print(train_df)

    image_ids = train_df['image_id'].unique()
    #print(image_ids) #['b6ab77fd7' 'b53afdf5c' '7b72ea0fb' ... 'a5c8d5f5c' 'e6b5e296d' '5e0747034']
    #print(isinstance(image_ids, list)) # false 不是list
    #print(type(image_ids)) # <class 'numpy.ndarray'>

    # 划分验证集和训练集
    valid_ids = image_ids[-665:]
    #print(len(valid_ids)) # 665 从倒数第665个id到最后
    #train_ids = image_ids[:-665]
    train_ids = image_ids  # 新一轮的pl训练
    #print(len(train_ids)) # 2708 从开头到倒数第665个id

    # 根据id筛选符合条件的df
    valid_df = train_df[train_df['image_id'].isin(valid_ids)]
    #train_df = train_df[train_df['image_id'].isin(train_ids)]

    #print(valid_df.shape)  # (25006, 8)
    #print(train_df.shape)  # (122787, 8)

    # pl训练
    frames = [train_df, test_df_pseudo]
    train_df = pd.concat(frames)

    return train_df, valid_df

train_df, valid_df= new_df()

if __name__ == "__main__":
    train_df, valid_df = new_df()
    records = train_df[train_df['image_id'] == 'b6ab77fd7']
    #print(records) # 依然是一个df

    boxes = records[['bbox_xmin', 'bbox_ymin', 'bbox_width', 'bbox_height']].values
    print(boxes.shape) #(47, 4)
    # 将2、3列变为x_max、y_max坐标值
    boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
    boxes[:, 3] = boxes[:, 1] + boxes[:, 3]

    #labels = np.ones((records.shape[0]), dtype=np.int64)
    #print(labels.shape)
    new_boxes = zip(*boxes)
    print(tuple(new_boxes))
