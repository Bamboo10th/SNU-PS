import torch
from torch.utils.data import Dataset
import cv2
import os
import numpy as np
import sys


# 读取文件下所有tif影像,全路径
def file_name(file_dir):
    L1 = []

    for root, dirs, files in os.walk(file_dir):
        for file in files:
            if os.path.splitext(file)[1] == '.tif':
                L1.append(file)

    return L1


# 获取所有文件
def Read_files(file_path, dataset_id):
    img_t1_lsit = file_name(os.path.join(file_path, dataset_id, 'T1'))
    img_list = []
    label_list = []
    for i in range(len(img_t1_lsit)):
        img_list.append(os.path.join(file_path, dataset_id, 'T1', img_t1_lsit[i]))
        img_list.append(os.path.join(file_path, dataset_id, 'T2', img_t1_lsit[i]))
        label_list.append(os.path.join(file_path, dataset_id, 'T1_Label', img_t1_lsit[i]))
        label_list.append(os.path.join(file_path, dataset_id, 'T2_Label', img_t1_lsit[i]))
    return img_list, label_list


class TrainDataset(Dataset):
    def __init__(self, file_dir, dataset_id):
        super().__init__()
        self.img_dir, self.label_dir = Read_files(file_dir, dataset_id)

    def __len__(self):
        return len(self.img_dir)

    def __getitem__(self, index):

        img = cv2.imread(self.img_dir[index], 1)
        label = cv2.imread(self.label_dir[index], 0)

        if img is None:
            print(self.img_dir[index])
            sys.exit(1)
        if label is None:
            print(self.label_dir[index])
            sys.exit(1)

        if np.max(label) == 255:
            label = (label == 255) * 1

        img = torch.from_numpy(img.transpose((2, 0, 1)).copy()).float()
        label = torch.from_numpy(label.copy()).float()
        sample = {'img': img, 'label': label}
        return sample


class TestDataset(Dataset):
    def __init__(self, file_dir, dataset_id):
        super().__init__()
        self.img_dir, self.label_dir = Read_files(file_dir, dataset_id)

    def __len__(self):
        return len(self.img_dir)

    def __getitem__(self, index):

        img = cv2.imread(self.img_dir[index], 1)
        label = cv2.imread(self.label_dir[index], 0)
        name = self.img_dir[index]

        if img is None:
            print(self.img_dir[index])
            sys.exit(1)
        if label is None:
            print(self.label_dir[index])
            sys.exit(1)

        if np.max(label) == 255:
            label = (label == 255) * 1

        img = torch.from_numpy(img.transpose((2, 0, 1)).copy()).float()
        label = torch.from_numpy(label.copy()).float()
        sample = {'img': img, 'label': label, 'name': name}
        return sample
