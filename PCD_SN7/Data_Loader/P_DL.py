import os
import torch
import numpy as np
from skimage import io
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


# --------------------用于读取变化检测中直接检测的数据集-------------------------

# 读取数据list，获取图像名
def Read_files(file_path):
    with open(file_path, "r") as f:
        files = []
        for line in f.readlines():
            line = line.strip('\n')  # 去掉列表中每一个元素的换行符
            files.append(line)
    return files


# numpy 转化为 tensor
def numpy_to_tensor(img_name, f_path, dataset_id):
    img1 = io.imread(os.path.join(f_path, dataset_id, 'T1_HRNet_Pre', img_name))
    img2 = io.imread(os.path.join(f_path, dataset_id, 'T2_HRNet_Pre', img_name))
    label = io.imread(os.path.join(f_path, dataset_id, 'Change_Label', img_name))
    save_path = os.path.join(f_path, dataset_id, 'Change_Label', img_name)

    if np.max(label) == 255:
        label = (label == 255) * 1

    img1 = img1[np.newaxis, :]
    img2 = img2[np.newaxis, :]

    imgA = torch.from_numpy(img1.copy()).float()
    imgB = torch.from_numpy(img2.copy()).float()
    label = torch.from_numpy(label.copy()).float()

    sample = {'imgA': imgA, 'imgB': imgB, 'label': label, 'path': save_path}
    return sample


# 读取数据
class CD_Dataset(Dataset):
    def __init__(self, img_list_dir, files_dir, dataset_id):
        super().__init__()
        self.img_names = Read_files(img_list_dir)
        self.files_dir = files_dir
        self.dataset_id = dataset_id

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, index):
        sample = numpy_to_tensor(self.img_names[index], self.files_dir, self.dataset_id)
        return sample


# 加载数据
def get_train_loaders_p(opt, dataset_id):
    if dataset_id == 'Train':
        dataset = CD_Dataset(opt.train_list_dir, opt.files_dir, dataset_id)
        data_size = len(dataset)
        print("数据集的长度为：{}".format(data_size))
        dataset_loader = DataLoader(dataset, batch_size=opt.batch_size,
                                    shuffle=True, num_workers=opt.num_workers, pin_memory=True)
    elif dataset_id == 'Val':
        dataset = CD_Dataset(opt.val_list_dir, opt.files_dir, dataset_id)
        data_size = len(dataset)
        print("数据集的长度为：{}".format(data_size))
        dataset_loader = DataLoader(dataset, batch_size=1, shuffle=False,
                                    num_workers=opt.num_workers, pin_memory=True)
    else:
        dataset = CD_Dataset(opt.test_list_dir, opt.files_dir, dataset_id)
        data_size = len(dataset)
        print("数据集的长度为：{}".format(data_size))
        dataset_loader = DataLoader(dataset, batch_size=1, shuffle=False,
                                    num_workers=opt.num_workers, pin_memory=True)
    return dataset_loader
