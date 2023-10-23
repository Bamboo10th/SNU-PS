import cv2
import os
import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np
import tifffile as tf


def conf_m(output, target_th):
    output_softmax = F.softmax(output, dim=1)
    output_conf = ((output_softmax.data).transpose(1, 3)).transpose(1, 2)
    output_conf = (output_conf.contiguous()).view(output_conf.size(0) * output_conf.size(1) * output_conf.size(2),
                                                  output_conf.size(3))
    target_conf = target_th.data
    target_conf = (target_conf.contiguous()).view(target_conf.size(0) * target_conf.size(1) * target_conf.size(2))
    return output_conf, target_conf


# 储存为img
def save_img(output, img_ids):
    # 输出层
    output_softmax = F.softmax(output, dim=1)
    output_np = output_softmax.cpu().detach().numpy()
    # 建筑物归属概率
    pre = output_np[:, 1, :, :].reshape(128, 128)

    # G:\\Zhuchuanhai\\Attribution_probability_CD\\Data\\SN7\\Train\\T1\\L15-0331E-1257N_1327_3160_13_0.tif
    if '\\T1' in img_ids[0]:

        file_name = img_ids[0].split('\\T1')[0]
        img_name = os.path.basename(img_ids[0])

        pre_file_name = os.path.join(file_name, 'T1_HRNet3_All_Pre')

        if not os.path.exists(pre_file_name):
            os.mkdir(pre_file_name)

        cv2.imwrite(os.path.join(pre_file_name, img_name), pre)
        print("已保存")

    if '\\T2' in img_ids[0]:

        file_name = img_ids[0].split('\\T2')[0]
        img_name = os.path.basename(img_ids[0])

        pre_file_name = os.path.join(file_name, 'T2_HRNet3_All_Pre')

        if not os.path.exists(pre_file_name):
            os.mkdir(pre_file_name)

        cv2.imwrite(os.path.join(pre_file_name, img_name), pre)
        print("已保存")


# 储存为img
def save_img2(output, img_ids):
    # 输出层
    output_softmax = F.softmax(output, dim=1)
    output_np = output_softmax.cpu().detach().numpy()
    # 建筑物归属概率
    pre = output_np[0]
    pre = pre.transpose((1, 2, 0))

    # /chenlab/Seg/HRSCD/Test/T1/14-2012-0415-6890-LA93-0M50-E080_19.tif
    if '\\T1' in img_ids[0]:

        file_name = img_ids[0].split('\\T1')[0]
        img_name = os.path.basename(img_ids[0])

        pre_file_name = os.path.join(file_name, 'T1_HRNet_Pre')

        if not os.path.exists(pre_file_name):
            os.mkdir(pre_file_name)
        tf.imsave(os.path.join(pre_file_name, img_name), pre)
        print("已保存")

    if '\\T2' in img_ids[0]:

        file_name = img_ids[0].split('\\T2')[0]
        img_name = os.path.basename(img_ids[0])

        pre_file_name = os.path.join(file_name, 'T2_HRNet_Pre')

        if not os.path.exists(pre_file_name):
            os.mkdir(pre_file_name)

        tf.imsave(os.path.join(pre_file_name, img_name), pre)
        print("已保存")


# 数据标准化
class Normalization(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, img):
        img_mean = torch.mean(img, dim=(0, 2, 3), keepdim=True)
        img_mean = torch.squeeze(img_mean, dim=0)
        img_std = torch.std(img, dim=(0, 2, 3), keepdim=True)
        img_std = torch.squeeze(img_std, dim=0)

        img_n = img.sub_(img_mean).div_(img_std)

        return img_n


def index_calculation_f1(cm):
    tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]

    if fp + tp == 0:
        p = 0
    else:
        p = tp / (fp + tp)

    if fn + tp == 0:
        r = 0
    else:
        r = tp / (fn + tp)

    if r + p == 0:
        f1 = 0
    else:
        f1 = 2 * r * p / (r + p)
    return f1
