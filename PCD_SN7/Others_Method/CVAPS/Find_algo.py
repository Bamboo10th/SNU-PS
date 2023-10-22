
from skimage import io
import numpy as np
import os


# 读取文件下所有tif影像
def file_name(file_dir):
    L1 = []

    for root, dirs, files in os.walk(file_dir):
        for file in files:
            if os.path.splitext(file)[1] == '.tif':
                L1.append(file)

    return L1


def T1_T2_Diff(pre1_path, pre2_path, label_path):
    pre1 = io.imread(pre1_path)
    pre2 = io.imread(pre2_path)
    label = io.imread(label_path)

    pre11 = np.zeros((128, 128, 2), dtype=np.float32)
    pre11[:, :, 0] = 1 - pre1.astype(np.float32)
    pre11[:, :, 1] = pre1.astype(np.float32)

    pre22 = np.zeros((128, 128, 2), dtype=np.float32)
    pre22[:, :, 0] = 1 - pre2.astype(np.float32)
    pre22[:, :, 1] = pre2.astype(np.float32)

    deltaP = pre11 - pre22

    Pnorm = np.sqrt(np.sum(deltaP * deltaP, axis=2))
    Pnorm2 = Pnorm * (label == 255)

    return Pnorm2


# 基于直方图熵的无监督阈值确定方法查找灰度级图像的阈值
def KSW_algo(data):
    num = data.shape[0] * data.shape[1]
    minvalue = np.min(data)
    maxvalue = np.max(data)
    bin = 0.0001

    x_hit, _ = np.histogram(data, bins=np.arange(minvalue, maxvalue + bin, bin))
    x_prob = x_hit / num
    maxcyc = round(maxvalue / bin)
    Entrope = -x_prob * np.log(x_prob + (x_prob == 0))
    H = np.sum(Entrope)
    H_value_all = []

    for knum in range(4, maxcyc):
        Pt = np.count_nonzero(data <= (knum * bin)) / (num * num)
        Ht = np.sum(Entrope[0:knum])
        H_value = np.log(Pt * (1 - Pt)) + Ht / Pt + (H - Ht) / (1 - Pt)
        H_value_all.append(H_value)

    index = H_value_all.index(max(H_value_all))
    kk = index + 4
    threshold = kk * bin
    print(threshold)


img_path = r"H:\zhuchuanhai\PCD_SN7\Data\SN7\Test\T1_HRNet_Pre"
name_list = file_name(img_path)
dif_all = np.zeros((len(name_list), 128, 128), dtype=np.float32)
for i in range(len(name_list)):
    j = os.path.join(img_path, name_list[i])
    dif = T1_T2_Diff(j, j.replace('T1_HRNet_Pre', 'T2_HRNet_Pre'), j.replace('T1_HRNet_Pre', 'Change_Label'))
    dif_all[i] = dif

dif_all2 = np.reshape(dif_all, (-1, 128))
KSW_algo(dif_all2)
