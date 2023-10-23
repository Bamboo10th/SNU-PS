from skimage import io
import numpy as np
import cv2
import os


# 读取文件下所有tif影像
def file_name(file_dir):
    L1 = []

    for root, dirs, files in os.walk(file_dir):
        for file in files:
            if os.path.splitext(file)[1] == '.tif':
                L1.append(file)

    return L1


def HRNet_CVAPS(pre1_path, pre2_path, label_path, threshold, save_index=None):
    pre1 = io.imread(pre1_path)
    pre2 = io.imread(pre2_path)
    label = io.imread(label_path)
    label = (label == 255) * 1

    pre11 = np.zeros((128, 128, 2), dtype=np.float32)
    pre11[:, :, 0] = 1 - pre1.astype(np.float32)
    pre11[:, :, 1] = pre1.astype(np.float32)

    pre22 = np.zeros((128, 128, 2), dtype=np.float32)
    pre22[:, :, 0] = 1 - pre2.astype(np.float32)
    pre22[:, :, 1] = pre2.astype(np.float32)

    deltaP = pre11 - pre22
    Pnorm = np.sqrt(np.sum(deltaP * deltaP, axis=2))

    Changed = (Pnorm >= threshold) * 1

    if save_index == 'Save':
        Changed = Changed.astype(np.uint8)
        Changed = Changed * 255

        save_path = pre1_path.replace('T1_HRNet_Pre', 'HRNet_CVAPS')
        cv2.imwrite(save_path, Changed)
        print('已保存')
    else:
        return Changed, label


# ------------------------------------------------------------------

img_path = r"H:\zhuchuanhai\PCD_SN7\Data\SN7\Test\T1_HRNet_Pre"
name_list = file_name(img_path)
n = len(name_list)

for i in range(n):
    t1_path = os.path.join(img_path, name_list[i])
    t2_path = t1_path.replace('T1_HRNet_Pre', 'T2_HRNet_Pre')
    l_path = t1_path.replace('T1_HRNet_Pre', 'Change_Label')

    HRNet_CVAPS(j, j.replace('T1_HRNet_Pre', 'T2_HRNet_Pre'), 1.4000000000000001)
