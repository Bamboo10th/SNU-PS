import os
import sys
import cv2
import numpy as np
from skimage import io


# 读取原始文件夹所有文件名及路径————影像
def file_name(file_dir):
    L1 = []

    for root, dirs, files in os.walk(file_dir):
        for file in files:
            if os.path.splitext(file)[1] == '.tif':
                L1.append(file)

    return L1


def pcc_change(pre1, pre2):
    pre1_2v = (pre1 > 0.5) * 1
    pre1_2v = pre1_2v.astype(np.float32)
    pre2_2v = (pre2 > 0.5) * 1
    pre2_2v = pre2_2v.astype(np.float32)
    change_img = pre2_2v - pre1_2v
    change_img = (change_img != 0) * 255
    change_img = change_img.astype(np.uint8)
    return change_img


# 获取首尾影像
def get_data(img_name, read_path, save_path):
    pre1 = io.imread(os.path.join(read_path, 'T1_HRNet_Pre', img_name))
    pre2 = io.imread(os.path.join(read_path, 'T2_HRNet_Pre', img_name))

    if pre1 is None:
        print('error33:', os.path.join(read_path, 'T1_HRNet_Pre', img_name))
        sys.exit(1)
    if pre2 is None:
        print('error44:', os.path.join(read_path, 'T2_HRNet_Pre', img_name))
        sys.exit(1)

    p_change = pcc_change(pre2, pre1)

    cv2.imwrite(os.path.join(save_path, 'HRNet_PCC_Pre', img_name), p_change)

    print('已保存')


def main(name_list, rp, sp):
    for i in range(len(name_list)):
        get_data(name_list[i], rp, sp)


# 读取所有含img文件夹的文件夹路径
img_file_path_list = file_name(r"H:\zhuchuanhai\PCD_SN7\Data\SN7\Test\T1_HRNet_Pre")

print(len(img_file_path_list))

main(img_file_path_list, r'H:\zhuchuanhai\PCD_SN7\Data\SN7\Test', r'H:\zhuchuanhai\PCD_SN7\Data\SN7\Test')