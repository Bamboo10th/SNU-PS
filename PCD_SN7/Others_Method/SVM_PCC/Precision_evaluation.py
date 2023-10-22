import os
from skimage import io
import numpy as np
from sklearn.metrics import confusion_matrix


# 计算精度
def Precision(cm):
    # 混淆矩阵
    cm = cm.astype(np.float64)
    n_all = np.sum(cm)
    tn, fn, fp, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
    # 精度oa
    oa = (tn + tp) / n_all
    # 精确度 p
    if fp + tp == 0:
        p = 0
    else:
        p = tp / (fp + tp)
    # 召回率 r
    if fn + tp == 0:
        r = 0
    else:
        r = tp / (fn + tp)
    # f1分数
    if r + p == 0:
        f1 = 0
    else:
        f1 = 2 * r * p / (r + p)
    # 交并比 iou
    if fp + tp + fn == 0:
        iou = 0
    else:
        iou = tp / (fp + tp + fn)
    # kappa系数
    po = oa
    pe = ((tn + fn) * (tn + fp) + (fp + tp) * (fn + tp)) / (n_all * n_all)
    kappa = (po - pe) / (1 - pe)

    print("oa:", oa)
    print("p:", p)
    print("r:", r)
    print("f1:", f1)
    print("iou:", iou)
    print("kappa:", kappa)


def get_img(img_list, read_path1, read_path2):
    n = len(img_list)
    img_np = np.zeros((n, 128, 128), dtype=np.uint8)
    label_np = np.zeros((n, 128, 128), dtype=np.uint8)
    for i in range(n):
        img = io.imread(os.path.join(read_path1, img_list[i]))
        label = io.imread(os.path.join(read_path2, img_list[i]))
        img_np[i] = img
        label_np[i] = label
    img_np = img_np.flatten()
    label_np = label_np.flatten()
    return img_np, label_np


# 读取文件下所有tif影像,全路径
def file_name(file_dir):
    L1 = []

    for root, dirs, files in os.walk(file_dir):
        for file in files:
            if os.path.splitext(file)[1] == '.tif':
                L1.append(file)

    return L1


img_name_list = file_name(r"H:\zhuchuanhai\PCD_SN7\Data\SN7\Test\SVM_PCC_Pre")
img_np, label_np = get_img(img_name_list, r"H:\zhuchuanhai\PCD_SN7\Data\SN7\Test\SVM_PCC_Pre",
                           r"H:\zhuchuanhai\PCD_SN7\Data\SN7\Test\Change_Label")

C = confusion_matrix(label_np, img_np)
Precision(C)