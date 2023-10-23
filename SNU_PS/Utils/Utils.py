import json
import torch
import torch.nn as nn
import argparse as ag
import torch.nn.functional as F
import cv2
import numpy as np


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


# 数据处理，适合计算混淆矩阵
def conf_m(output, target_th):
    output_softmax = F.softmax(output, dim=1)
    output_conf = ((output_softmax.data).transpose(1, 3)).transpose(1, 2)
    output_conf = (output_conf.contiguous()).view(output_conf.size(0) * output_conf.size(1) * output_conf.size(2),
                                                  output_conf.size(3))
    target_conf = target_th.data
    target_conf = (target_conf.contiguous()).view(target_conf.size(0) * target_conf.size(1) * target_conf.size(2))
    return output_conf, target_conf


# 保存
def save_img(output, path, save_name):
    output_softmax = F.softmax(output, dim=1)
    output_np = output_softmax.cpu().detach().numpy()

    write_path = path.replace('Change_Label', save_name)

    pre = output_np[:, 1, :, :].reshape(128, 128)
    pre = (pre >= 0.5) * 255

    cv2.imwrite(write_path, pre.astype(np.uint8))
    print("已保存")


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
