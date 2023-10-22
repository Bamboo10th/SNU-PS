from skimage import io
import numpy as np
import os
import cv2
from sklearn.metrics import confusion_matrix


# 读取数据list，获取图像名
def Read_files(file_path):
    with open(file_path, "r") as f:
        files = []
        for line in f.readlines():
            line = line.strip('\n')  # 去掉列表中每一个元素的换行符
            files.append(line)
    return files


# 读取文件下所有tif影像
def file_name(file_dir):
    L1 = []

    for root, dirs, files in os.walk(file_dir):
        for file in files:
            if os.path.splitext(file)[1] == '.tif':
                L1.append(file)

    return L1


# 生成变化图、标签
def HRNet_CVAPS(pre1_path, pre2_path, label_path, threshold):
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

    return Changed, label


# 计算精度
def Precision(cm):
    # 混淆矩阵
    tn, fn, fp, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
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
    return f1


# 计算精度
def Precision2(cm):
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


# train集寻找阈值
def train_find(read_path, name_list):
    print("-----------开始查找--------")
    n = len(name_list)
    # 初始参数
    a = 0.1
    b = 2
    c_all = []
    f1_all = []
    for i in range(200):
        c = a + i * 0.01
        if c <= b:
            print('阈值为：', c)
            change_all = np.zeros((n, 128, 128), dtype=np.uint8)
            label_all = np.zeros((n, 128, 128), dtype=np.uint8)
            for j in range(n):
                t1_path = os.path.join(read_path, name_list[j])
                t2_path = t1_path.replace('T1_HRNet_Pre', 'T2_HRNet_Pre')
                l_path = t1_path.replace('T1_HRNet_Pre', 'Change_Label')
                change0, label0 = HRNet_CVAPS(t1_path, t2_path, l_path, c)
                change_all[j] = change0
                label_all[j] = label0
            change_np = change_all.flatten()
            label_np = label_all.flatten()
            cm0 = confusion_matrix(label_np, change_np)
            f10 = Precision(cm0)
            print('f1为：', f10)
            c_all.append(c)
            f1_all.append(f10)

    max_index = f1_all.index(max(f1_all))
    print("-----------查找结束--------")
    print("f1最大值对应阈值：", c_all[max_index])
    print("f1最大值：", f1_all[max_index])

    return c_all[max_index]


# 测试，精度评价
def test_find(read_path, name_list, max_c, save_index):
    print("-----------精度评价--------")
    n = len(name_list)
    change_all = np.zeros((n, 128, 128), dtype=np.uint8)
    label_all = np.zeros((n, 128, 128), dtype=np.uint8)
    for j in range(n):
        t1_path = os.path.join(read_path, name_list[j])
        t2_path = t1_path.replace('T1_HRNet_Pre', 'T2_HRNet_Pre')
        l_path = t1_path.replace('T1_HRNet_Pre', 'Change_Label')
        change0, label0 = HRNet_CVAPS(t1_path, t2_path, l_path, max_c)

        if save_index == 'Save':
            Changed = change0.astype(np.uint8)
            Changed = Changed * 255
            save_path = t1_path.replace('T1_HRNet_Pre', 'HRNet_CVAPS_Pre')
            cv2.imwrite(save_path, Changed)
            print('已保存')
        else:
            change_all[j] = change0
            label_all[j] = label0

    change_np = change_all.flatten()
    label_np = label_all.flatten()
    cm0 = confusion_matrix(label_np, change_np)
    Precision2(cm0)


# ---------------------------------主程序-------------------------------------
# 数据集信息，读取数据
read_train_path = r"H:\zhuchuanhai\PCD_SN7\Data\SN7\Train\T1_HRNet_Pre"
name_list_train = Read_files(r'H:\zhuchuanhai\PCD_SN7\Data\List\D010\Train_List1.txt')
max_c0 = train_find(read_train_path, name_list_train)

read_test_path = r"H:\zhuchuanhai\PCD_SN7\Data\SN7\Test\T1_HRNet_Pre"
name_list_test = file_name(read_test_path)
test_find(read_test_path, name_list_test, max_c0, save_index='Save')

