from skimage import io
import numpy as np
import pickle
import os
from sklearn.svm import SVC
import random
from sklearn.metrics import confusion_matrix


# 读取文件下所有tif影像
def file_name(file_dir):
    L1 = []

    for root, dirs, files in os.walk(file_dir):
        for file in files:
            if os.path.splitext(file)[1] == '.tif':
                L1.append(file)

    return L1


def evaluate_segmentation(segmentation, ground_truth):
    cm = confusion_matrix(ground_truth, segmentation)
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


def get_img_label(img_path):
    name_list = file_name(img_path)
    n = len(name_list)
    img_all = np.zeros((n, 128, 128, 3))
    label_all = np.zeros((n, 128, 128))
    for i in range(n):
        img_path0 = os.path.join(img_path, name_list[i])
        img0 = io.imread(img_path0)
        label0 = io.imread(img_path0.replace('T2', 'T2_Label'))
        img_all[i] = img0
        label_all[i] = label0
    img_all2 = img_all.reshape(-1, 3)
    label_all2 = label_all.reshape(-1)
    label_all2 = (label_all2 != 0) * 1
    return img_all2, label_all2


def sample_img(label, snumber):
    nc_list = []
    c_list = []
    for i in range(len(label)):
        if label[i] == 0:
            nc_list.append(i)
        else:
            c_list.append(i)
    index_nc = random.sample(nc_list, k=snumber)
    index_c = random.sample(c_list, k=snumber)
    index_nc.extend(index_c)
    return index_nc


def extract_elements_by_index(arr, indices):
    extracted_arr = [arr[i] for i in indices]
    return extracted_arr


x_train, y_train = get_img_label(r'H:\zhuchuanhai\PCD_SN7\Data\SN7\Train\T2')
# x_val, y_val = get_img_label(r'H:\zhuchuanhai\PCD_SN7\Data\SN7\Val\T2')
print('数据准备完毕')
index = sample_img(y_train, 3000)
print(len(index))
x_train2 = extract_elements_by_index(x_train, index)
y_train2 = extract_elements_by_index(y_train, index)

# 创建SVM分类器
svm = SVC(kernel='linear')
print('开始训练')
# 训练SVM分类器
svm.fit(x_train2, y_train2)
print('训练完成')

# 对每个像素进行分类
print('开始预测')
predicted_labels = svm.predict(x_train)
print('预测完成')
evaluate_segmentation(predicted_labels, y_train)

with open(r'H:\zhuchuanhai\PCD_SN7\Others_Method\SVM_PCC\Out\svm.pickle', 'wb') as f:
    pickle.dump(svm, f)
