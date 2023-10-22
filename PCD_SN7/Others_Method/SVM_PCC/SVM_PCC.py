import pickle
from skimage import io
import numpy as np
import pickle
import os
import cv2


# 读取文件下所有tif影像
def file_name(file_dir):
    L1 = []

    for root, dirs, files in os.walk(file_dir):
        for file in files:
            if os.path.splitext(file)[1] == '.tif':
                L1.append(file)

    return L1


def get_img_label(img_path, svm):
    name_list = file_name(img_path)
    n = len(name_list)
    for i in range(n):
        diff128_128 = np.zeros((128, 128))
        img_path1 = os.path.join(img_path, name_list[i])
        img1 = io.imread(img_path1)
        img2 = io.imread(img_path1.replace('T1', 'T2'))
        img11 = img1.reshape(-1, 3)
        img22 = img2.reshape(-1, 3)
        predicted1 = svm.predict(img11)
        predicted2 = svm.predict(img22)
        diff = predicted1.astype(np.float32) - predicted2.astype(np.float32)
        diff = (diff != 0) * 255
        diff = diff.astype(np.uint8)
        for s in range(128):
            for q in range(128):
                diff128_128[s, q] = diff[s * 128 + q]
        save_path = img_path1.replace('T1', 'SVM_PCC')

        cv2.imwrite(save_path, diff128_128.astype(np.uint8))
        print('已保存')


with open(r'H:\zhuchuanhai\PCD_SN7\Others_Method\SVM_PCC\Out\svm.pickle', 'rb') as fr:
    SVM = pickle.load(fr)
print('模型已加载')
get_img_label(r'H:\zhuchuanhai\PCD_SN7\Data\SN7\Test\T1', SVM)
