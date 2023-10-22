import torch
import argparse
import random
import numpy as np
import os
import torchnet as tnt
from torch.utils.data import DataLoader
from Utils.Tools import Normalization, conf_m, save_img2
from Utils.Dataset import TestDataset
from Utils.Model_Select import model_select


def eval(arg, device):
    # 选择模型、优化器、loss
    model = model_select(arg, device)
    model.load_state_dict(torch.load(arg.bestmodel))

    # 数据集加载
    testdata = TestDataset(arg.data_path, 'Test')
    test_data_size = len(testdata)
    print("验证集长度为：{}".format(test_data_size))
    testloader = DataLoader(testdata, batch_size=1)

    # 混淆矩阵
    test_confusion_matrix = tnt.meter.ConfusionMeter(arg.out_channels, normalized=False)

    # 验证
    model.eval()
    with torch.no_grad():
        for data in testloader:
            imgs, targets, name = data['img'], data['label'], data['name']
            imgs = imgs.to(device)
            targets = targets.to(device)

            outputs = model(imgs)
            save_img2(outputs, name)

            output_conf, target_conf = conf_m(outputs, targets)
            test_confusion_matrix.add(output_conf, target_conf)

    print(test_confusion_matrix.value())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", default='HRNet3')
    parser.add_argument("--out_channels", default=2)
    parser.add_argument("--data_path", default='H:/zhuchuanhai/PCD_Single/Data/SN7/')
    parser.add_argument("--bestmodel", default='H:/zhuchuanhai/CD_Seg/Out/SN7_All_HRNet3/Model/Model_27.pth')

    arg = parser.parse_args()
    device = torch.device('cuda:0')
    eval(arg, device)


if __name__ == '__main__':
    main()
