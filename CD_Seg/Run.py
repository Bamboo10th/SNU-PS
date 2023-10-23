import torch
import time
import argparse
import random
import numpy as np
import os
import torchnet as tnt
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from Utils.Tools import Normalization, conf_m, index_calculation_f1
from Utils.Dataset_train import TrainDataset, TestDataset
from Utils.Model_Select import model_select, optimizer_select, scheduler_select, loss_select


def train(arg, device):
    # 选择模型、优化器、loss
    model = model_select(arg, device)
    model.load_state_dict(torch.load(arg.bestmodel))
    optimizer = optimizer_select(arg, model)
    scheduler = scheduler_select(arg, optimizer)
    loss_fn = loss_select(arg, device)

    # 数据集加载
    traindata = TrainDataset(arg.data_path, 'Train')
    train_data_size = len(traindata)
    print("训练集长度为：{}".format(train_data_size))
    trainloader = DataLoader(traindata, batch_size=arg.batch_size,
                             shuffle=True, num_workers=arg.num_workers, pin_memory=True)

    valdata = TestDataset(arg.data_path, 'Val')
    val_data_size = len(valdata)
    print("验证集长度为：{}".format(val_data_size))
    valloader = DataLoader(valdata, batch_size=arg.batch_size,
                           shuffle=False, num_workers=arg.num_workers, pin_memory=True)

    # 混淆矩阵
    writer = SummaryWriter(arg.log_dir)
    train_confusion_matrix = tnt.meter.ConfusionMeter(arg.out_channels, normalized=False)
    val_confusion_matrix = tnt.meter.ConfusionMeter(arg.out_channels, normalized=False)

    model.train()
    for i in range(0, arg.epochs):
        print("-------第 {} 轮训练开始-------".format(i + 1))
        start_time = time.time()
        train_batch_step = 0
        train_loss = torch.tensor(0).to(device)
        train_confusion_matrix.reset()
        val_batch_step = 0
        val_loss = 0
        val_confusion_matrix.reset()

        for data in trainloader:
            imgs, targets = data['img'], data['label']
            imgs = imgs.to(device)
            targets = targets.to(device)

            outputs = model(imgs)
            loss = loss_fn(outputs, targets.long())

            output_conf, target_conf = conf_m(outputs, targets)
            train_confusion_matrix.add(output_conf, target_conf)

            # 优化器优化模型
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss = train_loss + loss
            train_batch_step = train_batch_step + 1

            if train_batch_step % 2000 == 0:
                print("训练次数：{}, Loss: {}".format(train_batch_step, loss.item()))

        # 显示平均loss、学习率
        train_acc = (np.trace(train_confusion_matrix.conf) / float(np.ndarray.sum(train_confusion_matrix.conf)))
        train_f1 = index_calculation_f1(train_confusion_matrix.value())
        print("训练集平均的Loss: {},acc: {},f1: {},学习率Lr: {}".format(train_loss.item() / train_batch_step,
                                                               train_acc,
                                                               train_f1,
                                                               optimizer.state_dict()['param_groups'][0]['lr']))
        # scheduler.step(train_loss)
        scheduler.step()

        # 验证
        model.eval()
        with torch.no_grad():
            for data in valloader:
                imgs, targets = data['img'], data['label']
                imgs = imgs.to(device)
                targets = targets.to(device)

                outputs = model(imgs)
                loss = loss_fn(outputs, targets.long())

                output_conf, target_conf = conf_m(outputs, targets)
                val_confusion_matrix.add(output_conf, target_conf)

                val_loss = val_loss + loss
                val_batch_step = val_batch_step + 1

        # 显示平均loss、学习率
        val_acc = (np.trace(val_confusion_matrix.conf) / float(np.ndarray.sum(val_confusion_matrix.conf)))
        val_f1 = index_calculation_f1(val_confusion_matrix.value())
        print("验证集平均Loss: {}, acc: {}, f1: {}".format(val_loss / val_batch_step, val_acc, val_f1))
        # 画loss、acc曲线
        writer.add_scalars("Loss", {'Train': train_loss.item() / train_batch_step,
                                    'Val': val_loss.item() / val_batch_step}, i + 1)
        writer.add_scalars("Acc", {'Train': train_acc, 'Val': val_acc}, i + 1)
        writer.add_scalars("f1", {'Train': train_f1, 'Val': val_f1}, i + 1)

        # 保存
        # if (i + 1) % 2 == 0:
        #     # 保存模型
        torch.save(model.state_dict(), os.path.join(arg.model_save_path, 'Model_{}.pth'.format(i + 1)))
        print("模型已保存")

        # 计算一个epoch花费时间
        end_time = time.time()
        time_dif = end_time - start_time
        print("每个epoch的时间: {}".format(time_dif))
    writer.close()


def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def main():
    parser = argparse.ArgumentParser()
    # 模型名，选择模型
    parser.add_argument("--model_id", default='UNet_2Plus')
    # 选择loss
    parser.add_argument("--loss_function", default='CrossEntropyLoss')
    parser.add_argument("--optimizer", default='SGD')
    parser.add_argument("--scheduler", default='CosineAnnealingLR')
    parser.add_argument("--learning_rate", default=0.001)
    parser.add_argument("--epochs", default=200)
    parser.add_argument("--out_channels", default=2)
    parser.add_argument("--batch_size", default=6)
    parser.add_argument("--num_workers", default=4)
    # 数据路径名
    parser.add_argument("--data_path", default='xx')
    # 训练好的模型位置，训练时可忽略
    parser.add_argument("--bestmodel", default='')
    # 模型保存位置
    parser.add_argument("--model_save_path", default='xx')
    # 训练日志保存位置
    parser.add_argument("--log_dir", default='xx')

    arg = parser.parse_args()
    device = torch.device('cuda:0')
    seed_torch(seed=114514)
    train(arg, device)


if __name__ == '__main__':
    main()
