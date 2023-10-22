import os
import sys

sys.path.append("/chenlab/Seg_3090/")

import torch
import time
import argparse
import random
import numpy as np
import torchnet as tnt
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from HRNet import HighResolutionNet2
from Config import HRNET_48
from UNet import EfficientNet_Unet
from Utils import conf_m, index_calculation_f1types
from Dataset import TrainDataset
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn


def reduce_value(value, nprocs):
    with torch.no_grad():
        dist.all_reduce(value)
        value /= nprocs

    return value


def train(arg, local_rank):
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(local_rank)

    arg.batch_size = int(arg.batch_size / arg.nprocs)

    # model = HighResolutionNet2(HRNET_48, out_channels=arg.out_channels)
    model = EfficientNet_Unet(name='efficientnet-b6')
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda(local_rank)
    # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)

    # 选择优化器、loss
    optimizer = torch.optim.SGD(model.parameters(), lr=arg.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 20, eta_min=0, last_epoch=-1)
    loss_fn = nn.CrossEntropyLoss().cuda(local_rank)

    cudnn.benchmark = True

    # 数据集加载
    traindata = TrainDataset(arg.data_path, 'Train')
    train_sampler = torch.utils.data.distributed.DistributedSampler(traindata)
    trainloader = torch.utils.data.DataLoader(traindata, pin_memory=True, batch_size=arg.batch_size,
                                              sampler=train_sampler)

    valdata = TrainDataset(arg.data_path, 'Val')
    val_sampler = torch.utils.data.distributed.DistributedSampler(valdata)
    valloader = torch.utils.data.DataLoader(valdata, pin_memory=True, batch_size=arg.batch_size, sampler=val_sampler)

    if local_rank == 0:
        train_data_size = len(traindata)
        print("训练集长度为：{}".format(train_data_size))
        val_data_size = len(valdata)
        print("验证集长度为：{}".format(val_data_size))

    # 混淆矩阵
    writer = SummaryWriter(arg.log_dir)
    train_confusion_matrix = tnt.meter.ConfusionMeter(arg.out_channels, normalized=False)
    val_confusion_matrix = tnt.meter.ConfusionMeter(arg.out_channels, normalized=False)

    model.train()
    for i in range(0, arg.epochs):
        trainloader.sampler.set_epoch(i)
        if local_rank == 0:
            print("-------第 {} 轮训练开始-------".format(i + 1))
        start_time = time.time()
        train_batch_step = 0
        train_loss = 0
        train_confusion_matrix.reset()
        val_batch_step = 0
        val_loss = 0
        val_confusion_matrix.reset()

        for data in trainloader:
            imgs, targets = data['img'], data['label']
            imgs = imgs.cuda(local_rank)
            targets = targets.cuda(local_rank)

            outputs = model(imgs)
            loss = loss_fn(outputs, targets.long())

            torch.distributed.barrier()

            reduced_loss = reduce_value(loss, arg.nprocs)

            output_conf, target_conf = conf_m(outputs, targets)
            train_confusion_matrix.add(output_conf, target_conf)

            # 优化器优化模型
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss = train_loss + reduced_loss
            train_batch_step = train_batch_step + 1

            if local_rank == 0:
                if train_batch_step % 500 == 0:
                    print("训练次数: {},loss: {}".format(train_batch_step, train_loss.item() / train_batch_step))

        # 显示平均loss、学习率
        train_acc = (np.trace(train_confusion_matrix.conf) / float(np.ndarray.sum(train_confusion_matrix.conf)))
        train_f1 = index_calculation_f1types(train_confusion_matrix.value())
        if local_rank == 0:
            print("训练集平均的Loss: {},acc: {}, f1: {}, 学习率Lr: {}".format(train_loss.item() / train_batch_step,
                                                                              train_acc,
                                                                              train_f1,
                                                                              optimizer.state_dict()['param_groups'][0][
                                                                                  'lr']))
        scheduler.step()

        # 验证
        model.eval()
        with torch.no_grad():
            for data in valloader:
                imgs, targets = data['img'], data['label']
                imgs = imgs.cuda(local_rank)
                targets = targets.cuda(local_rank)

                outputs = model(imgs)
                loss = loss_fn(outputs, targets.long())

                torch.distributed.barrier()
                reduced_loss = reduce_value(loss, arg.nprocs)

                output_conf, target_conf = conf_m(outputs, targets)
                val_confusion_matrix.add(output_conf, target_conf)

                val_loss = val_loss + reduced_loss
                val_batch_step = val_batch_step + 1

        # 显示平均loss、学习率
        val_acc = (np.trace(val_confusion_matrix.conf) / float(np.ndarray.sum(val_confusion_matrix.conf)))
        val_f1 = index_calculation_f1types(val_confusion_matrix.value())
        if local_rank == 0:
            print("验证集平均Loss: {}, acc: {}, f1: {}".format(val_loss / val_batch_step, val_acc, val_f1))

        # 画loss、acc曲线
        if local_rank == 0:
            writer.add_scalars("Loss", {'Train': train_loss.item() / train_batch_step,
                                        'Val': val_loss.item() / val_batch_step}, i + 1)
            writer.add_scalars("Acc", {'Train': train_acc, 'Val': val_acc}, i + 1)
            writer.add_scalars("f1", {'Train': train_f1, 'Val': val_f1}, i + 1)

        # 保存
        if local_rank == 0:
            if (i + 1) % 2 == 0:
                # 保存模型
                torch.save(model.module.state_dict(), os.path.join(arg.model_save_path, 'Model_{}.pth'.format(i + 1)))
                print("模型已保存")

        # 计算一个epoch花费时间
        end_time = time.time()
        time_dif = end_time - start_time
        if local_rank == 0:
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
    parser.add_argument("--learning_rate", default=0.001)
    parser.add_argument("--epochs", default=150)
    parser.add_argument("--out_channels", default=5)
    parser.add_argument("--batch_size", default=16)
    parser.add_argument("--data_path", default='/chenlab/Seg_3090/HRSCD')
    parser.add_argument("--log_dir", default='/chenlab/Seg_3090/Out2/Log')
    parser.add_argument("--model_save_path", default='/chenlab/Seg_3090/Out2/Model')
    parser.add_argument('--local_rank', default=-1, type=int,
                        help='node rank for distributed training')

    arg = parser.parse_args()
    arg.nprocs = torch.cuda.device_count()

    random.seed(3407)
    torch.manual_seed(3407)
    cudnn.deterministic = True

    train(arg, arg.local_rank)


if __name__ == '__main__':
    main()
