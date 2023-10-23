import time
import os
import torch
import torch.nn as nn
import numpy as np
import torchnet as tnt
from tensorboardX import SummaryWriter

from Utils.Model_Select import model_select, loaders_select, loss_select, optimizer_select, scheduler_select
from Utils.Calculation_Selcet import calculation_select
from Utils.Utils import index_calculation_f1, conf_m
from Utils.Utils import save_img, Precision


def trainer(opt, device):
    writer = SummaryWriter(opt.log_dir)
    # 混淆矩阵
    train_confusion_matrix = tnt.meter.ConfusionMeter(opt.out_channels, normalized=False)
    val_confusion_matrix = tnt.meter.ConfusionMeter(opt.out_channels, normalized=False)
    # 参数设置
    model = model_select(opt, device)
    loss_fn = loss_select(opt, device)
    train_dataloader = loaders_select(opt, 'Train')
    val_dataloader = loaders_select(opt, 'Val')
    # 优化器
    optimizer = optimizer_select(opt, model)
    scheduler = scheduler_select(opt, optimizer)

    for i in range(0, opt.epochs):
        print("-------第 {} 轮训练开始-------".format(i + 1))
        start_time = time.time()

        train_batch_step = 0
        train_loss = torch.tensor(0).to(device)
        train_confusion_matrix.reset()
        val_batch_step = 0
        val_loss = 0
        val_confusion_matrix.reset()

        # 训练
        model.train()
        for data in train_dataloader:
            outputs, targets, loss = calculation_select(opt, data, 'Train', model, loss_fn, device)

            output_conf, target_conf = conf_m(outputs, targets)
            train_confusion_matrix.add(output_conf, target_conf)

            # 优化器优化模型
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss = train_loss + loss
            train_batch_step = train_batch_step + 1

        # 显示平均loss、学习率
        train_acc = (np.trace(train_confusion_matrix.conf) / float(np.ndarray.sum(train_confusion_matrix.conf)))
        train_f1 = index_calculation_f1(train_confusion_matrix.value())
        print("训练集平均的Loss: {},acc: {}, f1: {}, 学习率Lr: {}".format(train_loss.item() / train_batch_step,
                                                                 train_acc,
                                                                 train_f1,
                                                                 optimizer.state_dict()['param_groups'][0][
                                                                     'lr']))
        # 根据平均loss动态调整学习率lr5
        scheduler.step(train_loss)
        # scheduler.step()

        # 验证
        model.eval()
        with torch.no_grad():
            for data in val_dataloader:
                outputs, targets, loss = calculation_select(opt, data, 'Val', model, loss_fn, device)

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

        # # 保存
        # if (i + 1) % 1 == 0:
        # 保存模型
        torch.save(model.state_dict(), os.path.join(opt.model_save_path, 'Model_{}.pth'.format(i + 1)))
        print("模型已保存")

        # 计算一个epoch花费时间
        end_time = time.time()
        time_dif = end_time - start_time
        print("每个epoch的时间: {}".format(time_dif))
    writer.close()


def evaler(opt, device):

    # 混淆矩阵
    test_confusion_matrix = tnt.meter.ConfusionMeter(opt.out_channels, normalized=False)
    # 参数设置
    model = model_select(opt, device)
    model.load_state_dict(torch.load(opt.bestmodel))
    loss_fn = nn.CrossEntropyLoss()
    test_dataloader = loaders_select(opt, 'Test')

    # 训练
    model.eval()
    with torch.no_grad():
        for data in test_dataloader:
            outputs, targets, _ = calculation_select(opt, data, 'Test', model, loss_fn, device)
            output_conf, target_conf = conf_m(outputs, targets)
            test_confusion_matrix.add(output_conf, target_conf)
            save_img(outputs, data['path'][0], opt.save_name)
        # print(test_confusion_matrix.value())
        Precision(test_confusion_matrix.value())

