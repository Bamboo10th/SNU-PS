import torch.nn as nn
import torch
import sys


# # 数据标准化
# class Normalization(nn.Module):
#     def __init__(self):
#         super().__init__()
#
#     def forward(self, imga, imgb):
#         img = torch.stack([imga, imgb], dim=0)
#
#         img_mean = torch.mean(img, dim=(0, 1, 3, 4), keepdim=True)
#         img_mean = torch.squeeze(img_mean, dim=0)
#         img_std = torch.std(img, dim=(0, 1, 3, 4), keepdim=True)
#         img_std = torch.squeeze(img_std, dim=0)
#
#         imga_n = imga.sub_(img_mean).div_(img_std)
#         imgb_n = imgb.sub_(img_mean).div_(img_std)
#
#         return imga_n, imgb_n
#
#
# class Normalization2(nn.Module):
#     def __init__(self):
#         super().__init__()
#
#     def forward(self, img):
#
#         img_mean = torch.mean(img, dim=(0, 2, 3), keepdim=True)
#         img_mean = torch.squeeze(img_mean, dim=0)
#         img_std = torch.std(img, dim=(0, 2, 3), keepdim=True)
#         img_std = torch.squeeze(img_std, dim=0)
#
#         img_n = img.sub_(img_mean).div_(img_std)
#
#         return img_n


def model_calculation_rgb(data, dataset_id, model, loss_fn, device):
    if dataset_id == 'Train':
        imgas, imgbs, targets, name = data['imgA'], data['imgB'], data['label'], data['path']
        imgas = imgas.to(device, non_blocking=True)
        imgbs = imgbs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
    else:
        imgas, imgbs, targets, name = data['imgA'], data['imgB'], data['label'], data['path']
        imgas = imgas.to(device, non_blocking=False)
        imgbs = imgbs.to(device, non_blocking=False)
        targets = targets.to(device, non_blocking=False)

    outputs = model(imgas, imgbs)
    loss = loss_fn(outputs, targets.long())
    return outputs, targets, loss


def model_calculation_p(data, dataset_id, model, loss_fn, device):
    if dataset_id == 'Train':
        imgas, imgbs, targets, name = data['imgA'], data['imgB'], data['label'], data['path']
        imgas = imgas.to(device, non_blocking=True)
        imgbs = imgbs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
    else:
        imgas, imgbs, targets, name = data['imgA'], data['imgB'], data['label'], data['path']
        imgas = imgas.to(device, non_blocking=False)
        imgbs = imgbs.to(device, non_blocking=False)
        targets = targets.to(device, non_blocking=False)

    outputs = model(imgas, imgbs)
    loss = loss_fn(outputs, targets.long())
    return outputs, targets, loss


def model_calculation_fusion(data, dataset_id, model, loss_fn, device):
    if dataset_id == 'Train':
        imgs, targets, name = data['img'], data['label'], data['path']
        imgs = imgs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
    else:
        imgs, targets, name = data['img'], data['label'], data['path']
        imgs = imgs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

    outputs = model(imgs)
    loss = loss_fn(outputs, targets.long())

    return outputs, targets, loss


def model_calculation_pcfn(data, dataset_id, model, loss_fn, device):
    if dataset_id == 'Train':
        imgas, imgbs, targetas, targetbs, targets, name = \
            data['imgA'], data['imgB'], data['labelA'], data['labelB'], data['label'], data['path']
        imgas = imgas.to(device, non_blocking=True)
        imgbs = imgbs.to(device, non_blocking=True)
        targetas = targetas.to(device, non_blocking=True)
        targetbs = targetbs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
    else:
        imgas, imgbs, targetas, targetbs, targets, name = \
            data['imgA'], data['imgB'], data['labelA'], data['labelB'], data['label'], data['path']
        imgas = imgas.to(device, non_blocking=True)
        imgbs = imgbs.to(device, non_blocking=True)
        targetas = targetas.to(device, non_blocking=True)
        targetbs = targetbs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

    outputas, outputbs, outputs = model(imgas, imgbs)
    loss_fn1 = nn.CrossEntropyLoss()
    loss1 = loss_fn1(outputas, targetas.long())
    loss2 = loss_fn1(outputbs, targetbs.long())
    loss3 = loss_fn(outputs, targets.long())
    loss = loss1 + loss2 + loss3
    return outputs, targets, loss


def calculation_select(opt, data, dataset_id, model, loss_fn, device):
    if opt.model_id == 'FC_EF':
        outputs, targets, loss = model_calculation_fusion(data, dataset_id, model, loss_fn, device)
    elif opt.model_id == 'SNU_RGB':
        outputs, targets, loss = model_calculation_rgb(data, dataset_id, model, loss_fn, device)
    elif opt.model_id == 'SNU_P':
        outputs, targets, loss = model_calculation_p(data, dataset_id, model, loss_fn, device)
    elif opt.model_id == 'BIT':
        outputs, targets, loss = model_calculation_rgb(data, dataset_id, model, loss_fn, device)
    elif opt.model_id == 'PCFN':
        outputs, targets, loss = model_calculation_pcfn(data, dataset_id, model, loss_fn, device)
    else:
        print('model id is error!')
        sys.exit(1)
    return outputs, targets, loss
