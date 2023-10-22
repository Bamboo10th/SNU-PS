from Model.HRNet1 import HighResolutionNet1
from Model.HRNet2 import HighResolutionNet2
from Model.HRNet3 import HighResolutionNet3
from Model.HRNet_Config import HRNET_48
from Model.UNet import UNet
from Model.UNet_2Plus import UNet_2Plus
from Model.UperNet import UPerNet
from DeeplabV3PLlus_Model.modeling import deeplabv3plus_resnet50
from Model.UNet3Plus import UNet_3Plus
from Model.EfficientNet_Unet import EfficientNet_Unet
from Model.EfficientNet_Unet4 import EfficientNet_Unet4

from Model.TransUNet import VisionTransformer as ViT_seg
from Model.TransUNet import CONFIGS as CONFIGS_ViT_seg

from Utils.Hybrid_loss import hybrid_loss, FocalLoss
from Utils.Dice_Loss import DiceLoss, Dice_CrossEntropyLoss
import torch.nn as nn
import torch
import sys


def model_select(opt, device):
    if opt.model_id == 'HRNet1':
        model = HighResolutionNet1(HRNET_48, out_channels=opt.out_channels).to(device)
    elif opt.model_id == 'HRNet2':
        model = HighResolutionNet2(HRNET_48, out_channels=opt.out_channels).to(device)
    elif opt.model_id == 'HRNet3':
        model = HighResolutionNet3(HRNET_48, out_channels=opt.out_channels).to(device)
    elif opt.model_id == 'UNet':
        model = UNet(n_channels=3, n_classes=opt.out_channels).to(device)
    elif opt.model_id == 'EfficientNet_Unet':
        model = EfficientNet_Unet(name='efficientnet-b6', oc=opt.out_channels).to(device)
    elif opt.model_id == 'EfficientNet_Unet4':
        model = EfficientNet_Unet4(name='efficientnet-b6', oc=opt.out_channels).to(device)
    elif opt.model_id == 'UperNet':
        model = UPerNet(num_classes=opt.out_channels).to(device)
    elif opt.model_id == 'UNet_2Plus':
        model = UNet_2Plus(in_channels=3, n_classes=opt.out_channels).to(device)
    elif opt.model_id == 'UNet_3Plus':
        model = UNet_3Plus(in_channels=3, n_classes=opt.out_channels).to(device)
    elif opt.model_id == 'DeeplabV3PLlus':
        model = deeplabv3plus_resnet50(num_classes=opt.out_channels).to(device)
    elif opt.model_id == 'TransUNet':
        config_vit = CONFIGS_ViT_seg['R50-ViT-B_16']
        config_vit.n_classes = opt.out_channels
        config_vit.n_skip = 3
        config_vit.patches.grid = (int(128 / 16), int(128 / 16))
        model = ViT_seg(config_vit, img_size=128, num_classes=config_vit.n_classes).to(device)
    else:
        print('model id is error!')
        sys.exit(1)
    return model


def loss_select(opt, device):
    if opt.loss_function == 'hybrid':
        loss_fn = hybrid_loss
    elif opt.loss_function == 'CrossEntropyLoss':
        loss_fn = nn.CrossEntropyLoss()
        loss_fn = loss_fn.to(device)
    elif opt.loss_function == 'FocalLoss':
        loss_fn = FocalLoss(gamma=2, alpha=0.25).to(device)
    elif opt.loss_function == 'DiceLoss':
        loss_fn = DiceLoss(2).to(device)
    elif opt.loss_function == 'Dice_CrossEntropyLoss':
        loss_fn = Dice_CrossEntropyLoss
    else:
        print('loss id is error!')
        sys.exit(1)
    return loss_fn


def optimizer_select(opt, model):
    if opt.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), lr=opt.learning_rate)
    elif opt.optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=opt.learning_rate)
    else:
        print('optimizer id is error!')
        sys.exit(1)
    return optimizer


def scheduler_select(opt, optimizer):
    if opt.scheduler == 'ReduceLROnPlateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                               factor=0.5, patience=3, verbose=True, threshold=0.001,
                                                               threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)
    elif opt.scheduler == 'StepLR':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.5)

    elif opt.scheduler == 'CosineAnnealingLR':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 20, eta_min=0, last_epoch=-1)
    else:
        print('scheduler id is error!')
        sys.exit(1)

    return scheduler
