from Model.FC_EF.FCEF import FCEF
from Model.SNU_RGB.SNU_Model_RGB import SNUNet_ECAM as SNUNet_ECAM_RGB
from Model.SNU_P.SNU_Model_P import SNUNet_ECAM as SNUNet_ECAM_P
from Model.BIT.networks import BASE_Transformer
from Model.PCFN.PCFN import PCFN
from Utils.Hybrid_loss import hybrid_loss

from Data_Loader.RGB_DL import get_train_loaders_rgb
from Data_Loader.P_DL import get_train_loaders_p
from Data_Loader.Fusion_DL import get_train_loaders_fusion
from Data_Loader.PCFN_DL import get_train_loaders_pcfn

import sys
import torch
import torch.nn as nn


def model_select(opt, device):
    if opt.model_id == 'FC_EF':
        model = FCEF(2 * opt.input_channels).to(device)
    elif opt.model_id == 'SNU_RGB':
        model = SNUNet_ECAM_RGB(opt.input_channels, opt.out_channels).to(device)
    elif opt.model_id == 'SNU_P':
        model = SNUNet_ECAM_P(opt.input_channels, opt.out_channels).to(device)
    elif opt.model_id == 'BIT':
        model = BASE_Transformer(input_nc=opt.input_channels, output_nc=opt.out_channels, token_len=4,
                                 resnet_stages_num=4, with_pos='learned', enc_depth=1, dec_depth=8).to(device)
    elif opt.model_id == 'PCFN':
        model = PCFN(opt.out_channels, opt.out_channels).to(device)
    else:
        print('model id is error!')
        sys.exit(1)

    return model


def loaders_select(opt, dataset_id):
    if opt.model_id == 'FC_EF':
        dataset_loader = get_train_loaders_fusion(opt, dataset_id)
    elif opt.model_id == 'SNU_RGB':
        dataset_loader = get_train_loaders_rgb(opt, dataset_id)
    elif opt.model_id == 'SNU_P':
        dataset_loader = get_train_loaders_p(opt, dataset_id)
    elif opt.model_id == 'BIT':
        dataset_loader = get_train_loaders_rgb(opt, dataset_id)
    elif opt.model_id == 'PCFN':
        dataset_loader = get_train_loaders_pcfn(opt, dataset_id)
    else:
        print('model id is error!')
        sys.exit(1)

    return dataset_loader


def loss_select(opt, device):
    if opt.loss_function == 'hybrid':
        loss_fn = hybrid_loss
    elif opt.loss_function == 'CrossEntropyLoss':
        loss_fn = nn.CrossEntropyLoss()
        loss_fn = loss_fn.to(device)
    else:
        print('model id is error!')
        sys.exit(1)

    return loss_fn


def optimizer_select(opt, model):
    if opt.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), lr=opt.learning_rate)
    elif opt.optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=opt.learning_rate)
    else:
        print('model id is error!')
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
        print('model id is error!')
        sys.exit(1)

    return scheduler
