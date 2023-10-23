import torch
import argparse
import random
import numpy as np
import os
from Utils.Tools import trainer, evaler


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

    parser.add_argument("--loss_function", default='')
    parser.add_argument("--optimizer", default='')
    parser.add_argument("--scheduler", default='')
    parser.add_argument("--learning_rate", default='')
    parser.add_argument("--epochs", default='')
    parser.add_argument("--batch_size", default='')
    parser.add_argument("--num_workers", default=0)  # 卡2设置为0

    # 不同网路修改
    parser.add_argument("--train_id", default='Eval')
    parser.add_argument("--model_id", default='SNU_P')
    parser.add_argument("--input_channels", default=1)
    parser.add_argument("--out_channels", default=2)

    # 文件路径
    parser.add_argument("--files_dir", default=r'H:\zhuchuanhai\PCD_SN7\Data\SN7')
    parser.add_argument("--val_list_dir", default=r'')
    parser.add_argument("--test_list_dir", default=r'H:\zhuchuanhai\PCD_SN7\Data\List\Test_List.txt')

    parser.add_argument("--train_list_dir", default=r'')

    parser.add_argument("--save_name", default='SNU_P_Pre010')
    parser.add_argument("--bestmodel", default=r'H:\zhuchuanhai\PCD_SN7\Out\D010\T3\SNU_P\Model\Model_50.pth')

    parser.add_argument("--model_save_path", default=r'')
    parser.add_argument("--log_dir", default=r'')

    arg = parser.parse_args()
    device = torch.device('cuda:0')
    if arg.train_id == 'Train':
        seed_torch(seed=114514)
        trainer(arg, device)
    else:
        evaler(arg, device)


if __name__ == '__main__':
    main()
