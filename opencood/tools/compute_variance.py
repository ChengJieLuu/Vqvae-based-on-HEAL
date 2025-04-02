"""
将point pillar中vfe模块的pfn网络修改成平均（也可以直接32*10，命名上以avg统称）后，训练vqvae模型
但训练时不分开调用原来的heal模型（提取point pillar的feature）和vqvae模型（训练vqvae），而是将vqvae模型集成在原先的heal模型中
loss可以下降（最开始会先上升）
"""

import argparse
import os
import statistics

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
from torch.utils.data import DataLoader, Subset
from tensorboardX import SummaryWriter

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils
from opencood.data_utils.datasets import build_dataset

from icecream import ic


def train_parser():
    parser = argparse.ArgumentParser(description="synthetic data generation")
    parser.add_argument("--hypes_yaml", "-y", type=str, required=True,
                        help='data generation yaml file needed ')
    parser.add_argument('--model_dir', default='',
                        help='Continued training path')
    parser.add_argument('--fusion_method', '-f', default="intermediate",
                        help='passed to inference.')
    opt = parser.parse_args()
    return opt


def main():
    opt = train_parser()
    hypes = yaml_utils.load_yaml(opt.hypes_yaml, opt)

    print('Dataset Building')
    opencood_train_dataset = build_dataset(hypes, visualize=False, train=True)
    opencood_validate_dataset = build_dataset(hypes,
                                              visualize=False,
                                              train=False)

    train_loader = DataLoader(opencood_train_dataset,
                              batch_size=hypes['train_params']['batch_size'],
                              num_workers=4,
                              collate_fn=opencood_train_dataset.collate_batch_train,
                              shuffle=True,
                              pin_memory=True,
                              drop_last=True,
                              prefetch_factor=2)
    val_loader = DataLoader(opencood_validate_dataset,
                            batch_size=hypes['train_params']['batch_size'],
                            num_workers=4,
                            collate_fn=opencood_train_dataset.collate_batch_train,
                            shuffle=True,
                            pin_memory=True,
                            drop_last=True,
                            prefetch_factor=2)

    print('Creating Model')
    model = train_utils.create_model(hypes)

    # # 查看模型的所有子模块
    # for name, module in model.named_children():
    #     print(f"Module name: {name}")
    #     print(f"Module structure: {module}")
    #     print("------------------------")

    # 指定设备
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # 将模型移到指定设备
    model = model.to(device)
    
    # 确保数据在正确设备上
    def to_device(batch_data):
        if isinstance(batch_data, dict):
            return {k: to_device(v) for k, v in batch_data.items()}
        elif isinstance(batch_data, list):
            return [to_device(v) for v in batch_data]
        elif isinstance(batch_data, torch.Tensor):
            return batch_data.to(device)
        return batch_data
    
    # record lowest validation loss checkpoint.
    lowest_val_loss = 1e5
    lowest_val_epoch = -1

    # define the loss
    criterion = train_utils.create_loss(hypes)

    # optimizer setup
    optimizer = train_utils.setup_optimizer(hypes, model)
    # lr scheduler setup
    

    # if we want to train from last checkpoint.
    if opt.model_dir:
        saved_path = opt.model_dir
        init_epoch, model = train_utils.load_saved_model(saved_path, model)
        lowest_val_epoch = init_epoch
        scheduler = train_utils.setup_lr_schedular(hypes, optimizer, init_epoch=init_epoch)
        print(f"resume from {init_epoch} epoch.")

    else:
        init_epoch = 0
        # if we train the model from scratch, we need to create a folder
        # to save the model,
        saved_path = train_utils.setup_train(hypes)
        scheduler = train_utils.setup_lr_schedular(hypes, optimizer)

    # record training
    writer = SummaryWriter(saved_path)

    print('Training start')
    epoches = hypes['train_params']['epoches']
    supervise_single_flag = False if not hasattr(opencood_train_dataset, "supervise_single") else opencood_train_dataset.supervise_single
    # used to help schedule learning rate

    # 计算数据集的方差 - 使用 Welford's online algorithm
    with torch.no_grad():
        count = 0
        mean = 0
        M2 = 0  # 二阶矩
        
        for i, batch_data in enumerate(train_loader):
            if batch_data is None or batch_data['ego']['object_bbx_mask'].sum()==0:
                continue
                
            model.zero_grad()
            optimizer.zero_grad()
            batch_data = to_device(batch_data)
            ouput_dict = model(batch_data['ego'])
            features = ouput_dict['vqvae_feature']  # [B, C, H, W]
            
            # 将特征展平为一维向量
            features_flat = features.view(-1)
            
            # 更新计数
            batch_count = features_flat.size(0)
            
            # Welford's online algorithm
            delta = features_flat - mean
            count += batch_count
            mean += torch.sum(delta) / count
            delta2 = features_flat - mean
            M2 += torch.sum(delta * delta2)
            
            # 打印进度
            if i % 10 == 0:  # 每10个batch打印一次
                print(f"Processing batch {i}, current mean: {mean.item():.6f}")
        
        # 计算最终方差
        variance = M2 / (count - 1)  # 使用无偏估计
        
        print(f"Total number of elements processed: {count}")
        print(f"Final mean: {mean.item():.6f}")
        print(f"Computed variance: {variance.item():.6f}")
        
        # 保存方差
        torch.save({'data_variance': variance.item()}, 'variance.pth')
        
            


if __name__ == '__main__':
    main()
