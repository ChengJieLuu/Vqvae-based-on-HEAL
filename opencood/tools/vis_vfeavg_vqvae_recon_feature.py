"""
对训练好的vqvae模型，可视化重构特征
因为是高维，不能像三维一样直接可视化，另写了一个文件单独实现
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

    # lcj change
    # 设置使用第二块GPU (索引为1)
    torch.cuda.set_device(0)
    print(f"Using GPU with ID: {torch.cuda.current_device()}")

    # 指定设备
    device = torch.device(f"cuda:{torch.cuda.current_device()}")
    
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

    with torch.no_grad():
        for epoch in range(init_epoch, max(epoches, init_epoch)):
            for param_group in optimizer.param_groups:
                print('learning rate %f' % param_group["lr"])
            # the model will be evaluation mode during validation
            model.train()
            try: # heter_model stage2
                model.model_train_init()
            except:
                print("No model_train_init function")
            for i, batch_data in enumerate(train_loader):
                if batch_data is None or batch_data['ego']['object_bbx_mask'].sum()==0:
                    continue
                model.zero_grad()
                optimizer.zero_grad()
                batch_data = to_device(batch_data)
                batch_data['ego']['epoch'] = epoch
                ouput_dict = model(batch_data['ego'])

                vqvae_feature = ouput_dict['vqvae_feature']
                recon_feature = ouput_dict['reconstructed_feature']

                # 1. 计算重构误差图
                reconstruction_error = torch.abs(vqvae_feature - recon_feature)
                
                # 2. 特征图可视化
                def visualize_feature_maps(orig_feat, recon_feat, error_map, batch_idx=0, save_path='feature_vis'):
                    import matplotlib.pyplot as plt
                    import numpy as np
                    import random
                    os.makedirs(save_path, exist_ok=True)
                    
                    # 选择第一个batch的数据
                    orig = orig_feat[batch_idx].detach().cpu().numpy()  # [320,256,512]
                    recon = recon_feat[batch_idx].detach().cpu().numpy()  # [320,256,512]
                    error = error_map[batch_idx].detach().cpu().numpy()  # [320,256,512]
                    
                    # 1. 平均值可视化 - 对通道维度(320)取平均
                    orig_mean = np.mean(orig, axis=0)  # [256,512]
                    recon_mean = np.mean(recon, axis=0)  # [256,512]
                    error_mean = np.mean(error, axis=0)  # [256,512]
                    
                    plt.figure(figsize=(15, 10))
                    
                    plt.subplot(231)
                    plt.imshow(orig_mean, cmap='viridis')
                    plt.colorbar()
                    plt.title('Original Feature (Mean)')
                    
                    plt.subplot(232)
                    plt.imshow(recon_mean, cmap='viridis')
                    plt.colorbar()
                    plt.title('Reconstructed Feature (Mean)')
                    
                    plt.subplot(233)
                    plt.imshow(error_mean, cmap='hot')
                    plt.colorbar()
                    plt.title('Reconstruction Error (Mean)')
                    
                    # 2. 随机通道可视化
                    random_channel = random.randint(0, orig.shape[0]-1)  # 在320个通道中随机选择
                    orig_random = orig[random_channel]  # [256,512]
                    recon_random = recon[random_channel]  # [256,512]
                    error_random = error[random_channel]  # [256,512]
                    
                    plt.subplot(234)
                    plt.imshow(orig_random, cmap='viridis')
                    plt.colorbar()
                    plt.title(f'Original Feature (Channel {random_channel})')
                    
                    plt.subplot(235)
                    plt.imshow(recon_random, cmap='viridis')
                    plt.colorbar()
                    plt.title(f'Reconstructed Feature (Channel {random_channel})')
                    
                    plt.subplot(236)
                    plt.imshow(error_random, cmap='hot')
                    plt.colorbar()
                    plt.title(f'Reconstruction Error (Channel {random_channel})')
                    
                    plt.tight_layout()
                    # 保存图片
                    plt.savefig(os.path.join(save_path, f'feature_comparison_epoch{epoch}_iter{i}_channel{random_channel}.png'))
                    plt.close()
                    
                    # 计算统计信息
                    mse = np.mean((orig - recon) ** 2)
                    mae = np.mean(np.abs(orig - recon))
                    mse_random = np.mean((orig_random - recon_random) ** 2)
                    mae_random = np.mean(np.abs(orig_random - recon_random))
                    
                    print(f"Overall - MSE: {mse:.6f}, MAE: {mae:.6f}")
                    print(f"Channel {random_channel} - MSE: {mse_random:.6f}, MAE: {mae_random:.6f}")
                    
                    # 记录到tensorboard
                    writer.add_scalar('Reconstruction/MSE_Overall', mse, epoch * len(train_loader) + i)
                    writer.add_scalar('Reconstruction/MAE_Overall', mae, epoch * len(train_loader) + i)
                    writer.add_scalar('Reconstruction/MSE_RandomChannel', mse_random, epoch * len(train_loader) + i)
                    writer.add_scalar('Reconstruction/MAE_RandomChannel', mae_random, epoch * len(train_loader) + i)
                
                # 每隔一定步数进行可视化
                if i % 1 == 0:  # 可以调整这个频率
                    visualize_feature_maps(vqvae_feature, recon_feature, reconstruction_error, batch_idx=0, save_path=os.path.join(saved_path, 'recon_vis'))

            if epoch % hypes['train_params']['save_freq'] == 0:
                torch.save(model.state_dict(),
                        os.path.join(saved_path,
                                        'net_epoch%d.pth' % (epoch + 1)))

            if epoch % hypes['train_params']['eval_freq'] == 0:
                valid_ave_loss = []
                valid_ave_loss = statistics.mean(valid_ave_loss)
                print('At epoch %d, the validation loss is %f' % (epoch,
                                                                valid_ave_loss))
                writer.add_scalar('Validate_Loss', valid_ave_loss, epoch)

                # lowest val loss
                if valid_ave_loss < lowest_val_loss:
                    lowest_val_loss = valid_ave_loss
                    torch.save(model.state_dict(),
                        os.path.join(saved_path,
                                        'net_epoch_bestval_at%d.pth' % (epoch + 1)))
                    if lowest_val_epoch != -1 and os.path.exists(os.path.join(saved_path,
                                        'net_epoch_bestval_at%d.pth' % (lowest_val_epoch))):
                        os.remove(os.path.join(saved_path,
                                        'net_epoch_bestval_at%d.pth' % (lowest_val_epoch)))
                    lowest_val_epoch = epoch + 1

            scheduler.step(epoch)

            opencood_train_dataset.reinitialize()

    print('Training Finished, checkpoints saved to %s' % saved_path)

    run_test = True
    if run_test:
        fusion_method = opt.fusion_method
        cmd = f"python opencood/tools/inference.py --model_dir {saved_path} --fusion_method {fusion_method}"
        print(f"Running command: {cmd}")
        os.system(cmd)

if __name__ == '__main__':
    main()
