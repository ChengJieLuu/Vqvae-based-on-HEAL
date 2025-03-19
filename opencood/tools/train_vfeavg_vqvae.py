"""
将point pillar中vfe模块的pfn网络修改成平均（也可以直接32*10，命名上以avg统称）后，训练vqvae模型
但训练时分开调用原来的heal模型（提取point pillar的feature）和vqvae模型（训练vqvae）
结果一直不收敛，loss保持小数点后两位不变
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
from opencood.models.vq_vae import VQVAE

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
    vqvae_model = VQVAE(in_channels=10, embedding_dim=256, num_embeddings=256, num_res_blocks=4)
    

    # # 查看模型的所有子模块
    # for name, module in model.named_children():
    #     print(f"Module name: {name}")
    #     print(f"Module structure: {module}")
    #     print("------------------------")

    # lcj change
    # 设置使用第二块GPU (索引为1)
    torch.cuda.set_device(1)
    print(f"Using GPU with ID: {torch.cuda.current_device()}")

    # 指定设备
    device = torch.device(f"cuda:{torch.cuda.current_device()}")
    
    # 将模型移到指定设备
    model = model.to(device)
    vqvae_model = vqvae_model.to(device)
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
            with torch.no_grad():
                # ouput_dict, vqvae_feature = model(batch_data['ego'])
                vqvae_feature = model.encoder_m1(batch_data['ego'],'m1')     

                # 添加归一化操作
                # 方法1：使用 min-max 归一化
                # vqvae_feature_min = vqvae_feature.min()
                # vqvae_feature_max = vqvae_feature.max()
                # vqvae_feature = (vqvae_feature - vqvae_feature_min) / (vqvae_feature_max - vqvae_feature_min)
                
                # 或者方法2：使用 z-score 标准化
                # mean = vqvae_feature.mean()
                # std = vqvae_feature.std()
                # vqvae_feature = (vqvae_feature - mean) / std 

                vqvae_feature_min = vqvae_feature.min()
                vqvae_feature_max = vqvae_feature.max()
                vqvae_feature = 2 * (vqvae_feature - vqvae_feature_min) / (vqvae_feature_max - vqvae_feature_min) - 1

            recon, loss, recon_loss = vqvae_model(vqvae_feature)
            vq_loss = loss - recon_loss

            writer.add_scalar('Loss/step/total', loss.item(), i)
            writer.add_scalar('Loss/step/reconstruction', recon_loss.item(), i)
            writer.add_scalar('Loss/step/vq', vq_loss.item(), i)

            final_loss = loss
            print(f'At epoch {epoch} iter {i}, final VQ loss: {final_loss.item():.8f}')

            # back-propagation
            final_loss.backward()
            optimizer.step()

            torch.cuda.empty_cache()  # it will destroy memory buffer

        if epoch % hypes['train_params']['save_freq'] == 0:
            torch.save(vqvae_model.state_dict(),
                       os.path.join(saved_path,
                                    'vqvae_net_epoch%d.pth' % (epoch + 1)))    

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
