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
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    
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

            # #排查数据是否有异常值
            # writer.add_scalar('Training/vqvae_feature_max', ouput_dict['vqvae_feature'].max(), epoch * len(train_loader) + i)
            # writer.add_scalar('Training/vqvae_feature_min', ouput_dict['vqvae_feature'].min(), epoch * len(train_loader) + i)
            # writer.add_scalar('Training/vqvae_feature_mean', ouput_dict['vqvae_feature'].mean(), epoch * len(train_loader) + i)

            # if i==0:    
            #     dummy_input = torch.randn(1, 8, 256, 512).to(device)
            #     writer.add_graph(model.vqvae_model, dummy_input)

            # lcj change
            vqvae_training_mode = False
            if 'model' in hypes and 'args' in hypes['model']:
                model_args = hypes['model']['args']
                if 'vqvae_params' in model_args and model_args['vqvae_params']['vqvae_training'] == True:
                    vqvae_training_mode = True
            if vqvae_training_mode:
                # final_loss = criterion(ouput_dict, batch_data['ego']['label_dict'])
                # criterion.logging(epoch, i, len(train_loader), writer)
                # final_loss += ouput_dict['vq_loss']
                final_loss = ouput_dict['vq_loss']
                writer.add_scalar('Training/Total_Loss', final_loss.item(), 
                    epoch * len(train_loader) + i)
                # 记录详细的损失组成
                if 'recon_loss' in ouput_dict:
                    writer.add_scalar('Training/Reconstruction_Loss', 
                                    ouput_dict['recon_loss'].item(),
                                    epoch * len(train_loader) + i)
                    writer.add_scalar('Training/VQ_Loss', ouput_dict['vq_loss']-ouput_dict['recon_loss'], 
                                    epoch * len(train_loader) + i)
                print(f'At epoch {epoch} iter {i}, VQ loss: {final_loss.item():.8f}')
            else:
                final_loss = criterion(ouput_dict, batch_data['ego']['label_dict'])
                criterion.logging(epoch, i, len(train_loader), writer)

                if supervise_single_flag:
                    final_loss += criterion(ouput_dict, batch_data['ego']['label_dict_single'], suffix="_single") * hypes['train_params'].get("single_weight", 1)
                    criterion.logging(epoch, i, len(train_loader), writer, suffix="_single")

            # back-propagation
            final_loss.backward()
            optimizer.step()

            torch.cuda.empty_cache()  # it will destroy memory buffer

        if epoch % hypes['train_params']['save_freq'] == 0:
            torch.save(model.state_dict(),
                       os.path.join(saved_path,
                                    'net_epoch%d.pth' % (epoch + 1)))

        if epoch % hypes['train_params']['eval_freq'] == 0:
            valid_ave_loss = []

            with torch.no_grad():
                for i, batch_data in enumerate(val_loader):
                    if batch_data is None:
                        continue
                    model.zero_grad()
                    optimizer.zero_grad()
                    model.eval()

                    batch_data = to_device(batch_data)
                    batch_data['ego']['epoch'] = epoch
                    ouput_dict = model(batch_data['ego'])
                    
                    # lcj change
                    if vqvae_training_mode:
                        final_loss = ouput_dict['vq_loss']
                        print(f'val vq_loss {final_loss:.8f}')

                    else:
                        final_loss = criterion(ouput_dict,
                                           batch_data['ego']['label_dict'])
                        print(f'val loss {final_loss:.3f}')
                    valid_ave_loss.append(final_loss.item())

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
