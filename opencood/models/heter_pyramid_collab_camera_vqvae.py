""" Author: Chengjie Lu

将point pillar中vfe模块的pfn网络修改成罗老师提供的点云簇的特征，包括质心坐标范数、点云方差、最大点间距离
不使用下游检测头单纯训练vqvae模型时调用的模型类

支持归一化操作，但并未使用，模型收敛效果较差

"""

import torch
import torch.nn as nn
import numpy as np
from icecream import ic
from collections import OrderedDict, Counter
from opencood.models.sub_modules.base_bev_backbone_resnet import ResNetBEVBackbone 
from opencood.models.sub_modules.feature_alignnet import AlignNet
from opencood.models.sub_modules.downsample_conv import DownsampleConv
from opencood.models.sub_modules.naive_compress import NaiveCompressor
from opencood.models.fuse_modules.pyramid_fuse import PyramidFusion
from opencood.utils.transformation_utils import normalize_pairwise_tfm
from opencood.utils.model_utils import check_trainable_module, fix_bn, unfix_bn
import importlib
import torchvision
from opencood.models.vq_vae import VQVAE
import math
import torchvision.utils as vutils
import os

class HeterPyramidCollabCameraVQVAE(nn.Module):
    def __init__(self, args):
        super(HeterPyramidCollabCameraVQVAE, self).__init__()
        self.args = args
        modality_name_list = list(args.keys())
        modality_name_list = [x for x in modality_name_list if x.startswith("m") and x[1:].isdigit()] 
        self.modality_name_list = modality_name_list

        self.cav_range = args['lidar_range']
        self.sensor_type_dict = OrderedDict()

        self.cam_crop_info = {} 

        # setup each modality model
        for modality_name in self.modality_name_list:
            model_setting = args[modality_name]
            sensor_name = model_setting['sensor_type']
            self.sensor_type_dict[modality_name] = sensor_name

            # import model
            encoder_filename = "opencood.models.heter_encoders"
            encoder_lib = importlib.import_module(encoder_filename)
            encoder_class = None
            target_model_name = model_setting['core_method'].replace('_', '')

            for name, cls in encoder_lib.__dict__.items():
                if name.lower() == target_model_name.lower():
                    encoder_class = cls

            """
            Encoder building
            """
            setattr(self, f"encoder_{modality_name}", encoder_class(model_setting['encoder_args']))
            if model_setting['encoder_args'].get("depth_supervision", False):
                setattr(self, f"depth_supervision_{modality_name}", True)
            else:
                setattr(self, f"depth_supervision_{modality_name}", False)

            """
            Backbone building 
            """
            setattr(self, f"backbone_{modality_name}", ResNetBEVBackbone(model_setting['backbone_args']))

            """
            Aligner building
            """
            setattr(self, f"aligner_{modality_name}", AlignNet(model_setting['aligner_args']))
            if sensor_name == "camera":
                camera_mask_args = model_setting['camera_mask_args']
                setattr(self, f"crop_ratio_W_{modality_name}", (self.cav_range[3]) / (camera_mask_args['grid_conf']['xbound'][1]))
                setattr(self, f"crop_ratio_H_{modality_name}", (self.cav_range[4]) / (camera_mask_args['grid_conf']['ybound'][1]))
                setattr(self, f"xdist_{modality_name}", (camera_mask_args['grid_conf']['xbound'][1] - camera_mask_args['grid_conf']['xbound'][0]))
                setattr(self, f"ydist_{modality_name}", (camera_mask_args['grid_conf']['ybound'][1] - camera_mask_args['grid_conf']['ybound'][0]))
                self.cam_crop_info[modality_name] = {
                    f"crop_ratio_W_{modality_name}": eval(f"self.crop_ratio_W_{modality_name}"),
                    f"crop_ratio_H_{modality_name}": eval(f"self.crop_ratio_H_{modality_name}"),
                }

        """for vfe33:change feature dim size from 5 to 64"""
        self.avg_conv = nn.Conv2d(5, 64, kernel_size=1)


        """For feature transformation"""
        self.H = (self.cav_range[4] - self.cav_range[1])
        self.W = (self.cav_range[3] - self.cav_range[0])
        self.fake_voxel_size = 1

        """
        Fusion, by default multiscale fusion: 
        Note the input of PyramidFusion has downsampled 2x. (SECOND required)
        """
        self.pyramid_backbone = PyramidFusion(args['fusion_backbone'])


        """
        Shrink header
        """
        self.shrink_flag = False
        if 'shrink_header' in args:
            self.shrink_flag = True
            self.shrink_conv = DownsampleConv(args['shrink_header'])

        # self.vqvae_model = VQVAE(in_channels=3, embedding_dim=64, hidden_dim=64, num_embeddings=512, num_res_blocks=4)
        self.vqvae_model = VQVAE(in_channels=3, embedding_dim=64, hidden_dim=128, num_embeddings=512, num_res_blocks=2)
        # print(self.vqvae_model)
        
        # compressor will be only trainable
        self.compress = False
        if 'compressor' in args:
            self.compress = True
            self.compressor = NaiveCompressor(args['compressor']['input_dim'],
                                              args['compressor']['compress_ratio'])

        self.model_train_init()
        # check again which module is not fixed.
        check_trainable_module(self)


    def model_train_init(self):
        # if compress, only make compressor trainable
        if self.compress:
            # freeze all
            self.eval()
            for p in self.parameters():
                p.requires_grad_(False)
            # unfreeze compressor
            self.compressor.train()
            for p in self.compressor.parameters():
                p.requires_grad_(True)

    def positionalencoding2d(self,d_model, height, width):
        """
        生成二维位置编码
        :param d_model: 特征通道数 (8)
        :param height: 特征图高度 (256)
        :param width: 特征图宽度 (512)
        :return: [d_model, height, width]
        """
        if d_model % 4 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with "
                            "odd dimension (got dim={:d})".format(d_model))
        pe = torch.zeros(d_model, height, width)
        # 每个维度使用一半的通道
        d_model = int(d_model / 2)
        div_term = torch.exp(torch.arange(0., d_model, 2) *
                            -(math.log(10000.0) / d_model))
        pos_w = torch.arange(0., width).unsqueeze(1)
        pos_h = torch.arange(0., height).unsqueeze(1)
        
        # 编码宽度方向
        pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
        pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
        
        # 编码高度方向
        pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
        pe[d_model+1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
        
        return pe

    def forward(self, data_dict):
        output_dict = {'pyramid': 'collab'}
        agent_modality_list = data_dict['agent_modality_list'] 
        modality_count_dict = Counter(agent_modality_list)
        modality_feature_dict = {}

        for modality_name in self.modality_name_list:
            if modality_name not in modality_count_dict:
                continue
            
            image_inputs_dict = data_dict[f'inputs_{modality_name}']
            x = image_inputs_dict['imgs']
            
            # 保存第一个batch的第一个视角的RGB图像
            sample_img = x[0, 0, :3, :, :]  # [3, H, W]
            
            # 确保值在0-1之间
            sample_img = (sample_img - sample_img.min()) / (sample_img.max() - sample_img.min())
            
            # 保存图像
            # save_path = "debug_images"
            # os.makedirs(save_path, exist_ok=True)
            # vutils.save_image(sample_img, os.path.join(save_path, "sample_view.png"))
            
            # x的形状: [batch, n_views, channels, H, W]
            batch_size, n_views = x.shape[0], x.shape[1]
            
            # 只取RGB通道，去掉深度通道
            feature = x[..., :3, :, :]  # [batch, n_views, 3, H, W]
            
            # 遍历每个视角
            view_features = []
            for view_idx in range(n_views):
                view_feature = feature[:, view_idx]  # [batch, 3, H, W]
                
                # 归一化操作
                # feature_min = view_feature.min()
                # feature_max = view_feature.max()
                # view_feature = (view_feature - feature_min) / (feature_max - feature_min)
                
                # 添加位置编码
                # pe = self.positionalencoding2d(3, 384, 512).to(view_feature.device)
                # pe = pe.unsqueeze(0).repeat(batch_size, 1, 1, 1)
                # view_feature = view_feature + pe
                
                view_features.append(view_feature)
            
            # 将所有视角的特征存储
            modality_feature_dict[modality_name] = torch.stack(view_features, dim=1)  # [batch, n_views, 128, H, W]

        # Crop/Padd camera feature map
        for modality_name in self.modality_name_list:
            if modality_name in modality_count_dict:
                if self.sensor_type_dict[modality_name] == "camera":
                    feature = modality_feature_dict[modality_name]
                    _, _, _, H, W = feature.shape
                    target_H = int(H*eval(f"self.crop_ratio_H_{modality_name}"))
                    target_W = int(W*eval(f"self.crop_ratio_W_{modality_name}"))

                    # 对每个视角进行裁剪
                    crop_func = torchvision.transforms.CenterCrop((target_H, target_W))
                    cropped_features = []
                    for view_idx in range(n_views):
                        cropped_features.append(crop_func(feature[:, view_idx]))
                    modality_feature_dict[modality_name] = torch.stack(cropped_features, dim=1)

        # Assemble heter features
        counting_dict = {modality_name:0 for modality_name in self.modality_name_list}
        heter_feature_2d_list = []
        
        for modality_name in agent_modality_list:
            feat_idx = counting_dict[modality_name]
            # 遍历该模态的所有视角
            for view_idx in range(n_views):
                heter_feature_2d_list.append(modality_feature_dict[modality_name][feat_idx, view_idx])
            counting_dict[modality_name] += 1

        heter_feature_2d = torch.stack(heter_feature_2d_list)

        output_dict.update({'vqvae_feature': heter_feature_2d})
        recon, loss, recon_loss = self.vqvae_model(heter_feature_2d)
        output_dict.update({'reconstructed_feature': recon})
        output_dict.update({'vq_loss': loss})
        output_dict.update({'recon_loss': recon_loss})
        return output_dict
