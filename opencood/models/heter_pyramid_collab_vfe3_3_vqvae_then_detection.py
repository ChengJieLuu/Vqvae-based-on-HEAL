""" Author: Chengjie Lu

将point pillar中vfe模块的质心坐标范数、点云方差、最大点间距离进一步修改，质心坐标范数改成相对质心坐标的偏移量并保留三维，添加有效点的个数作为新的一维
训练好vqvae模型后，使用下游检测头训练时调用的模型类

支持归一化操作，但并未使用，模型收敛效果较差
使用了相对位置编码

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

class HeterPyramidCollabVFE33VQVAEThenDetection(nn.Module):
    def __init__(self, args):
        super(HeterPyramidCollabVFE33VQVAEThenDetection, self).__init__()
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
        self.avg_conv_32 = nn.Conv2d(32, 64, kernel_size=1)
        self.up_conv_32 = nn.ConvTranspose2d(32, 64, kernel_size=2, stride=2)
        self.avg_conv_64 = nn.Conv2d(64, 64, kernel_size=1)
        self.avg_conv_128 = nn.Conv2d(128, 64, kernel_size=1)
        self.avg_conv_256 = nn.Conv2d(256, 64, kernel_size=1)
        self.avg_conv_512 = nn.Conv2d(512, 64, kernel_size=1)



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

        self.vqvae_model = VQVAE(in_channels=8, embedding_dim=512, num_embeddings=1024, num_res_blocks=4)
        # self.vqvae_model = VQVAE(in_channels=8, embedding_dim=256, hidden_dim=512, num_embeddings=1024, num_res_blocks=4)

        """
        Shared Heads
        """
        self.cls_head = nn.Conv2d(args['in_head'], args['anchor_number'],
                                  kernel_size=1)
        self.reg_head = nn.Conv2d(args['in_head'], 7 * args['anchor_number'],
                                  kernel_size=1)
        self.dir_head = nn.Conv2d(args['in_head'], args['dir_args']['num_bins'] * args['anchor_number'],
                                  kernel_size=1) # BIN_NUM = 2
        
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
        affine_matrix = normalize_pairwise_tfm(data_dict['pairwise_t_matrix'], self.H, self.W, self.fake_voxel_size)
        record_len = data_dict['record_len'] 
        # print(agent_modality_list)
        modality_count_dict = Counter(agent_modality_list)
        modality_feature_dict = {}

        for modality_name in self.modality_name_list:
            if modality_name not in modality_count_dict:
                continue
            feature = eval(f"self.encoder_{modality_name}")(data_dict, modality_name)

            # 添加归一化操作
            # 方法1：使用 min-max 归一化
            # feature_min = feature.min()
            # feature_max = feature.max()
            # feature = (feature - feature_min) / (feature_max - feature_min)
            
            # 或者方法2：使用 z-score 标准化
            # mean = vqvae_feature.mean()
            # std = vqvae_feature.std()
            # vqvae_feature = (vqvae_feature - mean) / std

            # 添加位置编码
            pe = self.positionalencoding2d(8, 256, 512).to(feature.device)
            pe = pe.unsqueeze(0).repeat(feature.shape[0], 1, 1, 1)
            feature = feature + pe

            modality_feature_dict[modality_name] = feature

        """
        Crop/Padd camera feature map.
        """
        for modality_name in self.modality_name_list:
            if modality_name in modality_count_dict:
                if self.sensor_type_dict[modality_name] == "camera":
                    # should be padding. Instead of masking
                    feature = modality_feature_dict[modality_name]
                    _, _, H, W = feature.shape
                    target_H = int(H*eval(f"self.crop_ratio_H_{modality_name}"))
                    target_W = int(W*eval(f"self.crop_ratio_W_{modality_name}"))

                    crop_func = torchvision.transforms.CenterCrop((target_H, target_W))
                    modality_feature_dict[modality_name] = crop_func(feature)
                    if eval(f"self.depth_supervision_{modality_name}"):
                        output_dict.update({
                            f"depth_items_{modality_name}": eval(f"self.encoder_{modality_name}").depth_items
                        })

        """
        Assemble heter features
        """
        counting_dict = {modality_name:0 for modality_name in self.modality_name_list}
        heter_feature_2d_list = []
        for modality_name in agent_modality_list:
            feat_idx = counting_dict[modality_name]
            heter_feature_2d_list.append(modality_feature_dict[modality_name][feat_idx])
            counting_dict[modality_name] += 1

        heter_feature_2d = torch.stack(heter_feature_2d_list)

        with torch.set_grad_enabled(False):
            # 使用encoder部分获取特征
            z = self.vqvae_model.encoder(heter_feature_2d)
            # 通过VQ层获取量化特征
            z_q, _ = self.vqvae_model.vq(z)
            vqvae_quantized_feature = z_q


        # 添加位置编码
        # pe = self.positionalencoding2d(32, 128, 256).to(vqvae_quantized_feature.device)
        # pe = pe.unsqueeze(0).repeat(vqvae_quantized_feature.shape[0], 1, 1, 1)
        # vqvae_quantized_feature = vqvae_quantized_feature + pe

        feature = vqvae_quantized_feature
        with torch.set_grad_enabled(True):
            feature = self.avg_conv_512(feature)


            # feature = self.up_conv_32(feature)
            #原始的resnet backbone会下采样，为了尺寸不变这里不用
            # feature = eval(f"self.backbone_{modality_name}")({"spatial_features": feature})['spatial_features_2d']
            # feature = eval(f"self.aligner_{modality_name}")(feature)

        heter_feature_2d = feature
        # Pyramid backbone处理 - 在两个阶段都需要更新
        with torch.set_grad_enabled(True):  # 始终启用梯度
            fused_feature, occ_outputs = self.pyramid_backbone.forward_collab(
                heter_feature_2d,
                record_len, 
                affine_matrix, 
                agent_modality_list, 
                self.cam_crop_info
            )

            if self.shrink_flag:
                fused_feature = self.shrink_conv(fused_feature)

        # 检测头处理 - 只在检测头训练时更新
        with torch.set_grad_enabled(True):
            cls_preds = self.cls_head(fused_feature)
            reg_preds = self.reg_head(fused_feature)
            dir_preds = self.dir_head(fused_feature)

            output_dict.update({
                'cls_preds': cls_preds,
                'reg_preds': reg_preds,
                'dir_preds': dir_preds,
                'occ_single_list': occ_outputs
            })
        return output_dict
