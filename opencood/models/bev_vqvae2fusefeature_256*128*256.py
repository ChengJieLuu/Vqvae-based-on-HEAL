""" Author: Chengjie Lu

因为使用的是张丰谦师弟的原始bev特征，所以在进行vqvae重构后，需要将重构后的特征调整到后续网络的深层尺寸
这里是只加深通道而保持尺寸与后续网络对齐，所以只下采样一倍，尺寸为256*128*256

使用时将命名中的后缀尺寸去掉

"""

import torch.nn as nn

class VQVAEFeatureProcessor(nn.Module):
    def __init__(self, in_channels=256, out_channels=256):
        super(VQVAEFeatureProcessor, self).__init__()
        
        # 初始特征提取
        self.init_conv = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        # 使用多个残差块替代上采样块
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(256) for _ in range(6)  # 可以根据需要调整残差块数量
        ])
        
        # 最终调整
        self.final_conv = nn.Sequential(
            nn.Conv2d(256, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        # x: [batch, 256, 128, 256]
        
        # 初始特征提取
        x = self.init_conv(x)  # [batch, 256, 128, 256]
        
        # 通过多个残差块
        for block in self.residual_blocks:
            x = block(x)  # [batch, 256, 128, 256]
        
        # 最终调整
        out = self.final_conv(x)  # [batch, 256, 128, 256]
        
        return out

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        
        self.conv_block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels)
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        identity = x
        out = self.conv_block(x)
        out += identity
        out = self.relu(out)
        return out