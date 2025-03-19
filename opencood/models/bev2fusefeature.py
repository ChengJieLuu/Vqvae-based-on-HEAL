""" Author: Chengjie Lu

这里是用张丰谦师弟的原始bev特征做基线测试，不使用vqvae
所以需要将最初的原始bev特征调整到后续网络的深层尺寸

"""

import torch.nn as nn

class BEVFeatureProcessor(nn.Module):
    def __init__(self, in_channels=3, out_channels=256):
        super(BEVFeatureProcessor, self).__init__()
        
        # 初始特征提取 + 第一次下采样
        self.init_conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1, stride=2),  # 256x512 -> 128x256
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # 残差块设计 + 第二次下采样
        self.res_block1 = ResBlock(64, 128, downsample=True)    # 128x256 -> 128x256
        self.res_block2 = ResBlock(128, 192)                    # 保持尺寸
        self.res_block3 = ResBlock(192, 256)                    # 保持尺寸
        
        # 特征金字塔网络
        self.fpn = FPN(
            in_channels=[64, 128, 192, 256],
            out_channels=out_channels
        )
        
        # 将整个网络转换为双精度
        self.double()
        
    def forward(self, x):
        # x: [batch, 3, 256, 512]
        
        # 初始特征提取 + 第一次下采样
        x1 = self.init_conv(x)  # [batch, 64, 128, 256]
        
        # 残差特征提取 + 第二次下采样
        x2 = self.res_block1(x1)  # [batch, 128, 64, 128]
        x3 = self.res_block2(x2)  # [batch, 192, 64, 128]
        x4 = self.res_block3(x3)  # [batch, 256, 64, 128]
        
        # 特征融合
        out = self.fpn([x1, x2, x3, x4])  # [batch, 256, 128, 256]
        
        return out

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=False):
        super(ResBlock, self).__init__()
        
        stride = 2 if downsample else 1
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                              padding=1, stride=stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                              padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # 残差连接需要考虑下采样
        if downsample or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.skip = nn.Identity()
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        identity = self.skip(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += identity
        out = self.relu(out)
        
        return out

class FPN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FPN, self).__init__()
        
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(c, out_channels, kernel_size=1)
            for c in in_channels
        ])
        
        # 添加上采样层
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        self.output_conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, features):
        # features: [x1(128,256), x2(64,128), x3(64,128), x4(64,128)]
        
        # 将所有特征转换为相同的通道数
        laterals = []
        for i, (conv, feat) in enumerate(zip(self.lateral_convs, features)):
            lat = conv(feat)
            if i > 0:  # 对x2, x3, x4进行上采样
                lat = self.upsample(lat)
            laterals.append(lat)
        
        # 特征融合
        fused = sum(laterals)
        
        # 最终处理
        out = self.output_conv(fused)  # [batch, 256, 128, 256]
        
        return out