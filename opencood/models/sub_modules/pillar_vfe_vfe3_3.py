""" Author: Chengjie Lu

将point pillar中vfe模块的质心坐标范数、点云方差、最大点间距离进一步修改，质心坐标范数改成相对质心坐标的偏移量并保留三维，添加有效点的个数作为新的一维
这里处于point pillar的vfe模块

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np


class PFNLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 use_norm=True,
                 last_layer=False):
        super().__init__()

        self.last_vfe = last_layer
        self.use_norm = use_norm
        if not self.last_vfe:
            out_channels = out_channels // 2

        if self.use_norm:
            self.linear = nn.Linear(in_channels, out_channels, bias=False)
            self.norm = nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01)
        else:
            self.linear = nn.Linear(in_channels, out_channels, bias=True)

        self.part = 50000

    def forward(self, inputs):
        if inputs.shape[0] > self.part:
            # nn.Linear performs randomly when batch size is too large
            num_parts = inputs.shape[0] // self.part
            part_linear_out = [self.linear(
                inputs[num_part * self.part:(num_part + 1) * self.part])
                for num_part in range(num_parts + 1)]
            x = torch.cat(part_linear_out, dim=0)
        else:
            x = self.linear(inputs)
        torch.backends.cudnn.enabled = False
        x = self.norm(x.permute(0, 2, 1)).permute(0, 2,
                                                  1) if self.use_norm else x
        torch.backends.cudnn.enabled = True
        x = F.relu(x)
        x_max = torch.max(x, dim=1, keepdim=True)[0]

        if self.last_vfe:
            return x_max
        else:
            x_repeat = x_max.repeat(1, inputs.shape[1], 1)
            x_concatenated = torch.cat([x, x_repeat], dim=2)
            return x_concatenated


class PillarVFEVFENine(nn.Module):
    def __init__(self, model_cfg, num_point_features, voxel_size,
                 point_cloud_range):
        super().__init__()
        self.model_cfg = model_cfg

        self.use_norm = self.model_cfg['use_norm']
        self.with_distance = self.model_cfg['with_distance']

        self.use_absolute_xyz = self.model_cfg['use_absolute_xyz']
        num_point_features += 6 if self.use_absolute_xyz else 3
        if self.with_distance:
            num_point_features += 1

        self.num_filters = self.model_cfg['num_filters']
        assert len(self.num_filters) > 0

        self.voxel_x = voxel_size[0]
        self.voxel_y = voxel_size[1]
        self.voxel_z = voxel_size[2]
        self.x_offset = self.voxel_x / 2 + point_cloud_range[0]
        self.y_offset = self.voxel_y / 2 + point_cloud_range[1]
        self.z_offset = self.voxel_z / 2 + point_cloud_range[2]


    def get_output_feature_dim(self):
        return self.num_filters[-1]

    @staticmethod
    def get_paddings_indicator(actual_num, max_num, axis=0):
        actual_num = torch.unsqueeze(actual_num, axis + 1)
        max_num_shape = [1] * len(actual_num.shape)
        max_num_shape[axis + 1] = -1
        max_num = torch.arange(max_num,
                               dtype=torch.int,
                               device=actual_num.device).view(max_num_shape)
        paddings_indicator = actual_num.int() > max_num
        return paddings_indicator

    def analyze_features(self, statistical_features):
        """分析特征的统计分布
        Args:
            statistical_features: [M, 7] tensor, 包含相对偏移[0:3]，方差[3:6]，最大距离[6]
        """
        # 转换为numpy进行分析
        features = statistical_features.detach().cpu().numpy()
        
        # 创建子图
        fig, axes = plt.subplots(3, 1, figsize=(10, 15))
        
        # 1. 分析相对偏移量
        offsets = features[:, :3]
        axes[0].boxplot([offsets[:, i] for i in range(3)], labels=['x', 'y', 'z'])
        axes[0].set_title('Relative Offsets Distribution')
        axes[0].set_ylabel('Offset Value')
        
        # 2. 分析方差
        vars = features[:, 3:6]
        axes[1].boxplot([vars[:, i] for i in range(3)], labels=['x', 'y', 'z'])
        axes[1].set_title('Variance Distribution')
        axes[1].set_ylabel('Variance Value')
        
        # 3. 分析最大距离
        max_dist = features[:, 6]
        axes[2].hist(max_dist, bins=50)
        axes[2].set_title('Max Distance Distribution')
        axes[2].set_xlabel('Distance')
        axes[2].set_ylabel('Count')
        
        # 打印基本统计信息
        print("\nFeature Statistics:")
        print("\nRelative Offsets (x, y, z):")
        print(f"Mean: {np.mean(offsets, axis=0)}")
        print(f"Std: {np.std(offsets, axis=0)}")
        print(f"Min: {np.min(offsets, axis=0)}")
        print(f"Max: {np.max(offsets, axis=0)}")
        
        print("\nVariance (x, y, z):")
        print(f"Mean: {np.mean(vars, axis=0)}")
        print(f"Std: {np.std(vars, axis=0)}")
        print(f"Min: {np.min(vars, axis=0)}")
        print(f"Max: {np.max(vars, axis=0)}")
        
        print("\nMax Distance:")
        print(f"Mean: {np.mean(max_dist)}")
        print(f"Std: {np.std(max_dist)}")
        print(f"Min: {np.min(max_dist)}")
        print(f"Max: {np.max(max_dist)}")
        
        plt.tight_layout()
        plt.savefig('feature_analysis.png')
        plt.close()

    def forward(self, batch_dict):
        voxel_features, voxel_num_points, coords = \
            batch_dict['voxel_features'], batch_dict['voxel_num_points'], \
            batch_dict['voxel_coords']
        
        # 创建mask来标识有效点 [M, 32, 1]
        points_mask = torch.arange(voxel_features.shape[1], device=voxel_features.device) \
            .unsqueeze(0).expand(voxel_features.shape[0], -1) \
            < voxel_num_points.unsqueeze(-1)
        points_mask = points_mask.unsqueeze(-1)
        
        # 使用mask获取有效点的xyz坐标 [M, 32, 3]
        valid_points = voxel_features[:, :, :3] * points_mask.float()
        
        # 1. 计算相对于质心的平均偏移量 [M, 3]
        # 首先计算质心
        points_mean = valid_points.sum(dim=1) / voxel_num_points.unsqueeze(-1).float()  # [M, 3]
        
        # 计算每个点到质心的相对偏移
        relative_offsets = valid_points - points_mean.unsqueeze(1)  # [M, 32, 3]
        
        # 计算有效点的平均相对偏移（使用mask确保只考虑有效点）
        mean_relative_offsets = (relative_offsets * points_mask.float()).sum(dim=1) / \
                               voxel_num_points.unsqueeze(-1).float()  # [M, 3]
        
        # 2. 计算点云方差 Variance [M, 3]
        # 广播减法计算每个点到均值的差
        centered_points = valid_points - points_mean.unsqueeze(1)  # [M, 32, 3]
        # 计算平方
        squared_diff = (centered_points * points_mask.float()) ** 2  # [M, 32, 3]
        # 使用sum和voxel_num_points计算方差
        variance = squared_diff.sum(dim=1) / voxel_num_points.unsqueeze(-1).float()  # [M, 3]
        
        # 3. 计算最大点间距离 Diameter [M, 1]
        # 使用广播计算所有点对之间的距离
        # reshape以便使用cdist
        points_expanded = valid_points.view(-1, voxel_features.shape[1], 3)  # [M, 32, 3]
        # 计算每个柱体内所有点对的距离
        distances = torch.cdist(points_expanded, points_expanded)  # [M, 32, 32]
        # 创建mask来处理无效点对
        mask_matrix = points_mask.squeeze(-1).unsqueeze(-1) * points_mask.squeeze(-1).unsqueeze(1)  # [M, 32, 32]
        # 将无效点对的距离设为0
        distances = distances * mask_matrix
        # 获取每个柱体内的最大距离
        max_distances = torch.max(distances.view(distances.shape[0], -1), dim=1)[0].unsqueeze(-1)  # [M, 1]
        
        # 将voxel_num_points转换为浮点数并增加维度 [M, 1]
        points_count = voxel_num_points.float().unsqueeze(-1)
        
        # 拼接所有特征 [M, 8] (原来是[M, 7])
        statistical_features = torch.cat([mean_relative_offsets, variance, max_distances, points_count], dim=1)

        # 将voxel_num_points进行二值化 [M, 1]
        # points_count = voxel_num_points.float()  # 首先转换为浮点数
        # points_mask = (points_count > 10).float()  # 大于0的设为1，否则为0
        # points_mask = points_mask.unsqueeze(-1)   # 增加维度 [M, 1]
        # statistical_features = points_mask
        # # 统计并打印1和0的数量
        # num_ones = torch.sum(points_mask == 1.0).item()
        # num_zeros = torch.sum(points_mask == 0.0).item()
        # print(f"Number of ones (有点的位置): {num_ones}")
        # print(f"Number of zeros (无点的位置): {num_zeros}")
        # print(f"Total positions (总位置数): {num_ones + num_zeros}")
        # print(f"Percentage of ones (有点位置占比): {(num_ones/(num_ones + num_zeros))*100:.2f}%")

        
        # 处理只有一个点的情况（方差和最大距离应为0）
        single_point_mask = (voxel_num_points == 1).unsqueeze(-1)  # [M, 1]
        statistical_features[:, 1:4] = statistical_features[:, 1:4] * (~single_point_mask)  # 注意这里改为1:4，因为points_count不需要置零

        # 添加归一化操作
        # 方法1：使用 min-max 归一化
        feature_min = statistical_features.min()
        feature_max = statistical_features.max()
        statistical_features = (statistical_features - feature_min) / (feature_max - feature_min)
        statistical_features = statistical_features * 2 - 1
        
        # 或者方法2：使用 z-score 标准化
        # mean = statistical_features.mean()
        # std = statistical_features.std()
        # statistical_features = (statistical_features - mean) / std
        # statistical_features = statistical_features *2 -1


        
        # 将统计特征添加到batch_dict中
        batch_dict['pillar_features'] = statistical_features
        
        # 在返回之前添加分析
        if False:  # 可以选择只在训练时分析
            self.analyze_features(statistical_features)
        
        return batch_dict
