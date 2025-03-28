""" Author: Chengjie Lu

github上找到的cifar10数据集的vqvae实现

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25):
        super(VectorQuantizer, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        
        # 创建嵌入表
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1/num_embeddings, 1/num_embeddings)
        
    def forward(self, inputs):
        # 保持BCHW->BHWC的转换
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape
        flat_input = inputs.view(-1, self.embedding_dim)
        
        # 计算距离和编码（保持不变）
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self.embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self.embedding.weight.t()))
        
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings).to(inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        # 量化
        quantized = torch.matmul(encodings, self.embedding.weight)
        
        # 损失计算（在展平空间进行）
        e_latent_loss = F.mse_loss(quantized.detach(), flat_input)
        q_latent_loss = F.mse_loss(quantized, flat_input.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss
        
        # Straight-through estimator
        quantized = flat_input + (quantized - flat_input).detach()
        
        # 恢复形状并返回
        quantized = quantized.view(input_shape)
        quantized = quantized.permute(0, 3, 1, 2).contiguous()
        
        return quantized, loss

class ResBlock(nn.Module):
    def __init__(self, channels, bottleneck_ratio=1):
        super().__init__()
        bottleneck_channels = channels // bottleneck_ratio

        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(channels, bottleneck_channels, 3)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(bottleneck_channels, bottleneck_channels, 1)
        self.relu2 = nn.ReLU(inplace=True)
        

        
    def forward(self, x):
        identity = x
        
        out = self.relu1(x)
        out = self.conv1(out)

        out = self.relu2(out)  
        out = self.conv2(out)

        
        out += identity
        out = self.relu(out)
        
        return out
    
class ResidualStack(nn.Module):
    def __init__(self, num_hiddens, num_residual_layers, num_residual_hiddens):
        super().__init__()
        # See Section 4.1 of "Neural Discrete Representation Learning".
        layers = []
        for i in range(num_residual_layers):
            layers.append(
                nn.Sequential(
                    nn.ReLU(),
                    nn.Conv2d(
                        in_channels=num_hiddens,
                        out_channels=num_residual_hiddens,
                        kernel_size=3,
                        padding=1,
                    ),
                    nn.ReLU(),
                    nn.Conv2d(
                        in_channels=num_residual_hiddens,
                        out_channels=num_hiddens,
                        kernel_size=1,
                    ),
                )
            )

        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        h = x
        for layer in self.layers:
            h = h + layer(h)

        # ResNet V1-style.
        return torch.relu(h)

class VQVAE(nn.Module):
    def __init__(self, in_channels, embedding_dim=256, hidden_dim=512 ,num_embeddings=1024, num_res_blocks=4):
        super().__init__()
        self.embedding_dim = embedding_dim
        
        # 编码器
        encoder_layers = [
            # 初始特征提取 [B,3,256,512] -> [B,128,128,256]
            nn.Conv2d(in_channels, hidden_dim // 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            
            # 增加通道数到embedding_dim [B,128,128,256] -> [B,256,128,256]
            nn.Conv2d(hidden_dim // 2, hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),

            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1),
        ]
        
        # 添加残差块
        residual_hiddens_ratio = 4
        for _ in range(num_res_blocks):
            encoder_layers.append(ResidualStack(
            hidden_dim, num_res_blocks, hidden_dim // residual_hiddens_ratio
        ))
            
        # encoder_layers.append(nn.BatchNorm2d(hidden_dim))
        encoder_layers.append(nn.Conv2d(hidden_dim, embedding_dim, kernel_size=3, stride=1, padding=1))

        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Vector Quantizer
        self.vq = VectorQuantizer(num_embeddings, embedding_dim)

        
        # 解码器
        decoder_layers = [
            # 初始特征处理
            nn.Conv2d(embedding_dim, hidden_dim, kernel_size=3, stride=1, padding=1),
        ]
        
        # 添加残差块
        for _ in range(num_res_blocks):
            decoder_layers.append( ResidualStack(
            hidden_dim, num_res_blocks, hidden_dim // residual_hiddens_ratio
        ))
            
        # 最终输出层
        decoder_layers.extend([
            # 通道数调整 [B,256,128,256] -> [B,128,128,256]
            nn.Conv2d(hidden_dim, hidden_dim // 2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            
            # 上采样恢复原始尺寸 [B,128,128,256] -> [B,3,256,512]
            nn.ConvTranspose2d(hidden_dim // 2, in_channels, kernel_size=4, stride=2, padding=1),
        ])
        
        self.decoder = nn.Sequential(*decoder_layers)
    
    def forward(self, x):
        # 编码
        z = self.encoder(x)
        
        # 量化
        z_q, vq_loss = self.vq(z)
        
        # 解码
        x_recon = self.decoder(z_q)
        
        # 重建损失
        recon_loss = F.mse_loss(x_recon, x)
        
        # 总损失
        total_loss = recon_loss + vq_loss
        
        return x_recon, total_loss, recon_loss