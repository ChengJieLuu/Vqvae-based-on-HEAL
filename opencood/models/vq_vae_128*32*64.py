import torch
import torch.nn as nn
import torch.nn.functional as F

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
    def __init__(self, channels, bottleneck_ratio=4):
        super().__init__()
        bottleneck_channels = channels // bottleneck_ratio
        
        self.conv1 = nn.Conv2d(channels, bottleneck_channels, 1)
        self.bn1 = nn.BatchNorm2d(bottleneck_channels)
        self.conv2 = nn.Conv2d(bottleneck_channels, bottleneck_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(bottleneck_channels)
        self.conv3 = nn.Conv2d(bottleneck_channels, channels, 1)
        self.bn3 = nn.BatchNorm2d(channels)
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        
        out += identity
        out = self.relu(out)
        
        return out

class VQVAE(nn.Module):
    def __init__(self, in_channels, embedding_dim=128, num_embeddings=512, num_res_blocks=4):
        super().__init__()
        self.embedding_dim = embedding_dim
        
        # 编码器
        encoder_layers = [
            # 初始特征提取 256->128
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # 增加通道数 128->64
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            # 进一步增加通道数到embedding_dim 64->32
            nn.Conv2d(128, embedding_dim, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(embedding_dim),
            nn.ReLU(inplace=True),
        ]
        
        # 添加残差块（在32x32分辨率下处理）
        for _ in range(num_res_blocks):
            encoder_layers.append(ResBlock(embedding_dim))
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Vector Quantizer（在低分辨率特征上进行量化）
        self.vq = VectorQuantizer(num_embeddings, self.embedding_dim)
        
        # 解码器
        decoder_layers = [
            # 保持高维特征处理
            nn.Conv2d(embedding_dim, embedding_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(embedding_dim),
            nn.ReLU(inplace=True),
        ]
        
        # 添加残差块（在32x32分辨率下处理）
        for _ in range(num_res_blocks):
            decoder_layers.append(ResBlock(embedding_dim))
            
        # 最终输出层（逐步上采样并降低通道数）
        decoder_layers.extend([
            # 32->64
            nn.ConvTranspose2d(embedding_dim, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            # 64->128
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # 128->256
            nn.ConvTranspose2d(64, in_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(in_channels)
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