#!/bin/bash

# 设置工作目录
cd /home/ubuntu/HEAL

# 运行训练脚本
nohup python opencood/tools/train_vfeavg_vqvae_one_model.py \
    -y None \
    --model_dir opencood/logs/HEAL_m1_based/stage1/m1_vfe8_vqvae_new_256_1024_ratio=4_blk=2_normal_1_batchnorm_train \
    > m1_vfe8_vqvae_new_256_1024_ratio=4_blk=2_normal_1_batchnorm_train_$(date +"%Y%m%d_%H%M%S").log 2>&1 &

# 打印进程ID
echo "Training process started with PID: $!" 