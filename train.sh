#!/bin/bash

# 设置工作目录
cd /home/ubuntu/HEAL

# 运行训练脚本
nohup python opencood/tools/train_vfeavg_vqvae_one_model.py \
    -y None \
    --model_dir opencood/logs/HEAL_m1_based/stage1/m1_vfe8_vqvae_512_1024_train_normolaize_1_detection \
    > m1_vfe8_vqvae_512_1024_train_normolaize_1_detection_$(date +"%Y%m%d_%H%M%S").log 2>&1 &

# 打印进程ID
echo "Training process started with PID: $!" 