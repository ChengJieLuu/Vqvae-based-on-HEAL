#!/bin/bash

# 设置工作目录
cd /home/ubuntu/HEAL

# 运行训练脚本
nohup python opencood/tools/train.py \
    -y None \
    --model_dir opencood/logs/HEAL_m1_based/stage1/m1_vfe8_base \
    > vfe8_base_$(date +"%Y%m%d_%H%M%S").log 2>&1 &

# 打印进程ID
echo "Training process started with PID: $!" 