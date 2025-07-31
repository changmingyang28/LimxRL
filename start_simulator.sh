#!/bin/bash

# 启动MuJoCo仿真器的脚本
echo "=== 启动PointFoot MuJoCo仿真器 ==="

# 激活conda环境
source /home/cmy/miniconda3/etc/profile.d/conda.sh
conda activate isaacgym

# 进入pointfootMujoco目录
cd /home/cmy/Desktop/LimxRL/pointfootMujoco

# 设置机器人类型
export ROBOT_TYPE=PF_TRON1A

echo "环境配置完成，启动仿真器..."
echo "机器人类型: $ROBOT_TYPE"
echo "当前目录: $(pwd)"

# 启动仿真器
python simulator.py