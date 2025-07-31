#!/bin/bash

# 启动RL控制器的脚本
echo "=== 启动PointFoot RL控制器 ==="

# 激活conda环境
source /home/cmy/miniconda3/etc/profile.d/conda.sh
conda activate isaacgym

# 进入pointfootMujoco目录
cd /home/cmy/Desktop/LimxRL/pointfootMujoco

# 设置机器人类型
export ROBOT_TYPE=PF_TRON1A

echo "环境配置完成，启动控制器..."
echo "机器人类型: $ROBOT_TYPE"
echo "模型路径: policy/PF_TRON1A/policy/policy.onnx"
echo "当前目录: $(pwd)"

# 启动控制器
python rl_controller.py