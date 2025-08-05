#!/bin/bash

# 启动站立控制器的脚本
echo "=== 启动PointFoot 站立控制器 ==="

# 激活conda环境
source /home/cmy/miniconda3/etc/profile.d/conda.sh
conda activate isaacgym

# 进入pointfootMujoco目录
cd /home/cmy/Desktop/LimxRL/pointfootMujoco

# 设置机器人类型
export ROBOT_TYPE=PF_TRON1A

echo "环境配置完成，启动站立控制器..."
echo "机器人类型: $ROBOT_TYPE"
echo "控制模式: 键盘控制（神经网络+键盘输入）"
echo "当前目录: $(pwd)"
echo ""
echo "控制说明:"
echo "- 机器人将自动站立，然后自动导航到目标点 (5.0, 0.0)"
echo "- C: 取消导航，切换到手动控制"
echo "- T: 设置新目标点"
echo "- G: 开始导航到目标点"
echo "- 手动控制: W/S(前后) A/D(左右) Q/E(转向)"
echo "- ESC/X: 退出程序"
echo ""

# 启动站立控制器
python stand_controller.py