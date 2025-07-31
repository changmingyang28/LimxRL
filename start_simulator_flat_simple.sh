#!/bin/bash

# 启动平坦地面MuJoCo仿真器（使用原始simulator.py）
echo "=== 启动PointFoot MuJoCo仿真器 (平坦地面) ==="

# 激活conda环境
source /home/cmy/miniconda3/etc/profile.d/conda.sh
conda activate isaacgym

# 进入pointfootMujoco目录
cd /home/cmy/Desktop/LimxRL/pointfootMujoco

# 设置机器人类型
export ROBOT_TYPE=PF_TRON1A

echo "环境配置完成，启动平坦地面仿真器..."
echo "机器人类型: $ROBOT_TYPE"
echo "地形类型: 平坦地面"
echo "当前目录: $(pwd)"

# 临时修改simulator.py使用平坦地面模型
sed 's/robot.xml/robot_flat.xml/' simulator.py > temp_flat_simulator.py

# 运行修改后的仿真器
python temp_flat_simulator.py