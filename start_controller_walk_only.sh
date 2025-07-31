#!/bin/bash

# 直接进入步行模式的控制器
echo "=== 启动PointFoot RL控制器 (直接步行模式) ==="

# 激活conda环境
source /home/cmy/miniconda3/etc/profile.d/conda.sh
conda activate isaacgym

# 进入pointfootMujoco目录
cd /home/cmy/Desktop/LimxRL/pointfootMujoco

# 设置机器人类型
export ROBOT_TYPE=PF_TRON1A

echo "环境配置完成，启动控制器 (直接步行模式)..."
echo "机器人类型: $ROBOT_TYPE"
echo "模型路径: policy/PF_TRON1A/policy/policy.onnx"
echo "当前目录: $(pwd)"

# 创建直接步行模式的控制器文件
cat > walk_only_controller.py << 'EOF'
import os
import sys
sys.path.append('.')
from rl_controller import *

# 修改PointfootController类，直接进入WALK模式
class WalkOnlyController(PointfootController):
    def __init__(self, model_dir, robot, robot_type, start_controller):
        super().__init__(model_dir, robot, robot_type, start_controller)
        # 直接设置为WALK模式，跳过STAND
        self.mode = "WALK"
        self.stand_percent = 1.0  # 标记站立已完成

if __name__ == '__main__':
    robot_type = os.getenv('ROBOT_TYPE')
    if not robot_type:
        print('Error: Please set the ROBOT_TYPE using export ROBOT_TYPE=<robot_type>.')
        sys.exit(1)
    
    robot = Robot(RobotType.PointFoot)
    robot_ip = '127.0.0.1'
    
    if not robot.init(robot_ip):
        sys.exit()
    
    start_controller = robot_ip == '127.0.0.1'
    
    # 使用步行专用控制器
    controller = WalkOnlyController(f'{os.path.dirname(os.path.abspath(__file__))}/policy', robot, robot_type, start_controller)
    controller.run()
EOF

# 运行步行模式控制器
python walk_only_controller.py