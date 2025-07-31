#!/bin/bash

# 只站立模式的控制器
echo "=== 启动PointFoot RL控制器 (仅站立模式) ==="

# 激活conda环境
source /home/cmy/miniconda3/etc/profile.d/conda.sh
conda activate isaacgym

# 进入pointfootMujoco目录
cd /home/cmy/Desktop/LimxRL/pointfootMujoco

# 设置机器人类型
export ROBOT_TYPE=PF_TRON1A

echo "环境配置完成，启动控制器 (仅站立模式)..."
echo "机器人类型: $ROBOT_TYPE"
echo "模型路径: policy/PF_TRON1A/policy/policy.onnx"
echo "当前目录: $(pwd)"

# 创建临时的站立模式控制器文件
cat > stand_only_controller.py << 'EOF'
import os
import sys
sys.path.append('.')
from rl_controller import *

# 修改PointfootController类,让它永久保持STAND模式
class StandOnlyController(PointfootController):
    def handle_stand_mode(self):
        # 永远保持站立姿势,不切换到WALK模式
        for j in range(len(self.joint_names)):
            pos_des = self.init_state[self.joint_names[j]]
            self.set_joint_command(j, pos_des, 0, 0, self.control_cfg['stiffness'], self.control_cfg['damping'])

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
    
    # 使用站立专用控制器
    controller = StandOnlyController(f'{os.path.dirname(os.path.abspath(__file__))}/policy', robot, robot_type, start_controller)
    controller.run()
EOF

# 运行站立模式控制器
python stand_only_controller.py