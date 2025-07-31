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
