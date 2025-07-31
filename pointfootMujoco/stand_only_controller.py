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
