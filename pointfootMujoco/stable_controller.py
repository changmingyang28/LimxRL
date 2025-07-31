import os
import sys
import numpy as np
sys.path.append('.')
from rl_controller import *

# 稳定版控制器，增加安全保护
class StableController(PointfootController):
    def compute_actions(self):
        """带安全保护的动作计算"""
        # 调用原始的动作计算
        super().compute_actions()
        
        # 严格限制动作范围到 [-2.0, 2.0]
        self.actions = np.clip(self.actions, -2.0, 2.0)
        
        # 平滑处理：与上一次动作混合
        if hasattr(self, 'prev_actions'):
            alpha = 0.3  # 混合系数
            self.actions = alpha * self.actions + (1 - alpha) * self.prev_actions
        self.prev_actions = self.actions.copy()
        
    def handle_walk_mode(self):
        """更保守的步行模式"""
        # 调用原始的步行处理
        super().handle_walk_mode()
        
        # 额外的安全检查：如果机器人倾斜过大，停止动作
        if hasattr(self, 'imu_data_tmp'):
            # 检查重力向量，判断倾斜程度
            gravity_vector = np.array([0, 0, -1])
            imu_orientation = np.array(self.imu_data_tmp.quat)
            from scipy.spatial.transform import Rotation as R
            q_wi = R.from_quat(imu_orientation).as_euler('zyx')
            inverse_rot = R.from_euler('zyx', q_wi).inv().as_matrix()
            projected_gravity = np.dot(inverse_rot, gravity_vector)
            
            # 如果倾斜角度过大（z分量小于0.7），使用保守动作
            if abs(projected_gravity[2]) < 0.7:
                print("检测到过度倾斜，使用保守控制")
                # 回到接近站立的姿势
                for i in range(len(self.joint_names)):
                    self.actions[i] *= 0.1  # 大幅降低动作幅度

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
    
    # 使用稳定版控制器
    controller = StableController(f'{os.path.dirname(os.path.abspath(__file__))}/policy', robot, robot_type, start_controller)
    controller.run()
