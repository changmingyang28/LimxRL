import os
import sys
import numpy as np
sys.path.append('.')
from rl_controller import *

# 原地踏步控制器
class MarchInPlaceController(PointfootController):
    def __init__(self, model_dir, robot, robot_type, start_controller):
        super().__init__(model_dir, robot, robot_type, start_controller)
        # 设置原地踏步命令：无前进速度，无转向，可能有轻微的垂直运动
        self.march_commands = np.array([0.0, 0.0, 0.0])  # [lin_vel_x, lin_vel_y, ang_vel_yaw]
        print("原地踏步模式已激活")
        print("命令设置: 前进=0, 侧移=0, 转向=0")
    
    def compute_observation(self):
        """重写观测计算，强制使用原地踏步命令"""
        # 调用父类的观测计算
        super().compute_observation()
        
        # 但是强制命令为原地踏步
        self.commands = self.march_commands.copy()
        
        # 重新计算观测的命令部分（确保一致性）
        command_scaler = np.diag([
            self.user_cmd_cfg['lin_vel_x'],
            self.user_cmd_cfg['lin_vel_y'], 
            self.user_cmd_cfg['ang_vel_yaw']
        ])
        scaled_commands = np.dot(command_scaler, self.commands)
        
        # 更新观测中的命令部分（观测的最后3个元素是命令）
        self.observations[-3:] = scaled_commands
        
        print(f"原地踏步命令: {scaled_commands}")
    
    def handle_walk_mode(self):
        """原地踏步的步行模式处理"""
        # 更新机器人状态
        self.robot_state_tmp = copy.deepcopy(self.robot_state)
        self.imu_data_tmp = copy.deepcopy(self.imu_data)

        # 执行动作计算（每decimation次迭代）
        if self.loop_count % self.control_cfg['decimation'] == 0:
            self.compute_observation()
            self.compute_actions()
            
            # 限制动作幅度（原地踏步应该更温和）
            action_limit = 2.0  # 比正常步行更小的动作幅度
            self.actions = np.clip(self.actions, -action_limit, action_limit)

        # 应用动作到关节
        joint_pos = np.array(self.robot_state_tmp.q)
        joint_vel = np.array(self.robot_state_tmp.dq)

        for i in range(len(joint_pos)):
            if self.is_point_foot or (i + 1) % 4 != 0:
                # 计算动作限制
                action_min = (joint_pos[i] - self.init_joint_angles[i] +
                              (self.control_cfg['damping'] * joint_vel[i] - self.control_cfg['user_torque_limit']) /
                              self.control_cfg['stiffness'])
                action_max = (joint_pos[i] - self.init_joint_angles[i] +
                              (self.control_cfg['damping'] * joint_vel[i] + self.control_cfg['user_torque_limit']) /
                              self.control_cfg['stiffness'])

                # 裁剪动作
                self.actions[i] = max(action_min / self.control_cfg['action_scale_pos'],
                                      min(action_max / self.control_cfg['action_scale_pos'], self.actions[i]))

                # 计算期望关节位置
                pos_des = self.actions[i] * self.control_cfg['action_scale_pos'] + self.init_joint_angles[i]
                
                # 应用关节命令（可以稍微增加阻尼以获得更稳定的踏步）
                damping = self.control_cfg['damping'] * 1.2  # 增加20%阻尼
                self.set_joint_command(i, pos_des, 0, 0, self.control_cfg['stiffness'], damping)

                self.last_actions[i] = self.actions[i]

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
    
    # 使用原地踏步控制器
    controller = MarchInPlaceController(f'{os.path.dirname(os.path.abspath(__file__))}/policy', robot, robot_type, start_controller)
    controller.run()
