import os
import sys
import numpy as np
import copy
sys.path.append('.')
from rl_controller import *
from scipy.spatial.transform import Rotation as R

# 稳定站立控制器
class StableStandController(PointfootController):
    def __init__(self, model_dir, robot, robot_type, start_controller):
        super().__init__(model_dir, robot, robot_type, start_controller)
        
        # 永远保持站立模式
        self.mode = "STAND"
        self.stand_percent = 0.0
        
        # 增强的PD控制参数
        self.enhanced_stiffness = 60.0  # 增加刚度
        self.enhanced_damping = 8.0     # 增加阻尼
        
        # 目标关节角度（稍微弯曲以提高稳定性）
        self.target_angles = {
            "abad_L_Joint": 0.0,
            "hip_L_Joint": -0.15,   # 髋关节稍微向后，降低重心
            "knee_L_Joint": 0.3,    # 膝关节弯曲，增加稳定性
            "abad_R_Joint": 0.0,
            "hip_R_Joint": -0.15,   
            "knee_R_Joint": 0.3
        }
        
        # 平衡控制参数
        self.balance_gains = {
            'pitch_kp': 10.0,      # 俯仰角控制增益
            'roll_kp': 10.0,       # 横滚角控制增益
            'pitch_kd': 2.0,       # 俯仰角速度控制增益
            'roll_kd': 2.0         # 横滚角速度控制增益
        }
        
        print("稳定站立控制器已启动")
        print(f"目标关节角度: {self.target_angles}")
        print("特点: 主动平衡 + 增强PD控制 + 稳定姿态")
    
    def compute_balance_compensation(self):
        """计算平衡补偿"""
        if not hasattr(self, 'imu_data') or not self.imu_data:
            return np.zeros(6)
        
        try:
            # 获取IMU姿态
            imu_orientation = np.array(self.imu_data.quat)
            if np.linalg.norm(imu_orientation) < 0.1:  # 检查数据有效性
                return np.zeros(6)
            
            # 转换为欧拉角
            rotation = R.from_quat(imu_orientation)
            euler_angles = rotation.as_euler('xyz')  # roll, pitch, yaw
            
            roll, pitch, yaw = euler_angles
            
            # 获取角速度
            gyro = np.array(self.imu_data.gyro) if hasattr(self.imu_data, 'gyro') else np.zeros(3)
            roll_rate, pitch_rate, yaw_rate = gyro
            
            # 计算平衡补偿
            compensation = np.zeros(6)
            
            # 俯仰平衡（前后倾斜）- 主要通过髋关节和膝关节调节
            pitch_compensation = -(self.balance_gains['pitch_kp'] * pitch + 
                                 self.balance_gains['pitch_kd'] * pitch_rate)
            
            # 横滚平衡（左右倾斜）- 主要通过髋关节差分调节
            roll_compensation = -(self.balance_gains['roll_kp'] * roll + 
                                self.balance_gains['roll_kd'] * roll_rate)
            
            # 分配到各关节
            # 俯仰控制：髋关节和膝关节同向调节
            compensation[1] += pitch_compensation * 0.3  # hip_L
            compensation[2] += pitch_compensation * 0.5  # knee_L  
            compensation[4] += pitch_compensation * 0.3  # hip_R
            compensation[5] += pitch_compensation * 0.5  # knee_R
            
            # 横滚控制：左右髋关节反向调节
            compensation[1] += roll_compensation * 0.2   # hip_L
            compensation[4] -= roll_compensation * 0.2   # hip_R
            
            # 限制补偿幅度
            compensation = np.clip(compensation, -0.2, 0.2)
            
            if abs(pitch) > 0.1 or abs(roll) > 0.1:
                print(f"平衡补偿: roll={roll:.3f}, pitch={pitch:.3f}, compensation={compensation}")
            
            return compensation
            
        except Exception as e:
            print(f"平衡计算错误: {e}")
            return np.zeros(6)
    
    def handle_stand_mode(self):
        """增强的站立模式处理"""
        try:
            # 获取平衡补偿
            balance_comp = self.compute_balance_compensation()
            
            # 对每个关节设置控制命令
            for j, joint_name in enumerate(self.joint_names):
                if joint_name in self.target_angles:
                    # 基础目标角度
                    base_angle = self.target_angles[joint_name]
                    
                    # 添加平衡补偿
                    target_angle = base_angle + balance_comp[j]
                    
                    # 使用增强的PD参数
                    self.set_joint_command(
                        j, 
                        target_angle, 
                        0,  # 目标速度为0
                        0,  # 前馈扭矩为0
                        self.enhanced_stiffness, 
                        self.enhanced_damping
                    )
                else:
                    # 如果关节名不在目标角度中，使用默认值
                    self.set_joint_command(j, 0, 0, 0, self.enhanced_stiffness, self.enhanced_damping)
                    
        except Exception as e:
            print(f"站立控制错误: {e}")
            # 回退到基础站立控制
            for j, joint_name in enumerate(self.joint_names):
                target_angle = self.target_angles.get(joint_name, 0.0)
                self.set_joint_command(j, target_angle, 0, 0, 
                                     self.enhanced_stiffness, self.enhanced_damping)
    
    def handle_walk_mode(self):
        """重写步行模式，强制保持站立"""
        self.handle_stand_mode()
    
    def update(self):
        """更新控制器状态"""
        # 强制保持站立模式
        if self.mode != "STAND":
            self.mode = "STAND"
            print("强制切换回站立模式")
        
        # 调用站立处理
        self.handle_stand_mode()
        
        # 增加循环计数
        self.loop_count += 1
        
        # 发布机器人命令
        self.robot.publishRobotCmd(self.robot_cmd)

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
    
    # 使用稳定站立控制器
    controller = StableStandController(f'{os.path.dirname(os.path.abspath(__file__))}/policy', robot, robot_type, start_controller)
    controller.run()
