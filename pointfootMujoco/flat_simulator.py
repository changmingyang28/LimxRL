import os
import sys
import time
import mujoco
import mujoco.viewer as viewer
from functools import partial
import limxsdk
import limxsdk.robot.Rate as Rate
import limxsdk.robot.Robot as Robot
import limxsdk.robot.RobotType as RobotType
import limxsdk.datatypes as datatypes

class FlatSimulatorMujoco:
    def __init__(self, asset_path, joint_sensor_names, robot): 
        self.robot = robot
        self.joint_sensor_names = joint_sensor_names
        self.joint_num = len(joint_sensor_names)
        
        # 加载平坦地面的模型文件
        print(f"加载平坦地面模型: {asset_path}")
        self.mujoco_model = mujoco.MjModel.from_xml_path(asset_path)
        self.mujoco_data = mujoco.MjData(self.mujoco_model)
        
        # 启动MuJoCo viewer
        self.viewer = viewer.launch_passive(self.mujoco_model, self.mujoco_data, key_callback=self.key_callback, show_left_ui=True, show_right_ui=True)
        self.viewer.cam.distance = 5  # 调整相机距离，适合平地观察
        self.viewer.cam.elevation = -15  # 调整相机角度
        
        self.dt = self.mujoco_model.opt.timestep
        self.fps = 1 / self.dt
        
        # 确保控制数组大小正确
        if self.mujoco_model.nu < self.joint_num:
            print(f"警告: 模型执行器数量({self.mujoco_model.nu}) < 关节数量({self.joint_num})")
            self.joint_num = self.mujoco_model.nu

        # 初始化命令和状态数据结构
        self.robot_cmd = datatypes.RobotCmd()
        self.robot_cmd.mode = [0. for x in range(0, self.joint_num)]
        self.robot_cmd.q = [0. for x in range(0, self.joint_num)]
        self.robot_cmd.dq = [0. for x in range(0, self.joint_num)]
        self.robot_cmd.tau = [0. for x in range(0, self.joint_num)]
        self.robot_cmd.Kp = [0. for x in range(0, self.joint_num)]
        self.robot_cmd.Kd = [0. for x in range(0, self.joint_num)]

        self.robot_state = datatypes.RobotState()
        self.robot_state.tau = [0. for x in range(0, self.joint_num)]
        self.robot_state.q = [0. for x in range(0, self.joint_num)]
        self.robot_state.dq = [0. for x in range(0, self.joint_num)]

        self.imu_data = datatypes.ImuData()

        self.robotCmdCallbackPartial = partial(self.robotCmdCallback)
        self.robot.subscribeRobotCmdForSim(self.robotCmdCallbackPartial)

    def robotCmdCallback(self, robot_cmd: datatypes.RobotCmd):
        self.robot_cmd = robot_cmd

    def key_callback(self, keycode):
        pass

    def run(self):
        frame_count = 0
        self.rate = Rate(self.fps)
        print("平坦地面仿真器启动成功！等待控制器连接...")
        print(f"模型信息: nq={self.mujoco_model.nq}, nv={self.mujoco_model.nv}, nu={self.mujoco_model.nu}")
        print(f"关节数量: {self.joint_num}")
        
        while self.viewer.is_running():    
            mujoco.mj_step(self.mujoco_model, self.mujoco_data)

            # 更新机器人状态 - 添加边界检查
            for i in range(self.joint_num):
                if i + 7 < len(self.mujoco_data.qpos):
                    self.robot_state.q[i] = self.mujoco_data.qpos[i + 7]
                if i + 6 < len(self.mujoco_data.qvel):
                    self.robot_state.dq[i] = self.mujoco_data.qvel[i + 6]
                if i < len(self.mujoco_data.ctrl):
                    self.robot_state.tau[i] = self.mujoco_data.ctrl[i]

                    # 应用控制命令
                    self.mujoco_data.ctrl[i] = (
                        self.robot_cmd.Kp[i] * (self.robot_cmd.q[i] - self.robot_state.q[i]) + 
                        self.robot_cmd.Kd[i] * (self.robot_cmd.dq[i] - self.robot_state.dq[i]) + 
                        self.robot_cmd.tau[i]
                    )
        
            self.robot_state.stamp = time.time_ns()
            self.robot.publishRobotStateForSim(self.robot_state)

            # 更新IMU数据
            imu_quat_id = mujoco.mj_name2id(self.mujoco_model, mujoco.mjtObj.mjOBJ_SENSOR, "quat")
            self.imu_data.quat[0] = self.mujoco_data.sensordata[self.mujoco_model.sensor_adr[imu_quat_id] + 0]
            self.imu_data.quat[1] = self.mujoco_data.sensordata[self.mujoco_model.sensor_adr[imu_quat_id] + 1]
            self.imu_data.quat[2] = self.mujoco_data.sensordata[self.mujoco_model.sensor_adr[imu_quat_id] + 2]
            self.imu_data.quat[3] = self.mujoco_data.sensordata[self.mujoco_model.sensor_adr[imu_quat_id] + 3]

            imu_gyro_id = mujoco.mj_name2id(self.mujoco_model, mujoco.mjtObj.mjOBJ_SENSOR, "gyro")
            self.imu_data.gyro[0] = self.mujoco_data.sensordata[self.mujoco_model.sensor_adr[imu_gyro_id] + 0]
            self.imu_data.gyro[1] = self.mujoco_data.sensordata[self.mujoco_model.sensor_adr[imu_gyro_id] + 1]
            self.imu_data.gyro[2] = self.mujoco_data.sensordata[self.mujoco_model.sensor_adr[imu_gyro_id] + 2]

            imu_acc_id = mujoco.mj_name2id(self.mujoco_model, mujoco.mjtObj.mjOBJ_SENSOR, "acc")
            self.imu_data.acc[0] = self.mujoco_data.sensordata[self.mujoco_model.sensor_adr[imu_acc_id] + 0]
            self.imu_data.acc[1] = self.mujoco_data.sensordata[self.mujoco_model.sensor_adr[imu_acc_id] + 1]
            self.imu_data.acc[2] = self.mujoco_data.sensordata[self.mujoco_model.sensor_adr[imu_acc_id] + 2]

            self.imu_data.stamp = time.time_ns()
            self.robot.publishImuDataForSim(self.imu_data)

            if frame_count % 20 == 0:
                self.viewer.sync()

            frame_count += 1
            self.rate.sleep()

if __name__ == '__main__': 
    robot_type = os.getenv("ROBOT_TYPE")

    if not robot_type:
        print("Error: Please set the ROBOT_TYPE using 'export ROBOT_TYPE=<robot_type>'.")
        sys.exit(1)

    robot = Robot(RobotType.PointFoot, True)

    if not robot.init("127.0.0.1"):
        sys.exit()

    script_dir = os.path.dirname(os.path.abspath(__file__))

    # 使用平坦地面的模型文件
    model_path = f'{script_dir}/robot_description/{robot_type}/xml/robot_flat.xml'

    if not os.path.exists(model_path):
        print(f"Error: The file {model_path} does not exist.")
        sys.exit(1)

    print(f"*** 平坦地面模型已加载: robot_description/{robot_type}/xml/robot_flat.xml ***")

    joint_sensor_names = [
        "abad_L_Joint", "hip_L_Joint", "knee_L_Joint", "abad_R_Joint", "hip_R_Joint", "knee_R_Joint"
    ]

    simulator = FlatSimulatorMujoco(model_path, joint_sensor_names, robot)
    simulator.run()
