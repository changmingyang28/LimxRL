import os
import sys
import copy
import numpy as np
import yaml
import time
import threading
import termios
import tty
import select
import onnxruntime as ort
from scipy.spatial.transform import Rotation as R
from functools import partial
import limxsdk
import limxsdk.robot.Rate as Rate
import limxsdk.robot.Robot as Robot
import limxsdk.robot.RobotType as RobotType
import limxsdk.datatypes as datatypes

class PointfootController:
    def __init__(self, model_dir, robot, robot_type, start_controller):
        # Initialize robot and type information
        self.robot = robot
        self.robot_type = robot_type
        self.is_point_foot = self.robot_type.startswith("PF")
        self.is_wheel_foot = self.robot_type.startswith("WF")
        self.is_sole_foot = self.robot_type.startswith("SF")

        # Load configuration and model file paths based on robot type
        self.config_file = f'{model_dir}/{self.robot_type}/params.yaml'
        self.model_file = f'{model_dir}/{self.robot_type}/policy/policy.onnx'

        # Load configuration settings from the YAML file
        self.load_config(self.config_file)

        # Load the ONNX model and set up input and output names
        self.policy_session = ort.InferenceSession(self.model_file)
        self.policy_input_names = [self.policy_session.get_inputs()[0].name]
        self.policy_output_names = [self.policy_session.get_outputs()[0].name]

        # Prepare robot command structure with default values for mode, q, dq, tau, Kp, Kd
        self.robot_cmd = datatypes.RobotCmd()
        self.robot_cmd.mode = [0. for x in range(0, self.joint_num)]
        self.robot_cmd.q = [0. for x in range(0, self.joint_num)]
        self.robot_cmd.dq = [0. for x in range(0, self.joint_num)]
        self.robot_cmd.tau = [0. for x in range(0, self.joint_num)]
        self.robot_cmd.Kp = [self.control_cfg['stiffness'] for x in range(0, self.joint_num)]
        self.robot_cmd.Kd = [self.control_cfg['damping'] for x in range(0, self.joint_num)]

        # Prepare robot state structure
        self.robot_state = datatypes.RobotState()
        self.robot_state.tau = [0. for x in range(0, self.joint_num)]
        self.robot_state.q = [0. for x in range(0, self.joint_num)]
        self.robot_state.dq = [0. for x in range(0, self.joint_num)]
        self.robot_state_tmp = copy.deepcopy(self.robot_state)

        # Initialize IMU (Inertial Measurement Unit) data structure
        self.imu_data = datatypes.ImuData()
        self.imu_data.quat[0] = 0
        self.imu_data.quat[1] = 0
        self.imu_data.quat[2] = 0
        self.imu_data.quat[3] = 1
        self.imu_data_tmp = copy.deepcopy(self.imu_data)

        # Set up a callback to receive updated robot state data
        self.robot_state_callback_partial = partial(self.robot_state_callback)
        self.robot.subscribeRobotState(self.robot_state_callback_partial)

        # Set up a callback to receive updated IMU data
        self.imu_data_callback_partial = partial(self.imu_data_callback)
        self.robot.subscribeImuData(self.imu_data_callback_partial)

        # Set up a callback to receive updated SensorJoy
        self.sensor_joy_callback_partial = partial(self.sensor_joy_callback)
        self.robot.subscribeSensorJoy(self.sensor_joy_callback_partial)

        # Set up a callback to receive diagnostic data
        self.robot_diagnostic_callback_partial = partial(self.robot_diagnostic_callback)
        self.robot.subscribeDiagnosticValue(self.robot_diagnostic_callback_partial)

        # Initialize the calibration state to -1, indicating no calibration has occurred.
        self.calibration_state = -1

        # Flag to start the controller
        self.start_controller = start_controller

    # Load the configuration from a YAML file
    def load_config(self, config_file):
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)

        # Assign configuration parameters to controller variables
        self.joint_names = config['PointfootCfg']['joint_names']
        self.init_state = config['PointfootCfg']['init_state']['default_joint_angle']
        self.stand_duration = config['PointfootCfg']['stand_mode']['stand_duration']
        self.control_cfg = config['PointfootCfg']['control']
        self.rl_cfg = config['PointfootCfg']['normalization']
        self.obs_scales = config['PointfootCfg']['normalization']['obs_scales']
        self.actions_size = config['PointfootCfg']['size']['actions_size']
        self.observations_size = config['PointfootCfg']['size']['observations_size']
        self.imu_orientation_offset = np.array(list(config['PointfootCfg']['imu_orientation_offset'].values()))
        self.user_cmd_cfg = config['PointfootCfg']['user_cmd_scales']
        self.loop_frequency = config['PointfootCfg']['loop_frequency']
        
        # Initialize variables for actions, observations, and commands
        self.actions = np.zeros(self.actions_size)
        self.observations = np.zeros(self.observations_size)
        self.last_actions = np.zeros(self.actions_size)
        self.commands = np.zeros(3)  # command to the robot (e.g., velocity, rotation)
        self.scaled_commands = np.zeros(3)
        self.base_lin_vel = np.zeros(3)  # base linear velocity
        self.base_position = np.zeros(3)  # robot base position
        self.loop_count = 0  # loop iteration count
        self.stand_percent = 0  # percentage of time the robot has spent in stand mode
        self.policy_session = None  # ONNX model session for policy inference
        self.joint_num = len(self.joint_names)  # number of joints
        self.commands = np.zeros(3)

        if self.is_wheel_foot:
          self.joint_pos_idxs = config['PointfootCfg']['size']['jointpos_idxs']
          self.wheel_joint_damping = config['PointfootCfg']['control']['wheel_joint_damping']
          self.wheel_joint_torque_limit = config['PointfootCfg']['control']['wheel_joint_torque_limit']

        # Initialize joint angles based on the initial configuration
        self.init_joint_angles = np.zeros(len(self.joint_names))
        for i in range(len(self.joint_names)):
            self.init_joint_angles[i] = self.init_state[self.joint_names[i]]
        
        # Set initial mode to "STAND"
        self.mode = "STAND"
        
        # Keyboard control variables
        self.keyboard_thread = None
        self.running = True
        self.keys_pressed = set()
        
        # Navigation system variables
        self.navigation_mode = False  # True when navigating to target
        self.target_position = np.array([5.0, 0.0])  # Default target: 5m forward
        self.current_position = np.array([0.0, 0.0])  # Current x, y position
        self.current_yaw = 0.0  # Current heading angle
        self.position_tolerance = 0.2  # Distance tolerance to consider target reached
        self.heading_tolerance = 0.1  # Heading tolerance in radians
        
        # Navigation control parameters
        self.max_linear_speed = 1.0
        self.max_angular_speed = 0.8
        self.position_gain = 1.5  # Proportional gain for position control
        self.heading_gain = 2.0   # Proportional gain for heading control
        
        # Start keyboard input thread
        self.start_keyboard_thread()
    
    def start_keyboard_thread(self):
        """Start the keyboard input thread"""
        self.keyboard_thread = threading.Thread(target=self.keyboard_input_loop, daemon=True)
        self.keyboard_thread.start()
        print("Keyboard control started:")
        print("  W/S - Forward/Backward")
        print("  A/D - Left/Right")  
        print("  Q/E - Turn Left/Right")
        print("  R - Reset commands")
        print("  T - Set target position")
        print("  G - Go to target (auto navigation)")
        print("  C - Cancel navigation")
        print("  ESC/X - Exit")
    
    def keyboard_input_loop(self):
        """Keyboard input loop running in separate thread"""
        # Set terminal to raw mode for immediate key detection
        if sys.stdin.isatty():
            old_settings = termios.tcgetattr(sys.stdin)
            tty.setraw(sys.stdin.fileno())
        
        try:
            while self.running:
                if select.select([sys.stdin], [], [], 0.1)[0]:
                    key = sys.stdin.read(1).lower()
                    
                    # Handle special keys
                    if ord(key) == 27:  # ESC key
                        print("\nESC pressed - Stopping controller...")
                        self.start_controller = False
                        self.running = False
                        break
                    elif key == 'x':
                        print("\nX pressed - Stopping controller...")
                        self.start_controller = False
                        self.running = False
                        break
                    
                    # Handle navigation commands
                    if key == 't':
                        self.set_target_position()
                        continue
                    elif key == 'g':
                        self.start_navigation()
                        continue
                    elif key == 'c':
                        self.cancel_navigation()
                        continue
                    
                    # Update key states
                    if key in 'wasdeqr':
                        self.keys_pressed.add(key)
                        
                    # Process keys immediately
                    self.process_keyboard_input()
                    
        except Exception as e:
            print(f"Keyboard input error: {e}")
        finally:
            # Restore terminal settings
            if sys.stdin.isatty():
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
    
    def process_keyboard_input(self):
        """Process keyboard input and update commands"""
        # If in navigation mode, let navigation system control the robot
        if self.navigation_mode:
            self.update_navigation()
            return
            
        # Manual control mode
        # Start with default forward movement if in WALK mode and no keys pressed
        if self.mode == "WALK" and len(self.keys_pressed) == 0:
            linear_x = 1.0  # Default forward movement
        else:
            linear_x = 1.0  # Start with default forward movement
        
        linear_y = 0.0  
        angular_z = 0.0
        
        # Process current keys - modify the default movement
        if 'w' in self.keys_pressed:
            linear_x = 1.5  # Faster forward
        if 's' in self.keys_pressed:
            linear_x = -1.0  # Backward
        if 'a' in self.keys_pressed:
            linear_y += 0.5  # Left
        if 'd' in self.keys_pressed:
            linear_y -= 0.5  # Right
        if 'q' in self.keys_pressed:
            angular_z += 0.8  # Turn left
        if 'e' in self.keys_pressed:
            angular_z -= 0.8  # Turn right
        if 'r' in self.keys_pressed:
            # Reset to default forward movement
            linear_x = 1.0
            linear_y = 0.0
            angular_z = 0.0
            self.keys_pressed.clear()
            print("Commands reset to default forward movement (1.0 m/s)")
        
        # Update robot commands
        self.commands[0] = linear_x
        self.commands[1] = linear_y  
        self.commands[2] = angular_z
        
        # Clear processed keys (except for continuous movement)
        if 'r' in self.keys_pressed:
            self.keys_pressed.remove('r')
    
    def update_position_tracking(self):
        """Update current position based on IMU and movement integration"""
        # Simple position integration using velocity commands and time
        # This is a basic approximation - for real applications, use SLAM or visual odometry
        dt = 1.0 / self.loop_frequency
        
        # Get current heading from IMU
        imu_orientation = np.array(self.imu_data_tmp.quat)
        q_wi = R.from_quat(imu_orientation).as_euler('zyx')
        self.current_yaw = q_wi[0]  # Yaw angle
        
        # Integrate velocity commands to estimate position
        # Convert body-frame velocities to world-frame
        cos_yaw = np.cos(self.current_yaw)
        sin_yaw = np.sin(self.current_yaw)
        
        world_vel_x = self.commands[0] * cos_yaw - self.commands[1] * sin_yaw
        world_vel_y = self.commands[0] * sin_yaw + self.commands[1] * cos_yaw
        
        self.current_position[0] += world_vel_x * dt
        self.current_position[1] += world_vel_y * dt
    
    def set_target_position(self):
        """Set target position via console input"""
        try:
            print(f"\nCurrent position: ({self.current_position[0]:.2f}, {self.current_position[1]:.2f})")
            print("Enter target position:")
            x = float(input("Target X: "))
            y = float(input("Target Y: "))
            self.target_position = np.array([x, y])
            print(f"Target set to: ({x:.2f}, {y:.2f})")
        except ValueError:
            print("Invalid input. Please enter numeric values.")
        except:
            print("Input cancelled.")
    
    def start_navigation(self):
        """Start autonomous navigation to target"""
        distance = np.linalg.norm(self.target_position - self.current_position)
        if distance < self.position_tolerance:
            print("Already at target position!")
            return
            
        self.navigation_mode = True
        print(f"Navigation started to ({self.target_position[0]:.2f}, {self.target_position[1]:.2f})")
        print(f"Distance: {distance:.2f}m")
    
    def cancel_navigation(self):
        """Cancel autonomous navigation"""
        if self.navigation_mode:
            self.navigation_mode = False
            print("Navigation cancelled. Returning to manual control.")
        else:
            print("No active navigation to cancel.")
    
    def update_navigation(self):
        """Update navigation control - compute commands to reach target"""
        # Calculate distance and angle to target
        target_vector = self.target_position - self.current_position
        distance_to_target = np.linalg.norm(target_vector)
        
        # Check if target is reached
        if distance_to_target < self.position_tolerance:
            self.navigation_mode = False
            self.commands[0] = 0.0
            self.commands[1] = 0.0
            self.commands[2] = 0.0
            print(f"Target reached! Final position: ({self.current_position[0]:.2f}, {self.current_position[1]:.2f})")
            return
        
        # Calculate desired heading to target
        desired_heading = np.arctan2(target_vector[1], target_vector[0])
        heading_error = desired_heading - self.current_yaw
        
        # Normalize heading error to [-pi, pi]
        while heading_error > np.pi:
            heading_error -= 2 * np.pi
        while heading_error < -np.pi:
            heading_error += 2 * np.pi
        
        # Compute control commands
        # If heading error is large, prioritize turning
        if abs(heading_error) > self.heading_tolerance:
            # Turn towards target
            angular_vel = np.clip(self.heading_gain * heading_error, 
                                -self.max_angular_speed, self.max_angular_speed)
            # Reduce linear speed when turning
            linear_speed = min(self.max_linear_speed * 0.5, 
                             self.position_gain * distance_to_target)
        else:
            # Move towards target
            linear_speed = np.clip(self.position_gain * distance_to_target, 
                                 0.0, self.max_linear_speed)
            angular_vel = np.clip(self.heading_gain * heading_error * 0.5, 
                                -self.max_angular_speed, self.max_angular_speed)
        
        # Update commands
        self.commands[0] = linear_speed  # Forward velocity
        self.commands[1] = 0.0          # No lateral movement for simplicity
        self.commands[2] = angular_vel  # Angular velocity
        
        # Print status every 50 loops (10Hz status update)
        if self.loop_count % 50 == 0:
            print(f"Nav: pos({self.current_position[0]:.2f},{self.current_position[1]:.2f}) "
                  f"dist={distance_to_target:.2f}m heading_err={np.degrees(heading_error):.1f}°")

    # Main control loop
    def run(self):
        # Wait until the controller is started
        while not self.start_controller:
          time.sleep(1)

        # Initialize default joint angles for standing
        self.default_joint_angles = np.array([0.0] * len(self.joint_names))
        self.stand_percent += 1 / (self.stand_duration * self.loop_frequency)
        self.mode = "STAND"
        self.loop_count = 0

        print("Controller started! Use keyboard to control:")
        print("  W/S - Forward/Backward")
        print("  A/D - Left/Right")  
        print("  Q/E - Turn Left/Right")
        print("  R - Reset commands")
        print("  T - Set target position")
        print("  G - Go to target (auto navigation)")
        print("  C - Cancel navigation")
        print("  ESC/X - Exit")
        
        # Set the loop rate based on the frequency in the configuration
        rate = Rate(self.loop_frequency)
        try:
            while self.start_controller and self.running:
                self.update()
                rate.sleep()
        except KeyboardInterrupt:
            print("\nKeyboard interrupt received")
        finally:
            # Stop keyboard thread
            self.running = False
            if self.keyboard_thread and self.keyboard_thread.is_alive():
                self.keyboard_thread.join(timeout=1.0)
            
            # Reset robot command values to ensure a safe stop when exiting the loop
            self.robot_cmd.q = [0. for x in range(0, self.joint_num)]
            self.robot_cmd.dq = [0. for x in range(0, self.joint_num)]
            self.robot_cmd.tau = [0. for x in range(0, self.joint_num)]
            self.robot_cmd.Kp = [0. for x in range(0, self.joint_num)]
            self.robot_cmd.Kd = [1.0 for x in range(0, self.joint_num)]
            self.robot.publishRobotCmd(self.robot_cmd)
            time.sleep(1)
            print("Controller stopped safely")

    # Handle the stand mode for smoothly transitioning the robot into standing
    def handle_stand_mode(self):
        if self.stand_percent < 1:
            for j in range(len(self.joint_names)):
                # Interpolate between initial and default joint angles during stand mode
                pos_des = self.default_joint_angles[j] * (1 - self.stand_percent) + self.init_state[self.joint_names[j]] * self.stand_percent
                self.set_joint_command(j, pos_des, 0, 0, self.control_cfg['stiffness'], self.control_cfg['damping'])
            # Increment the stand percentage over time
            self.stand_percent += 1 / (self.stand_duration * self.loop_frequency)
        else:
            # Switch to walk mode and start navigation to default target
            self.mode = "WALK"
            self.navigation_mode = True
            print(f"Switched to WALK mode - Auto navigating to target ({self.target_position[0]:.1f}, {self.target_position[1]:.1f})")
            print("Press 'C' to cancel navigation and use manual control")

    # Handle the walk mode where the robot moves based on computed actions
    def handle_walk_mode(self):
        # Update the temporary robot state and IMU data
        self.robot_state_tmp = copy.deepcopy(self.robot_state)
        self.imu_data_tmp = copy.deepcopy(self.imu_data)
        
        # Update position tracking
        self.update_position_tracking()

        # Execute actions every 'decimation' iterations
        if self.loop_count % self.control_cfg['decimation'] == 0:
            self.compute_observation()
            self.compute_actions()
            # Clip the actions within predefined limits
            action_min = -self.rl_cfg['clip_scales']['clip_actions']
            action_max = self.rl_cfg['clip_scales']['clip_actions']
            self.actions = np.clip(self.actions, action_min, action_max)

        # Iterate over the joints and set commands based on actions
        joint_pos = np.array(self.robot_state_tmp.q)
        joint_vel = np.array(self.robot_state_tmp.dq)

        for i in range(len(joint_pos)):
            if self.is_point_foot or (i + 1) % 4 != 0:
                # Compute the limits for the action based on joint position and velocity
                action_min = (joint_pos[i] - self.init_joint_angles[i] +
                              (self.control_cfg['damping'] * joint_vel[i] - self.control_cfg['user_torque_limit']) /
                              self.control_cfg['stiffness'])
                action_max = (joint_pos[i] - self.init_joint_angles[i] +
                              (self.control_cfg['damping'] * joint_vel[i] + self.control_cfg['user_torque_limit']) /
                              self.control_cfg['stiffness'])

                # Clip action within limits
                self.actions[i] = max(action_min / self.control_cfg['action_scale_pos'],
                                      min(action_max / self.control_cfg['action_scale_pos'], self.actions[i]))

                # Compute the desired joint position and set it
                pos_des = self.actions[i] * self.control_cfg['action_scale_pos'] + self.init_joint_angles[i]
                self.set_joint_command(i, pos_des, 0, 0, self.control_cfg['stiffness'], self.control_cfg['damping'])

                # Save the last action for reference
                self.last_actions[i] = self.actions[i]
            elif self.is_wheel_foot:
                action_min = joint_vel[i] - self.wheel_joint_torque_limit / self.wheel_joint_damping
                action_max = joint_vel[i] + self.wheel_joint_torque_limit / self.wheel_joint_damping
                self.last_actions[i] = self.actions[i]
                self.actions[i] = max(action_min / self.wheel_joint_damping,
                                      min(action_max / self.wheel_joint_damping, self.actions[i]))
                velocity_des = self.actions[i] * self.wheel_joint_damping
                self.set_joint_command(i, 0, velocity_des, 0, 0, self.wheel_joint_damping)

    def compute_observation(self):
        # Convert IMU orientation from quaternion to Euler angles (ZYX convention)
        imu_orientation = np.array(self.imu_data_tmp.quat)
        q_wi = R.from_quat(imu_orientation).as_euler('zyx')  # Quaternion to Euler ZYX conversion
        inverse_rot = R.from_euler('zyx', q_wi).inv().as_matrix()  # Get the inverse rotation matrix

        # Project the gravity vector (pointing downwards) into the body frame
        gravity_vector = np.array([0, 0, -1])  # Gravity in world frame (z-axis down)
        projected_gravity = np.dot(inverse_rot, gravity_vector)  # Transform gravity into body frame

        # Retrieve base angular velocity from the IMU data
        base_ang_vel = np.array(self.imu_data_tmp.gyro)
        # Apply IMU orientation offset correction (using Euler angles)
        rot = R.from_euler('zyx', self.imu_orientation_offset).as_matrix()  # Rotation matrix for offset correction
        base_ang_vel = np.dot(rot, base_ang_vel)  # Apply correction to angular velocity
        projected_gravity = np.dot(rot, projected_gravity)  # Apply correction to projected gravity

        # Retrieve joint positions and velocities from the robot state
        joint_positions = np.array(self.robot_state_tmp.q)
        joint_velocities = np.array(self.robot_state_tmp.dq)

        # Retrieve the last actions that were applied to the robot
        actions = np.array(self.last_actions)

        # Create a command scaler matrix for linear and angular velocities
        command_scaler = np.diag([
            self.user_cmd_cfg['lin_vel_x'],  # Scale factor for linear velocity in x direction
            self.user_cmd_cfg['lin_vel_y'],  # Scale factor for linear velocity in y direction
            self.user_cmd_cfg['ang_vel_yaw']  # Scale factor for yaw (angular velocity)
        ])

        # Apply scaling to the command inputs (velocity commands)
        scaled_commands = np.dot(command_scaler, self.commands)

        # Populate observation vector
        joint_pos_value = (joint_positions - self.init_joint_angles) * self.obs_scales['dof_pos']

        # In WF, joint pos does not include wheel speed, index(3, 7) needs to be removed
        if self.is_wheel_foot:
            joint_pos_input = np.array([joint_pos_value[idx] for idx in self.joint_pos_idxs])
        else:
            joint_pos_input = joint_pos_value

        # Create the observation vector by concatenating various state variables:
        # - Base angular velocity (scaled)
        # - Projected gravity vector
        # - Joint positions (difference from initial angles, scaled)
        # - Joint velocities (scaled)
        # - Last actions applied to the robot
        # - Scaled command inputs
        obs = np.concatenate([
            base_ang_vel * self.obs_scales['ang_vel'],  # Scaled base angular velocity
            projected_gravity,  # Projected gravity vector in body frame
            joint_pos_input,  # Scaled joint positions
            joint_velocities * self.obs_scales['dof_vel'],  # Scaled joint velocities
            actions,  # Last actions taken by the robot
            scaled_commands  # Scaled velocity commands from user input
        ])
        
        # Clip the observation values to within the specified limits for stability
        self.observations = np.clip(
            obs, 
            -self.rl_cfg['clip_scales']['clip_observations'],  # Lower limit for clipping
            self.rl_cfg['clip_scales']['clip_observations']  # Upper limit for clipping
        )
    
    def compute_actions(self):
        """
        Computes the actions based on the current observations using the policy session.
        """
        # Concatenate observations into a single tensor and convert to float32
        input_tensor = np.concatenate([self.observations], axis=0)
        input_tensor = input_tensor.astype(np.float32)
        
        # Create a dictionary of inputs for the policy session
        inputs = {self.policy_input_names[0]: input_tensor}
        
        # Run the policy session and get the output
        output = self.policy_session.run(self.policy_output_names, inputs)
        
        # Flatten the output and store it as actions
        self.actions = np.array(output).flatten()
        
    def set_joint_command(self, joint_index, q, dq, tau, kp, kd):
        """
        Sends a command to configure the state of a specific joint.
        This method updates the joint's desired position, velocity, torque, and control gains.
        Replace this implementation with the actual communication logic for your hardware.

        Parameters:
        joint_index (int): The index of the joint to be controlled.
        q (float): The desired joint position, typically in radians or degrees.
        dq (float): The desired joint velocity, typically in radians/second or degrees/second.
        tau (float): The desired joint torque, typically in Newton-meters (Nm).
        kp (float): The proportional gain for position control.
        kd (float): The derivative gain for velocity control.
        """
        self.robot_cmd.q[joint_index] = q
        self.robot_cmd.dq[joint_index] = dq
        self.robot_cmd.tau[joint_index] = tau
        self.robot_cmd.Kp[joint_index] = kp
        self.robot_cmd.Kd[joint_index] = kd

    def update(self):
        """
        Updates the robot's state based on the current mode and publishes the robot command.
        """
        if self.mode == "STAND":
            self.handle_stand_mode()
        elif self.mode == "WALK":
            self.handle_walk_mode()
        
        # Increment the loop count
        self.loop_count += 1

        # Publish the robot command
        self.robot.publishRobotCmd(self.robot_cmd)
        
    # Callback function for receiving robot command data
    def robot_state_callback(self, robot_state: datatypes.RobotState):
        """
        Callback function to update the robot state from incoming data.
        
        Parameters:
        robot_state (datatypes.RobotState): The current state of the robot.
        """
        self.robot_state = robot_state

    # Callback function for receiving imu data
    def imu_data_callback(self, imu_data: datatypes.ImuData):
        """
        Callback function to update IMU data from incoming data.
        
        Parameters:
        imu_data (datatypes.ImuData): The IMU data containing stamp, acceleration, gyro, and quaternion.
        """
        self.imu_data.stamp = imu_data.stamp
        self.imu_data.acc = imu_data.acc
        self.imu_data.gyro = imu_data.gyro
        
        # Rotate quaternion values
        self.imu_data.quat[0] = imu_data.quat[1]
        self.imu_data.quat[1] = imu_data.quat[2]
        self.imu_data.quat[2] = imu_data.quat[3]
        self.imu_data.quat[3] = imu_data.quat[0]

    # Callback function for receiving sensor joy data
    def sensor_joy_callback(self, sensor_joy: datatypes.SensorJoy):
        # Keyboard control replaces joystick - ignore joystick input
        # Keep emergency stop functionality if joystick is connected
        if self.start_controller and sensor_joy.buttons[4] == 1 and sensor_joy.buttons[2] == 1:
          print(f"L1 + X: stop_controller...")
          self.start_controller = False
          self.running = False
        
        # Commands are now controlled by keyboard input
        # All joystick movement commands are ignored

    # Callback function for receiving diagnostic data
    def robot_diagnostic_callback(self, diagnostic_value: datatypes.DiagnosticValue):
      # Check if the received diagnostic data is related to calibration.
      if diagnostic_value.name == "calibration":
        print(f"Calibration state: {diagnostic_value.code}")
        self.calibration_state = diagnostic_value.code

if __name__ == '__main__':
    # Get the robot type from the environment variable
    robot_type = os.getenv("ROBOT_TYPE")
    
    # Check if the ROBOT_TYPE environment variable is set, otherwise exit with an error
    if not robot_type:
        print("Error: Please set the ROBOT_TYPE using 'export ROBOT_TYPE=<robot_type>'.")
        sys.exit(1)

    # Create a Robot instance of the specified type
    robot = Robot(RobotType.PointFoot)

    # Default IP address for the robot
    robot_ip = "127.0.0.1"
    
    # Check if command-line argument is provided for robot IP
    if len(sys.argv) > 1:
        robot_ip = sys.argv[1]

    # Initialize the robot with the provided IP address
    if not robot.init(robot_ip):
        sys.exit()

    # Determine if the simulation is running
    start_controller = robot_ip == "127.0.0.1"

    # Create and run the PointfootController
    controller = PointfootController(f'{os.path.dirname(os.path.abspath(__file__))}/policy', robot, robot_type, start_controller)
    controller.run()
