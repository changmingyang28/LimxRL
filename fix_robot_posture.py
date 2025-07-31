#!/usr/bin/env python3

import os
import shutil

def fix_base_com():
    """修复基座质心偏移问题"""
    print("=== 修复机器人姿态问题 ===\n")
    
    # 备份原文件
    robot_xml_path = "/home/cmy/Desktop/LimxRL/pointfootMujoco/robot_description/PF_TRON1A/xml/robot.xml"
    backup_path = robot_xml_path + ".backup"
    
    if not os.path.exists(backup_path):
        shutil.copy2(robot_xml_path, backup_path)
        print(f"✅ 已备份原文件到: {backup_path}")
    
    # 读取文件内容
    with open(robot_xml_path, 'r') as f:
        content = f.read()
    
    # 修复base_Link的质心位置
    # 原始: pos="0.0457123 0.000139854 -0.163809"
    # 修正: pos="0.0 0.0 -0.163809" (只保持Z方向偏移，消除X,Y偏移)
    old_base_inertial = 'pos="0.0457123 0.000139854 -0.163809"'
    new_base_inertial = 'pos="0.0 0.0 -0.163809"'
    
    if old_base_inertial in content:
        content = content.replace(old_base_inertial, new_base_inertial)
        print("✅ 修正base_Link质心位置")
    else:
        print("⚠️  未找到base_Link质心位置，可能已经修改过")
    
    # 也修复flat版本
    robot_flat_path = "/home/cmy/Desktop/LimxRL/pointfootMujoco/robot_description/PF_TRON1A/xml/robot_flat.xml"
    if os.path.exists(robot_flat_path):
        with open(robot_flat_path, 'r') as f:
            flat_content = f.read()
        
        if old_base_inertial in flat_content:
            flat_content = flat_content.replace(old_base_inertial, new_base_inertial)
            with open(robot_flat_path, 'w') as f:
                f.write(flat_content)
            print("✅ 同时修正robot_flat.xml")
    
    # 写入修正后的内容
    with open(robot_xml_path, 'w') as f:
        f.write(content)
    
    print("\n修正完成！")
    print("建议:")
    print("1. 重启仿真器查看效果")
    print("2. 如果还有问题，可能需要调整初始关节角度")
    print("3. 如果需要恢复原文件，使用备份文件robot.xml.backup")

def suggest_joint_angle_adjustments():
    """建议关节角度调整"""
    print(f"\n=== 关节角度调整建议 ===")
    print("如果修正质心后仍有倾斜，可以尝试调整初始关节角度:")
    print()
    print("在params.yaml中修改 default_joint_angle:")
    print("  hip_L_Joint: -0.1   # 左髋关节稍微向后")
    print("  hip_R_Joint: -0.1   # 右髋关节稍微向后") 
    print("  knee_L_Joint: 0.2   # 左膝关节稍微弯曲")
    print("  knee_R_Joint: 0.2   # 右膝关节稍微弯曲")
    print()
    print("这样可以让机器人采用更稳定的蹲姿")

if __name__ == "__main__":
    fix_base_com()
    suggest_joint_angle_adjustments()