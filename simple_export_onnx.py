#!/usr/bin/env python3

import os
import sys
import torch
import torch.onnx
import yaml
import copy

def export_model_to_onnx():
    # 模型路径
    model_path = "/home/cmy/Desktop/LimxRL/pointfootGym/logs/model_21000.pt"
    
    # 加载模型
    print(f"Loading model from: {model_path}")
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # 从checkpoint中提取actor网络
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    # 根据pointfoot配置创建网络结构
    # 这些数值来自params.yaml配置文件
    num_observations = 27  # pointfoot的观测维度 (从params.yaml读取)
    num_actions = 6       # pointfoot的动作维度
    
    # 创建与训练模型完全匹配的Actor网络
    class RealActor(torch.nn.Module):
        def __init__(self, num_obs, num_actions):
            super().__init__()
            # 与训练模型完全相同的网络结构: 27->512->256->128->6
            self.actor = torch.nn.Sequential(
                torch.nn.Linear(num_obs, 512),
                torch.nn.ELU(),
                torch.nn.Linear(512, 256), 
                torch.nn.ELU(),
                torch.nn.Linear(256, 128),
                torch.nn.ELU(),
                torch.nn.Linear(128, num_actions)
            )
        
        def forward(self, x):
            return self.actor(x)
    
    # 创建网络实例
    actor = RealActor(num_observations, num_actions)
    
    # 正确加载训练好的actor权重
    actor_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('actor.'):
            # 将 'actor.0.weight' 转换为 'actor.0.weight'
            new_key = key  # 保持原始的层次结构
            actor_state_dict[new_key] = value
            print(f"Loading weight: {key} -> {new_key}, shape: {value.shape}")
    
    if actor_state_dict:
        try:
            actor.load_state_dict(actor_state_dict, strict=True)
            print("✅ Successfully loaded all trained actor weights!")
        except Exception as e:
            print(f"❌ Error loading weights: {e}")
            print("Available keys in checkpoint:")
            for key in actor_state_dict.keys():
                print(f"  {key}: {actor_state_dict[key].shape}")
            print("Expected keys in model:")
            for key in actor.state_dict().keys():
                print(f"  {key}: {actor.state_dict()[key].shape}")
            return
    else:
        print("❌ No actor weights found in checkpoint!")
        return
    
    # 设置为评估模式
    actor.eval()
    
    # 创建虚拟输入
    dummy_input = torch.randn(num_observations)
    
    # 导出为ONNX
    output_path = "/home/cmy/Desktop/LimxRL/pointfootMujoco/policy/PF_TRON1A/policy/policy.onnx"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    print(f"Exporting to: {output_path}")
    
    torch.onnx.export(
        actor,
        dummy_input,
        output_path,
        verbose=True,
        input_names=["nn_input"],
        output_names=["nn_output"],
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        dynamic_axes={
            'nn_input': {0: 'batch_size'},
            'nn_output': {0: 'batch_size'}
        }
    )
    
    print(f"Model exported successfully to: {output_path}")
    
    # 验证导出的模型
    import onnxruntime as ort
    try:
        session = ort.InferenceSession(output_path)
        test_input = dummy_input.numpy().astype('float32')
        result = session.run(None, {'nn_input': test_input})
        print(f"ONNX model test passed. Output shape: {result[0].shape}")
    except Exception as e:
        print(f"ONNX model test failed: {e}")

if __name__ == "__main__":
    export_model_to_onnx()