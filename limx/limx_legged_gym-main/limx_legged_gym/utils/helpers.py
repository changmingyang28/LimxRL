# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import os
import copy
import numpy as np
import random
from isaacgym import gymapi
from isaacgym import gymutil
import torch
from limx_legged_gym import LIMX_LEGGED_GYM_ROOT_DIR, LIMX_LEGGED_GYM_ENVS_DIR


def tuple_type(values):
    return tuple(map(int, values.split(',')))


def class_to_dict(obj) -> dict:
    if not hasattr(obj, "__dict__"):
        return obj
    result = {}
    for key in dir(obj):
        if key.startswith("_"):
            continue
        element = []
        val = getattr(obj, key)
        if isinstance(val, list):
            for item in val:
                element.append(class_to_dict(item))
        else:
            element = class_to_dict(val)
        result[key] = element
    return result


def update_class_from_dict(instance, dict_for_update):
    """
    This function updates an object from a dictionary.
    It uses the setattr function to set an attribute of an object from a dictionary.
    """
    for key, value in dict_for_update.items():
        if isinstance(value, dict):  # if value itself a dictionary
            inner_instance = getattr(instance, key, None)  # get existing value
            if inner_instance is None or isinstance(inner_instance, dict):
                setattr(instance, key, value)
            else:
                # recursion
                update_class_from_dict(inner_instance, value)
        else:
            setattr(instance, key, value)


def set_seed(seed):
    if seed == -1:
        seed = np.random.randint(0, 10000)
    print("Setting seed: {}".format(seed))

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_sim_params(args, cfg):
    # code from Isaac Gym Preview 2
    # initialize sim params
    sim_params = gymapi.SimParams()

    # set some values from args
    if args.physics_engine == gymapi.SIM_FLEX:
        if args.device != "cpu":
            print("WARNING: Using Flex with GPU instead of PHYSX!")
    elif args.physics_engine == gymapi.SIM_PHYSX:
        sim_params.physx.use_gpu = args.use_gpu
        sim_params.physx.num_subscenes = args.subscenes
    sim_params.use_gpu_pipeline = args.use_gpu_pipeline

    # if sim options are provided in cfg, parse them and update/override above:
    if "sim" in cfg:
        gymutil.parse_sim_config(cfg["sim"], sim_params)

    # Override num_threads if passed on the command line
    if args.physics_engine == gymapi.SIM_PHYSX and args.num_threads > 0:
        sim_params.physx.num_threads = args.num_threads

    return sim_params


def get_load_path(root, load_run=-1, checkpoint=-1):
    last_run = get_last_run_path(root)
    if load_run == -1:
        load_run = os.path.join(last_run, "checkpoints")
    else:
        load_run = os.path.join(root, load_run, "checkpoints")
    if checkpoint == -1:
        models = [file for file in os.listdir(load_run) if 'model' in file]
        models.sort(key=lambda m: '{0:0>15}'.format(m))
        model = models[-1]
    else:
        model = "model_{}.pt".format(checkpoint)

    load_path = os.path.join(load_run, model)
    return load_path


def get_last_run_path(root):
    try:
        runs = os.listdir(root)
        # TODO sort by date to handle change of month
        runs.sort()
        if 'exported' in runs: runs.remove('exported')
        last_run = os.path.join(root, runs[-1])
    except:
        raise ValueError("No runs in this directory: " + root)

    return last_run


def update_cfg_from_args(env_cfg, cfg_train, args):
    # seed
    if env_cfg is not None:
        # num envs
        if args.num_envs is not None:
            env_cfg.env.num_envs = args.num_envs
    if cfg_train is not None:
        if args.seed is not None:
            cfg_train.seed = args.seed
        # alg runner parameters
        if args.max_iterations is not None:
            cfg_train.runner.max_iterations = args.max_iterations
        if args.experiment_name is not None:
            cfg_train.runner.experiment_name = args.experiment_name
        if args.logger is not None:
            cfg_train.logger = args.logger
        if args.update_model is not None:
            cfg_train.update_model = args.update_model
        if args.video is not None:
            cfg_train.video = args.video
        if args.record_length is not None:
            cfg_train.record_length = args.record_length
        if args.record_interval is not None:
            cfg_train.record_interval = args.record_interval
        if args.frame_size is not None:
            cfg_train.frame_size = args.frame_size
        if args.fps is not None:
            cfg_train.fps = args.fps
        if args.camera_offset is not None:
            cfg_train.camera_offset = args.camera_offset
        if args.camera_rotation is not None:
            cfg_train.camera_rotation = args.camera_rotation
        if args.env_idx_record is not None:
            cfg_train.env_idx_record = args.env_idx_record
        if args.actor_idx_record is not None:
            cfg_train.actor_idx_record = args.actor_idx_record
        if args.rigid_body_idx_record is not None:
            cfg_train.rigid_body_idx_record = args.rigid_body_idx_record

    return env_cfg, cfg_train


def get_args():
    custom_parameters = [
        {"name": "--task", "type": str, "default": "anymal_c_flat",
         "help": "Resume training or start testing from a checkpoint. Overrides config file if provided."},
        {"name": "--resume", "action": "store_true", "default": False, "help": "Resume training from a checkpoint"},
        {"name": "--experiment_name", "type": str,
         "help": "Name of the experiment to run or load. Overrides config file if provided."},
        {"name": "--run_name", "type": str, "default": "",
         "help": "Name of the run. Overrides config file if provided."},
        {"name": "--load_run", "type": str, "default": -1,
         "help": "Name of the run to load when resume=True. If -1: will load the last run. Overrides config file if provided."},
        {"name": "--checkpoint", "type": int, "default": -1,
         "help": "Saved model checkpoint number. If -1: will load the last checkpoint. Overrides config file if provided."},

        {"name": "--headless", "action": "store_true", "default": False, "help": "Force display off at all times."},
        {"name": "--horovod", "action": "store_true", "default": False, "help": "Use horovod for multi-gpu training."},
        {"name": "--video", "action": "store_true", "default": False, "help": "Record video during training."},
        {"name": "--record_length", "type": int, "default": 200, "help": "The number of steps to record for videos."},
        {"name": "--record_interval", "type": int, "default": 50,
         "help": "The number of iterations before each recording."},
        {"name": "--fps", "type": int, "default": 50, "help": "The fps of recorded videos."},
        {"name": "--frame_size", "type": tuple_type, "default": (1280, 720), "help": "The size of recorded frame."},
        {"name": "--camera_offset", "type": tuple_type, "default": (0, -2, 0),
         "help": "The offset of the video filming camera."},
        {"name": "--camera_rotation", "type": tuple_type, "default": (0, 0, 90),
         "help": "The rotation of the video filming camera."},
        {"name": "--env_idx_record", "type": int, "default": 0,
         "help": "The env idx to record."},
        {"name": "--actor_idx_record", "type": int, "default": 0,
         "help": "The actor idx to record."},
        {"name": "--rigid_body_idx_record", "type": int, "default": 0,
         "help": "The rigid_body idx to record."},
        {"name": "--rl_device", "type": str, "default": "cuda:0",
         "help": 'Device used by the RL algorithm, (cpu, gpu, cuda:0, cuda:1 etc..)'},
        {"name": "--num_envs", "type": int,
         "help": "Number of environments to create. Overrides config file if provided."},
        {"name": "--seed", "type": int, "help": "Random seed. Overrides config file if provided."},
        {"name": "--max_iterations", "type": int,
         "help": "Maximum number of training iterations. Overrides config file if provided."},
        {"name": "--logger", "type": str, "default": None, "choices": ["tensorboard", "wandb", "neptune"],
         "help": "Logger module to use.", },
        {"name": "--update_model", "action": "store_true", "default": False, help: "upload models to Wandb or Neptune."}
    ]
    # parse arguments
    args = gymutil.parse_arguments(
        description="RL Policy",
        custom_parameters=custom_parameters)

    # name allignment
    args.sim_device_id = args.compute_device_id
    args.sim_device = args.sim_device_type
    if args.sim_device == 'cuda':
        args.sim_device += f":{args.sim_device_id}"

    # value alignment
    if args.load_run == "-1":
        args.load_run = -1
    return args


def export_policy_as_jit(actor_critic, path):
    if hasattr(actor_critic, 'memory_a'):
        # assumes LSTM: TODO add GRU
        exporter = PolicyExporterLSTM(actor_critic)
        exporter.export(path)
    else:
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, 'policy.pt')
        model = actor_critic.actor
        traced_script_module = torch.jit.script(model)
        traced_script_module.save(path)

    print('Exported policy as jit script to: ', path)


def export_nn_as_onnx(model, path, name, input_dim):
    os.makedirs(path, exist_ok=True)
    path = os.path.join(path, name + ".onnx")
    model.eval()

    dummy_input = torch.randn(input_dim)
    input_names = ["nn_input"]
    output_names = ["nn_output"]

    torch.onnx.export(
        model,
        dummy_input,
        path,
        verbose=True,
        input_names=input_names,
        output_names=output_names,
        export_params=True,
        opset_version=13,
    )
    print("Exported policy as onnx script to: ", path)


class PolicyExporterLSTM(torch.nn.Module):
    def __init__(self, actor_critic):
        super().__init__()
        self.actor = copy.deepcopy(actor_critic.actor)
        self.is_recurrent = actor_critic.is_recurrent
        self.memory = copy.deepcopy(actor_critic.memory_a.rnn)
        self.memory.cpu()
        self.register_buffer(f'hidden_state', torch.zeros(self.memory.num_layers, 1, self.memory.hidden_size))
        self.register_buffer(f'cell_state', torch.zeros(self.memory.num_layers, 1, self.memory.hidden_size))

    def forward(self, x):
        out, (h, c) = self.memory(x.unsqueeze(0), (self.hidden_state, self.cell_state))
        self.hidden_state[:] = h
        self.cell_state[:] = c
        return self.actor(out.squeeze(0))

    @torch.jit.export
    def reset_memory(self):
        self.hidden_state[:] = 0.
        self.cell_state[:] = 0.

    def export(self, path):
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, 'policy_lstm.pt')
        self.to('cpu')
        traced_script_module = torch.jit.script(self)
        traced_script_module.save(path)
