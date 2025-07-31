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

from limx_legged_gym import LIMX_LEGGED_GYM_ROOT_DIR

import os
import moviepy.editor as mpy

from isaacgym import gymapi
from limx_legged_gym.envs import *
from limx_legged_gym.utils import get_args, class_to_dict, get_load_path, export_nn_as_onnx, export_policy_as_jit, \
    task_registry
from limx_rl.modules import ActorCritic, ActorCriticRecurrent

import torch


def export_policy(args):
    # prepare environment
    args.resume = True
    if args.experiment_name is None:
        raise ValueError('Please specify an experiment name.')
    args, task_class, log_dir, log_root, env_cfg, train_cfg = task_registry.get_env_train_cfg(name=args.task, args=args)
    resume_path = get_load_path(log_root, load_run=args.load_run, checkpoint=args.checkpoint)
    loaded_dict = torch.load(resume_path)
    actor_critic_class = eval(train_cfg.policy.class_name)
    if env_cfg.env.num_privileged_obs is None:
        env_cfg.env.num_privileged_obs = env_cfg.env.num_proprioceptive_obs
    actor_critic = actor_critic_class(
        env_cfg.env.num_proprioceptive_obs, env_cfg.env.num_privileged_obs, env_cfg.env.num_actions,
        **class_to_dict(train_cfg.policy)
    ).to("cpu")
    actor_critic.load_state_dict(loaded_dict['model_state_dict'])
    export_path = os.path.join(LIMX_LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name,
                               train_cfg.runner.run_name, 'exported')
    path = os.path.join(export_path, 'policies')
    if actor_critic_class is ActorCritic:
        export_nn_as_onnx(
            actor_critic.actor,
            path,
            "policy",
            env_cfg.env.num_proprioceptive_obs
        )
    elif actor_critic_class is ActorCriticRecurrent:
        export_nn_as_onnx(
            actor_critic.memory_a,
            path,
            "rnn",
            env_cfg.env.num_proprioceptive_obs)
        export_nn_as_onnx(
            actor_critic.actor,
            path,
            "policy",
            train_cfg.policy.rnn_hidden_size
        )
    export_policy_as_jit(actor_critic, path)


if __name__ == '__main__':
    args = get_args()
    export_policy(args)
