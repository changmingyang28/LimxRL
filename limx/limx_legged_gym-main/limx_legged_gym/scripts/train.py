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


from isaacgym import gymapi
from limx_legged_gym.envs import *
from limx_legged_gym.utils import get_args, task_registry, start_learning
from limx_legged_gym.utils.VecGymWrapper import VecGymWrapper
from limx_legged_gym.utils.RecordVideoWrapper import RecordVideoWrapper


def train(args):
    args, task_class, log_dir, log_root, env_cfg, train_cfg = task_registry.get_env_train_cfg(name=args.task, args=args)
    env, env_cfg = task_registry.make_env(task_class, args, env_cfg, log_dir)
    env = VecGymWrapper(env)
    env = RecordVideoWrapper(env)
    ppo_runner, train_cfg = task_registry.make_alg_runner(env, args, train_cfg, log_root, log_dir)
    start_learning(ppo_runner, num_learning_iterations=train_cfg.runner.max_iterations,
                   init_at_random_ep_len=True)


if __name__ == '__main__':
    args = get_args()
    train(args)
