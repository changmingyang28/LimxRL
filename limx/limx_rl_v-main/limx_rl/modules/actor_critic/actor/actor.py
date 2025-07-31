#  Copyright 2021 ETH Zurich, NVIDIA CORPORATION
#  SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions import Normal
from abc import ABC, abstractmethod


class Actor(nn.Module, ABC):

    def __init__(
            self,
            num_obs,
            num_commands,
            num_actions,
            init_noise_std=1.0
    ):
        super().__init__()
        # Action noise
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False

    @abstractmethod
    def reset(self, dones=None):
        raise NotImplementedError

    @abstractmethod
    def forward(self, observations, obs_history, masks=None, hidden_states=None):
        raise NotImplementedError

    def get_parameters(self):
        return None, list(self.parameters())

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, observations, obs_history, masks, hidden_states):
        mean = self(observations=observations, obs_history=obs_history, masks=masks, hidden_states=hidden_states)
        self.distribution = Normal(mean, mean * 0.0 + self.std)

    def act(self, observations, obs_history, masks=None, hidden_states=None, **kwargs):
        self.update_distribution(observations, obs_history, masks, hidden_states)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations, obs_history):
        actions_mean = self(observations, obs_history)
        return actions_mean
