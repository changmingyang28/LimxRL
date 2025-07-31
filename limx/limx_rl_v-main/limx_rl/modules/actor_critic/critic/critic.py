#  Copyright 2021 ETH Zurich, NVIDIA CORPORATION
#  SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import torch.nn as nn
from abc import ABC, abstractmethod


class Critic(nn.Module, ABC):

    def __init__(
            self,
            num_critic_obs,
            num_commands
    ):
        super().__init__()
    @abstractmethod
    def reset(self, dones=None):
        raise NotImplementedError
    @abstractmethod
    def forward(self, critic_observations, critic_obs_history, masks, hidden_states):
        raise NotImplementedError
    def get_parameters(self):
        return None, list(self.parameters())
    def evaluate(self, critic_observations, critic_obs_history, masks, hidden_states):
        value = self(critic_observations, critic_obs_history, masks, hidden_states)
        return value
