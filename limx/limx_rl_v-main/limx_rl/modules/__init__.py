#  Copyright 2021 ETH Zurich, NVIDIA CORPORATION
#  SPDX-License-Identifier: BSD-3-Clause

"""Definitions for neural-network components for RL-agents."""

from limx_rl.modules.actor_critic.actor_critic import ActorCritic
from .normalizer import EmpiricalNormalization
from limx_rl.modules.actor_critic.mlp import MLP
from limx_rl.modules.actor_critic.encoder.proprioceptive_obs_encoder import ProprioceptiveObsEncoder


__all__ = ["ActorCritic", "MLP"]
