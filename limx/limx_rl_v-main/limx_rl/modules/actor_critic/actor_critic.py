#  Copyright 2021 ETH Zurich, NVIDIA CORPORATION
#  SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import torch.nn as nn


class ActorCritic(nn.Module):
    is_recurrent = False

    def __init__(
            self,
            actor,
            critic,
            **kwargs,
    ):
        if kwargs:
            print(
                "ActorCritic.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()
        self.actor = actor
        self.critic = critic

        print(f"Actor: {self.actor}")
        print(f"Critic: {self.critic}")

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.actor.action_mean

    @property
    def action_std(self):
        return self.actor.action_std

    @property
    def entropy(self):
        return self.actor.entropy

    def update_distribution(self, observations):
        self.actor.update_distribution(observations)

    def act(self, observations, obs_history, masks=None, hidden_states=None, **kwargs):
        return self.actor.act(observations=observations, obs_history=obs_history, masks=masks, hidden_states=hidden_states, **kwargs)

    def get_actions_log_prob(self, actions):
        return self.actor.get_actions_log_prob(actions)

    def act_inference(self, observations):
        return self.actor.act_inference(observations)

    def evaluate(self, critic_observations, critic_obs_history, **kwargs):
        return self.critic.evaluate(critic_observations, critic_obs_history, **kwargs)
