#  Copyright 2021 ETH Zurich, NVIDIA CORPORATION
#  SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.optim as optim

from limx_rl.modules import ActorCritic
from limx_rl.storage import RolloutStorage
from limx_rl.utils import unpad_trajectories


class PPO:
    actor_critic: ActorCritic

    def __init__(
            self,
            actor_critic,
            num_learning_epochs=1,
            num_mini_batches=1,
            clip_param=0.2,
            gamma=0.998,
            lam=0.95,
            value_loss_coef=1.0,
            entropy_coef=0.0,
            learning_rate=1e-3,
            encoder_learning_rate=1e-4,
            max_grad_norm=1.0,
            use_clipped_value_loss=True,
            schedule="fixed",
            desired_kl=0.01,
            device="cpu",
            **kwargs
    ):
        if kwargs:
            print(
                "PPO.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        self.device = device

        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate

        # PPO components
        self.actor_critic = actor_critic
        self.actor_critic.to(self.device)
        self.storage = None  # initialized later
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=self.learning_rate)
        self.extra_optimizer = optim.Adam(
            self.actor_critic.actor.encoder.parameters(), lr=encoder_learning_rate
        )

        self.transition = RolloutStorage.Transition()
        # PPO parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

    def init_storage(self, num_envs, num_transitions_per_env, num_commands, actor_obs_shape, critic_obs_shape,
                     obs_history_shape,
                     critic_obs_history_shape, action_shape):
        self.storage = RolloutStorage(
            num_envs, num_transitions_per_env, num_commands, actor_obs_shape, critic_obs_shape, obs_history_shape,
            critic_obs_history_shape, action_shape, self.device
        )

    def test_mode(self):
        self.actor_critic.eval()

    def train_mode(self):
        self.actor_critic.train()

    def act(self, obs, critic_obs, commands, obs_history, critic_obs_history):
        if getattr(self.actor_critic.actor, "is_recurrent", False):
            self.transition.hidden_states_a = self.actor_critic.actor.get_hidden_states()
        if getattr(self.actor_critic.critic, "is_recurrent", False):
            self.transition.hidden_states_c = self.actor_critic.critic.get_hidden_states()
        # Compute the actions and values
        self.transition.actions = self.actor_critic.act(torch.cat((obs, commands), dim=-1), obs_history).detach()
        self.transition.values = self.actor_critic.evaluate(torch.cat((critic_obs, commands), dim=-1),
                                                            critic_obs_history).detach()
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()
        # need to record obs and critic_obs before env.step()
        self.transition.observations = obs
        self.transition.critic_observations = critic_obs
        self.transition.obs_history = obs_history
        self.transition.critic_obs_history = critic_obs_history
        self.transition.commands = commands
        return self.transition.actions

    def process_env_step(self, rewards, dones, infos):
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        # Bootstrapping on timeouts
        if "time_outs" in infos:
            self.transition.rewards += self.gamma * torch.squeeze(
                self.transition.values * infos["time_outs"].unsqueeze(1).to(self.device), 1
            )

        # Record the transition
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.actor_critic.reset(dones)

    def compute_returns(self, last_critic_obs, last_critic_obs_history):
        last_values = self.actor_critic.evaluate(last_critic_obs, last_critic_obs_history).detach()
        self.storage.compute_returns(last_values, self.gamma, self.lam)

    def update(self):
        # todo: finish critic_encoder detach mode
        mean_value_loss = 0
        mean_surrogate_loss = 0
        if getattr(self.actor_critic.actor, "is_recurrent", False) or getattr(self.actor_critic.critic, "is_recurrent",
                                                                              False):
            generator = self.storage.reccurent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        for (
                obs_batch,
                critic_obs_batch,
                obs_history_batch,
                critic_obs_history_batch,
                command_batch,
                actions_batch,
                target_values_batch,
                advantages_batch,
                returns_batch,
                old_actions_log_prob_batch,
                old_mu_batch,
                old_sigma_batch,
                hid_states_batch,
                masks_batch,
        ) in generator:
            if getattr(self.actor_critic.actor, "is_recurrent", False) or getattr(self.actor_critic.critic,
                                                                                  "is_recurrent", False):
                if not getattr(self.actor_critic.actor, "is_recurrent", False):
                    obs_batch = unpad_trajectories(obs_batch, masks_batch)
                if not getattr(self.actor_critic.critic, "is_recurrent", False):
                    critic_obs_batch = unpad_trajectories(critic_obs_batch, masks_batch)
            self.actor_critic.act(torch.cat((obs_batch, command_batch), dim=-1), obs_history_batch, masks=masks_batch,
                                  hidden_states=hid_states_batch[0])


            actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
            value_batch = self.actor_critic.evaluate(
                torch.cat((critic_obs_batch, command_batch), dim=-1), critic_obs_history_batch, masks=masks_batch,
                hidden_states=hid_states_batch[1]
            )
            mu_batch = self.actor_critic.action_mean
            sigma_batch = self.actor_critic.action_std
            entropy_batch = self.actor_critic.entropy

            # KL
            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
                        + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch))
                        / (2.0 * torch.square(sigma_batch))
                        - 0.5,
                        axis=-1,
                    )
                    kl_mean = torch.mean(kl)

                    if kl_mean > self.desired_kl * 2.0:
                        self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                    elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                        self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            # Surrogate loss
            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # Value function loss
            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                    -self.clip_param, self.clip_param
                )
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            ppo_loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()

            self.optimizer.zero_grad()
            ppo_loss.backward()
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
            self.optimizer.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()

        num_updates_extra = 0
        mean_extra_loss = 0
        if self.extra_optimizer is not None:
            generator = self.storage.encoder_mini_batch_generator(
                self.num_mini_batches, self.num_learning_epochs
            )
            for (
                    obs_batch, critic_obs_batch, _, _
            ) in generator:
                encode_batch = self.actor_critic.actor.encoder(obs_history_batch)
                extra_loss = (
                    (encode_batch[:, 0:3] - critic_obs_batch[:, 0:3]).pow(2).mean()
                )

                self.extra_optimizer.zero_grad()
                extra_loss.backward()
                self.extra_optimizer.step()

                num_updates_extra += 1
                mean_extra_loss += extra_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        self.storage.clear()

        return None, None, mean_value_loss, mean_surrogate_loss
