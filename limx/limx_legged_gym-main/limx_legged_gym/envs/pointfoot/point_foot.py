from isaacgym.torch_utils import *

import torch
from limx_legged_gym.envs import LeggedRobot
from limx_legged_gym.utils import class_to_dict
from limx_legged_gym.utils.math import wrap_to_pi


class PointFoot(LeggedRobot):
    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        self._resample_gaits(env_ids)

        self.gait_indices[env_ids] = 0

        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf | self.edge_reset_buf

    def compute_proprioception_observations(self):
        # note that observation noise need to modified accordingly !!!
        obs_buf = torch.cat(
            (
                self.base_ang_vel * self.obs_scales.ang_vel,
                self.projected_gravity,
                (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                self.dof_vel * self.obs_scales.dof_vel,
                self.actions,
                self.clock_inputs_sin.view(self.num_envs, 1),
                self.clock_inputs_cos.view(self.num_envs, 1),
                self.gaits,
            ),
            dim=-1,
        )
        return obs_buf

    def _post_physics_step_callback(self):
        """Callback called before computing terminations, rewards, and observations
        Default behaviour: Compute ang vel command based on target and heading, compute measured terrain heights and randomly push robots
        """
        env_ids = (
            (
                    self.episode_length_buf
                    % int(self.cfg.commands.resampling_time / self.dt)
                    == 0
            )
            .nonzero(as_tuple=False)
            .flatten()
        )
        self._resample_commands(env_ids)
        self._resample_gaits(env_ids)
        self._step_contact_targets()

        if self.cfg.commands.heading_command:
            forward = quat_apply(self.base_quat, self.forward_vec)
            heading = torch.atan2(forward[:, 1], forward[:, 0])
            self.commands[:, 2] = 1.0 * wrap_to_pi(self.commands[:, 3] - heading)

        if self.cfg.terrain.measure_heights or self.cfg.terrain.critic_measure_heights:
            self.measured_heights = self._get_heights()

        self.base_height = torch.mean(
            self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1
        )

    def _step_contact_targets(self):
        frequencies = self.gaits[:, 0]
        offsets = self.gaits[:, 1]
        durations = torch.cat(
            [
                self.gaits[:, 2].view(self.num_envs, 1),
                self.gaits[:, 2].view(self.num_envs, 1),
            ],
            dim=1,
        )
        self.gait_indices = torch.remainder(
            self.gait_indices + self.dt * frequencies, 1.0
        )

        self.clock_inputs_sin = torch.sin(2 * np.pi * self.gait_indices)
        self.clock_inputs_cos = torch.cos(2 * np.pi * self.gait_indices)
        # self.doubletime_clock_inputs_sin = torch.sin(4 * np.pi * foot_indices)
        # self.halftime_clock_inputs_sin = torch.sin(np.pi * foot_indices)

        # von mises distribution
        kappa = self.cfg.rewards.kappa_gait_probs
        smoothing_cdf_start = torch.distributions.normal.Normal(0, kappa).cdf

        foot_indices = torch.remainder(
            torch.cat(
                [
                    self.gait_indices.view(self.num_envs, 1),
                    (self.gait_indices + offsets + 1).view(self.num_envs, 1),
                ],
                dim=1,
            ),
            1.0,
        )
        stance_idxs = foot_indices < durations
        swing_idxs = foot_indices > durations

        # 感觉这两句可以不要
        foot_indices[stance_idxs] = torch.remainder(foot_indices[stance_idxs], 1) * (
                0.5 / durations[stance_idxs]
        )
        foot_indices[swing_idxs] = 0.5 + (
                torch.remainder(foot_indices[swing_idxs], 1) - durations[swing_idxs]
        ) * (0.5 / (1 - durations[swing_idxs]))

        # 就是为了拟合论文中的曲线，也可以用更正常的方法
        self.desired_contact_states = smoothing_cdf_start(foot_indices) * (
                1 - smoothing_cdf_start(foot_indices - 0.5)
        ) + smoothing_cdf_start(foot_indices - 1) * (
                                              1 - smoothing_cdf_start(foot_indices - 1.5)
                                      )

    def _parse_cfg(self, cfg):
        super()._parse_cfg(cfg)
        self.gaits_ranges = class_to_dict(self.cfg.gait.ranges)

    def _resample_gaits(self, env_ids):
        if len(env_ids) == 0:
            return
        self.gaits[env_ids, 0] = torch_rand_float(
            self.gaits_ranges["frequencies"][0],
            self.gaits_ranges["frequencies"][1],
            (len(env_ids), 1),
            device=self.device,
        ).squeeze(1)

        self.gaits[env_ids, 1] = torch_rand_float(
            self.gaits_ranges["offsets"][0],
            self.gaits_ranges["offsets"][1],
            (len(env_ids), 1),
            device=self.device,
        ).squeeze(1)
        # parts = 4
        # self.gaits[env_ids, 1] = (self.gaits[env_ids, 1] * parts).round() / parts
        self.gaits[env_ids, 1] = 0.5

        self.gaits[env_ids, 2] = torch_rand_float(
            self.gaits_ranges["durations"][0],
            self.gaits_ranges["durations"][1],
            (len(env_ids), 1),
            device=self.device,
        ).squeeze(1)
        # parts = 2
        # self.gaits[env_ids, 2] = (self.gaits[env_ids, 2] * parts).round() / parts

        self.gaits[env_ids, 3] = torch_rand_float(
            self.gaits_ranges["swing_height"][0],
            self.gaits_ranges["swing_height"][1],
            (len(env_ids), 1),
            device=self.device,
        ).squeeze(1)

    def _init_buffers(self):
        super()._init_buffers()

        self.gaits = torch.zeros(
            self.num_envs,
            self.cfg.gait.num_gait_params,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.desired_contact_states = torch.zeros(
            self.num_envs,
            len(self.feet_indices),
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.gait_indices = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.clock_inputs_sin = torch.zeros(
            self.num_envs,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.clock_inputs_cos = torch.zeros(
            self.num_envs,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.doubletime_clock_inputs_sin = torch.zeros(
            self.num_envs,
            len(self.feet_indices),
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.halftime_clock_inputs_sin = torch.zeros(
            self.num_envs,
            len(self.feet_indices),
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )

    def _reward_single_contact(self):
        contacts = self.contact_forces[:, self.feet_indices, 2] > 0.1
        single_contact = torch.sum(1.0 * contacts, dim=1) == 1
        return 1.0 * single_contact

    def _reward_no_fly(self):
        contacts = self.contact_forces[:, self.feet_indices, 2] > 0.1
        zero_contact = torch.sum(1.0 * contacts, dim=1) == 0
        return 1.0 * zero_contact

    def _reward_feet_height(self):
        feet_height = self.cfg.rewards.base_height_target * 0.05

        # penalize stand still
        reward = torch.sum(
            torch.exp(-self.foot_heights / feet_height)
            * torch.exp(-torch.norm(self.commands[:, :3], dim=1, keepdim=True)).repeat(1, len(self.feet_indices)),
            dim=1,
        )

        reward += torch.sum(
            torch.exp(-self.foot_heights / feet_height)
            * torch.square(torch.norm(self.foot_velocities[:, :, :2], dim=-1)),
            dim=1,
        )
        feet_height *= 0.5
        reward += torch.sum(
            torch.exp(-self.foot_heights / feet_height)
            * torch.square(torch.abs(self.foot_velocities[:, :, 2])),
            dim=1,
        )

        return reward

    def _reward_tracking_contacts_shaped_force(self):
        foot_forces = torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1)
        desired_contact = self.desired_contact_states

        reward = 0
        if self.reward_scales["tracking_contacts_shaped_force"] > 0:
            for i in range(len(self.feet_indices)):
                reward += (1 - desired_contact[:, i]) * torch.exp(
                    -foot_forces[:, i] ** 2 / self.cfg.rewards.gait_force_sigma
                )
        else:
            for i in range(len(self.feet_indices)):
                reward += (1 - desired_contact[:, i]) * (
                        1
                        - torch.exp(
                    -foot_forces[:, i] ** 2 / self.cfg.rewards.gait_force_sigma
                )
                )

        return reward / len(self.feet_indices)

    def _reward_tracking_contacts_shaped_vel(self):
        foot_velocities = torch.norm(self.foot_velocities, dim=-1)
        desired_contact = self.desired_contact_states
        reward = 0
        if self.reward_scales["tracking_contacts_shaped_vel"] > 0:
            for i in range(len(self.feet_indices)):
                reward += desired_contact[:, i] * torch.exp(
                    -foot_velocities[:, i] ** 2 / self.cfg.rewards.gait_vel_sigma
                )
        else:
            for i in range(len(self.feet_indices)):
                reward += desired_contact[:, i] * (
                        1
                        - torch.exp(
                    -foot_velocities[:, i] ** 2 / self.cfg.rewards.gait_vel_sigma
                )
                )
        return reward / len(self.feet_indices)

    def _reward_feet_distance(self):
        # Penalize base height away from target
        feet_distance = torch.norm(
            self.foot_positions[:, 0, :2] - self.foot_positions[:, 1, :2], dim=-1
        )
        reward = torch.clip(self.cfg.rewards.min_feet_distance - feet_distance, 0, 1)
        return reward
