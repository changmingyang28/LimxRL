#  Copyright 2021 ETH Zurich, NVIDIA CORPORATION
#  SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os, sys
import statistics
import time
import torch
from collections import deque
from torch.utils.tensorboard import SummaryWriter as TensorboardSummaryWriter

import limx_rl
from limx_rl.algorithms import PPO
from limx_rl.env import VecEnv, VideoEnv
from limx_rl.modules import EmpiricalNormalization
from limx_rl.modules.actor_critic import *
from limx_rl.utils import store_code_state


class OnPolicyRunner:
    """On-policy runner for training and evaluation."""

    def __init__(self, env: VecEnv | VideoEnv, train_cfg, log_dir=None, device="cpu"):
        self.cfg = train_cfg
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.device = device
        self.env = env
        obs, infos = self.env.get_observations()
        num_obs = obs.shape[1]
        if "critic" in infos["observations"]:
            num_critic_obs = infos["observations"]["critic"].shape[1]
        else:
            num_critic_obs = num_obs
        if "encoder" in self.policy_cfg["actor"]:
            encoder_class = eval(self.policy_cfg["actor"]["encoder"]["class_name"])
            if getattr(encoder_class, "is_recurrent", False):
                actor_encoder = encoder_class(input_dim=self.env.num_obs,
                                              **self.policy_cfg["actor"]["encoder"]).to(self.device)
            else:
                actor_encoder = encoder_class(input_dim=self.env.num_obs * self.env.obs_history_length,
                                              **self.policy_cfg["actor"]["encoder"]).to(self.device)
            del self.policy_cfg["actor"]["encoder"]
        else:
            actor_encoder = None
        if "encoder" in self.policy_cfg["critic"]:
            encoder_class = eval(self.policy_cfg["critic"]["encoder"]["class_name"])
            if getattr(encoder_class, "is_recurrent", False):
                critic_encoder = encoder_class(
                    input_dim=self.env.num_privileged_obs,
                    **self.policy_cfg["critic"]["encoder"]).to(self.device)
            else:
                critic_encoder = encoder_class(
                    input_dim=self.env.num_privileged_obs * self.env.critic_obs_history_length,
                    **self.policy_cfg["critic"]["encoder"]).to(self.device)
            del self.policy_cfg["critic"]["encoder"]
        else:
            critic_encoder = None
        actor_class = eval(self.policy_cfg["actor"]["class_name"])
        critic_class = eval(self.policy_cfg["critic"]["class_name"])
        if actor_encoder is not None:
            actor = actor_class(encoder=actor_encoder, num_obs=num_obs, num_commands=self.env.num_commands,
                                num_actions=self.env.num_actions,
                                **self.policy_cfg["actor"])
        else:
            actor = actor_class(num_obs=num_obs, num_commands=self.env.num_commands, num_actions=self.env.num_actions,
                                **self.policy_cfg["actor"])
        if critic_encoder is not None:
            critic = critic_class(encoder=critic_encoder, num_critic_obs=num_critic_obs,
                                  num_commands=self.env.num_commands, **self.policy_cfg["critic"])
        else:
            critic = critic_class(num_critic_obs=num_critic_obs, num_commands=self.env.num_commands,
                                  **self.policy_cfg["critic"])
        actor_critic = ActorCritic(actor, critic).to(self.device)
        alg_class = eval(self.alg_cfg["class_name"])
        self.alg: PPO = alg_class(actor_critic, device=self.device, **self.alg_cfg)
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]

        obs_history_shape = [infos["obs_history"]["actor"].shape[1]]
        critic_obs_history_shape = [(infos["obs_history"].get("critic", infos["obs_history"]["actor"])).shape[1]]
        self.empirical_normalization = self.cfg["empirical_normalization"]
        if self.empirical_normalization:
            self.obs_normalizer = EmpiricalNormalization(shape=[num_obs], until=1.0e8).to(self.device)
            self.critic_obs_normalizer = EmpiricalNormalization(shape=[num_critic_obs], until=1.0e8).to(self.device)
            self.obs_history_normalizer = EmpiricalNormalization(
                shape=obs_history_shape, until=1.0e8).to(self.device)
            self.critic_obs_history_normalizer = EmpiricalNormalization(shape=critic_obs_history_shape, until=1.0e8).to(
                self.device)
        else:
            self.obs_normalizer = torch.nn.Identity()  # no normalization
            self.critic_obs_normalizer = torch.nn.Identity()  # no normalization
            self.obs_history_normalizer = torch.nn.Identity()  # no normalization
            self.critic_obs_history_normalizer = torch.nn.Identity()  # no normalization
        # init storage and model
        self.alg.init_storage(
            self.env.num_envs,
            self.num_steps_per_env,
            self.env.num_commands,
            [num_obs],
            [num_critic_obs],
            obs_history_shape,
            critic_obs_history_shape,
            [self.env.num_actions],
        )

        # Log
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0
        self.git_status_repos = [limx_rl.__file__]

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False):
        # initialize writer
        if self.log_dir is not None and self.writer is None:
            # Launch either Tensorboard or Neptune & Tensorboard summary writer(s), default: Tensorboard.
            self.logger_type = self.cfg.get("logger", "tensorboard")
            self.logger_type = self.logger_type.lower()

            if self.logger_type == "neptune":
                from limx_rl.utils.neptune_utils import NeptuneSummaryWriter

                self.writer = NeptuneSummaryWriter(log_dir=self.log_dir, flush_secs=10, cfg=self.cfg)
                self.writer.log_config(self.env.cfg, self.cfg, self.alg_cfg, self.policy_cfg)
            elif self.logger_type == "wandb":
                from limx_rl.utils.wandb_utils import WandbSummaryWriter

                self.writer = WandbSummaryWriter(log_dir=self.log_dir, flush_secs=10, cfg=self.cfg)
                self.writer.log_config(self.env.cfg, self.cfg, self.alg_cfg, self.policy_cfg)
            elif self.logger_type == "tensorboard":
                self.writer = TensorboardSummaryWriter(log_dir=self.log_dir, flush_secs=10)
            else:
                raise AssertionError("logger type not found")
        checkpoint_path = os.path.join(self.log_dir, "")
        if not os.path.isdir(checkpoint_path):
            os.makedirs(checkpoint_path, exist_ok=True)
        if self.cfg["video"]:
            if self.cfg["record_length"] > self.num_steps_per_env * self.cfg["record_interval"]:
                print(
                    "Warning: record_length is larger than (num_steps_per_env * record_interval), which reuslts the actual record_interval is larger than the set.",
                    file=sys.stderr)
            video_path = os.path.join(self.log_dir, "videos")
            if not os.path.isdir(video_path):
                os.makedirs(video_path, exist_ok=True)
            self.env.set_camera_video_props(frame_size=self.cfg["frame_size"], camera_offset=self.cfg["camera_offset"],
                                            camera_rotation=self.cfg["camera_rotation"],
                                            env_idx=self.cfg["env_idx_record"], actor_idx=self.cfg["actor_idx_record"],
                                            rigid_body_idx=self.cfg["rigid_body_idx_record"], fps=self.cfg["fps"])
            frame_idx = -1
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )
        obs, infos = self.env.get_observations()
        obs_history = infos["obs_history"]["actor"]
        critic_obs = infos["observations"].get("critic", obs)
        critic_obs_history = infos["obs_history"].get("critic", obs_history)
        commands = infos["commands"]
        obs, critic_obs, obs_history, critic_obs_history, commands = obs.to(self.device), critic_obs.to(
            self.device), obs_history.to(self.device), critic_obs_history.to(self.device), commands.to(self.device)
        self.train_mode()  # switch to train mode (for dropout for example)

        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        start_iter = self.current_learning_iteration
        tot_iter = start_iter + num_learning_iterations

        for it in range(start_iter, tot_iter):
            start = time.time()
            # Rollout
            with (torch.inference_mode()):
                for i in range(self.num_steps_per_env):
                    if self.cfg["video"]:
                        if it % self.cfg["record_interval"] == 0 and frame_idx == -1:
                            video_save_it = it
                            frame_idx = 0
                            self.env.start_recording_video()
                        elif frame_idx != -1:
                            frame_idx += 1
                        if frame_idx == self.cfg["record_length"] - 1:
                            self.env.end_and_save_recording_video(os.path.join(self.log_dir, "videos"),
                                                                  f"{video_save_it}.mp4")
                            frame_idx = -1

                    actions = self.alg.act(obs, critic_obs, commands, obs_history, critic_obs_history)
                    obs, rewards, dones, infos = self.env.step(actions)
                    obs_history = infos["obs_history"]["actor"]
                    obs = self.obs_normalizer(obs)
                    obs_history = self.obs_history_normalizer(obs_history)
                    if "critic" in infos["observations"]:
                        critic_obs = self.critic_obs_normalizer(infos["observations"]["critic"])
                        critic_obs_history = infos["obs_history"]["critic"]
                    else:
                        critic_obs = obs
                        critic_obs_history = obs_history
                    commands = infos["commands"]
                    obs, critic_obs, obs_history, critic_obs_history, rewards, dones, commands = (
                        obs.to(self.device),
                        critic_obs.to(self.device),
                        obs_history.to(self.device),
                        critic_obs_history.to(self.device),
                        rewards.to(self.device),
                        dones.to(self.device),
                        commands.to(self.device)
                    )
                    self.alg.process_env_step(rewards, dones, infos)

                    if self.log_dir is not None:
                        # Book keeping
                        # note: we changed logging to use "log" instead of "episode" to avoid confusion with
                        # different types of logging data (rewards, curriculum, etc.)
                        if "episode" in infos:
                            ep_infos.append(infos["episode"])
                        elif "log" in infos:
                            ep_infos.append(infos["log"])
                        cur_reward_sum += rewards
                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

                stop = time.time()
                collection_time = stop - start

                # Learning step
                start = stop
                self.alg.compute_returns(torch.cat((critic_obs, commands), dim=-1), critic_obs_history)

            mean_actor_encoder_loss, mean_critic_encoder_loss, mean_value_loss, mean_surrogate_loss = self.alg.update()
            stop = time.time()
            learn_time = stop - start
            self.current_learning_iteration = it
            if self.log_dir is not None:
                self.log(locals())
            if it % self.save_interval == 0:
                self.save(os.path.join(self.log_dir, f"model_{it}.pt"))
            ep_infos.clear()
            if it == start_iter:
                # obtain all the diff files
                git_file_paths = store_code_state(self.log_dir, self.git_status_repos)
                # if possible store them to wandb
                if self.logger_type in ["wandb", "neptune"] and git_file_paths:
                    for path in git_file_paths:
                        self.writer.save_file(path)

        self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))

    def log(self, locs: dict, width: int = 80, pad: int = 35):
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += locs["collection_time"] + locs["learn_time"]
        iteration_time = locs["collection_time"] + locs["learn_time"]

        ep_string = ""
        if locs["ep_infos"]:
            for key in locs["ep_infos"][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs["ep_infos"]:
                    # handle scalar and zero dimensional tensor infos
                    if key not in ep_info:
                        continue
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                # log to logger and terminal
                if "/" in key:
                    self.writer.add_scalar(key, value, locs["it"])
                    ep_string += f"""{f'{key}:':>{pad}} {value:.4f}\n"""
                else:
                    self.writer.add_scalar("Episode/" + key, value, locs["it"])
                    ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""
        mean_std = self.alg.actor_critic.actor.std.mean()
        fps = int(self.num_steps_per_env * self.env.num_envs / (locs["collection_time"] + locs["learn_time"]))

        if locs["mean_actor_encoder_loss"] is not None:
            self.writer.add_scalar("Loss/actor_encoder", locs["mean_actor_encoder_loss"], locs["it"])
        if locs["mean_critic_encoder_loss"] is not None:
            self.writer.add_scalar("Loss/critic_encoder", locs["mean_critic_encoder_loss"], locs["it"])
        self.writer.add_scalar("Loss/value_function", locs["mean_value_loss"], locs["it"])
        self.writer.add_scalar("Loss/surrogate", locs["mean_surrogate_loss"], locs["it"])
        self.writer.add_scalar("Loss/learning_rate", self.alg.learning_rate, locs["it"])
        self.writer.add_scalar("Policy/mean_noise_std", mean_std.item(), locs["it"])
        self.writer.add_scalar("Perf/total_fps", fps, locs["it"])
        self.writer.add_scalar("Perf/collection time", locs["collection_time"], locs["it"])
        self.writer.add_scalar("Perf/learning_time", locs["learn_time"], locs["it"])
        if len(locs["rewbuffer"]) > 0:
            self.writer.add_scalar("Train/mean_reward", statistics.mean(locs["rewbuffer"]), locs["it"])
            self.writer.add_scalar("Train/mean_episode_length", statistics.mean(locs["lenbuffer"]), locs["it"])
            if self.logger_type != "wandb":  # wandb does not support non-integer x-axis logging
                self.writer.add_scalar("Train/mean_reward/time", statistics.mean(locs["rewbuffer"]), self.tot_time)
                self.writer.add_scalar(
                    "Train/mean_episode_length/time", statistics.mean(locs["lenbuffer"]), self.tot_time
                )

        str = f" \033[1m Learning iteration {locs['it']}/{locs['tot_iter']} \033[0m "

        if len(locs["rewbuffer"]) > 0:
            log_string = (
                f"""{'#' * width}\n"""
                f"""{str.center(width, ' ')}\n\n"""
                f"""{'Run name:':>{pad}} {self.cfg["runner"]["run_name"]} \n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                    'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n"""
            )
            #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
            #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")
        else:
            log_string = (
                f"""{'#' * width}\n"""
                f"""{str.center(width, ' ')}\n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                    'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
            )
            #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
            #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")

        log_string += ep_string
        log_string += (
            f"""{'-' * width}\n"""
            f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
            f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
            f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
            f"""{'ETA:':>{pad}} {self.tot_time / (locs['it'] - locs['start_iter'] + 1) * (
                    locs['tot_iter'] - locs['it']):.1f}s\n"""
        )
        print(log_string)

    def save(self, path, infos=None):
        extra_optimizer_state_dict = self.alg.extra_optimizer.state_dict() if self.alg.extra_optimizer is not None else None

        saved_dict = {
            "model_state_dict": self.alg.actor_critic.state_dict(),
            "optimizer_state_dict": self.alg.optimizer.state_dict(),
            "extra_optimizer_state_dict": extra_optimizer_state_dict,
            "iter": self.current_learning_iteration,
            "infos": infos,
        }
        if self.empirical_normalization:
            saved_dict["obs_norm_state_dict"] = self.obs_normalizer.state_dict()
            saved_dict["critic_obs_norm_state_dict"] = self.critic_obs_normalizer.state_dict()
        torch.save(saved_dict, path)

        # Upload model to external logging service
        if self.logger_type in ["neptune", "wandb"]:
            if self.cfg["update_model"]:
                self.writer.save_model(path, self.current_learning_iteration)

    def load(self, path, load_optimizer=True):
        loaded_dict = torch.load(path)
        self.alg.actor_critic.load_state_dict(loaded_dict["model_state_dict"])
        if self.empirical_normalization:
            self.obs_normalizer.load_state_dict(loaded_dict["obs_norm_state_dict"])
            self.critic_obs_normalizer.load_state_dict(loaded_dict["critic_obs_norm_state_dict"])
        if load_optimizer:
            self.alg.ppo_optimizer.load_state_dict(loaded_dict["ppo_optimizer_state_dict"])
            if loaded_dict["actor_encoder_optimizer_state_dict"] is not None:
                self.alg.actor_encoder_optimizer.load_state_dict(loaded_dict["actor_encoder_optimizer_state_dict"])
            if loaded_dict["critic_encoder_optimizer_state_dict"] is not None:
                self.alg.critic_encoder_optimizer.load_state_dict(loaded_dict["critic_encoder_optimizer_state_dict"])
        self.current_learning_iteration = loaded_dict["iter"]
        return loaded_dict["infos"]

    def get_inference_policy(self, device=None):
        self.eval_mode()  # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.actor_critic.to(device)
        policy = self.alg.actor_critic.act_inference
        if self.cfg["empirical_normalization"]:
            if device is not None:
                self.obs_normalizer.to(device)
            policy = lambda x: self.alg.actor_critic.act_inference(self.obs_normalizer(x))  # noqa: E731
        return policy

    def train_mode(self):
        self.alg.actor_critic.train()
        if self.empirical_normalization:
            self.obs_normalizer.train()
            self.critic_obs_normalizer.train()

    def eval_mode(self):
        self.alg.actor_critic.eval()
        if self.empirical_normalization:
            self.obs_normalizer.eval()
            self.critic_obs_normalizer.eval()

    def add_git_repo_to_log(self, repo_file_path):
        self.git_status_repos.append(repo_file_path)
