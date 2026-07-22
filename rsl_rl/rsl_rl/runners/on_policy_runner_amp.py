# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
import statistics
import time
import torch
import warnings
from collections import deque

import rsl_rl
from rsl_rl.algorithms import AMPPPO, AMPROAPPO
from rsl_rl.env import VecEnv
from rsl_rl.modules import ActorCritic, ActorCriticRecurrent, ActorCriticROA, resolve_rnd_config, resolve_symmetry_config, Discriminator
from rsl_rl.utils import resolve_obs_groups, store_code_state, AMPLoader, Normalizer


class OnPolicyRunnerAmp:
    """On-policy runner for training and evaluation of actor-critic methods."""

    _save_iteration_as_next = True

    def __init__(self, env: VecEnv, train_cfg: dict, log_dir: str | None = None, device="cpu"):
        self.cfg = train_cfg
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.device = device
        self.env = env

        # check if multi-gpu is enabled
        self._configure_multi_gpu()

        # store training configuration
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]

        # query observations from environment for algorithm construction
        obs = self.env.get_observations()
        default_sets = ["critic", "amp"]

        self.cfg["obs_groups"] = resolve_obs_groups(obs, self.cfg["obs_groups"], default_sets)

        # create the algorithm
        self.alg = self._construct_algorithm(obs)

        # Decide whether to disable logging
        # We only log from the process with rank 0 (main process)
        self.disable_logs = self.is_distributed and self.gpu_global_rank != 0

        # Logging
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0
        self.git_status_repos = [rsl_rl.__file__]

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False):  # noqa: C901
        # initialize writer
        self._prepare_logging_writer()

        # randomize initial episode lengths (for exploration)
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        # start learning
        obs = self.env.get_observations().to(self.device)

        amp_obs = None
        if hasattr(self.alg, "discriminator"):
            amp_obs = obs["amp"].to(self.device)
            if len(amp_obs.shape) == 3:
                amp_obs = amp_obs.view(amp_obs.shape[0], -1)

        self.train_mode()  # switch to train mode (for dropout for example)

        # Book keeping
        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        # Ensure all parameters are in-synced
        if self.is_distributed:
            print(f"Synchronizing parameters for rank {self.gpu_global_rank}...")
            self.alg.broadcast_parameters()

        # 记录上一帧的 AMP 观测
        current_amp_obs = amp_obs.clone() if amp_obs is not None else None

        # Start training
        start_iter = self.current_learning_iteration
        tot_iter = start_iter + num_learning_iterations
        for it in range(start_iter, tot_iter):
            start = time.time()
            # Rollout
            with torch.inference_mode():
                for _ in range(self.num_steps_per_env):
                    # Sample actions
                    actions = self.alg.act(obs, amp_obs=amp_obs)
                    # Step the environment
                    obs, rewards, dones, extras = self.env.step(actions.to(self.env.device))
                    # Move to device
                    obs, rewards, dones = (obs.to(self.device), rewards.to(self.device), dones.to(self.device))

                    # 1. 获取下一帧 AMP 观测
                    next_amp_obs = obs["amp"].to(self.device)
                    if len(next_amp_obs.shape) == 3:
                        next_amp_obs = next_amp_obs.view(next_amp_obs.shape[0], -1)
                        
                    # 2. 处理 Terminal States (关键：防止重置干扰判别器)
                    # 我们需要构建一个用于判别器训练的 "next_amp_obs"，其中重置的环境使用其重置前的最后一帧
                    next_amp_obs_with_term = next_amp_obs.clone()
                    amp_transition_valid = ~dones.bool()
                    # 通过 dones 找到哪些环境刚重置了
                    reset_env_ids = dones.nonzero(as_tuple=False).flatten()

                    if len(reset_env_ids) > 0:
                        # 安全获取 observations 字典，如果没有则返回空字典
                        obs_extras = extras.get("observations", {})
                        
                        # 从 observations 字典里安全获取 terminal_obs
                        terminal_obs_dict = obs_extras.get("terminal_obs")
                        
                        # 有些 Isaac Lab 版本/Wrapper 会把 terminal_obs 直接放在 extras 顶层
                        if terminal_obs_dict is None:
                            terminal_obs_dict = extras.get("terminal_obs")
                        # -----------------------------------
                        if terminal_obs_dict is not None and "amp" in terminal_obs_dict:
                            term_amp = terminal_obs_dict["amp"][reset_env_ids].to(self.device)
                            if len(term_amp.shape) == 3:
                                term_amp = term_amp.view(term_amp.shape[0], -1)
                            next_amp_obs_with_term[reset_env_ids] = term_amp
                            amp_transition_valid[reset_env_ids] = True
                        else:
                            # 上一有效窗口只用于终止步的保守 AMP reward，
                            # 不作为伪造的 (old, old) transition 写入 replay。
                            next_amp_obs_with_term[reset_env_ids] = current_amp_obs[reset_env_ids]

                    # 3. 调用算法处理 (传入 amp_obs)
                    # 在 AMPPPO.process_env_step 内部已经封装好了奖励计算逻辑
                    self.alg.process_env_step(
                        obs,
                        rewards,
                        dones,
                        extras,
                        next_amp_obs_with_term,
                        amp_transition_valid=amp_transition_valid,
                    )
                        
                    # 4. 更新当前帧
                    amp_obs = next_amp_obs
                    current_amp_obs = next_amp_obs.clone() 

                    # book keeping
                    if self.log_dir is not None:
                        if "episode" in extras:
                            ep_infos.append(extras["episode"])
                        elif "log" in extras:
                            ep_infos.append(extras["log"])

                        cur_reward_sum += rewards
                        # Update episode length
                        cur_episode_length += 1
                        # Clear data for completed episodes
                        # -- common
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

                stop = time.time()
                collection_time = stop - start
                start = stop

                # compute returns
                self.alg.compute_returns(obs)

            # update policy
            loss_dict = self.alg.update()

            stop = time.time()
            learn_time = stop - start
            # Store the next iteration to execute so resume does not repeat the
            # rollout/update that has just completed.
            self.current_learning_iteration = it + 1
            # log info
            if self.log_dir is not None and not self.disable_logs:
                # Log information
                self.log(locals())
                # Save model
                if it % self.save_interval == 0:
                    self.save(os.path.join(self.log_dir, f"model_{it}.pt"))

            # Clear episode infos
            ep_infos.clear()
            # Save code state
            if self.log_dir is not None and it == start_iter and not self.disable_logs:
                # obtain all the diff files
                git_file_paths = store_code_state(self.log_dir, self.git_status_repos)
                # if possible store them to wandb
                if self.logger_type in ["wandb", "neptune"] and git_file_paths:
                    for path in git_file_paths:
                        self.writer.save_file(path)

        # Save the final model after training
        if self.log_dir is not None and not self.disable_logs:
            self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))

    def log(self, locs: dict, width: int = 80, pad: int = 35):
        # Compute the collection size
        collection_size = self.num_steps_per_env * self.env.num_envs * self.gpu_world_size
        # Update total time-steps and time
        self.tot_timesteps += collection_size
        self.tot_time += locs["collection_time"] + locs["learn_time"]
        iteration_time = locs["collection_time"] + locs["learn_time"]

        # -- Episode info
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

        mean_std = self.alg.policy.action_std.mean()
        fps = int(collection_size / (locs["collection_time"] + locs["learn_time"]))

        # -- Losses
        for key, value in locs["loss_dict"].items():
            self.writer.add_scalar(f"Loss/{key}", value, locs["it"])
        self.writer.add_scalar("Loss/learning_rate", self.alg.learning_rate, locs["it"])

        # -- Policy
        self.writer.add_scalar("Policy/mean_noise_std", mean_std.item(), locs["it"])

        # -- Performance
        self.writer.add_scalar("Perf/total_fps", fps, locs["it"])
        self.writer.add_scalar("Perf/collection time", locs["collection_time"], locs["it"])
        self.writer.add_scalar("Perf/learning_time", locs["learn_time"], locs["it"])

        # -- Training
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
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                    'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
            )
            # -- Losses
            for key, value in locs["loss_dict"].items():
                log_string += f"""{f'Mean {key} loss:':>{pad}} {value:.4f}\n"""
            # -- Rewards
            log_string += f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
            # -- episode info
            log_string += f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n"""
        else:
            log_string = (
                f"""{'#' * width}\n"""
                f"""{str.center(width, ' ')}\n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                    'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
            )
            for key, value in locs["loss_dict"].items():
                log_string += f"""{f'{key}:':>{pad}} {value:.4f}\n"""

        log_string += ep_string
        log_string += (
            f"""{'-' * width}\n"""
            f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
            f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
            f"""{'Time elapsed:':>{pad}} {time.strftime("%H:%M:%S", time.gmtime(self.tot_time))}\n"""
            f"""{'ETA:':>{pad}} {time.strftime(
                "%H:%M:%S",
                time.gmtime(
                    self.tot_time / (locs['it'] - locs['start_iter'] + 1)
                    * (locs['start_iter'] + locs['num_learning_iterations'] - locs['it'])
                )
            )}\n"""
        )
        print(log_string)

    def save(self, path: str, infos=None):
        # 先创建基础字典
        saved_dict = {
            "model_state_dict": self.alg.policy.state_dict(),
            "optimizer_state_dict": self.alg.optimizer.state_dict(),
            "iter": self.current_learning_iteration,
            "iteration_is_next": getattr(self, "_save_iteration_as_next", False),
            "infos": infos,
        }
        # AMP判别器和归一化器状态
        if hasattr(self.alg, "discriminator"):
            saved_dict["discriminator_state_dict"] = self.alg.discriminator.state_dict()
            saved_dict["amp_normalizer"] = self.alg.amp_normalizer
        if hasattr(self.alg, "amp_optimizer"):
            saved_dict["amp_optimizer_state_dict"] = self.alg.amp_optimizer.state_dict()
        if getattr(self.alg, "hist_encoder_optimizer", None) is not None:
            saved_dict["hist_encoder_optimizer_state_dict"] = self.alg.hist_encoder_optimizer.state_dict()
        if hasattr(self.alg, "counter"):
            saved_dict["algorithm_counter"] = self.alg.counter
        torch.save(saved_dict, path)

        # upload model to external logging service
        if self.logger_type in ["neptune", "wandb"] and not self.disable_logs:
            self.writer.save_model(path, self.current_learning_iteration)

    def load(self, path: str, load_optimizer: bool = True, map_location: str | None = None):
        loaded_dict = torch.load(path, weights_only=False, map_location=map_location)
        self._loaded_checkpoint_metadata = {
            "iteration_is_next": bool(loaded_dict.get("iteration_is_next", False)),
            "algorithm_counter_present": "algorithm_counter" in loaded_dict,
        }
        # -- Load model
        resumed_training = self.alg.policy.load_state_dict(loaded_dict["model_state_dict"])
         # 加载 AMP 状态
        if "discriminator_state_dict" in loaded_dict:
            self.alg.discriminator.load_state_dict(loaded_dict["discriminator_state_dict"])
        else:
            print("[Warning] 'discriminator_state_dict' not found in checkpoint. Skipping.")
                
        if "amp_normalizer" in loaded_dict:
            self.alg.amp_normalizer = loaded_dict["amp_normalizer"]
        else:
            print("[Warning] 'amp_normalizer' not found in checkpoint. Skipping.")
        # -- load optimizer if used
        if load_optimizer and resumed_training:
            # -- algorithm optimizer
            self.alg.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
            self.alg.learning_rate = self.alg.optimizer.param_groups[0]["lr"]
            if hasattr(self.alg, "amp_optimizer"):
                if "amp_optimizer_state_dict" in loaded_dict:
                    self.alg.amp_optimizer.load_state_dict(loaded_dict["amp_optimizer_state_dict"])
                else:
                    print("[Warning] 'amp_optimizer_state_dict' not found in checkpoint. Using a fresh optimizer.")
            if getattr(self.alg, "hist_encoder_optimizer", None) is not None:
                if "hist_encoder_optimizer_state_dict" in loaded_dict:
                    self.alg.hist_encoder_optimizer.load_state_dict(
                        loaded_dict["hist_encoder_optimizer_state_dict"]
                    )
                else:
                    print(
                        "[Warning] 'hist_encoder_optimizer_state_dict' not found in checkpoint. "
                        "Using a fresh optimizer."
                    )
        # -- load current learning iteration
        if resumed_training:
            self.current_learning_iteration = loaded_dict["iter"]
            if hasattr(self.alg, "counter"):
                if "algorithm_counter" in loaded_dict:
                    self.alg.counter = int(loaded_dict["algorithm_counter"])
                else:
                    # Backward compatibility: old AMP-ROA checkpoints did not save
                    # the regularization schedule counter.
                    self.alg.counter = int(loaded_dict["iter"])
                    print(
                        "[Warning] 'algorithm_counter' not found in checkpoint. "
                        f"Falling back to iter={self.alg.counter}."
                    )
            if not self._loaded_checkpoint_metadata["iteration_is_next"]:
                legacy_iteration = self.current_learning_iteration
                self.current_learning_iteration = legacy_iteration + 1
                if hasattr(self.alg, "counter") and not self._loaded_checkpoint_metadata["algorithm_counter_present"]:
                    self.alg.counter = self.current_learning_iteration
                print(
                    "[AMP Resume] Migrated legacy checkpoint iteration "
                    f"from {legacy_iteration} to {self.current_learning_iteration}."
                )
        return loaded_dict.get("infos")

    def get_inference_policy(self, device=None):
        self.eval_mode()  # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.policy.to(device)
        return self.alg.policy.act_inference

    def train_mode(self):
        # -- PPO
        self.alg.policy.train()
        self.alg.discriminator.train()

    def eval_mode(self):
        # -- PPO
        self.alg.policy.eval()
        # 切换 Discriminator 模式
        self.alg.discriminator.eval()

    def add_git_repo_to_log(self, repo_file_path):
        self.git_status_repos.append(repo_file_path)

    """
    Helper functions.
    """

    def _configure_multi_gpu(self):
        """Configure multi-gpu training."""
        # check if distributed training is enabled
        self.gpu_world_size = int(os.getenv("WORLD_SIZE", "1"))
        self.is_distributed = self.gpu_world_size > 1

        # if not distributed training, set local and global rank to 0 and return
        if not self.is_distributed:
            self.gpu_local_rank = 0
            self.gpu_global_rank = 0
            self.multi_gpu_cfg = None
            return

        # get rank and world size
        self.gpu_local_rank = int(os.getenv("LOCAL_RANK", "0"))
        self.gpu_global_rank = int(os.getenv("RANK", "0"))

        # make a configuration dictionary
        self.multi_gpu_cfg = {
            "global_rank": self.gpu_global_rank,  # rank of the main process
            "local_rank": self.gpu_local_rank,  # rank of the current process
            "world_size": self.gpu_world_size,  # total number of processes
        }

        # check if user has device specified for local rank
        if self.device != f"cuda:{self.gpu_local_rank}":
            raise ValueError(
                f"Device '{self.device}' does not match expected device for local rank '{self.gpu_local_rank}'."
            )
        # validate multi-gpu configuration
        if self.gpu_local_rank >= self.gpu_world_size:
            raise ValueError(
                f"Local rank '{self.gpu_local_rank}' is greater than or equal to world size '{self.gpu_world_size}'."
            )
        if self.gpu_global_rank >= self.gpu_world_size:
            raise ValueError(
                f"Global rank '{self.gpu_global_rank}' is greater than or equal to world size '{self.gpu_world_size}'."
            )

        # initialize torch distributed
        torch.distributed.init_process_group(backend="nccl", rank=self.gpu_global_rank, world_size=self.gpu_world_size)
        # set device to the local rank
        torch.cuda.set_device(self.gpu_local_rank)

    def _construct_algorithm(self, obs) -> AMPPPO:
        """Construct the actor-critic algorithm."""
        
        # resolve RND config
        self.alg_cfg = resolve_rnd_config(self.alg_cfg, obs, self.cfg["obs_groups"], self.env)

        # resolve symmetry config
        self.alg_cfg = resolve_symmetry_config(self.alg_cfg, self.env)

        # resolve deprecated normalization config
        if self.cfg.get("empirical_normalization") is not None:
            warnings.warn(
                "The `empirical_normalization` parameter is deprecated. Please set `actor_obs_normalization` and "
                "`critic_obs_normalization` as part of the `policy` configuration instead.",
                DeprecationWarning,
            )
            if self.policy_cfg.get("actor_obs_normalization") is None:
                self.policy_cfg["actor_obs_normalization"] = self.cfg["empirical_normalization"]
            if self.policy_cfg.get("critic_obs_normalization") is None:
                self.policy_cfg["critic_obs_normalization"] = self.cfg["empirical_normalization"]

        # initialize the actor-critic
        actor_critic_class = eval(self.policy_cfg.pop("class_name"))
        actor_critic: ActorCritic = actor_critic_class(
            obs, self.cfg["obs_groups"], self.env.num_actions, **self.policy_cfg
        ).to(self.device)

        # initialize the algorithm
        alg_class = eval(self.alg_cfg.pop("class_name"))
        try:
            if hasattr(self.env, "step_dt"):
                # 如果环境对象已经有了计算好的 step_dt
                step_dt = self.env.step_dt
            elif hasattr(self.env, "unwrapped") and hasattr(self.env.unwrapped, "step_dt"):
                # 尝试从原始环境中获取
                step_dt = self.env.unwrapped.step_dt
            else:
                # 如果没有，根据配置手动计算
                physics_dt = self.env.cfg.sim.dt
                # 尝试获取 decimation，如果找不到则默认为 1 (即 RL 频率 = 物理频率)
                decimation = getattr(self.env.cfg, "decimation", 1)
                step_dt = physics_dt * decimation
        except Exception as e:
            print(f"[Warning] Could not automatically determine step_dt, using default 0.02s. Error: {e}")
            step_dt = 0.02
        print(f"[AMP] Time between frames (step_dt) set to: {step_dt}s")
        # 初始化 AMP 专用组件
        amp_data = AMPLoader(
            self.device,
            time_between_frames=step_dt,
            preload_transitions=True,
            num_preload_transitions=self.cfg["amp_num_preload_transitions"],
            motion_files=self.cfg["amp_motion_files"],
            history_length=self.cfg.get("amp_history_length", 1),
        )
        print(f"AMP Observation Dim: {amp_data.observation_dim}")
        print(f"AMP Frame Dim for Normalizer: {amp_data.frame_dim}")
        amp_normalizer = Normalizer(amp_data.frame_dim)
        use_history_window = self.cfg.get("amp_discriminator_history_window", False)
        discriminator_input_dim = amp_data.observation_dim if use_history_window else amp_data.observation_dim * 2
        discriminator = Discriminator(
            discriminator_input_dim,
            self.cfg["amp_reward_coef"],
            self.cfg["amp_discr_hidden_dims"],
            self.device,
            self.cfg["amp_task_reward_lerp"],
            dt=step_dt,
            use_history_window=use_history_window,
        ).to(self.device)
        min_std = torch.zeros(len(self.cfg["min_normalized_std"]), device=self.device, requires_grad=False)

        alg = alg_class(
            actor_critic,
            discriminator=discriminator,
            amp_data=amp_data,
            amp_normalizer=amp_normalizer,
            min_std=min_std,
            # 确保这些 key 在你的 config 文件中存在，或者在这里给默认值
            amp_replay_buffer_size=self.cfg.get("amp_replay_buffer_size", 100000), 
            amp_reward_coef=self.cfg.get("amp_reward_coef", 2.0),
            amp_task_reward_lerp=self.cfg.get("amp_task_reward_lerp", 0.3),
            disc_learning_rate=self.cfg.get("disc_learning_rate", 1e-4),
            device=self.device,
            **self.alg_cfg,
            multi_gpu_cfg=self.multi_gpu_cfg,
        )

        # initialize the storage
        alg.init_storage(
            "rl",
            self.env.num_envs,
            self.num_steps_per_env,
            obs,
            [self.env.num_actions],
        )

        return alg

    def _prepare_logging_writer(self):
        """Prepares the logging writers."""
        if self.log_dir is not None and self.writer is None and not self.disable_logs:
            # Launch either Tensorboard or Neptune & Tensorboard summary writer(s), default: Tensorboard.
            self.logger_type = self.cfg.get("logger", "tensorboard")
            self.logger_type = self.logger_type.lower()

            if self.logger_type == "neptune":
                from rsl_rl.utils.neptune_utils import NeptuneSummaryWriter

                self.writer = NeptuneSummaryWriter(log_dir=self.log_dir, flush_secs=10, cfg=self.cfg)
                self.writer.log_config(self.env.cfg, self.cfg, self.alg_cfg, self.policy_cfg)
            elif self.logger_type == "wandb":
                from rsl_rl.utils.wandb_utils import WandbSummaryWriter

                self.writer = WandbSummaryWriter(log_dir=self.log_dir, flush_secs=10, cfg=self.cfg)
                self.writer.log_config(self.env.cfg, self.cfg, self.alg_cfg, self.policy_cfg)
            elif self.logger_type == "tensorboard":
                from torch.utils.tensorboard import SummaryWriter

                self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
            else:
                raise ValueError("Logger type not found. Please choose 'neptune', 'wandb' or 'tensorboard'.")
