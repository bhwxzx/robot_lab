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
from rsl_rl.algorithms import AMPDWAQPPO
from rsl_rl.env import VecEnv
from rsl_rl.modules import ActorCriticDwaq, Discriminator, resolve_rnd_config, resolve_symmetry_config
from rsl_rl.utils import resolve_obs_groups, store_code_state, AMPLoader, Normalizer


class OnPolicyRunnerAmpDwaq:
    """
    OnPolicyRunnerAmpDwaq: 结合了 AMP 和 DWAQ 的运行器。
    
    功能整合：
    1. AMP (Adversarial Motion Prior): 管理 Motion Loader, Discriminator, 处理 (s, s') 转换和风格奖励。
    2. DWAQ (Domain Randomization with VAE): 管理 prev_critic_obs 用于 VAE 的速度估计监督。
    """

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
        
        # 必须同时包含 DWAQ 所需的 policy/critic 和 AMP 所需的 amp
        default_sets = ["critic", "policy", "amp"]
        self.cfg["obs_groups"] = resolve_obs_groups(obs, self.cfg["obs_groups"], default_sets)

        # create the algorithm
        self.alg = self._construct_algorithm(obs)

        # Decide whether to disable logging
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

        # randomize initial episode lengths
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        # start learning
        obs = self.env.get_observations().to(self.device)

        # --- [Init: AMP Data] ---
        # amp_obs 用于判别器输入，需要记录当前帧(s)和下一帧(s')
        amp_obs = obs["amp"].to(self.device)
        if len(amp_obs.shape) == 3:
            amp_obs = amp_obs.view(amp_obs.shape[0], -1)
        # 记录上一帧 AMP 观测 (用于处理 reset 时的 terminal state 近似)
        current_amp_obs = amp_obs.clone()

        # --- [Init: DWAQ Data] ---
        # 初始化“上一时刻特权观测”，第一步用当前时刻填充
        # prev_critic_obs 用于 VAE 估计当前速度 (v_t 依赖于 o_t 和 o_{t-1})
        prev_critic_obs = obs["critic"].clone()

        self.train_mode()

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

        # Start training
        start_iter = self.current_learning_iteration
        tot_iter = start_iter + num_learning_iterations
        for it in range(start_iter, tot_iter):
            start = time.time()
            # Rollout
            with torch.inference_mode():
                for _ in range(self.num_steps_per_env):
                    # 1. [DWAQ & AMP] Act
                    # 同时传入 obs, prev_critic_obs (DWAQ用), amp_obs (AMP判别器用, 此时为 s_t)
                    actions = self.alg.act(obs, prev_critic_obs, amp_obs=amp_obs)

                    # 2. [DWAQ] Backup Critic Obs before step
                    # 在环境步进前，备份当前的 critic obs，它将在下一步成为 "prev"
                    last_critic_before_step = obs["critic"].clone()

                    # 3. Step Environment
                    obs, rewards, dones, extras = self.env.step(actions.to(self.env.device))
                    # Move to device
                    obs, rewards, dones = (obs.to(self.device), rewards.to(self.device), dones.to(self.device))

                    # 4. [AMP] Handle Next Obs & Terminal States
                    # 获取 s_{t+1}
                    next_amp_obs = obs["amp"].to(self.device)
                    if len(next_amp_obs.shape) == 3:
                        next_amp_obs = next_amp_obs.view(next_amp_obs.shape[0], -1)
                    
                    # 构建用于判别器训练的 next_amp_obs_with_term
                    # 如果环境重置了，不能直接用新 Episode 的第一帧作为 s_{t+1}
                    # 必须使用 episode 结束前的最后一帧 (terminal_obs)
                    next_amp_obs_with_term = next_amp_obs.clone()
                    amp_transition_valid = ~dones.bool()
                    reset_env_ids = dones.nonzero(as_tuple=False).flatten()

                    if len(reset_env_ids) > 0:
                        obs_extras = extras.get("observations", {})
                        terminal_obs_dict = obs_extras.get("terminal_obs")
                        if terminal_obs_dict is None: # 兼容不同版本的 IsaacLab/RSL_RL
                            terminal_obs_dict = extras.get("terminal_obs")
                        
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

                    # 5. [DWAQ] Update Prev Critic Obs
                    prev_critic_obs = last_critic_before_step
                    # 如果环境重置了，"上一帧"不存在，将 prev_critic_obs 重置为当前新帧
                    # 这避免了跨 Episode 的错误速度计算
                    if len(reset_env_ids) > 0:
                        prev_critic_obs[reset_env_ids] = obs["critic"][reset_env_ids]

                    # 6. [Process Step]
                    # AMPDWAQPPO 内部处理：
                    # - 存入 DWAQ Storage (obs, actions, prev_critic_obs...)
                    # - 存入 AMP Buffer (s_t, s_{t+1})
                    # - 计算 AMP 风格奖励并加到 rewards 上
                    self.alg.process_env_step(
                        obs,
                        rewards,
                        dones,
                        extras,
                        next_amp_obs_with_term,
                        amp_transition_valid=amp_transition_valid,
                    )

                    # 7. Update pointers for next loop
                    amp_obs = next_amp_obs
                    current_amp_obs = next_amp_obs.clone()

                    # book keeping
                    if self.log_dir is not None:
                        if "episode" in extras:
                            ep_infos.append(extras["episode"])
                        elif "log" in extras:
                            ep_infos.append(extras["log"])

                        cur_reward_sum += rewards
                        cur_episode_length += 1
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
                self.log(locals())
                if it % self.save_interval == 0:
                    self.save(os.path.join(self.log_dir, f"model_{it}.pt"))

            ep_infos.clear()
            
            if self.log_dir is not None and it == start_iter and not self.disable_logs:
                git_file_paths = store_code_state(self.log_dir, self.git_status_repos)
                if self.logger_type in ["wandb", "neptune"] and git_file_paths:
                    for path in git_file_paths:
                        self.writer.save_file(path)

        if self.log_dir is not None and not self.disable_logs:
            self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))

    def log(self, locs: dict, width: int = 80, pad: int = 35):
        # Compute the collection size
        collection_size = self.num_steps_per_env * self.env.num_envs * self.gpu_world_size
        self.tot_timesteps += collection_size
        self.tot_time += locs["collection_time"] + locs["learn_time"]
        iteration_time = locs["collection_time"] + locs["learn_time"]

        ep_string = ""
        if locs["ep_infos"]:
            for key in locs["ep_infos"][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs["ep_infos"]:
                    if key not in ep_info: continue
                    if not isinstance(ep_info[key], torch.Tensor): ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0: ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                if "/" in key:
                    self.writer.add_scalar(key, value, locs["it"])
                    ep_string += f"""{f'{key}:':>{pad}} {value:.4f}\n"""
                else:
                    self.writer.add_scalar("Episode/" + key, value, locs["it"])
                    ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""

        mean_std = self.alg.policy.action_std.mean()
        fps = int(collection_size / (locs["collection_time"] + locs["learn_time"]))

        # -- Losses (Includes PPO, AMP, DWAQ/VAE losses)
        for key, value in locs["loss_dict"].items():
            self.writer.add_scalar(f"Loss/{key}", value, locs["it"])
        self.writer.add_scalar("Loss/learning_rate", self.alg.learning_rate, locs["it"])

        self.writer.add_scalar("Policy/mean_noise_std", mean_std.item(), locs["it"])
        self.writer.add_scalar("Perf/total_fps", fps, locs["it"])
        self.writer.add_scalar("Perf/collection time", locs["collection_time"], locs["it"])
        self.writer.add_scalar("Perf/learning_time", locs["learn_time"], locs["it"])

        if len(locs["rewbuffer"]) > 0:
            self.writer.add_scalar("Train/mean_reward", statistics.mean(locs["rewbuffer"]), locs["it"])
            self.writer.add_scalar("Train/mean_episode_length", statistics.mean(locs["lenbuffer"]), locs["it"])
            if self.logger_type != "wandb":
                self.writer.add_scalar("Train/mean_reward/time", statistics.mean(locs["rewbuffer"]), self.tot_time)
                self.writer.add_scalar("Train/mean_episode_length/time", statistics.mean(locs["lenbuffer"]), self.tot_time)

        str_art = f" \033[1m Learning iteration {locs['it']}/{locs['tot_iter']} \033[0m "
        
        log_string = (f"""{'#' * width}\n"""
                      f"""{str_art.center(width, ' ')}\n\n"""
                      f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs['collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                      f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n""")
        
        for key, value in locs["loss_dict"].items():
            log_string += f"""{f'Mean {key} loss:':>{pad}} {value:.4f}\n"""
        
        if len(locs["rewbuffer"]) > 0:
            log_string += f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
            log_string += f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n"""
        
        log_string += ep_string
        log_string += (f"""{'-' * width}\n"""
                       f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
                       f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
                       f"""{'Time elapsed:':>{pad}} {time.strftime("%H:%M:%S", time.gmtime(self.tot_time))}\n"""
                       f"""{'ETA:':>{pad}} {time.strftime("%H:%M:%S", time.gmtime(self.tot_time / (locs['it'] - locs['start_iter'] + 1) * (locs['start_iter'] + locs['num_learning_iterations'] - locs['it'])))}\n""")
        print(log_string)

    def save(self, path: str, infos=None):
        saved_dict = {
            "model_state_dict": self.alg.policy.state_dict(),
            "optimizer_state_dict": self.alg.optimizer.state_dict(),
            "iter": self.current_learning_iteration,
            "iteration_is_next": self._save_iteration_as_next,
            "infos": infos,
            # [AMP 特有]
            "discriminator_state_dict": self.alg.discriminator.state_dict(),
            "amp_normalizer": self.alg.amp_normalizer,
            "amp_optimizer_state_dict": self.alg.amp_optimizer.state_dict(),
            "vae_optimizer_state_dict": self.alg.vae_optimizer.state_dict(),
        }
        torch.save(saved_dict, path)
        if self.logger_type in ["neptune", "wandb"] and not self.disable_logs:
            self.writer.save_model(path, self.current_learning_iteration)

    def load(self, path: str, load_optimizer: bool = True, map_location: str | None = None):
        loaded_dict = torch.load(path, weights_only=False, map_location=map_location)
        resumed_training = self.alg.policy.load_state_dict(loaded_dict["model_state_dict"])
        
        # [AMP 特有]
        if "discriminator_state_dict" in loaded_dict:
            self.alg.discriminator.load_state_dict(loaded_dict["discriminator_state_dict"])
        else:
            print("[Warning] 'discriminator_state_dict' not found in checkpoint. Skipping.")
        if "amp_normalizer" in loaded_dict:
            self.alg.amp_normalizer = loaded_dict["amp_normalizer"]
        else:
            print("[Warning] 'amp_normalizer' not found in checkpoint. Skipping.")

        if load_optimizer and resumed_training:
            self.alg.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
            self.alg.learning_rate = self.alg.optimizer.param_groups[0]["lr"]
            if "amp_optimizer_state_dict" in loaded_dict:
                self.alg.amp_optimizer.load_state_dict(loaded_dict["amp_optimizer_state_dict"])
            else:
                print("[Warning] 'amp_optimizer_state_dict' not found in checkpoint. Using a fresh optimizer.")
            if "vae_optimizer_state_dict" in loaded_dict:
                self.alg.vae_optimizer.load_state_dict(loaded_dict["vae_optimizer_state_dict"])
            else:
                print("[Warning] 'vae_optimizer_state_dict' not found in checkpoint. Using a fresh optimizer.")
        if resumed_training:
            self.current_learning_iteration = loaded_dict["iter"]
            if not bool(loaded_dict.get("iteration_is_next", False)):
                legacy_iteration = self.current_learning_iteration
                self.current_learning_iteration = legacy_iteration + 1
                print(
                    "[AMP-DWAQ Resume] Migrated legacy checkpoint iteration "
                    f"from {legacy_iteration} to {self.current_learning_iteration}."
                )
        return loaded_dict.get("infos")

    def get_inference_policy(self, device=None):
        self.eval_mode()
        if device is not None:
            self.alg.policy.to(device)
        return self.alg.policy.act_inference

    def train_mode(self):
        self.alg.policy.train()
        self.alg.discriminator.train() # [AMP]

    def eval_mode(self):
        self.alg.policy.eval()
        self.alg.discriminator.eval() # [AMP]

    def add_git_repo_to_log(self, repo_file_path):
        self.git_status_repos.append(repo_file_path)

    """
    Helper functions.
    """

    def _configure_multi_gpu(self):
        self.gpu_world_size = int(os.getenv("WORLD_SIZE", "1"))
        self.is_distributed = self.gpu_world_size > 1
        if not self.is_distributed:
            self.gpu_local_rank = 0
            self.gpu_global_rank = 0
            self.multi_gpu_cfg = None
            return
        self.gpu_local_rank = int(os.getenv("LOCAL_RANK", "0"))
        self.gpu_global_rank = int(os.getenv("RANK", "0"))
        self.multi_gpu_cfg = {
            "global_rank": self.gpu_global_rank,
            "local_rank": self.gpu_local_rank,
            "world_size": self.gpu_world_size,
        }
        if self.device != f"cuda:{self.gpu_local_rank}":
            raise ValueError(f"Device '{self.device}' does not match expected device for local rank '{self.gpu_local_rank}'.")
        torch.distributed.init_process_group(backend="nccl", rank=self.gpu_global_rank, world_size=self.gpu_world_size)
        torch.cuda.set_device(self.gpu_local_rank)

    def _construct_algorithm(self, obs) -> AMPDWAQPPO:
        """构建 AMPDWAQPPO 算法实例。"""
        
        # 0. Resolve Configurations (Symmetry & RND)
        self.alg_cfg = resolve_rnd_config(self.alg_cfg, obs, self.cfg["obs_groups"], self.env)
        self.alg_cfg = resolve_symmetry_config(self.alg_cfg, self.env)

        # 1. [DWAQ] Determine dims
        policy_tensor = obs["policy"]
        single_obs_dim = policy_tensor.shape[-1]
        print(f"[OnPolicyRunnerAmpDwaq] Derived Single Obs Dim for VAE: {single_obs_dim}")

        # 2. [AMP] Determine dt
        try:
            if hasattr(self.env, "step_dt"):
                step_dt = self.env.step_dt
            elif hasattr(self.env, "unwrapped") and hasattr(self.env.unwrapped, "step_dt"):
                step_dt = self.env.unwrapped.step_dt
            else:
                physics_dt = self.env.cfg.sim.dt
                decimation = getattr(self.env.cfg, "decimation", 1)
                step_dt = physics_dt * decimation
        except Exception as e:
            print(f"[Warning] Could not automatically determine step_dt, using default 0.02s. Error: {e}")
            step_dt = 0.02
        print(f"[AMP] Time between frames (step_dt) set to: {step_dt}s")

        # 3. [AMP] Components
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

        # 4. Initialize Policy (ActorCriticDwaq)
        policy_params = self.policy_cfg.copy()
        # 确保使用 DWAQ 的 ActorCritic
        actor_critic_class = eval(policy_params.pop("class_name", "ActorCriticDwaq"))
        
        actor_critic: ActorCriticDwaq = actor_critic_class(
            obs, self.cfg["obs_groups"], self.env.num_actions, **policy_params
        ).to(self.device)

        # 5. Initialize Algorithm (AMPDWAQPPO)
        alg_params = self.alg_cfg.copy()
        alg_params.pop("class_name", None)
        alg_params.pop("obs_dim", None) # 避免重复传递，我们下面显式传

        alg = AMPDWAQPPO(
            actor_critic,
            # AMP params
            discriminator=discriminator,
            amp_data=amp_data,
            amp_normalizer=amp_normalizer,
            min_std=torch.zeros(len(self.cfg["min_normalized_std"]), device=self.device, requires_grad=False),
            amp_replay_buffer_size=self.cfg.get("amp_replay_buffer_size", 100000),
            amp_reward_coef=self.cfg.get("amp_reward_coef", 2.0),
            amp_task_reward_lerp=self.cfg.get("amp_task_reward_lerp", 0.3),
            disc_learning_rate=self.cfg.get("disc_learning_rate", 1e-4),
            # DWAQ params
            obs_dim=single_obs_dim,
            # Common params
            device=self.device,
            multi_gpu_cfg=self.multi_gpu_cfg,
            **alg_params
        )

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
            self.logger_type = self.cfg.get("logger", "tensorboard").lower()

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
