from __future__ import annotations

import os
import statistics
import time
import torch
import warnings
from collections import deque

import rsl_rl
from rsl_rl.algorithms.amp_roa_ppo import AMPROAPPO
from rsl_rl.runners.on_policy_runner_amp import OnPolicyRunnerAmp
from rsl_rl.runners.on_policy_runner_roa import OnPolicyRunnerROA
from rsl_rl.utils import store_code_state
from rsl_rl.utils.velocity_diagnostics import VelocityDiagnosticConfig, critic_diagnostic_layout

class OnPolicyRunnerAmpROA(OnPolicyRunnerAmp):
    alg: AMPROAPPO
    """On-policy runner for combining AMP (Adversarial Motion Prior) and ROA (Regularized Online Adaptation).
    """

    _collect_velocity_diagnostics = OnPolicyRunnerROA._collect_velocity_diagnostics

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False):  # noqa: C901
        self._prepare_logging_writer()

        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        obs = self.env.get_observations().to(self.device)
        amp_obs = obs["amp"].to(self.device) if "amp" in obs else None
        if amp_obs is not None and len(amp_obs.shape) == 3:
            amp_obs = amp_obs.view(amp_obs.shape[0], -1)

        self.train_mode()
        diagnostic_config = self.cfg.get("velocity_diagnostics")
        if not self.alg.policy.use_velocity_estimation:
            diagnostic_config = None
        diagnostic_setup = None if diagnostic_config is None else (
            VelocityDiagnosticConfig(**diagnostic_config), critic_diagnostic_layout(self.env, self.alg.policy)
        )

        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        if self.is_distributed:
            print(f"Synchronizing parameters for rank {self.gpu_global_rank}...")
            self.alg.broadcast_parameters()

        start_iter = self.current_learning_iteration
        tot_iter = start_iter + num_learning_iterations
        
        # 获取 DAgger 更新频率
        dagger_update_freq = getattr(self.alg, "dagger_update_freq", 20)

        for it in range(start_iter, tot_iter):
            start = time.time()
            completed_reward_sums = []
            completed_episode_lengths = []
            completed_episode_masks = []
            
            # 是否在这一步让网络强制使用 History Encoder
            hist_encoding = (it % dagger_update_freq == 0)

            with torch.inference_mode():
                for _ in range(self.num_steps_per_env):
                    # Actor 推理，传入 hist_encoding
                    actions = self.alg.act(obs, amp_obs=amp_obs, hist_encoding=hist_encoding)
                    
                    obs, rewards, dones, extras = self.env.step(actions.to(self.env.device))
                    obs, rewards, dones = (obs.to(self.device), rewards.to(self.device), dones.to(self.device))

                    next_amp_obs = obs["amp"].to(self.device)
                    if len(next_amp_obs.shape) == 3:
                        next_amp_obs = next_amp_obs.view(next_amp_obs.shape[0], -1)
                    done_mask = dones.bool().reshape(-1)

                    if hist_encoding:
                        # DAgger rollouts do not query AMP rewards or write AMP replay.
                        next_amp_obs_with_term = next_amp_obs
                        amp_transition_valid = None
                    else:
                        obs_extras = extras.get("observations", {})
                        terminal_obs_dict = obs_extras.get("terminal_obs")
                        if terminal_obs_dict is None:
                            terminal_obs_dict = extras.get("terminal_obs")
                            
                        if terminal_obs_dict is not None and "amp" in terminal_obs_dict:
                            term_amp = terminal_obs_dict["amp"].to(self.device)
                            if len(term_amp.shape) == 3:
                                term_amp = term_amp.view(term_amp.shape[0], -1)

                            if term_amp.shape[0] == next_amp_obs.shape[0]:
                                next_amp_obs_with_term = torch.where(
                                    done_mask.unsqueeze(-1),
                                    term_amp,
                                    next_amp_obs,
                                )
                            else:
                                # Compatibility path for wrappers that return only
                                # the terminal observations of reset environments.
                                reset_env_ids = done_mask.nonzero(as_tuple=False).flatten()
                                next_amp_obs_with_term = next_amp_obs.clone()
                                next_amp_obs_with_term[reset_env_ids] = term_amp
                            amp_transition_valid = torch.ones_like(done_mask)
                        else:
                            # IsaacLab 当前不会返回 pre-reset terminal_obs。上一有效窗口
                            # 只用于计算该终止步的保守 AMP reward，不写入 replay buffer。
                            next_amp_obs_with_term = torch.where(
                                done_mask.unsqueeze(-1),
                                amp_obs,
                                next_amp_obs,
                            )
                            amp_transition_valid = ~done_mask

                    self.alg.process_env_step(
                        obs,
                        rewards,
                        dones,
                        extras,
                        next_amp_obs_with_term,
                        amp_transition_valid=amp_transition_valid,
                        process_amp=not hist_encoding,
                        defer_amp_reward=not hist_encoding,
                    )
                        
                    amp_obs = next_amp_obs

                    if self.log_dir is not None:
                        if "episode" in extras:
                            ep_infos.append(extras["episode"])
                        elif "log" in extras:
                            ep_infos.append(extras["log"])

                        cur_reward_sum += rewards
                        cur_episode_length += 1
                        completed_reward_sums.append(torch.where(done_mask, cur_reward_sum, 0.0))
                        completed_episode_lengths.append(torch.where(done_mask, cur_episode_length, 0.0))
                        completed_episode_masks.append(done_mask)
                        cur_reward_sum *= ~done_mask
                        cur_episode_length *= ~done_mask

                if not hist_encoding:
                    self.alg.finalize_amp_rollout_rewards()

                if completed_episode_masks:
                    completed_mask = torch.stack(completed_episode_masks).flatten()
                    completed_rewards = torch.stack(completed_reward_sums).flatten()
                    completed_lengths = torch.stack(completed_episode_lengths).flatten()
                    rewbuffer.extend(completed_rewards[completed_mask].cpu().tolist())
                    lenbuffer.extend(completed_lengths[completed_mask].cpu().tolist())

                stop = time.time()
                collection_time = stop - start
                start = stop

                self.alg.compute_returns(obs)

            velocity_diagnostics = self._collect_velocity_diagnostics(diagnostic_setup)
            self.last_velocity_diagnostics = velocity_diagnostics
            velocity_curriculum = self.alg.velocity_curriculum_report(hist_encoding)
            self.last_velocity_curriculum = velocity_curriculum

            # 学生策略采集的 rollout 只用于 DAgger 蒸馏，避免使用教师策略
            # 重算学生动作的 log_prob，导致 PPO importance ratio 失效。
            if hist_encoding:
                loss_dict = self.alg.update_dagger()
            else:
                loss_dict = self.alg.update()

            stop = time.time()
            learn_time = stop - start
            # Store the next iteration to execute. This prevents checkpoint resume
            # from repeating the rollout/update that has just completed.
            self.current_learning_iteration = it + 1
            
            if self.log_dir is not None and not self.disable_logs:
                self.log(locals())
                controller = "student" if hist_encoding else "teacher"
                for name, value in velocity_diagnostics.items():
                    self.writer.add_scalar(f"VelocityDiagnostics/{controller}/{name}", value, it)
                for name, value in velocity_curriculum.items():
                    self.writer.add_scalar(f"VelocityCurriculum/{controller}/{name}", value, it)
                if diagnostic_setup is not None and self.alg.hist_encoder_optimizer is not None:
                    self.writer.add_scalar("VelocityDiagnostics/history_learning_rate",
                                           self.alg.hist_encoder_optimizer.param_groups[0]["lr"], it)
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

    def get_inference_policy(self, device=None, hist_encoding=True):
        """部署模式：默认使用历史编码器"""
        self.eval_mode()
        if device is not None:
            self.alg.policy.to(device)
            
        def act_inference_wrapper(obs):
            return self.alg.policy.act_inference(obs, hist_encoding=hist_encoding)
            
        return act_inference_wrapper

    def save(self, path: str, infos=None):
        """Save both AMP state and the same ROA continuation contract as pure ROA."""
        saved_dict = {
            "model_state_dict": self.alg.policy.state_dict(),
            "optimizer_state_dict": self.alg.optimizer.state_dict(),
            "discriminator_state_dict": self.alg.discriminator.state_dict(),
            "amp_normalizer": self.alg.amp_normalizer,
            "amp_optimizer_state_dict": self.alg.amp_optimizer.state_dict(),
            "iter": self.current_learning_iteration,
            "iteration_is_next": True,
            "algorithm_counter": self.alg.counter,
            "priv_reg_coef_schedule": list(self.alg.priv_reg_coef_schedule),
            "use_velocity_estimation": self.alg.policy.use_velocity_estimation,
            "estimated_velocity_schedule": self.alg.estimated_velocity_schedule,
            "infos": infos,
        }
        if self.alg.hist_encoder_optimizer is not None:
            saved_dict["hist_encoder_optimizer_state_dict"] = self.alg.hist_encoder_optimizer.state_dict()
        torch.save(saved_dict, path)
        if getattr(self, "logger_type", None) in ["neptune", "wandb"] and not self.disable_logs:
            self.writer.save_model(path, self.current_learning_iteration)

    def load(self, path: str, load_optimizer: bool = True, map_location=None):
        """Validate ROA overrides before mutating weights; preserve AMP load behavior."""
        # A file-like checkpoint is also supported by the parent and CPU tests.
        position = path.tell() if hasattr(path, "tell") else None
        loaded_dict = torch.load(path, weights_only=False, map_location=map_location)
        if position is not None:
            path.seek(position)
        mode = loaded_dict.get("use_velocity_estimation", True)
        if type(mode) is not bool or mode != self.alg.policy.use_velocity_estimation:
            raise ValueError("ROA checkpoint velocity architecture mismatch; use the matching "
                             "policy.use_velocity_estimation configuration, or train a separate baseline")
        schedule = self.alg_cfg.get("priv_reg_coef_schedule_resume")
        if schedule is None:
            schedule = loaded_dict.get("priv_reg_coef_schedule", self.alg.priv_reg_coef_schedule)
        schedule = self.alg.validate_priv_reg_schedule(schedule)
        velocity_schedule = self.alg_cfg.get("estimated_velocity_schedule_resume")
        if velocity_schedule is None:
            velocity_schedule = loaded_dict.get("estimated_velocity_schedule", self.alg.estimated_velocity_schedule)
        velocity_schedule = self.alg.validate_velocity_schedule(velocity_schedule)
        has_saved_schedule = "priv_reg_coef_schedule" in loaded_dict
        # The parent restores AMP/PPO/history optimizers, iteration and counter,
        # including the legacy completed-iteration -> next-iteration migration.
        del loaded_dict
        infos = super().load(path, load_optimizer=load_optimizer, map_location=map_location)
        if not has_saved_schedule:
            if self.alg_cfg.get("priv_reg_coef_schedule_resume") is None:
                warnings.warn("ROA checkpoint lacks regularization schedule; using the configured training schedule.")
        self.alg.priv_reg_coef_schedule = schedule
        self.alg.set_estimated_velocity_schedule(velocity_schedule)
        return infos
