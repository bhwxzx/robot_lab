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
from rsl_rl.utils import store_code_state

class OnPolicyRunnerAmpROA(OnPolicyRunnerAmp):
    alg: AMPROAPPO
    """
    On-policy runner for combining AMP (Adversarial Motion Prior) and ROA (Regularized Online Adaptation).
    """

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False):  # noqa: C901
        self._prepare_logging_writer()

        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        obs = self.env.get_observations().to(self.device)
        amp_obs = obs["amp"].to(self.device) if "amp" in obs else None

        self.train_mode()

        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        if self.is_distributed:
            print(f"Synchronizing parameters for rank {self.gpu_global_rank}...")
            self.alg.broadcast_parameters()

        current_amp_obs = amp_obs.clone() if amp_obs is not None else None

        start_iter = self.current_learning_iteration
        tot_iter = start_iter + num_learning_iterations
        
        # 获取 DAgger 更新频率
        dagger_update_freq = getattr(self.alg, "dagger_update_freq", 20)

        for it in range(start_iter, tot_iter):
            start = time.time()
            
            # 是否在这一步让网络强制使用 History Encoder
            hist_encoding = (it % dagger_update_freq == 0)

            with torch.inference_mode():
                for _ in range(self.num_steps_per_env):
                    # Actor 推理，传入 hist_encoding
                    actions = self.alg.act(obs, amp_obs=amp_obs, hist_encoding=hist_encoding)
                    
                    obs, rewards, dones, extras = self.env.step(actions.to(self.env.device))
                    obs, rewards, dones = (obs.to(self.device), rewards.to(self.device), dones.to(self.device))

                    next_amp_obs = obs["amp"].to(self.device)
                    next_amp_obs_with_term = next_amp_obs.clone()
                    reset_env_ids = dones.nonzero(as_tuple=False).flatten()

                    if len(reset_env_ids) > 0:
                        obs_extras = extras.get("observations", {})
                        terminal_obs_dict = obs_extras.get("terminal_obs")
                        if terminal_obs_dict is None:
                            terminal_obs_dict = extras.get("terminal_obs")
                            
                        if terminal_obs_dict is not None and "amp" in terminal_obs_dict:
                            next_amp_obs_with_term[reset_env_ids] = terminal_obs_dict["amp"][reset_env_ids].to(self.device)
                        else:
                            next_amp_obs_with_term[reset_env_ids] = current_amp_obs[reset_env_ids]

                    self.alg.process_env_step(obs, rewards, dones, extras, next_amp_obs_with_term)
                        
                    amp_obs = next_amp_obs
                    current_amp_obs = next_amp_obs.clone() 

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

                self.alg.compute_returns(obs)

            # --- [主更新] ---
            loss_dict = self.alg.update()
            
            # --- [ROA DAgger 蒸馏] ---
            if hist_encoding:
                dagger_loss_dict = self.alg.update_dagger()
                loss_dict.update(dagger_loss_dict)

            stop = time.time()
            learn_time = stop - start
            self.current_learning_iteration = it
            
            if self.log_dir is not None and not self.disable_logs:
                self.log(locals())
                if it % self.save_interval == 0:
                    self.save(os.path.join(self.log_dir, f"model_{it}.pt"))

            ep_infos.clear()
            if it == start_iter and not self.disable_logs:
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

    def load(self, path: str, load_optimizer: bool = True, map_location=None):
        """
        重写原有的 load 方法。在恢复训练 (resume) 时，检查配置中是否存在
        'priv_reg_coef_schedule_resume'，并将其覆盖到特权正则化调度参数中，
        从而避免 resume 之后系数突然掉回 0。
        """
        super().load(path, load_optimizer=load_optimizer, map_location=map_location)

        if "priv_reg_coef_schedule_resume" in self.alg_cfg:
            resume_schedule = self.alg_cfg["priv_reg_coef_schedule_resume"]
            self.alg.priv_reg_coef_schedule = resume_schedule
            print(f"[AMP ROA Resume] Overriding priv_reg_coef_schedule with {resume_schedule}")
