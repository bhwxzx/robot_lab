# on_policy_runner_roa.py
from __future__ import annotations

import os
import statistics
import time
import torch
import warnings
from collections import deque

import rsl_rl
from rsl_rl.runners.on_policy_runner import OnPolicyRunner
from rsl_rl.utils import store_code_state
from rsl_rl.utils.velocity_diagnostics import (
    VelocityDiagnosticConfig, GroupedVelocityDiagnostics, critic_diagnostic_layout,
)

import typing
if typing.TYPE_CHECKING:
    from rsl_rl.algorithms.roa_ppo import ROAPPO

class OnPolicyRunnerROA(OnPolicyRunner):
    alg: ROAPPO
    """
    专为支持 ROA (Regularized Online Adaptation) 算法改进的 On-policy 训练管理器。
    
    采用面向对象继承机制，全盘继承自原版 OnPolicyRunner 的所有功能：
    - 多 GPU 通信、模型/日志存取、RND(好奇心探索) 以及 Symmetry(对称性增广) 的初始化机制完全复用。
    - 仅重写核心的 learn() 和 get_inference_policy() 方法，以支持 ROA 的 Teacher-Student 交替训练和 DAgger 蒸馏过程。
    """

    @torch.inference_mode()
    def _collect_velocity_diagnostics(self, diagnostic_setup):
        if diagnostic_setup is None or not self.alg.policy.use_velocity_estimation:
            return {}
        config, layout = diagnostic_setup
        accumulator = GroupedVelocityDiagnostics(config, device=self.device)
        storage = self.alg.storage
        observations = storage.observations[:storage.step].flatten(0, 1)
        for start in range(0, len(observations), config.chunk_size):
            batch = observations[start:start + config.chunk_size]
            critic = batch["critic"]
            command, command_clipped = layout["velocity_commands"].decode(critic)
            actual, velocity_clipped = layout["base_lin_vel"].decode(critic)
            angular, angular_clipped = layout["base_ang_vel"].decode(critic)
            _, prediction = self.alg.policy.infer_hist_latent(batch, return_vel=True)
            scale = layout["base_lin_vel"].scale.to(prediction.device)
            accumulator.add(command, actual, prediction.to(torch.float64) / scale, angular[:, 2],
                            clipped=command_clipped | velocity_clipped | angular_clipped)
        return accumulator.report(distributed=self.is_distributed)

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False):  # noqa: C901
        # 初始化记录器 (继承自父类)
        self._prepare_logging_writer()

        # 随机初始化初始回合长度，有助于初期更好的探索 (继承自父类逻辑)
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        # 获取环境初始观测值，并开启训练模式
        obs = self.env.get_observations().to(self.device)
        self.train_mode()  # 确保 Actor Critic 网络，包含特权编码器，均处于训练模式
        diagnostic_config = self.cfg.get("velocity_diagnostics")
        if not self.alg.policy.use_velocity_estimation:
            diagnostic_config = None
        diagnostic_setup = None if diagnostic_config is None else (
            VelocityDiagnosticConfig(**diagnostic_config), critic_diagnostic_layout(self.env, self.alg.policy)
        )

        # 初始化数据统计 Buffer
        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        # 如果框架启用了 RND 探索，创建专属的内在奖励与外在奖励记录 Buffer
        if hasattr(self.alg, "rnd") and self.alg.rnd:
            erewbuffer = deque(maxlen=100)
            irewbuffer = deque(maxlen=100)
            cur_ereward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
            cur_ireward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        # 多显卡模式下的参数强制同步
        if self.is_distributed:
            print(f"Synchronizing parameters for rank {self.gpu_global_rank}...")
            self.alg.broadcast_parameters()

        # 训练迭代初始化
        start_iter = self.current_learning_iteration
        tot_iter = start_iter + num_learning_iterations
        
        # =================================================================================
        # ROA 核心控制变量：
        # 获取特权蒸馏到历史特征的频率 (DAgger 频率)。默认 1 表示每次迭代都进行。
        # 如果是 N，表示每 N 次 PPO 迭代才进行一次纯监督的历史编码器更新。
        # =================================================================================
        dagger_update_freq = getattr(self.alg, "dagger_update_freq", 20)

        for it in range(start_iter, tot_iter):
            start = time.time()
            
            # =============================================================================
            # ROA 阶段切换：
            # 在某些训练步骤中，我们希望环境交互强行只利用 History Encoder。
            # 这有助于在训练期间即刻暴露历史编码器的问题，并收集分布外(OOD)数据以提高鲁棒性。
            # =============================================================================
            hist_encoding = (it % dagger_update_freq == 0)

            # Rollout 阶段 (与环境交互收集数据)
            with torch.inference_mode():
                for _ in range(self.num_steps_per_env):
                    # 获取动作。特别注意：我们把 hist_encoding 标志传入，控制网络使用哪种 Encoder
                    actions = self.alg.act(obs, hist_encoding=hist_encoding)
                    
                    # 步进仿真环境
                    obs, rewards, dones, extras = self.env.step(actions.to(self.env.device))
                    obs, rewards, dones = (obs.to(self.device), rewards.to(self.device), dones.to(self.device))
                    
                    # 算法层处理步进数据 (主要存入 Storage Buffer)
                    self.alg.process_env_step(obs, rewards, dones, extras)
                    
                    # 提取内在好奇心探索奖励 (支持原框架的 RND)
                    intrinsic_rewards = self.alg.intrinsic_rewards if (hasattr(self.alg, "rnd") and self.alg.rnd) else None
                    
                    # 数据簿记 (Book keeping)
                    if self.log_dir is not None:
                        if "episode" in extras:
                            ep_infos.append(extras["episode"])
                        elif "log" in extras:
                            ep_infos.append(extras["log"])
                        
                        # 更新累计奖励
                        if hasattr(self.alg, "rnd") and self.alg.rnd:
                            cur_ereward_sum += rewards
                            cur_ireward_sum += intrinsic_rewards  # type: ignore
                            cur_reward_sum += rewards + intrinsic_rewards
                        else:
                            cur_reward_sum += rewards
                        
                        cur_episode_length += 1
                        
                        # 处理当前回合结束的机器人环境 (done = True)
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0
                        
                        if hasattr(self.alg, "rnd") and self.alg.rnd:
                            erewbuffer.extend(cur_ereward_sum[new_ids][:, 0].cpu().numpy().tolist())
                            irewbuffer.extend(cur_ireward_sum[new_ids][:, 0].cpu().numpy().tolist())
                            cur_ereward_sum[new_ids] = 0
                            cur_ireward_sum[new_ids] = 0

                stop = time.time()
                collection_time = stop - start
                start = stop

                # 基于最后一步的 Critic Value 计算 GAE 优势函数与回报
                self.alg.compute_returns(obs)

            # Diagnose the frozen pre-update estimator once per original rollout sample.
            # Never count the repeated optimizer epochs as additional observations.
            velocity_diagnostics = self._collect_velocity_diagnostics(diagnostic_setup)
            self.last_velocity_diagnostics = velocity_diagnostics
            velocity_curriculum = self.alg.velocity_curriculum_report(hist_encoding)
            self.last_velocity_curriculum = velocity_curriculum

            # ROA Algorithm 1: student rollouts train the history encoder only.
            # Teacher rollouts train PPO with the student-latent regularizer.
            if hist_encoding:
                loss_dict = self.alg.update_dagger()
            else:
                loss_dict = self.alg.update()

            stop = time.time()
            learn_time = stop - start
            self.current_learning_iteration = it + 1
            
            # 使用继承的强大 log 函数进行性能输出
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
            
            # 代码状态快照存储 (支持复现)
            if self.log_dir is not None and it == start_iter and not self.disable_logs:
                git_file_paths = store_code_state(self.log_dir, self.git_status_repos)
                if self.logger_type in ["wandb", "neptune"] and git_file_paths:
                    for path in git_file_paths:
                        self.writer.save_file(path)

        # 训练结束保存最终模型
        if num_learning_iterations > 0 and self.log_dir is not None and not self.disable_logs:
            self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration - 1}.pt"))

    def get_inference_policy(self, device=None, hist_encoding=True):
        """
        覆盖原有的获取部署策略方法。
        
        为什么重写？
        因为一旦你将模型导出到实车部署阶段，机器人没有“上帝视角”(不知道物理摩擦力等 privileged info)。
        通过包裹一层 Wrapper，我们将推理模式 (inference) 强制锁定在 hist_encoding=True。
        这样网络在实车中便会自动调用 History Encoder 来根据本体时序状态推测隐变量。
        """
        self.eval_mode()  # 切换到测试模式，关闭 Dropout/BatchNorm 等的影响
        if device is not None:
            self.alg.policy.to(device)
            
        def act_inference_wrapper(obs):
            # 将外部传入的真机感知观测包裹，注入 hist_encoding=True 开关
            return self.alg.policy.act_inference(obs, hist_encoding=hist_encoding)
            
        return act_inference_wrapper

    def save(self, path: str, infos=None):
        """Save ROA optimizer/schedule state, with iter naming the next rollout."""
        saved_dict = {
            "model_state_dict": self.alg.policy.state_dict(),
            "optimizer_state_dict": self.alg.optimizer.state_dict(),
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
        if self.alg.rnd:
            saved_dict["rnd_state_dict"] = self.alg.rnd.state_dict()
            saved_dict["rnd_optimizer_state_dict"] = self.alg.rnd_optimizer.state_dict()
        torch.save(saved_dict, path)
        if getattr(self, "logger_type", None) in ["neptune", "wandb"] and not self.disable_logs:
            self.writer.save_model(path, self.current_learning_iteration)

    def load(self, path: str, load_optimizer: bool = True, map_location: str | None = None):
        """Resume ROA state; an explicit schedule override uses the restored counter."""
        loaded_dict = torch.load(path, weights_only=False, map_location=map_location)
        checkpoint_velocity_mode = loaded_dict.get("use_velocity_estimation", True)
        if (type(checkpoint_velocity_mode) is not bool
                or checkpoint_velocity_mode != self.alg.policy.use_velocity_estimation):
            raise ValueError("ROA checkpoint velocity architecture mismatch; use the matching "
                             "policy.use_velocity_estimation configuration, or train a separate baseline")
        velocity_schedule = self.alg_cfg.get("estimated_velocity_schedule_resume")
        if velocity_schedule is None:
            velocity_schedule = loaded_dict.get("estimated_velocity_schedule", self.alg.estimated_velocity_schedule)
        velocity_schedule = self.alg.validate_velocity_schedule(velocity_schedule)
        schedule = self.alg_cfg.get("priv_reg_coef_schedule_resume")
        if schedule is None:
            schedule = loaded_dict.get("priv_reg_coef_schedule", self.alg.priv_reg_coef_schedule)
        schedule = self.alg.validate_priv_reg_schedule(schedule)
        resumed_training = self.alg.policy.load_state_dict(loaded_dict["model_state_dict"])
        if self.alg.rnd:
            self.alg.rnd.load_state_dict(loaded_dict["rnd_state_dict"])
        if resumed_training:
            if load_optimizer:
                self.alg.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
                self.alg.learning_rate = self.alg.optimizer.param_groups[0]["lr"]
                if self.alg.hist_encoder_optimizer is not None:
                    if "hist_encoder_optimizer_state_dict" in loaded_dict:
                        self.alg.hist_encoder_optimizer.load_state_dict(loaded_dict["hist_encoder_optimizer_state_dict"])
                    else:
                        warnings.warn("ROA checkpoint lacks history optimizer state; exact optimizer continuation is unavailable.")
                if self.alg.rnd:
                    self.alg.rnd_optimizer.load_state_dict(loaded_dict["rnd_optimizer_state_dict"])
            self.current_learning_iteration = int(loaded_dict["iter"])
            if not loaded_dict.get("iteration_is_next", False):
                self.current_learning_iteration += 1
                warnings.warn("Legacy ROA checkpoint: migrated completed iteration to next iteration (+1).")
            if "algorithm_counter" not in loaded_dict:
                warnings.warn("ROA checkpoint lacks algorithm_counter; inferring it from the next iteration.")
            self.alg.counter = int(loaded_dict.get("algorithm_counter", self.current_learning_iteration))
            if "priv_reg_coef_schedule" not in loaded_dict and self.alg_cfg.get("priv_reg_coef_schedule_resume") is None:
                warnings.warn("ROA checkpoint lacks regularization schedule; using the configured training schedule.")
            self.alg.priv_reg_coef_schedule = schedule
            self.alg.set_estimated_velocity_schedule(velocity_schedule)
            if self.alg_cfg.get("priv_reg_coef_schedule_resume") is not None:
                print(f"[ROA Resume] Overriding priv_reg_coef_schedule with {schedule} at counter={self.alg.counter}")
        return loaded_dict.get("infos")
