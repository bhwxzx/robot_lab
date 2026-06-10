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

            # =============================================================================
            # ROA 算法核心 1：PPO 基础策略更新
            # 此过程会顺带计算 Privileged Regularization Loss (特权信息不要距离历史信息过远)
            # =============================================================================
            loss_dict = self.alg.update()

            # =============================================================================
            # ROA 算法核心 2：DAgger 蒸馏
            # 当本次回合是由历史编码器驱动探索时，进行监督学习：
            # 冻结所有策略，独占式地让 History Encoder 去模仿(L2 Loss) Privileged Encoder 的输出
            # =============================================================================
            if hist_encoding:
                dagger_loss_dict = self.alg.update_dagger()
                # 这一步非常精妙！我们将新产生的 Loss 塞入字典，父类的 log() 会在打印表格和
                # wandb 上传时自动提取并记录。无侵入式整合。
                loss_dict.update(dagger_loss_dict)

            stop = time.time()
            learn_time = stop - start
            self.current_learning_iteration = it
            
            # 使用继承的强大 log 函数进行性能输出
            if self.log_dir is not None and not self.disable_logs:
                self.log(locals())
                if it % self.save_interval == 0:
                    self.save(os.path.join(self.log_dir, f"model_{it}.pt"))

            ep_infos.clear()
            
            # 代码状态快照存储 (支持复现)
            if it == start_iter and not self.disable_logs:
                git_file_paths = store_code_state(self.log_dir, self.git_status_repos)
                if self.logger_type in ["wandb", "neptune"] and git_file_paths:
                    for path in git_file_paths:
                        self.writer.save_file(path)

        # 训练结束保存最终模型
        if self.log_dir is not None and not self.disable_logs:
            self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))

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
