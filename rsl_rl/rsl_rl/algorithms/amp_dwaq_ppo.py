# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim
from itertools import chain

from rsl_rl.modules import ActorCriticDwaq
from rsl_rl.modules.rnd import RandomNetworkDistillation
from rsl_rl.storage import RolloutStorageDwaq, ReplayBuffer

class AMPDWAQPPO:
    """
    AMPDWAQPPO: 结合了 Adversarial Motion Prior (AMP) 和 VAE Context Encoder (DWAQ) 的 PPO 算法。
    """
    policy: ActorCriticDwaq

    def __init__(
        self,
        policy,
        # --- [AMP 参数] ---
        discriminator,
        amp_data,
        amp_normalizer,
        # --- [DWAQ 参数] ---
        obs_dim=41,          # 原始本体感受观察值维度（用于提取速度真值标签）
        vae_beta=1.0,        # KL 散度权重系数
        vae_learning_rate=1e-3, # VAE 专属学习率
        # --- [通用 PPO 参数] ---
        num_learning_epochs=5,
        num_mini_batches=4,
        clip_param=0.2,
        gamma=0.99,
        lam=0.95,
        value_loss_coef=1.0,
        entropy_coef=0.01,
        learning_rate=0.001,
        max_grad_norm=1.0,
        use_clipped_value_loss=True,
        schedule="adaptive",
        desired_kl=0.01,
        device="cpu",
        normalize_advantage_per_mini_batch=False,
        # --- [AMP 配置] ---
        amp_replay_buffer_size=100000,
        min_std=None,
        amp_reward_coef=2.0,
        amp_task_reward_lerp=0.3,
        amp_discr_hidden_dims=None,
        disc_learning_rate=1e-4,
        # Symmetry parameters
        symmetry_cfg: dict | None = None,
        # Distributed training parameters
        multi_gpu_cfg: dict | None = None,
        **kwargs
    ):
        if kwargs:
            print(f"[AMPDWAQPPO] 忽略了多余的配置参数: {list(kwargs.keys())}")

        self.device = device
        self.is_multi_gpu = multi_gpu_cfg is not None

        # for dwaq
        self.obs_dim = obs_dim
        self.vae_beta = vae_beta
        
        from rsl_rl.utils import string_to_callable
        # Symmetry components
        if symmetry_cfg is not None:
            # Check if symmetry is enabled
            use_symmetry = symmetry_cfg["use_data_augmentation"] or symmetry_cfg["use_mirror_loss"]
            if not use_symmetry:
                print("Symmetry not used for learning. We will use it for logging instead.")
            if isinstance(symmetry_cfg["data_augmentation_func"], str):
                symmetry_cfg["data_augmentation_func"] = string_to_callable(symmetry_cfg["data_augmentation_func"])
            if symmetry_cfg["use_data_augmentation"] and not callable(symmetry_cfg["data_augmentation_func"]):
                raise ValueError(
                    "Data augmentation enabled but the function is not callable:"
                    f" {symmetry_cfg['data_augmentation_func']}"
                )
            self.symmetry = symmetry_cfg
        else:
            self.symmetry = None

        # --- [AMP 初始化] ---
        self.discriminator = discriminator
        self.discriminator.to(self.device)
        self.amp_data = amp_data
        self.amp_normalizer = amp_normalizer
        self.amploss_coef = 1.0  # 判别器损失的权重
        self.amp_reward_coef = amp_reward_coef
        self.min_std = min_std

        # Multi-GPU parameters
        if multi_gpu_cfg is not None: # 如果是多GPU训练，记录当前进程的等级(rank)和总数(world_size), 用于后续在不同 GPU 之间同步梯度（all_reduce）
            self.gpu_global_rank = multi_gpu_cfg["global_rank"]
            self.gpu_world_size = multi_gpu_cfg["world_size"]
        else:
            self.gpu_global_rank = 0
            self.gpu_world_size = 1

        # 初始化 AMP 回放池 (用于存储 Policy 产生的动作序列，供判别器训练)
        # discriminator.input_dim // 2 是因为输入是 (state, next_state) 拼接的
        self.amp_storage = ReplayBuffer(
            discriminator.input_dim // 2, amp_replay_buffer_size, device
        )
        self.amp_transition = RolloutStorageDwaq.Transition() # 临时的 AMP transition 存储

        # PPO components
        self.policy = policy
        self.policy.to(self.device)
        # --- [优化器修改] ---
        # Create optimizer
        self.rl_parameters = list(self.policy.actor.parameters()) + \
                             list(self.policy.critic.parameters()) + \
                             [self.policy.std]
                             
        ppo_params = [
            {"params": self.rl_parameters, "name": "policy"},
        ]
        self.optimizer = optim.Adam(ppo_params, lr=learning_rate)
        
        self.vae_parameters = list(self.policy.encoder_backbone.parameters()) + \
                              list(self.policy.encode_mean_latent.parameters()) + \
                              list(self.policy.encode_logvar_latent.parameters()) + \
                              list(self.policy.encode_mean_vel.parameters()) + \
                              list(self.policy.encode_logvar_vel.parameters()) + \
                              list(self.policy.decoder.parameters())
        self.vae_optimizer = optim.Adam(self.vae_parameters, lr=vae_learning_rate)
        
        amp_params = [
            {"params": self.discriminator.trunk.parameters(), "weight_decay": 1e-4, "name": "amp_trunk"},
            {"params": self.discriminator.amp_linear.parameters(), "weight_decay": 1e-2, "name": "amp_head"},
        ]
        self.amp_optimizer = optim.Adam(amp_params, lr=disc_learning_rate)
        # Create rollout storage
        # 使用 DWAQ 专用存储
        self.storage: RolloutStorageDwaq = None  # type: ignore
        self.transition = RolloutStorageDwaq.Transition()

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
        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate
        self.normalize_advantage_per_mini_batch = normalize_advantage_per_mini_batch

    def init_storage(self, training_type, num_envs, num_transitions_per_env, obs, actions_shape):
        """初始化 DWAQ 专用存储。"""
        self.storage = RolloutStorageDwaq(
            training_type,
            num_envs,
            num_transitions_per_env,
            obs,
            actions_shape,
            self.device,
        )

    def act(self, obs, prev_critic_obs, amp_obs=None):
        # 直接传入 TensorDict，由模型内部自行提取 policy/critic
        self.transition.actions = self.policy.act(obs).detach()
        self.transition.values = self.policy.evaluate(obs).detach()
        self.transition.actions_log_prob = self.policy.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.policy.action_mean.detach()
        self.transition.action_sigma = self.policy.action_std.detach()

        # 记录观察值：obs 包含了历史和当前，prev_critic_obs 记录上一时刻用于速度监督
        self.transition.observations = obs
        self.transition.prev_critic_observations = prev_critic_obs
        # [AMP] 记录当前的 AMP 观测值
        if amp_obs is not None:
            self.amp_transition.observations = amp_obs
        
        return self.transition.actions
    
    def process_env_step(self, obs, rewards, dones, extras, amp_obs):
        # --- [常规处理] ---
        self.policy.update_normalization(obs)
        self.transition.dones = dones

        # --- [AMP 逻辑] ---
        # 1. 存入 AMP ReplayBuffer
        # 使用暂存在 amp_transition 中的 observations
        self.amp_storage.insert(self.amp_transition.observations, amp_obs)

        # 2. 计算风格奖励
        # predict_amp_reward 内部处理: 归一化 -> 判别器 -> 奖励计算 -> Lerp
        amp_rewards, policy_d = self.discriminator.predict_amp_reward(
            self.amp_transition.observations,  # state (s)
            amp_obs,                           # next_state (s')
            task_reward=rewards,               # 传入当前任务奖励
            normalizer=self.amp_normalizer
        )

        if self.storage.step == 0:
            with torch.no_grad():
                raw_amp = self.discriminator.dt * self.discriminator.amp_reward_coef * torch.clamp(1 - (1 / 4) * torch.square(policy_d - 1), min=0)
                print(f"\n[AMP DEBUG] Task Reward Mean: {rewards.mean().item():.4f} | Raw AMP Reward Mean: {raw_amp.mean().item():.4f}")

        # 3. 设置最终奖励
        if self.discriminator.task_reward_lerp > 0:
            # 如果开启了 Lerp，amp_rewards 已经是混合后的奖励
            final_rewards = amp_rewards
        else:
            # 否则手动相加
            final_rewards = rewards + amp_rewards

        self.transition.rewards = final_rewards.clone()

        # --- [Bootstrapping] ---
        if "time_outs" in extras:
            self.transition.rewards += self.gamma * torch.squeeze(
                self.transition.values * extras["time_outs"].unsqueeze(1).to(self.device), 1
            )

        # --- [存入主存储] ---
        # 此时 self.transition 包含了 observations, prev_critic_obs, actions, rewards 等所有信息
        self.storage.add_transitions(self.transition)
        
        # 清理临时存储
        self.transition.clear()
        self.policy.reset(dones)

    def compute_returns(self, obs):
        # compute value for the last step
        last_values = self.policy.evaluate(obs).detach()
        self.storage.compute_returns(
            last_values, self.gamma, self.lam, normalize_advantage=not self.normalize_advantage_per_mini_batch
        )

    def update(self):  # noqa: C901
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_entropy = 0
        # AMP 统计
        mean_amp_loss = 0
        mean_grad_pen_loss = 0
        mean_policy_pred = 0
        mean_expert_pred = 0
        # DWAQ 统计
        mean_autoenc_loss = 0
        mean_vel_loss = 0
        # -- Symmetry loss
        if self.symmetry:
            mean_symmetry_loss = 0
        else:
            mean_symmetry_loss = None

        # --- [Generators] ---
        # 1. 主生成器 (包含 DWAQ 所需的 prev_critic_obs)
        generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        # 2. AMP 生成器
        amp_policy_generator = self.amp_storage.feed_forward_generator(
            self.num_learning_epochs * self.num_mini_batches,
            self.storage.num_envs * self.storage.num_transitions_per_env // self.num_mini_batches,
        )
        amp_expert_generator = self.amp_data.feed_forward_generator(
            self.num_learning_epochs * self.num_mini_batches,
            self.storage.num_envs * self.storage.num_transitions_per_env // self.num_mini_batches,
        )

        # --- [Training Loop] ---
        for sample, sample_amp_policy, sample_amp_expert in zip(generator, amp_policy_generator, amp_expert_generator):
            (
                policy_obs_batch,      # Actor/VAE 输入
                critic_obs_batch,      # Critic 输入
                prev_critic_obs_batch, # DWAQ: 速度监督辅助
                actions_batch,
                target_values_batch,
                advantages_batch,
                returns_batch,
                old_actions_log_prob_batch,
                old_mu_batch,
                old_sigma_batch,
                live_batch,             # DWAQ VAE 掩码
                hid_states_batch,
                masks_batch,
            ) = sample

            # -----------------------------------------------------
            # 1. PPO Pre-processing (Advantage Normalization)
            # -----------------------------------------------------
            original_batch_size = policy_obs_batch.batch_size[0]
            num_aug = 1

            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    advantages_batch = (advantages_batch - advantages_batch.mean()) / (advantages_batch.std() + 1e-8)

            if self.symmetry and self.symmetry["use_data_augmentation"]:
                # augmentation using symmetry
                data_augmentation_func = self.symmetry["data_augmentation_func"]
                policy_obs_batch, actions_batch = data_augmentation_func(
                    obs=policy_obs_batch,
                    actions=actions_batch,
                    env=self.symmetry["_env"],
                )
                # recompute the number of augmentations
                num_aug = int(policy_obs_batch.batch_size[0] / original_batch_size)
                # repeat the other parts of the batch
                old_actions_log_prob_batch = old_actions_log_prob_batch.repeat(num_aug, 1)
                target_values_batch = target_values_batch.repeat(num_aug, 1)
                advantages_batch = advantages_batch.repeat(num_aug, 1)
                returns_batch = returns_batch.repeat(num_aug, 1)
                # For DWAQ specific observations
                critic_obs_batch = critic_obs_batch.repeat(num_aug, 1)
                prev_critic_obs_batch = prev_critic_obs_batch.repeat(num_aug, 1)
                live_batch = live_batch.repeat(num_aug, 1)

            # -----------------------------------------------------
            # 2. DWAQ / VAE Loss Calculation
            # -----------------------------------------------------
            # 运行编码器
            (latent_code, reconstruction, mu_v, logvar_v, mu_l, logvar_l, current_obs_synced) = self.policy.cenet_forward(policy_obs_batch)
            # 提取速度预测 (前3维)
            code_vel = latent_code[:, :3]
            # 速度真值 (从 critic_obs 中提取，假设 obs_dim 之后是速度)
            vel_target = critic_obs_batch[:, self.obs_dim : self.obs_dim + 3].detach()
            # 解码目标 (通常是当前观测)
            decode_target = current_obs_synced.detach()

            # 速度估计误差与重构误差，使用 live_batch 过滤 padding
            vel_loss_term = nn.MSELoss()(code_vel * live_batch, vel_target * live_batch)
            recon_loss_term = nn.MSELoss()(reconstruction * live_batch, decode_target * live_batch)

            # KL 散度 (修复 broadcasting bug)
            logvar_l_clamped = torch.clamp(logvar_l, min=-10.0, max=10.0)
            kl_divergence = -0.5 * (torch.sum(1 + logvar_l_clamped - mu_l.pow(2) - logvar_l_clamped.exp(), dim=-1) * live_batch.squeeze(-1)).mean()

            # VAE 总损失 
            autoenc_loss = vel_loss_term + recon_loss_term + self.vae_beta * kl_divergence

            # -----------------------------------------------------
            # 3. PPO Forward Pass
            # -----------------------------------------------------
            # 构建完整观测字典
            full_obs_batch = {
                "policy": policy_obs_batch,
                "critic": critic_obs_batch,
            }
            # Re-run policy
            self.policy.act(full_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[0])
            
            actions_log_prob_batch = self.policy.get_actions_log_prob(actions_batch)
            value_batch = self.policy.evaluate(full_obs_batch)
            
            mu_batch = self.policy.action_mean
            sigma_batch = self.policy.action_std
            entropy_batch = self.policy.entropy

            # Adaptive KL
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
                    
                    if self.is_multi_gpu:
                        torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
                        kl_mean /= self.gpu_world_size

                    if self.gpu_global_rank == 0:
                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                    if self.is_multi_gpu:
                        lr_tensor = torch.tensor(self.learning_rate, device=self.device)
                        torch.distributed.broadcast(lr_tensor, src=0)
                        self.learning_rate = lr_tensor.item()

                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            # Surrogate Loss
            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # Value Function Loss
            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                    -self.clip_param, self.clip_param
                )
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()
                
            loss = (
                surrogate_loss 
                + self.value_loss_coef * value_loss 
                - self.entropy_coef * entropy_batch.mean()
            )

            # Symmetry loss
            if self.symmetry:
                if not self.symmetry["use_data_augmentation"]:
                    data_augmentation_func = self.symmetry["data_augmentation_func"]
                    policy_obs_batch, _ = data_augmentation_func(obs=policy_obs_batch, actions=None, env=self.symmetry["_env"])
                    num_aug = int(policy_obs_batch.shape[0] / original_batch_size)

                mean_actions_batch = self.policy.act_inference(policy_obs_batch.detach().clone())
                action_mean_orig = mean_actions_batch[:original_batch_size]
                _, actions_mean_symm_batch = data_augmentation_func(
                    obs=None, actions=action_mean_orig, env=self.symmetry["_env"]
                )

                mse_loss = torch.nn.MSELoss()
                symmetry_loss = mse_loss(
                    mean_actions_batch[original_batch_size:], actions_mean_symm_batch.detach()[original_batch_size:]
                )

                if self.symmetry["use_mirror_loss"]:
                    loss += self.symmetry["mirror_loss_coeff"] * symmetry_loss
                else:
                    symmetry_loss = symmetry_loss.detach()

            # -----------------------------------------------------
            # 4. AMP Discriminator Loss
            # -----------------------------------------------------
            policy_state, policy_next_state = sample_amp_policy
            expert_state, expert_next_state = sample_amp_expert

            if self.amp_normalizer is not None:
                with torch.no_grad():
                    policy_state = self.amp_normalizer.normalize_torch(policy_state, self.device)
                    policy_next_state = self.amp_normalizer.normalize_torch(policy_next_state, self.device)
                    expert_state = self.amp_normalizer.normalize_torch(expert_state, self.device)
                    expert_next_state = self.amp_normalizer.normalize_torch(expert_next_state, self.device)
            
            policy_cat = torch.cat([policy_state, policy_next_state], dim=-1)
            expert_cat = torch.cat([expert_state, expert_next_state], dim=-1)
            
            policy_d = self.discriminator(policy_cat)
            expert_d = self.discriminator(expert_cat)

            # LSGAN Loss
            expert_loss = torch.nn.MSELoss()(expert_d, torch.ones_like(expert_d))
            policy_loss = torch.nn.MSELoss()(policy_d, -1 * torch.ones_like(policy_d))
            amp_loss = 0.5 * (expert_loss + policy_loss)
            
            # Gradient Penalty
            grad_pen_loss = self.discriminator.compute_grad_pen(expert_state, expert_next_state, lambda_=10)

            # -----------------------------------------------------
            # 5. Total Loss & Optimization
            # -----------------------------------------------------
            
            amp_total_loss = self.amploss_coef * amp_loss + self.amploss_coef * grad_pen_loss

            # -- For PPO
            self.optimizer.zero_grad()
            loss.backward()
            
            # -- For AMP
            self.amp_optimizer.zero_grad()
            amp_total_loss.backward()

            if self.is_multi_gpu:
                self.reduce_parameters(self.rl_parameters, self.discriminator.parameters())

            # -- For PPO
            nn.utils.clip_grad_norm_(self.rl_parameters, self.max_grad_norm)
            self.optimizer.step()
            
            # -- For VAE
            self.vae_optimizer.zero_grad()
            autoenc_loss.backward()
            
            if self.is_multi_gpu:
                self.reduce_parameters(self.vae_parameters, [])
                
            nn.utils.clip_grad_norm_(self.vae_parameters, self.max_grad_norm)
            self.vae_optimizer.step()
            
            # -- For AMP
            nn.utils.clip_grad_norm_(self.discriminator.parameters(), self.max_grad_norm)
            self.amp_optimizer.step()

            # Update AMP Normalizer
            if self.amp_normalizer is not None:
                self.amp_normalizer.update(sample_amp_policy[0].cpu().numpy())
                self.amp_normalizer.update(sample_amp_expert[0].cpu().numpy())

            # -----------------------------------------------------
            # 6. Logging
            # -----------------------------------------------------
            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy_batch.mean().item()
            mean_autoenc_loss += autoenc_loss.item()
            mean_vel_loss += vel_loss_term.item()
            mean_amp_loss += amp_loss.item()
            mean_grad_pen_loss += grad_pen_loss.item()
            mean_policy_pred += policy_d.mean().item()
            mean_expert_pred += expert_d.mean().item()
            # -- Symmetry loss
            if mean_symmetry_loss is not None:
                mean_symmetry_loss += symmetry_loss.item()

        # Average stats
        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates
        mean_autoenc_loss /= num_updates
        mean_vel_loss /= num_updates
        mean_amp_loss /= num_updates
        mean_grad_pen_loss /= num_updates
        mean_policy_pred /= num_updates
        mean_expert_pred /= num_updates
        # -- For Symmetry
        if mean_symmetry_loss is not None:
            mean_symmetry_loss /= num_updates

        self.storage.clear()

        loss_dict = {
            "value_function": mean_value_loss,
            "surrogate": mean_surrogate_loss,
            "entropy": mean_entropy,
            # DWAQ logs
            "autoencoder": mean_autoenc_loss,
            "velocity_loss": mean_vel_loss,
            # AMP logs
            "amp/loss": mean_amp_loss,
            "amp/grad_pen": mean_grad_pen_loss,
            "amp/policy_pred": mean_policy_pred,
            "amp/expert_pred": mean_expert_pred,
        }
        if self.symmetry:
            loss_dict["symmetry"] = mean_symmetry_loss

        return loss_dict
    
    def broadcast_parameters(self):
        """Broadcast model parameters to all GPUs."""
        model_params = [self.policy.state_dict(), self.discriminator.state_dict()]
        torch.distributed.broadcast_object_list(model_params, src=0)
        self.policy.load_state_dict(model_params[0])
        self.discriminator.load_state_dict(model_params[1])

    def reduce_parameters(self, params1, params2=[]):
        """Collect gradients from all GPUs and average them."""
        grads = [param.grad.view(-1) for param in chain(params1, params2) if param.grad is not None]
        
        if not grads:
            return 

        all_grads = torch.cat(grads)
        torch.distributed.all_reduce(all_grads, op=torch.distributed.ReduceOp.SUM)
        all_grads /= self.gpu_world_size

        all_params = chain(params1, params2)
        
        offset = 0
        for param in all_params:
            if param.grad is not None:
                numel = param.numel()
                param.grad.data.copy_(all_grads[offset : offset + numel].view_as(param.grad.data))
                offset += numel