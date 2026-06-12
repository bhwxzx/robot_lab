# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim
from itertools import chain

from rsl_rl.modules import ActorCritic
from rsl_rl.storage import RolloutStorage, ReplayBuffer


class AMPROAPPO:
    """Proximal Policy Optimization algorithm with AMP (Adversarial Motion Prior) and ROA (Regularized Online Adaptation)."""

    def __init__(
        self,
        policy,
        # --- [AMP 新增参数] ---
        discriminator,
        amp_data,
        amp_normalizer,
        # --- [ROA 新增参数] ---
        priv_reg_coef_schedule=[0.0, 0.1, 1000, 2000],
        dagger_update_freq=1,
        vel_loss_coef=1.0,
        # --- [PPO 通用参数] ---
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
        # --- [AMP 配置参数] ---
        amp_replay_buffer_size=100000,
        min_std=None,
        amp_reward_coef=2.0,
        amp_task_reward_lerp=0.3,
        amp_discr_hidden_dims=None,
        disc_learning_rate=1e-4,
        # Distributed training parameters
        multi_gpu_cfg: dict | None = None,
        **kwargs
    ):
        if kwargs:
            print(f"[AMP_ROA_PPO] 忽略了多余的配置参数: {list(kwargs.keys())}")
        
        self.device = device
        self.is_multi_gpu = multi_gpu_cfg is not None
        if multi_gpu_cfg is not None:
            self.gpu_global_rank = multi_gpu_cfg["global_rank"]
            self.gpu_world_size = multi_gpu_cfg["world_size"]
        else:
            self.gpu_global_rank = 0
            self.gpu_world_size = 1

        # --- [ROA 参数] ---
        self.priv_reg_coef_schedule = priv_reg_coef_schedule
        self.dagger_update_freq = dagger_update_freq

        # --- [AMP 初始化] ---
        self.discriminator = discriminator
        self.discriminator.to(self.device)
        self.amp_data = amp_data
        self.amp_normalizer = amp_normalizer
        self.amploss_coef = 1.0  
        self.amp_reward_coef = amp_reward_coef
        self.min_std = min_std

        self.amp_storage = ReplayBuffer(
            discriminator.input_dim // 2, amp_replay_buffer_size, device
        )
        self.amp_transition = RolloutStorage.Transition()

        # --- [Policy] ---
        self.policy = policy
        self.policy.to(self.device)

        # --- [解耦的主优化器] ---
        ppo_params = [
            {"params": self.policy.parameters(), "name": "policy"},
        ]
        self.optimizer = optim.Adam(ppo_params, lr=learning_rate)
        
        amp_params = [
            {"params": self.discriminator.trunk.parameters(), "weight_decay": 1e-4, "name": "amp_trunk"},
            {"params": self.discriminator.amp_linear.parameters(), "weight_decay": 1e-2, "name": "amp_head"},
        ]
        self.amp_optimizer = optim.Adam(amp_params, lr=disc_learning_rate)

        # --- [ROA专属历史编码器优化器] ---
        if hasattr(self.policy, "history_encoder"):
            self.hist_encoder_optimizer = optim.Adam(self.policy.history_encoder.parameters(), lr=learning_rate)
        else:
            self.hist_encoder_optimizer = None

        # PPO components
        self.storage: RolloutStorage = None  # type: ignore
        self.transition = RolloutStorage.Transition()

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
        
        self.counter = 0
        self.vel_loss_coef = vel_loss_coef

    def init_storage(self, training_type, num_envs, num_transitions_per_env, obs, actions_shape):
        self.storage = RolloutStorage(
            training_type, num_envs, num_transitions_per_env, obs, actions_shape, self.device,
        )

    def act(self, obs, amp_obs=None, hist_encoding=False):
        # 注意: 训练时通常默认 hist_encoding=False, 让特权编码器指引网络。只在蒸馏时/评估时开 True
        self.transition.actions = self.policy.act(obs, hist_encoding=hist_encoding).detach()
        self.transition.values = self.policy.evaluate(obs).detach()
        self.transition.actions_log_prob = self.policy.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.policy.action_mean.detach()
        self.transition.action_sigma = self.policy.action_std.detach()
        self.transition.observations = obs
        if amp_obs is not None:
            self.amp_transition.observations = amp_obs
        return self.transition.actions

    def process_env_step(self, obs, rewards, dones, extras, amp_obs):
        self.policy.update_normalization(obs)
        self.transition.dones = dones
        
        self.amp_storage.insert(self.amp_transition.observations, amp_obs)

        # AMP 奖励计算
        amp_rewards, policy_d = self.discriminator.predict_amp_reward(
            self.amp_transition.observations, 
            amp_obs,                           
            task_reward=rewards,               
            normalizer=self.amp_normalizer     
        )

        if self.storage.step == 0:
            with torch.no_grad():
                raw_amp = self.discriminator.dt * self.discriminator.amp_reward_coef * torch.clamp(1 - (1 / 4) * torch.square(policy_d - 1), min=0)
                print(f"\n[AMP DEBUG] Task Reward Mean: {rewards.mean().item():.4f} | Raw AMP Reward Mean: {raw_amp.mean().item():.4f}")

        if self.discriminator.task_reward_lerp > 0:
            self.transition.rewards = amp_rewards
        else:
            self.transition.rewards = rewards + amp_rewards

        if "time_outs" in extras:
            self.transition.rewards += self.gamma * torch.squeeze(
                self.transition.values * extras["time_outs"].unsqueeze(1).to(self.device), 1
            )

        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.policy.reset(dones)

    def compute_returns(self, obs):
        last_values = self.policy.evaluate(obs).detach()
        self.storage.compute_returns(
            last_values, self.gamma, self.lam, normalize_advantage=not self.normalize_advantage_per_mini_batch
        ) 

    def update(self):
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_entropy = 0
        mean_priv_reg_loss = 0
        mean_amp_loss = 0
        mean_grad_pen_loss = 0
        mean_policy_pred = 0
        mean_expert_pred = 0

        generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        amp_policy_generator = self.amp_storage.feed_forward_generator(
            self.num_learning_epochs * self.num_mini_batches,
            self.storage.num_envs * self.storage.num_transitions_per_env // self.num_mini_batches,
        )
        amp_expert_generator = self.amp_data.feed_forward_generator(
            self.num_learning_epochs * self.num_mini_batches,
            self.storage.num_envs * self.storage.num_transitions_per_env // self.num_mini_batches,
        )

        for sample, sample_amp_policy, sample_amp_expert in zip(generator, amp_policy_generator, amp_expert_generator):
            (
                obs_batch, actions_batch, target_values_batch, advantages_batch,
                returns_batch, old_actions_log_prob_batch, old_mu_batch, old_sigma_batch,
                hid_states_batch, masks_batch,
            ) = sample

            original_batch_size = obs_batch.batch_size[0]

            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    advantages_batch = (advantages_batch - advantages_batch.mean()) / (advantages_batch.std() + 1e-8)

            # Recompute
            self.policy.act(obs_batch, masks=masks_batch, hidden_states=hid_states_batch[0])
            actions_log_prob_batch = self.policy.get_actions_log_prob(actions_batch)
            value_batch = self.policy.evaluate(obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1])
            mu_batch = self.policy.action_mean[:original_batch_size]
            sigma_batch = self.policy.action_std[:original_batch_size]
            entropy_batch = self.policy.entropy[:original_batch_size]

            # KL Adaptation
            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
                        + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch))
                        / (2.0 * torch.square(sigma_batch)) - 0.5, axis=-1,
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
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # Value Loss
            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(-self.clip_param, self.clip_param)
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            # --- [ROA: Privileged Regularization Loss] ---
            priv_latent_batch = self.policy.infer_priv_latent(obs_batch)
            with torch.inference_mode():
                hist_latent_batch = self.policy.infer_hist_latent(obs_batch)
            priv_reg_loss = (priv_latent_batch - hist_latent_batch.detach()).norm(p=2, dim=1).mean()
            
            # Dynamic coeff schedule (4 parameters: [start_val, end_val, start_iter, fade_iters])
            stage = min(max((self.counter - self.priv_reg_coef_schedule[2]), 0) / (self.priv_reg_coef_schedule[3] + 1e-8), 1.0)
            priv_reg_coef = stage * (self.priv_reg_coef_schedule[1] - self.priv_reg_coef_schedule[0]) + self.priv_reg_coef_schedule[0]

            # --- [AMP: Discriminator Loss] ---
            policy_state, policy_next_state = sample_amp_policy
            expert_state, expert_next_state = sample_amp_expert

            if self.amp_normalizer is not None:
                with torch.no_grad():
                    policy_state = self.amp_normalizer.normalize_torch(policy_state, self.device)
                    policy_next_state = self.amp_normalizer.normalize_torch(policy_next_state, self.device)
                    expert_state = self.amp_normalizer.normalize_torch(expert_state, self.device)
                    expert_next_state = self.amp_normalizer.normalize_torch(expert_next_state, self.device)
            
            policy_d = self.discriminator(torch.cat([policy_state, policy_next_state], dim=-1))
            expert_d = self.discriminator(torch.cat([expert_state, expert_next_state], dim=-1))

            expert_loss = torch.nn.MSELoss()(expert_d, torch.ones_like(expert_d))
            policy_loss = torch.nn.MSELoss()(policy_d, -1 * torch.ones_like(policy_d))
            amp_loss = 0.5 * (expert_loss + policy_loss)
            
            grad_pen_loss = self.discriminator.compute_grad_pen(expert_state, expert_next_state, lambda_=10)

            # ====== 解耦的 Loss ======
            loss = (surrogate_loss + 
                    self.value_loss_coef * value_loss - 
                    self.entropy_coef * entropy_batch.mean() + 
                    priv_reg_coef * priv_reg_loss)
                    
            amp_total_loss = self.amploss_coef * amp_loss + self.amploss_coef * grad_pen_loss

            # Optimization
            # -- For PPO
            self.optimizer.zero_grad()
            loss.backward()
            
            # -- For AMP
            self.amp_optimizer.zero_grad()
            amp_total_loss.backward()

            if self.is_multi_gpu:
                self.reduce_parameters()

            # -- For PPO
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()
            
            # -- For AMP
            nn.utils.clip_grad_norm_(self.discriminator.parameters(), self.max_grad_norm)
            self.amp_optimizer.step()

            if self.amp_normalizer is not None:
                self.amp_normalizer.update(sample_amp_policy[0].cpu().numpy())
                self.amp_normalizer.update(sample_amp_expert[0].cpu().numpy())

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy_batch.mean().item()
            mean_priv_reg_loss += priv_reg_loss.item()
            mean_amp_loss += amp_loss.item()
            mean_grad_pen_loss += grad_pen_loss.item()
            mean_policy_pred += policy_d.mean().item()
            mean_expert_pred += expert_d.mean().item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates
        mean_priv_reg_loss /= num_updates
        mean_amp_loss /= num_updates
        mean_grad_pen_loss /= num_updates
        mean_policy_pred /= num_updates
        mean_expert_pred /= num_updates
        
        self.storage.clear()
        self.counter += 1

        return {
            "value_function": mean_value_loss,
            "surrogate": mean_surrogate_loss,
            "entropy": mean_entropy,
            "priv_reg": mean_priv_reg_loss,
            "amp/loss": mean_amp_loss,
            "amp/grad_pen": mean_grad_pen_loss,
            "amp/policy_pred": mean_policy_pred,
            "amp/expert_pred": mean_expert_pred,
        }

    def update_dagger(self):
        """ ROA 的监督蒸馏阶段 (History Encoder 学习阶段) """
        if self.hist_encoder_optimizer is None:
            return {}
        
        mean_hist_latent_loss = 0
        mean_vel_loss = 0
        generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
            
        for (obs_batch, _, _, _, _, _, _, _, hid_states_batch, masks_batch) in generator:
            with torch.inference_mode():
                self.policy.act(obs_batch, hist_encoding=True, masks=masks_batch, hidden_states=hid_states_batch[0] if hid_states_batch else None)
                priv_latent_batch = self.policy.infer_priv_latent(obs_batch)
                true_vel_batch = self.policy.get_true_vel(obs_batch)
                
            hist_latent_batch, pred_vel_batch = self.policy.infer_hist_latent(obs_batch, return_vel=True)
            hist_latent_loss = (priv_latent_batch.detach() - hist_latent_batch).norm(p=2, dim=1).mean()
            vel_loss = (true_vel_batch.detach() - pred_vel_batch).pow(2).mean()
            
            total_dagger_loss = hist_latent_loss + self.vel_loss_coef * vel_loss
            
            self.hist_encoder_optimizer.zero_grad()
            total_dagger_loss.backward()
            
            if self.is_multi_gpu:
                self.reduce_history_parameters()
                
            nn.utils.clip_grad_norm_(self.policy.history_encoder.parameters(), self.max_grad_norm)
            self.hist_encoder_optimizer.step()
            
            mean_hist_latent_loss += hist_latent_loss.item()
            mean_vel_loss += vel_loss.item()
            
        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_hist_latent_loss /= num_updates
        mean_vel_loss /= num_updates
        return {"hist_latent": mean_hist_latent_loss, "vel_loss": mean_vel_loss}

    def broadcast_parameters(self):
        model_params = [self.policy.state_dict(), self.discriminator.state_dict()]
        torch.distributed.broadcast_object_list(model_params, src=0)
        self.policy.load_state_dict(model_params[0])
        self.discriminator.load_state_dict(model_params[1])

    def reduce_parameters(self):
        grads = [param.grad.view(-1) for param in self.policy.parameters() if param.grad is not None]
        grads += [param.grad.view(-1) for param in self.discriminator.parameters() if param.grad is not None]
        
        all_grads = torch.cat(grads)
        torch.distributed.all_reduce(all_grads, op=torch.distributed.ReduceOp.SUM)
        all_grads /= self.gpu_world_size

        all_params = chain(self.policy.parameters(), self.discriminator.parameters())
        offset = 0
        for param in all_params:
            if param.grad is not None:
                numel = param.numel()
                param.grad.data.copy_(all_grads[offset : offset + numel].view_as(param.grad.data))
                offset += numel

    def reduce_history_parameters(self):
        grads = [param.grad.view(-1) for param in self.policy.history_encoder.parameters() if param.grad is not None]
        if len(grads) == 0:
            return
        all_grads = torch.cat(grads)
        torch.distributed.all_reduce(all_grads, op=torch.distributed.ReduceOp.SUM)
        all_grads /= self.gpu_world_size
        offset = 0
        for param in self.policy.history_encoder.parameters():
            if param.grad is not None:
                numel = param.numel()
                param.grad.data.copy_(all_grads[offset : offset + numel].view_as(param.grad.data))
                offset += numel
