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
from rsl_rl.storage import RolloutStorageDwaq
from rsl_rl.utils import string_to_callable


class DWAQPPO:
    """结合了 DWAQ (β-VAE 上下文编码器) 的 PPO 算法。"""

    policy: ActorCriticDwaq
    """The actor critic module."""

    def __init__(
        self,
        policy,
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
        # DWAQ 特有参数
        obs_dim=41,          # 原始本体感受观察值维度（用于提取速度真值标签）
        vae_beta=1.0,        # KL 散度权重系数
        # Distributed training parameters
        multi_gpu_cfg: dict | None = None,
        **kwargs, # 兼容多余参数传递
    ):
        if kwargs:
            print(f"[DWAQPPO] 忽略了多余的配置参数: {list(kwargs.keys())}")
        # device-related parameters 决定了模型运行在哪个硬件上，以及是否开启分布式训练
        self.device = device
        # for dwaq
        self.obs_dim = obs_dim
        self.vae_beta = vae_beta

        self.is_multi_gpu = multi_gpu_cfg is not None
        # Multi-GPU parameters
        if multi_gpu_cfg is not None: # 如果是多GPU训练，记录当前进程的等级(rank)和总数(world_size), 用于后续在不同 GPU 之间同步梯度（all_reduce）
            self.gpu_global_rank = multi_gpu_cfg["global_rank"]
            self.gpu_world_size = multi_gpu_cfg["world_size"]
        else:
            self.gpu_global_rank = 0
            self.gpu_world_size = 1

        # PPO components
        self.policy = policy
        self.policy.to(self.device)
        # Create optimizer
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
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

    def act(self, obs, prev_critic_obs):
        # 直接传入 TensorDict，由模型内部自行提取 policy/critic
        self.transition.actions = self.policy.act(obs).detach()
        self.transition.values = self.policy.evaluate(obs).detach()
        self.transition.actions_log_prob = self.policy.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.policy.action_mean.detach()
        self.transition.action_sigma = self.policy.action_std.detach()

        # 记录观察值：obs 包含了历史和当前，prev_critic_obs 记录上一时刻用于速度监督
        self.transition.observations = obs
        self.transition.prev_critic_observations = prev_critic_obs
        
        return self.transition.actions

    def process_env_step(self, obs, rewards, dones, extras):
        # update the normalizers
        self.policy.update_normalization(obs) # 为了让神经网络训练更稳定，通常需要对输入的观测值 obs 进行归一化

        # Record the rewards and dones
        # Note: we clone here because later on we bootstrap the rewards based on timeouts
        self.transition.rewards = rewards.clone() # 因为后面会对奖励进行修改（加上内在奖励或处理超时）。为了不影响原始环境返回的数据，先复制一份副本
        self.transition.dones = dones

        # Bootstrapping on time outs
        if "time_outs" in extras:
            self.transition.rewards += self.gamma * torch.squeeze( # 利用 Critic 网络预测的 values 来“补全”因为时间截止而拿不到的未来奖励
                self.transition.values * extras["time_outs"].unsqueeze(1).to(self.device), 1
            )

        # record the transition
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.policy.reset(dones)

    def compute_returns(self, obs):
        # compute value for the last step
        last_values = self.policy.evaluate(obs).detach()
        self.storage.compute_returns(
            last_values, self.gamma, self.lam, normalize_advantage=not self.normalize_advantage_per_mini_batch
        ) # 如果设置了“在每个小批次（mini-batch）内归一化优势”，那么在这里就不进行全局归一化。
          # 如果没有设置批次内归一化，那么在这里会对整个存储池（Storage）中的优势函数进行全局归一化（减去均值，除以标准差）

    def update(self):  # noqa: C901
        mean_value_loss = 0  # 初始化统计变量（用于记录日志）
        mean_surrogate_loss = 0
        mean_entropy = 0
        mean_autoenc_loss = 0
        mean_vel_loss = 0 # 记录真实速度估计误差

        # generator for mini batches
        # DWAQ 生成器解包 12 个参数
        generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        # iterate over batches
        for (
            policy_obs_batch,      # Actor 输入 以及 VAE输入
            critic_obs_batch,      # Critic 输入
            prev_critic_obs_batch, # 速度监督辅助
            actions_batch,
            target_values_batch,
            advantages_batch,
            returns_batch,
            old_actions_log_prob_batch,
            old_mu_batch,
            old_sigma_batch,
            hid_states_batch,
            masks_batch,
        ) in generator:

            # 1. 优势归一化
            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    advantages_batch = (advantages_batch - advantages_batch.mean()) / (advantages_batch.std() + 1e-8)

            # 2. VAE 损失计算 (修正点：使用 code_vel 和 变量名对齐)
            # 这里的 cenet_forward 内部已经包含了采样
            (latent_code, reconstruction, mu_v, logvar_v, mu_l, logvar_l, current_obs_synced) = self.policy.cenet_forward(policy_obs_batch)
            
            # 直接从 latent_code 中提取采样后的速度部分 (前3维)，作为 Loss 的预测值
            code_vel = latent_code[:, :3] 
            
            # 速度真值标签
            vel_target = critic_obs_batch[:, self.obs_dim : self.obs_dim + 3].detach()
            # 修正后的代码
            decode_target = current_obs_synced.detach()

            # 单独计算速度部分的 MSE Loss (注意：这里不除以 num_mini_batches，为了看真实的物理误差量级)
            vel_loss_term = nn.MSELoss()(code_vel, vel_target)

            # KL 计算 (完全对齐原代码逻辑)
            logvar_l_clamped = torch.clamp(logvar_l, min=-10.0, max=10.0)
            kl_divergence = -0.5 * torch.sum(1 + logvar_l_clamped - mu_l.pow(2) - logvar_l_clamped.exp())
            
            # Autoencoder Loss
            # PPO 的 Surrogate_Loss 通常非常小。
            # 如果 VAE 的 MSE 损失项太强，优化器会优先去优化“如何预测速度”，而忽视了“如何走得稳”。
            # 通过将 MSE 也除以 num_mini_batches，作者实际上是人工调低了 VAE 部分的学习速率，让它的更新节奏慢下来，从而不会带偏 PPO 的主任务。

            autoenc_loss = (
                nn.MSELoss()(code_vel, vel_target) + 
                nn.MSELoss()(reconstruction, decode_target) + 
                self.vae_beta * kl_divergence
            ) / self.num_mini_batches 

            # 3. PPO 调用
            # 构造完整的观察字典，防止模型内部 get_obs_from_group 报错
            full_obs_batch = {
                "policy": policy_obs_batch,
                "critic": critic_obs_batch,
            }
            self.policy.act(full_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[0])
            
            actions_log_prob_batch = self.policy.get_actions_log_prob(actions_batch)
            value_batch = self.policy.evaluate(full_obs_batch) 
            
            mu_batch = self.policy.action_mean
            sigma_batch = self.policy.action_std
            entropy_batch = self.policy.entropy

            # KL
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

                    # Reduce the KL divergence across all GPUs
                    if self.is_multi_gpu: # 在多 GPU 模式下，会使用 all_reduce 和 broadcast 确保所有显卡上的学习率完全同步
                        torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
                        kl_mean /= self.gpu_world_size

                    # Update the learning rate
                    # Perform this adaptation only on the main process
                    # TODO: Is this needed? If KL-divergence is the "same" across all GPUs,
                    #       then the learning rate should be the same across all GPUs.
                    if self.gpu_global_rank == 0:
                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                    # Update the learning rate for all GPUs
                    if self.is_multi_gpu:
                        lr_tensor = torch.tensor(self.learning_rate, device=self.device)
                        torch.distributed.broadcast(lr_tensor, src=0)
                        self.learning_rate = lr_tensor.item()

                    # Update the learning rate for all parameter groups
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            # Surrogate loss
            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # Value function loss
            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                    -self.clip_param, self.clip_param
                )
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            # 总损失与反向传播
            loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean() + autoenc_loss

            # Compute the gradients
            # -- For PPO
            self.optimizer.zero_grad()
            loss.backward()

            # Collect gradients from all GPUs
            if self.is_multi_gpu:
                self.reduce_parameters() # 将所有显卡计算出的梯度求平均

            # Apply the gradients
            # -- For PPO
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()

            # Store the losses
            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy_batch.mean().item()
            # vae loss
            mean_autoenc_loss += autoenc_loss.item()
            mean_vel_loss += vel_loss_term.item()

        # -- For PPO
        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates
        mean_autoenc_loss /= num_updates

        mean_vel_loss /= num_updates
        # -- Clear the storage
        self.storage.clear()

        # construct the loss dictionary
        loss_dict = {
            "value_function": mean_value_loss,
            "surrogate": mean_surrogate_loss,
            "entropy": mean_entropy,
            "autoencoder": mean_autoenc_loss,
            "velocity_loss": mean_vel_loss 
        }

        return loss_dict

    """
    Helper functions
    """

    def broadcast_parameters(self):
        """Broadcast model parameters to all GPUs."""
        # obtain the model parameters on current GPU
        model_params = [self.policy.state_dict()]

        # broadcast the model parameters
        torch.distributed.broadcast_object_list(model_params, src=0)
        # load the model parameters on all GPUs from source GPU
        self.policy.load_state_dict(model_params[0])


    def reduce_parameters(self):
        """Collect gradients from all GPUs and average them.

        This function is called after the backward pass to synchronize the gradients across all GPUs.
        """
        # Create a tensor to store the gradients
        grads = [param.grad.view(-1) for param in self.policy.parameters() if param.grad is not None]

        all_grads = torch.cat(grads)

        # Average the gradients across all GPUs
        torch.distributed.all_reduce(all_grads, op=torch.distributed.ReduceOp.SUM)
        all_grads /= self.gpu_world_size

        # Get all parameters
        all_params = self.policy.parameters()


        # Update the gradients for all parameters with the reduced gradients
        offset = 0
        for param in all_params:
            if param.grad is not None:
                numel = param.numel()
                # copy data back from shared buffer
                param.grad.data.copy_(all_grads[offset : offset + numel].view_as(param.grad.data))
                # update the offset for the next parameter
                offset += numel
