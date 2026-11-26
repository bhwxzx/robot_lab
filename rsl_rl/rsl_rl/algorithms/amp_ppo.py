# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim
from itertools import chain, repeat

from rsl_rl.modules import ActorCritic
from rsl_rl.modules.rnd import RandomNetworkDistillation
from rsl_rl.storage import RolloutStorage, ReplayBuffer
from rsl_rl.utils import string_to_callable


class AMPPPO:
    """Proximal Policy Optimization algorithm with AMP (Adversarial Motion Prior)."""

    amp_command_dim = 0
    policy: ActorCritic
    """The actor critic module."""

    def __init__(
        self,
        policy,
        # --- [AMP 新增参数] ---
        discriminator,
        amp_data,
        amp_normalizer,
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
        # --- [AMP 新增参数] ---
        amp_replay_buffer_size=100000,
        min_std=None,
        amp_reward_coef=2.0,  # AMP 奖励的权重系数，决定风格奖励占比
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
            print(f"[AMPPPO] 忽略了多余的配置参数: {list(kwargs.keys())}")
        # device-related parameters 决定了模型运行在哪个硬件上，以及是否开启分布式训练
        self.device = device
        self.is_multi_gpu = multi_gpu_cfg is not None
        # Multi-GPU parameters
        if multi_gpu_cfg is not None: # 如果是多GPU训练，记录当前进程的等级(rank)和总数(world_size), 用于后续在不同 GPU 之间同步梯度（all_reduce）
            self.gpu_global_rank = multi_gpu_cfg["global_rank"]
            self.gpu_world_size = multi_gpu_cfg["world_size"]
        else:
            self.gpu_global_rank = 0
            self.gpu_world_size = 1

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

        # 初始化 AMP 回放池 (用于存储 Policy 产生的动作序列，供判别器训练)
        self.amp_storage = ReplayBuffer(amp_data.observation_dim, amp_replay_buffer_size, device, self.amp_command_dim)
        self.amp_transition = RolloutStorage.Transition() # 临时的 AMP transition 存储

        # PPO components
        self.policy = policy
        self.policy.to(self.device)

        # --- [优化器修改] ---
        # 我们需要同时优化 Policy 和 Discriminator，且 Discriminator 的不同部分有不同的        # Create optimizer
        ppo_params = [
            {"params": self.policy.parameters(), "name": "policy"},
        ]
        self.optimizer = optim.Adam(ppo_params, lr=learning_rate)
        
        amp_params = [
            {"params": self.discriminator.trunk.parameters(), "weight_decay": 1e-4, "name": "amp_trunk"},
            {"params": self.discriminator.amp_linear.parameters(), "weight_decay": 1e-2, "name": "amp_head"},
        ]
        self.amp_optimizer = optim.Adam(amp_params, lr=disc_learning_rate)

        # Create rollout storage
        self.storage: RolloutStorage = None  # type: ignore
        self.transition = RolloutStorage.Transition()

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
        # create rollout storage
        self.storage = RolloutStorage(
            training_type,
            num_envs,
            num_transitions_per_env,
            obs,
            actions_shape,
            self.device,
        )

    def act(self, obs, amp_obs=None):
        # compute the actions and values                                    
        self.transition.actions = self.policy.act(obs).detach()
        self.transition.values = self.policy.evaluate(obs).detach()
        self.transition.actions_log_prob = self.policy.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.policy.action_mean.detach()
        self.transition.action_sigma = self.policy.action_std.detach()
        # need to record obs before env.step()
        self.transition.observations = obs
        # [AMP] 记录当前的 AMP 观测值
        if amp_obs is not None:
            self.amp_transition.observations = amp_obs
        return self.transition.actions

    def process_env_step(self, obs, rewards, dones, extras, amp_obs, amp_transition_valid=None) -> torch.Tensor:
        """Store the PPO transition and return its mixed reward before timeout bootstrapping."""
        # 1. 常规处理
        self.policy.update_normalization(obs)
        self.transition.dones = dones
        
        # 注意：这里的 rewards 是环境给出的纯任务奖励 (Task Reward)，
        # predict_amp_reward 会根据配置完成任务奖励与 AMP 奖励的插值。
        
        # 2. 只把带有真实 post-step AMP observation 的 transition 写入 ReplayBuffer。
        # IsaacLab 默认在返回观测前重置终止环境，因此这些样本通常需要排除。
        if amp_transition_valid is None:
            amp_transition_valid = ~dones.bool()
        amp_transition_valid = amp_transition_valid.reshape(-1)
        mixed_rewards = self._record_amp_step(rewards, amp_obs, amp_transition_valid, dones)
        self.transition.rewards = mixed_rewards

        # 5. Bootstrapping (超时处理)
        if "time_outs" in extras:
            self.transition.rewards = mixed_rewards + self.gamma * torch.squeeze(
                self.transition.values * extras["time_outs"].unsqueeze(1).to(self.device), 1
            )

        # 6. 记录和重置
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.policy.reset(dones)
        return mixed_rewards

    def _record_amp_step(self, rewards, amp_obs, valid, dones):
        state = self.amp_transition.observations
        if torch.any(valid):
            self.amp_storage.insert(state[valid], amp_obs[valid])
        total, _ = self.discriminator.predict_amp_reward(state, amp_obs, rewards, normalizer=self.amp_normalizer)
        return total

    def _amp_batches(self, count, size):
        policies = self.amp_storage.feed_forward_generator(count, size)
        experts = self.amp_data.feed_forward_generator(count, size)
        for policy, expert in zip(policies, experts):
            yield policy, expert, None

    def _discriminator_losses(self, policy, expert, command):
        def normalize(pair):
            state, next_state = pair
            if self.amp_normalizer is not None:
                with torch.no_grad():
                    next_state = self.discriminator.normalize_amp_observation(next_state, self.amp_normalizer)
                    if not self.discriminator.use_history_window:
                        state = self.discriminator.normalize_amp_observation(state, self.amp_normalizer)
            return state, next_state

        policy_state, policy_next = normalize(policy)
        expert_state, expert_next = normalize(expert)
        policy_d = self.discriminator(self.discriminator.prepare_input(policy_state, policy_next, command))
        # CommandAMP's normalized post-step history is the full motion input to both losses.
        reuse_expert_score = command is not None and self.discriminator.use_history_window
        if reuse_expert_score:
            expert_next = expert_next.detach().clone().requires_grad_(True)
        expert_d = self.discriminator(self.discriminator.prepare_input(expert_state, expert_next, command))
        amp_loss = 0.5 * ((expert_d - 1).square().mean() + (policy_d + 1).square().mean())
        if reuse_expert_score:
            penalty = self.discriminator.grad_pen_from_score(expert_d, expert_next, lambda_=10)
        else:
            penalty = self.discriminator.compute_grad_pen(expert_state, expert_next, lambda_=10, command=command)
        return amp_loss, penalty, policy_d, expert_d

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
        mean_amp_loss = 0       # [新增]
        mean_grad_pen_loss = 0  # [新增]
        mean_policy_pred = 0    # [新增]
        mean_expert_pred = 0    # [新增]
        # -- Symmetry loss
        if self.symmetry:
            mean_symmetry_loss = 0
        else:
            mean_symmetry_loss = None

        # generator for mini batches
        generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        num_updates = self.num_learning_epochs * self.num_mini_batches
        size = self.storage.num_envs * self.storage.num_transitions_per_env // self.num_mini_batches
        ready = self.amp_storage.num_samples > 0
        if self.is_multi_gpu:
            # All ranks must reduce the same gradients while the replay buffers warm up.
            ready_flag = torch.tensor(int(ready), device=self.device)
            torch.distributed.all_reduce(ready_flag, op=torch.distributed.ReduceOp.MIN)
            ready = bool(ready_flag.item())
        amp_batches = self._amp_batches(num_updates, size) if ready else repeat(None, num_updates)

        # iterate over batches
        for sample, amp_batch in zip(generator, amp_batches):
            (
                obs_batch,
                actions_batch,
                target_values_batch,
                advantages_batch,
                returns_batch,
                old_actions_log_prob_batch,
                old_mu_batch,
                old_sigma_batch,
                hid_states_batch,
                masks_batch,
            )=sample

            # original batch size
            # we assume policy group is always there and needs augmentation
            original_batch_size = obs_batch.batch_size[0]

            # we start with 1 and increase it if we use symmetry augmentation
            num_aug = 1

            # check if we should normalize advantages per mini batch
            if self.normalize_advantage_per_mini_batch: # 如果开启了 normalize_advantage_per_mini_batch，会在每个小批次内重新计算优势函数的均值和方差。这能让梯度更加稳定
                with torch.no_grad():
                    advantages_batch = (advantages_batch - advantages_batch.mean()) / (advantages_batch.std() + 1e-8)

            if self.symmetry and self.symmetry["use_data_augmentation"]:
                # augmentation using symmetry
                data_augmentation_func = self.symmetry["data_augmentation_func"]
                obs_batch, actions_batch = data_augmentation_func(
                    obs=obs_batch,
                    actions=actions_batch,
                    env=self.symmetry["_env"],
                )
                # recompute the number of augmentations
                num_aug = int(obs_batch.batch_size[0] / original_batch_size)
                # repeat the other parts of the batch
                old_actions_log_prob_batch = old_actions_log_prob_batch.repeat(num_aug, 1)
                target_values_batch = target_values_batch.repeat(num_aug, 1)
                advantages_batch = advantages_batch.repeat(num_aug, 1)
                returns_batch = returns_batch.repeat(num_aug, 1)

            # Recompute actions log prob and entropy for current batch of transitions
            # Note: we need to do this because we updated the policy with the new parameters
            # -- actor # 由于网络权重在不断更新，我们需要用当前最新的网络去重新运行一遍 obs_batch
            self.policy.act(obs_batch, masks=masks_batch, hidden_states=hid_states_batch[0])
            actions_log_prob_batch = self.policy.get_actions_log_prob(actions_batch)
            # -- critic
            value_batch = self.policy.evaluate(obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1])
            # -- entropy
            # we only keep the entropy of the first augmentation (the original one)
            mu_batch = self.policy.action_mean[:original_batch_size]
            sigma_batch = self.policy.action_std[:original_batch_size]
            entropy_batch = self.policy.entropy[:original_batch_size]

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

            loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()

            # Symmetry loss
            if self.symmetry:
                if not self.symmetry["use_data_augmentation"]:
                    data_augmentation_func = self.symmetry["data_augmentation_func"]
                    obs_batch, _ = data_augmentation_func(obs=obs_batch, actions=None, env=self.symmetry["_env"])
                    num_aug = int(obs_batch.shape[0] / original_batch_size)

                if (
                    self.amp_command_dim
                    and self.symmetry["use_data_augmentation"]
                    and type(self.policy) is ActorCritic
                ):
                    # act() already evaluated this feed-forward actor on the full augmented batch.
                    mean_actions_batch = self.policy.action_mean
                else:
                    mean_actions_batch = self.policy.act_inference(obs_batch.detach().clone())
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

            # Compute the gradients
            # -- For PPO
            self.optimizer.zero_grad()
            loss.backward()
            
            # -- For AMP
            self.amp_optimizer.zero_grad()
            if amp_batch is not None:
                policy_sample, expert_sample, command = amp_batch
                amp_loss, grad_pen_loss, policy_d, expert_d = self._discriminator_losses(
                    policy_sample, expert_sample, command
                )
                (self.amploss_coef * (amp_loss + grad_pen_loss)).backward()

            # Collect gradients from all GPUs
            if self.is_multi_gpu:
                self.reduce_parameters() # 将所有显卡计算出的梯度求平均

            # Apply the gradients
            # -- For PPO
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()
            
            # -- For AMP
            if amp_batch is not None:
                nn.utils.clip_grad_norm_(self.discriminator.parameters(), self.max_grad_norm)
                self.amp_optimizer.step()
                if self.amp_normalizer is not None:
                    self.discriminator.update_amp_normalizer(self.amp_normalizer, policy_sample[1])
                    self.discriminator.update_amp_normalizer(self.amp_normalizer, expert_sample[1])
                mean_amp_loss += amp_loss.item()
                mean_grad_pen_loss += grad_pen_loss.item()
                mean_policy_pred += policy_d.mean().item()
                mean_expert_pred += expert_d.mean().item()

            # Store the losses
            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy_batch.mean().item()
            # -- Symmetry loss
            if mean_symmetry_loss is not None:
                mean_symmetry_loss += symmetry_loss.item()

        # -- For PPO
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates
        mean_amp_loss /= num_updates
        mean_grad_pen_loss /= num_updates
        mean_policy_pred /= num_updates
        mean_expert_pred /= num_updates
        # -- For Symmetry
        if mean_symmetry_loss is not None:
            mean_symmetry_loss /= num_updates
        # -- Clear the storage
        self.storage.clear()

        # construct the loss dictionary
        loss_dict = {
            "value_function": mean_value_loss,
            "surrogate": mean_surrogate_loss,
            "entropy": mean_entropy,
            "amp/loss": mean_amp_loss,
            "amp/grad_pen": mean_grad_pen_loss,
            "amp/policy_pred": mean_policy_pred,
            "amp/expert_pred": mean_expert_pred,
        }
        if self.symmetry:
            loss_dict["symmetry"] = mean_symmetry_loss

        return loss_dict

    """
    Helper functions
    """

    def broadcast_parameters(self):
        """Broadcast model parameters to all GPUs."""
        # obtain the model parameters on current GPU
        model_params = [self.policy.state_dict(), self.discriminator.state_dict()]
        torch.distributed.broadcast_object_list(model_params, src=0)
        self.policy.load_state_dict(model_params[0])
        self.discriminator.load_state_dict(model_params[1])

    def reduce_parameters(self):
        """Collect gradients from all GPUs and average them.

        This function is called after the backward pass to synchronize the gradients across all GPUs.
        """
        """Collect gradients from all GPUs and average them."""
        # 收集 Policy 的梯度
        grads = [param.grad.view(-1) for param in self.policy.parameters() if param.grad is not None]
        # [新增] 收集 Discriminator 的梯度
        grads += [param.grad.view(-1) for param in self.discriminator.parameters() if param.grad is not None]
        
        all_grads = torch.cat(grads)
        torch.distributed.all_reduce(all_grads, op=torch.distributed.ReduceOp.SUM)
        all_grads /= self.gpu_world_size

        all_params = chain(self.policy.parameters(), self.discriminator.parameters()) # [新增] 链接参数
        
        offset = 0
        for param in all_params:
            if param.grad is not None:
                numel = param.numel()
                param.grad.data.copy_(all_grads[offset : offset + numel].view_as(param.grad.data))
                offset += numel
