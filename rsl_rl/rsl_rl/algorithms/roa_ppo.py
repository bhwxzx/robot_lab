# roa_ppo.py
import torch
import torch.nn as nn
import torch.optim as optim
from itertools import chain

from rsl_rl.algorithms.ppo import PPO


class ROAPPO(PPO):
    """
    带有 ROA (Regularized Online Adaptation) 机制的 PPO 算法改进版。
    
    在传统的强化学习流程中增加了：
    1. Privileged Regularization Loss (在 update 阶段): 强迫特权编码器不要学得“太超前”，必须保证提取的信息是历史数据能够推断出来的。
    2. DAgger 更新 (在 update_dagger 阶段): 监督学习，让历史观测编码器去模仿特权编码器的输出。
    """
    def __init__(self, policy, 
                 priv_reg_coef_schedule=[0, 0.1, 2000, 3000], 
                 dagger_update_freq=20, 
                 vel_loss_coef=1.0,
                 **kwargs):
        super().__init__(policy=policy, **kwargs)
        
        # ROA 专属参数
        # priv_reg_coef_schedule: 控制正则化系数的动态调度
        # 格式为: [初始系数值, 目标最大系数值, 开始增加的迭代次数(Delay), 持续增加的迭代跨度(Duration)]
        # 例如 [0, 0.1, 2000, 3000]
        self.priv_reg_coef_schedule = priv_reg_coef_schedule
        self.dagger_update_freq = dagger_update_freq
        self.counter = 0  # 记录当前的总更新次数，用于计算系数的阶段性调度
        
        # 新增测速损失系数配置
        self.vel_loss_coef = vel_loss_coef
        
        # 为历史编码器 (History Encoder) 创建一个独立的优化器
        # 原因是在 update_dagger 蒸馏阶段时，我们只更新历史编码器，不更新整个 Actor。
        if hasattr(self.policy, 'history_encoder') and self.policy.history_encoder is not None:
            self.hist_encoder_optimizer = optim.Adam(self.policy.history_encoder.parameters(), lr=self.learning_rate)
        else:
            self.hist_encoder_optimizer = None

    def act(self, obs, hist_encoding=False):
        """
        环境交互步骤，获取机器人当前的动作。
        训练阶段通常 hist_encoding=False (依靠特权数据训练出更好的上限)，但在部署测试或真机上需要置为 True。
        """
        if self.policy.is_recurrent:
            self.transition.hidden_states = self.policy.get_hidden_states()
        
        self.transition.actions = self.policy.act(obs, hist_encoding=hist_encoding).detach()
        self.transition.values = self.policy.evaluate(obs).detach()
        self.transition.actions_log_prob = self.policy.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.policy.action_mean.detach()
        self.transition.action_sigma = self.policy.action_std.detach()
        self.transition.observations = obs
        return self.transition.actions

    def update(self):
        """
        PPO 主力更新方法。
        在此方法内，会额外计算 ROA 损失（正则化项），限制 Privileged Encoder 的表达范围。
        """
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_entropy = 0
        mean_priv_reg_loss = 0
        mean_rnd_loss = 0 if self.rnd else None
        mean_symmetry_loss = 0 if self.symmetry else None

        if self.policy.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        for (
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
        ) in generator:
            
            # 兼容处理批次大小：无论是传入字典形式的数据还是张量形式
            if hasattr(obs_batch, 'batch_size'):
                original_batch_size = obs_batch.batch_size[0]
            elif isinstance(obs_batch, dict):
                original_batch_size = list(obs_batch.values())[0].shape[0]
            else:
                original_batch_size = obs_batch.shape[0]

            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    advantages_batch = (advantages_batch - advantages_batch.mean()) / (advantages_batch.std() + 1e-8)

            # 数据增广(Symmetry Data Augmentation) 处理...
            if self.symmetry and self.symmetry["use_data_augmentation"]:
                data_augmentation_func = self.symmetry["data_augmentation_func"]
                obs_batch, actions_batch = data_augmentation_func(
                    obs=obs_batch, actions=actions_batch, env=self.symmetry["_env"]
                )
                
                if hasattr(obs_batch, 'batch_size'):
                    num_aug = int(obs_batch.batch_size[0] / original_batch_size)
                elif isinstance(obs_batch, dict):
                    num_aug = int(list(obs_batch.values())[0].shape[0] / original_batch_size)
                else:
                    num_aug = int(obs_batch.shape[0] / original_batch_size)
                    
                old_actions_log_prob_batch = old_actions_log_prob_batch.repeat(num_aug, 1)
                target_values_batch = target_values_batch.repeat(num_aug, 1)
                advantages_batch = advantages_batch.repeat(num_aug, 1)
                returns_batch = returns_batch.repeat(num_aug, 1)

            # 重新计算前向传播
            # 注意：使用 hist_encoding=False，表示使用“特权信息”来训练主强化学习策略
            self.policy.act(obs_batch, hist_encoding=False, masks=masks_batch, hidden_states=hid_states_batch[0] if hid_states_batch else None)
            actions_log_prob_batch = self.policy.get_actions_log_prob(actions_batch)
            value_batch = self.policy.evaluate(obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1] if hid_states_batch else None)
            
            mu_batch = self.policy.action_mean[:original_batch_size]
            sigma_batch = self.policy.action_std[:original_batch_size]
            entropy_batch = self.policy.entropy[:original_batch_size]

            # =========================================================================
            # ROA 核心逻辑 1：计算特权正则化损失 (Privileged Regularization Loss)
            # =========================================================================
            
            # 1. 获取特权编码器的隐空间输出 (Priv Latent)，该张量在计算图中，会被梯度更新影响
            priv_latent_batch = self.policy.infer_priv_latent(obs_batch)
            
            # 2. 获取历史编码器的隐空间输出 (Hist Latent)。
            # 我们不希望正则化的梯度反向流过历史编码器，这部分完全由 update_dagger 来管，因此放进 inference_mode
            with torch.inference_mode():
                hist_latent_batch = self.policy.infer_hist_latent(obs_batch)
            
            # 3. 计算两者欧氏距离的惩罚。
            # 这是 ROA 的灵魂：它迫使“特权隐向量”不敢远离“历史隐向量当前能够推断出的范围”。
            # 进而约束 RL 策略不要去依赖那些过于刁钻、现实中根本算不出来的上帝视角信息。
            priv_reg_loss = (priv_latent_batch - hist_latent_batch.detach()).norm(p=2, dim=1).mean()
            
            # 4. 动态调整正则化强度的调度器
            # 格式: [init_val, target_val, start_iter, duration]
            # Dynamic coeff schedule (4 parameters: [start_val, end_val, start_iter, fade_iters])
            # priv_reg_stage: 0.0 -> 1.0 (linear warmup)
            stage = min(max((self.counter - self.priv_reg_coef_schedule[2]), 0) / (self.priv_reg_coef_schedule[3] + 1e-8), 1.0)
            priv_reg_coef = stage * (self.priv_reg_coef_schedule[1] - self.priv_reg_coef_schedule[0]) + self.priv_reg_coef_schedule[0]

            # PPO KL 散度调整自适应学习率
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

            # 计算 PPO Surrogate loss
            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # 计算 Critic Value loss
            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(-self.clip_param, self.clip_param)
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            # ====== 将 ROA 损失 (priv_reg_loss) 叠加到最终 Loss 中更新整个网络 ======
            loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean() + priv_reg_coef * priv_reg_loss

            # 其余扩展功能：计算镜像/对称损失 (Symmetry) ...
            symmetry_loss = 0
            if self.symmetry:
                if not self.symmetry["use_data_augmentation"]:
                    data_augmentation_func = self.symmetry["data_augmentation_func"]
                    obs_batch, _ = data_augmentation_func(obs=obs_batch, actions=None, env=self.symmetry["_env"])
                mean_actions_batch = self.policy.act_inference(obs_batch, hist_encoding=False)
                action_mean_orig = mean_actions_batch[:original_batch_size]
                _, actions_mean_symm_batch = data_augmentation_func(
                    obs=None, actions=action_mean_orig, env=self.symmetry["_env"]
                )
                mse_loss = torch.nn.MSELoss()
                symmetry_loss = mse_loss(mean_actions_batch[original_batch_size:], actions_mean_symm_batch.detach()[original_batch_size:])
                if self.symmetry["use_mirror_loss"]:
                    loss += self.symmetry["mirror_loss_coeff"] * symmetry_loss
                else:
                    symmetry_loss = symmetry_loss.detach()

            # 其余扩展功能：计算好奇心损失 (RND) ...
            rnd_loss = 0
            if self.rnd:
                with torch.no_grad():
                    if isinstance(obs_batch, dict):
                        rnd_state_batch = self.rnd.get_rnd_state({k: v[:original_batch_size] for k, v in obs_batch.items()})
                    else:
                        rnd_state_batch = self.rnd.get_rnd_state(obs_batch[:original_batch_size])
                    rnd_state_batch = self.rnd.state_normalizer(rnd_state_batch)
                predicted_embedding = self.rnd.predictor(rnd_state_batch)
                target_embedding = self.rnd.target(rnd_state_batch).detach()
                mseloss = torch.nn.MSELoss()
                rnd_loss = mseloss(predicted_embedding, target_embedding)

            # ====== 统一反向传播与梯度更新 ======
            self.optimizer.zero_grad()
            loss.backward()
            if self.rnd:
                self.rnd_optimizer.zero_grad()
                rnd_loss.backward()

            if self.is_multi_gpu:
                self.reduce_parameters()

            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()
            if self.rnd:
                self.rnd_optimizer.step()

            # 统计数据
            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy_batch.mean().item()
            mean_priv_reg_loss += priv_reg_loss.item()
            if mean_rnd_loss is not None: mean_rnd_loss += rnd_loss.item()
            if mean_symmetry_loss is not None: mean_symmetry_loss += symmetry_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates
        mean_priv_reg_loss /= num_updates
        if mean_rnd_loss is not None: mean_rnd_loss /= num_updates
        if mean_symmetry_loss is not None: mean_symmetry_loss /= num_updates
        
        self.storage.clear()
        self.counter += 1

        loss_dict = {
            "value_function": mean_value_loss,
            "surrogate": mean_surrogate_loss,
            "entropy": mean_entropy,
            "priv_reg": mean_priv_reg_loss,
        }
        if self.rnd: loss_dict["rnd"] = mean_rnd_loss
        if self.symmetry: loss_dict["symmetry"] = mean_symmetry_loss

        return loss_dict

    def update_dagger(self):
        """
        =========================================================================
        ROA 核心逻辑 2：历史编码器的监督蒸馏 (DAgger 阶段)
        =========================================================================
        该方法独立于 RL 策略的更新。用于将特权编码器蕴含的知识，“教”给历史编码器。
        在此过程中，特权编码器被彻底冻结（纯粹作为标签产生器），历史编码器进行监督学习，努力拉近两者的隐向量距离。
        """
        if self.hist_encoder_optimizer is None:
            return {}
        
        mean_hist_latent_loss = 0
        mean_vel_loss = 0
        if self.policy.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
            
        for (
            obs_batch, _, _, _, _, _, _, _, hid_states_batch, masks_batch
        ) in generator:
            
            # 1. 使用 inference_mode 完全冻结 Teacher (特权编码器) 的梯度更新
            with torch.inference_mode():
                # 这一步前向传播是为了维持网络里的一些隐状态（如归一化层等），同时可以预热网络
                self.policy.act(obs_batch, hist_encoding=True, masks=masks_batch, hidden_states=hid_states_batch[0] if hid_states_batch else None)
                # 提取出特权隐向量和真实速度，也就是我们的监督目标 (Target/Label)
                priv_latent_batch = self.policy.infer_priv_latent(obs_batch)
                true_vel_batch = self.policy.get_true_vel(obs_batch)
                
            # 2. 获取 Student 的预测输出 (因为需要学习，所以带有梯度轨迹)
            hist_latent_batch, pred_vel_batch = self.policy.infer_hist_latent(obs_batch, return_vel=True)
            
            # 3. 计算双重 Loss: 隐向量蒸馏 + 速度显式监督
            hist_latent_loss = (priv_latent_batch.detach() - hist_latent_batch).norm(p=2, dim=1).mean()
            vel_loss = (true_vel_batch.detach() - pred_vel_batch).pow(2).mean()
            
            total_dagger_loss = hist_latent_loss + self.vel_loss_coef * vel_loss
            
            # 4. 仅仅对 History Encoder 的专属优化器执行反向传播和梯度下降
            self.hist_encoder_optimizer.zero_grad()
            total_dagger_loss.backward()
            
            # 多GPU梯度同步 (修复原版中缺失的多卡适配)
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
        """多GPU下同步模型参数广播"""
        model_params = [self.policy.state_dict()]
        if self.rnd:
            model_params.append(self.rnd.predictor.state_dict())
        torch.distributed.broadcast_object_list(model_params, src=0)
        self.policy.load_state_dict(model_params[0])
        if self.rnd:
            self.rnd.predictor.load_state_dict(model_params[1])

    def reduce_parameters(self):
        """多GPU下梯度的平均和同步 (All-Reduce)"""
        grads = [param.grad.view(-1) for param in self.policy.parameters() if param.grad is not None]
        if self.rnd:
            grads += [param.grad.view(-1) for param in self.rnd.parameters() if param.grad is not None]
        all_grads = torch.cat(grads)

        torch.distributed.all_reduce(all_grads, op=torch.distributed.ReduceOp.SUM)
        all_grads /= self.gpu_world_size

        all_params = self.policy.parameters()
        if self.rnd:
            all_params = chain(all_params, self.rnd.parameters())

        offset = 0
        for param in all_params:
            if param.grad is not None:
                numel = param.numel()
                param.grad.data.copy_(all_grads[offset : offset + numel].view_as(param.grad.data))
                offset += numel

    def reduce_history_parameters(self):
        """多GPU下专门针对历史编码器(History Encoder)的梯度同步"""
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
