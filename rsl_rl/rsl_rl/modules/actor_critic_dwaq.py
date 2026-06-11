# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal

from rsl_rl.networks import MLP, EmpiricalNormalization


class ActorCriticDwaq(nn.Module):
    """Actor-Critic with DWAQ (Deep Variational Autoencoder for Walking) context encoder.
    
    The context encoder (β-VAE) infers velocity and latent state from observation history.
    """
    is_recurrent = False

    def __init__(
        self,
        obs,
        obs_groups,
        num_actions,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        init_noise_std=1.0,
        noise_std_type: str = "scalar",
        # DreamWAQ 特有参数
        vae_hidden_dims=[128, 64],
        latent_dim=16,      # 隐含状态维度
        velocity_dim=3,     # 预测速度维度 (通常是 xyz)
        **kwargs,
    ):
        if kwargs:
            print(
                "ActorCritic.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()

        # get the observation dimensions
        self.obs_groups = obs_groups
        # for DWAQ
        self.latent_dim = latent_dim
        self.velocity_dim = velocity_dim
        self.total_latent_dim = latent_dim + velocity_dim

        # --- 1. 维度计算 (针对 3D 输入适配 flatten_history_dim=false) --- 
        # 假设 obs["policy"] 是 [Batch, History, Dim]
        policy_tensor = self.get_obs_from_group(obs, "policy")
        if len(policy_tensor.shape) == 3:
            self.history_len = policy_tensor.shape[1]
            self.obs_dim = policy_tensor.shape[2] # 单帧维度
            self.num_history_obs = self.history_len * self.obs_dim # 展平后的总维度
        else:
            raise ValueError(f"Error Obs Type! Input shape is: {policy_tensor.shape}. Should be [3]")

        # Actor 最终输入维度 = Latent(19) + Policy Obs
        actor_input_dim = self.total_latent_dim + self.obs_dim
        
        num_critic_obs = sum(obs[group].shape[-1] for group in obs_groups["critic"])

        # 2. 初始化 VAE 模块 (DreamWAQ 核心)
        # 编码器主干
        self.encoder_backbone = MLP(self.num_history_obs, vae_hidden_dims[-1], vae_hidden_dims[:-1], activation)
        # 预测头：均值和方差
        self.encode_mean_latent = nn.Linear(vae_hidden_dims[-1], latent_dim)
        self.encode_logvar_latent = nn.Linear(vae_hidden_dims[-1], latent_dim)
        self.encode_mean_vel = nn.Linear(vae_hidden_dims[-1], velocity_dim)
        self.encode_logvar_vel = nn.Linear(vae_hidden_dims[-1], velocity_dim)

        # 解码器 (用于训练时的重构)
        # 解码器的目标是根据 Latent 还原当前的单帧 Observation (obs_dim)
        # 这里的 obs_dim 通常是 policy 组的单帧维度
        self.decoder = MLP(self.total_latent_dim, self.obs_dim, vae_hidden_dims[::-1], activation)

        # actor
        self.actor = MLP(actor_input_dim, num_actions, actor_hidden_dims, activation)
        # actor observation normalization
        self.actor_obs_normalization = actor_obs_normalization
        if actor_obs_normalization:
            self.actor_obs_normalizer = EmpiricalNormalization(actor_input_dim)
        else:
            self.actor_obs_normalizer = torch.nn.Identity()
        print(f"Actor MLP: {self.actor}")

        # critic
        self.critic = MLP(num_critic_obs, 1, critic_hidden_dims, activation)
        # critic observation normalization
        self.critic_obs_normalization = critic_obs_normalization
        if critic_obs_normalization:
            self.critic_obs_normalizer = EmpiricalNormalization(num_critic_obs)
        else:
            self.critic_obs_normalizer = torch.nn.Identity()
        print(f"Critic MLP: {self.critic}")

        # Action noise
        self.noise_std_type = noise_std_type
        if self.noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif self.noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")

        # Action distribution (populated in update_distribution)
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args(False)

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    # ================= DreamWAQ 核心方法 =================
    
    def reparameterise(self, mean, logvar):
        logvar = torch.clamp(logvar, min=-10.0, max=10.0)
        std = torch.exp(logvar * 0.5)
        eps = torch.randn_like(std)
        return mean + eps * std
    
    def _process_history(self, obs_history):
        """
        处理历史观测。
        输入: [Batch, History, Dim] (3D 张量，Time-First)
        输出: 
            obs_flat: [Batch, History*Dim] (Time-First 排列，用于 MLP)
            current_obs: [Batch, Dim] (最后一帧，用于 Actor)
        """
        # 1. 直接展平前两个维度 -> [Batch, Time*Dim]
        obs_flat = obs_history.flatten(1, 2)
            
        # 2. 提取最后一帧
        current_obs = obs_history[:, -1, :]
        return obs_flat, current_obs

    def cenet_forward(self, obs_history):
        """上下文编码器前向传播"""
        # --- 形状切分与处理 ---
        obs_time_first, current_obs = self._process_history(obs_history)
        
        feat = self.encoder_backbone(obs_time_first)
        mu_l, logvar_l = self.encode_mean_latent(feat), self.encode_logvar_latent(feat)
        mu_v, logvar_v = self.encode_mean_vel(feat), self.encode_logvar_vel(feat)
        
        z_l = self.reparameterise(mu_l, logvar_l)
        z_v = self.reparameterise(mu_v, logvar_v)
        
        latent_code = torch.cat((z_v, z_l), dim=-1)
        reconstruction = self.decoder(latent_code)
        
        return latent_code, reconstruction, mu_v, logvar_v, mu_l, logvar_l, current_obs

    # ================= 修改后的动作方法 =================

    def act(self, obs, **kwargs):
         # 提取全量观测 (包含历史)  [Batch, 5, 45]
        policy_obs_raw = self.get_obs_from_group(obs, "policy")
        
        # 1. 运行 CENet 并自动获取对齐后的“当前帧”
        latent_code, _, _, _, _, _, current_obs = self.cenet_forward(policy_obs_raw)
        
        # 2. 拼接 [Latent, Current_Obs]
        combined_obs = torch.cat((latent_code, current_obs), dim=-1)
        
        # 3. 归一化并更新分布
        combined_obs = self.actor_obs_normalizer(combined_obs)
        self.update_distribution(combined_obs)
        return self.distribution.sample()

    def act_inference(self, obs):
        policy_obs_raw = self.get_obs_from_group(obs, "policy")
        
        # 1. 确定性重排与提取
        obs_time_first, current_obs = self._process_history(policy_obs_raw)
        
        # 【方案 B：严谨的数学实现】
        # 编码时强制取均值 (mu_v, mu_l)，彻底消除部署和 Play 时由 VAE 采样带来的随机震荡
        feat = self.encoder_backbone(obs_time_first)
        mu_v = self.encode_mean_vel(feat)
        mu_l = self.encode_mean_latent(feat)
        latent_code_det = torch.cat((mu_v, mu_l), dim=-1)
        
        # ====================================================================
        # 【方案 A：原版 DreamWaQ 的 "玄学 Hack" (已注释)】
        # 原版在推理时仍调用带高斯噪声采样的 cenet_forward。
        # 由于方案 A 训练出的网络已习惯在巨大噪声下工作，如果恢复方案 A，建议也恢复带噪推理：
        # latent_code_det, _, _, _, _, _, _ = self.cenet_forward(policy_obs_raw)
        # ====================================================================
        
        # 3. 拼接特征 [Latent Code(均值) + Current Obs(单帧)]
        combined_obs = torch.cat((latent_code_det, current_obs), dim=-1)
        
        # 4. 归一化处理
        combined_obs = self.actor_obs_normalizer(combined_obs)
        
        # 5. 直接调用 Actor 网络输出均值
        actions_mean = self.actor(combined_obs)
        
        return torch.nan_to_num(actions_mean, nan=0.0)

    def evaluate(self, obs, **kwargs):
        critic_obs = self.get_obs_from_group(obs, "critic")
        critic_obs = self.critic_obs_normalizer(critic_obs)
        return self.critic(critic_obs)
    
    # ================= 辅助工具 =================

    def get_obs_from_group(self, obs_dict, group_name):
        """根据配置的组名提取并拼接 Tensor"""
        obs_list = []
        for group in self.obs_groups[group_name]:
            obs_list.append(obs_dict[group])
        return torch.cat(obs_list, dim=-1)

    def update_distribution(self, obs):
        """添加了全方位安全检查的分布更新"""
        # 计算均值并进行清洗
        mean = self.actor(obs)
        # 将 NaN/Inf 替换为合法数值，防止梯度爆炸
        mean = torch.nan_to_num(mean, nan=0.0, posinf=1e6, neginf=-1e6)

        # 计算并截断标准差
        if self.noise_std_type == "scalar":
            std_param = self.std
        else: # log 类型
            std_param = torch.exp(self.log_std)
        # 确保标准差在一个合理区间内，1e-6 防止除以0，1e3 防止探索过大
        std_param = torch.clamp(std_param, min=1e-6, max=1e3)
        std = std_param.expand_as(mean)

        # 终极安全检查：如果还有非法值，强制重置
        if not torch.isfinite(mean).all() or not torch.isfinite(std).all():
            mean = torch.where(torch.isfinite(mean), mean, torch.zeros_like(mean))
            std = torch.where(torch.isfinite(std), std, torch.full_like(std, 1e-3))

        self.distribution = Normal(mean, std)

    def update_normalization(self, obs):
        # 1. 处理 Actor 侧归一化
        # 只有在 self.actor_obs_normalization 为 True 时才更新
        if self.actor_obs_normalization and hasattr(self.actor_obs_normalizer, 'update'):
            with torch.no_grad():
                policy_obs_raw = self.get_obs_from_group(obs, "policy")
                latent_code, _, _, _, _, _, current_obs = self.cenet_forward(policy_obs_raw)
                combined_obs = torch.cat((latent_code, current_obs), dim=-1)
                self.actor_obs_normalizer.update(combined_obs)
        
        # 2. 处理 Critic 侧归一化
        # 同样增加判断
        if self.critic_obs_normalization and hasattr(self.critic_obs_normalizer, 'update'):
            critic_obs = self.get_obs_from_group(obs, "critic")
            self.critic_obs_normalizer.update(critic_obs)

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def load_state_dict(self, state_dict, strict=True):
        """Load the parameters of the actor-critic model.

        Args:
            state_dict (dict): State dictionary of the model.
            strict (bool): Whether to strictly enforce that the keys in state_dict match the keys returned by this
                           module's state_dict() function.

        Returns:
            bool: Whether this training resumes a previous training. This flag is used by the `load()` function of
                  `OnPolicyRunner` to determine how to load further parameters (relevant for, e.g., distillation).
        """

        super().load_state_dict(state_dict, strict=strict)
        return True  # training resumes
