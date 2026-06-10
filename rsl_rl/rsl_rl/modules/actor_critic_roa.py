# actor_critic_roa.py
from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal

from rsl_rl.networks import MLP, EmpiricalNormalization


class StateHistoryEncoder(nn.Module):
    """
    历史观测编码器 (Student Network):
    用于在没有特权信息（如摩擦力、精确地形）时，通过机器人本体的一段历史观测数据（时序数据）推断出当前环境的隐藏特征。
    通常采用一维卷积网络 (1D-CNN) 处理时序特征。
    """
    def __init__(self, activation_fn, input_size, tsteps, output_size):
        super(StateHistoryEncoder, self).__init__()
        self.activation_fn = activation_fn
        self.tsteps = tsteps # 历史序列的步长，如过去 50 帧
        channel_size = 10

        # 对输入的每一帧历史数据进行初步的线性特征提取
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 3 * channel_size), self.activation_fn,
        )

        # 针对不同长度的历史记录设计不同的一维卷积结构
        if tsteps == 50:
            self.conv_layers = nn.Sequential(
                nn.Conv1d(in_channels=3 * channel_size, out_channels=2 * channel_size, kernel_size=8, stride=4), self.activation_fn,
                nn.Conv1d(in_channels=2 * channel_size, out_channels=channel_size, kernel_size=5, stride=1), self.activation_fn,
                nn.Conv1d(in_channels=channel_size, out_channels=channel_size, kernel_size=5, stride=1), self.activation_fn, 
                nn.Flatten())
        elif tsteps == 10:
            self.conv_layers = nn.Sequential(
                nn.Conv1d(in_channels=3 * channel_size, out_channels=2 * channel_size, kernel_size=4, stride=2), self.activation_fn,
                nn.Conv1d(in_channels=2 * channel_size, out_channels=channel_size, kernel_size=2, stride=1), self.activation_fn,
                nn.Flatten())
        elif tsteps == 20:
            self.conv_layers = nn.Sequential(
                nn.Conv1d(in_channels=3 * channel_size, out_channels=2 * channel_size, kernel_size=6, stride=2), self.activation_fn,
                nn.Conv1d(in_channels=2 * channel_size, out_channels=channel_size, kernel_size=4, stride=2), self.activation_fn,
                nn.Flatten())
        else:
            raise ValueError("tsteps 必须是 10, 20 或 50 (支持的序列长度)")

        # 最终输出历史隐向量 (hist latent vector)
        self.linear_output = nn.Sequential(
            nn.Linear(channel_size * 3, output_size), self.activation_fn
        )
        # 测速头：新增输出线速度 (explicit velocity estimation)
        self.vel_output = nn.Sequential(
            nn.Linear(channel_size * 3, 3)
        )

    def forward(self, obs_flat):
        batch_size = obs_flat.shape[0]
        T = self.tsteps
        # 先把 [Batch, T * Prop] 拆开进行独立的特征投影
        projection = self.encoder(obs_flat.reshape([batch_size * T, -1]))
        # 把数据形状转换为 Conv1d 需要的格式 [Batch, Channels, T]，然后送入卷积网络
        output = self.conv_layers(projection.reshape([batch_size, T, -1]).permute((0, 2, 1)))
        hist_latent = self.linear_output(output)
        code_vel = self.vel_output(output)
        return hist_latent, code_vel


class ActorCriticROA(nn.Module):
    """
    带有 ROA (Regularized Online Adaptation) 机制的 Actor-Critic 网络。
    针对 IsaacLab Manager-Based 环境做了深度适配，支持 3D 张量的直接解析。
    """
    is_recurrent = False

    def __init__(
        self,
        obs,
        obs_groups,
        num_actions,
        priv_encoder_dims=[64, 20],
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        activation="elu",
        init_noise_std=1.0,
        noise_std_type: str = "scalar",
        vel_offset=None,
        **kwargs,
    ):
        if kwargs:
            print(
                "ActorCriticROA.__init__ 收到未知参数，将被忽略: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()

        self.obs_groups = obs_groups

        # ====== IsaacLab Manager-Based 观测组解析 ======
        # 推荐配置 3 个组，但也兼容只配 2 个组：
        # - "policy": 3D 张量 [Batch, History, Dim] (必填)
        # - "privileged": 纯物理特权参数 (如摩擦力、质量)，送入特权编码器 (选填，若无则退化为尝试从 critic 提取)
        # - "critic": Critic 网络的输入，推荐包含 [无噪声本体观测 + 物理特权参数] (必填)
        self.prop_groups = kwargs.get("prop_groups", ["policy"])
        self.priv_groups = kwargs.get("priv_groups", ["privileged"])
        
        policy_tensor = self.get_group_obs(obs, self.prop_groups)
        
        # 灵活推断 priv_groups: 如果用户没有配 "privileged" 组，回退到使用 "critic" 组
        if self.priv_groups[0] not in obs:
            self.priv_groups = ["critic"]
        critic_tensor = self.get_group_obs(obs, self.priv_groups)

        # 自动推断维度
        if len(policy_tensor.shape) == 3:
            # 输入格式为 [Batch, History, Dim]
            self.history_len = policy_tensor.shape[1]
            self.num_prop = policy_tensor.shape[2]
            self.num_hist_obs_total = self.history_len * self.num_prop
        else:
            # 回退到普通平铺结构或无历史的情况
            self.history_len = 0
            self.num_prop = policy_tensor.shape[-1]
            self.num_hist_obs_total = 0

        self.num_priv_obs = critic_tensor.shape[-1]
        
        # 读取速度索引配置。如果不传，默认等于本体观测的维度 (排在本体后面)
        self.vel_offset = vel_offset if vel_offset is not None else self.num_prop
        
        critic_backbone_tensor = self.get_group_obs(obs, obs_groups.get("critic", ["critic"]))
        num_critic_obs = critic_backbone_tensor.shape[-1]

        # 初始化激活函数
        if activation == "elu":
            act_fn = nn.ELU()
        elif activation == "relu":
            act_fn = nn.ReLU()
        elif activation == "lrelu":
            act_fn = nn.LeakyReLU()
        else:
            act_fn = nn.ELU()

        # ====== 1. 特权编码器 (Privileged Encoder - Teacher) ======
        # 将 critic 组里包含的上帝视角信息压缩为 Latent
        if len(priv_encoder_dims) > 0 and self.num_priv_obs > 0:
            priv_layers = []
            priv_layers.append(nn.Linear(self.num_priv_obs, priv_encoder_dims[0]))
            priv_layers.append(act_fn)
            for i in range(len(priv_encoder_dims) - 1):
                priv_layers.append(nn.Linear(priv_encoder_dims[i], priv_encoder_dims[i + 1]))
                priv_layers.append(act_fn)
            self.priv_encoder = nn.Sequential(*priv_layers)
            priv_out_dim = priv_encoder_dims[-1]
        else:
            self.priv_encoder = nn.Identity()
            priv_out_dim = self.num_priv_obs

        # ====== 2. 历史编码器 (History Encoder - Student) ======
        # 处理 3D 张量展平后的 history 数据
        if self.history_len > 0:
            self.history_encoder = StateHistoryEncoder(act_fn, self.num_prop, self.history_len, priv_out_dim)
        else:
            self.history_encoder = None

        # ====== 3. 策略网络主干 (Actor Backbone) ======
        # Actor 输入: 单帧本体观测 (Current Obs) + 线速度 (Vel) + Latent Code
        self.actor = MLP(self.num_prop + 3 + priv_out_dim, num_actions, actor_hidden_dims, activation)
        self.actor_obs_normalization = actor_obs_normalization
        if actor_obs_normalization:
            self.actor_obs_normalizer = EmpiricalNormalization(self.num_prop)
        else:
            self.actor_obs_normalizer = torch.nn.Identity()

        # ====== 4. 评价网络 (Critic) ======
        self.critic = MLP(num_critic_obs, 1, critic_hidden_dims, activation)
        self.critic_obs_normalization = critic_obs_normalization
        if critic_obs_normalization:
            self.critic_obs_normalizer = EmpiricalNormalization(num_critic_obs)
        else:
            self.critic_obs_normalizer = torch.nn.Identity()

        # 动作噪声分布参数
        self.noise_std_type = noise_std_type
        if self.noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif self.noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        else:
            raise ValueError(f"未知的标准差类型: {self.noise_std_type}")

        self.distribution = None
        Normal.set_default_validate_args(False)

    def reset(self, dones=None):
        pass

    def get_group_obs(self, obs, groups):
        """辅助函数：根据提供的组名将观测字段提取并拼接成张量"""
        return torch.cat([obs[g] for g in groups if g in obs], dim=-1)

    def _process_policy_obs(self, obs):
        """核心处理逻辑：从 policy 组智能分离出当前帧 (Current) 和历史展平帧 (History)"""
        policy_obs = self.get_group_obs(obs, self.prop_groups)
        if len(policy_obs.shape) == 3:
            # IsaacLab 格式 [Batch, History, Dim]
            hist_flat = policy_obs.flatten(1, 2)
            current_obs = policy_obs[:, -1, :]  # 取最后一帧作为当前本体感受
        else:
            # 兼容平铺格式
            current_obs = policy_obs
            hist_flat = policy_obs
        return current_obs, hist_flat

    def infer_priv_latent(self, obs):
        """推理：使用特权编码器生成隐向量"""
        priv_obs = self.get_group_obs(obs, self.priv_groups)
        return self.priv_encoder(priv_obs)

    def infer_hist_latent(self, obs, return_vel=False):
        """推理：使用历史编码器生成隐向量"""
        _, hist_flat = self._process_policy_obs(obs)
        hist_latent, code_vel = self.history_encoder(hist_flat)
        if return_vel:
            return hist_latent, code_vel
        return hist_latent
        
    def get_true_vel(self, obs):
        """获取批评家视野中的真实线速度"""
        critic_obs = self.get_group_obs(obs, ["critic"])
        # 根据配置的偏移量截取 3 维速度
        return critic_obs[:, self.vel_offset : self.vel_offset + 3]

    def update_distribution(self, obs, hist_encoding=False):
        """前向传递计算动作分布"""
        # 1. 解析观测并归一化本体感受
        current_obs, _ = self._process_policy_obs(obs)
        current_obs = self.actor_obs_normalizer(current_obs)

        # 2. 路由：选择对应的特征提取器获取 Latent 和 Vel
        if hist_encoding and self.history_encoder is not None:
            latent, vel = self.infer_hist_latent(obs, return_vel=True)
        else:
            latent = self.infer_priv_latent(obs)
            vel = self.get_true_vel(obs)

        # 3. 拼接传入主网络
        actor_input = torch.cat([current_obs, vel, latent], dim=-1)
        mean = self.actor(actor_input)

        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std).expand_as(mean)
            
        self.distribution = Normal(mean, std)

    def act(self, obs, hist_encoding=False, **kwargs):
        """采样获取动作"""
        self.update_distribution(obs, hist_encoding)
        return self.distribution.sample()

    def act_inference(self, obs, hist_encoding=False):
        """确定性推理获取动作"""
        current_obs, _ = self._process_policy_obs(obs)
        current_obs = self.actor_obs_normalizer(current_obs)
        
        if hist_encoding and self.history_encoder is not None:
            latent = self.infer_hist_latent(obs)
        else:
            latent = self.infer_priv_latent(obs)
            
        actor_input = torch.cat([current_obs, latent], dim=-1)
        return self.actor(actor_input)

    def evaluate(self, obs, **kwargs):
        """Critic评估"""
        critic_obs = self.get_group_obs(obs, self.obs_groups["critic"])
        critic_obs = self.critic_obs_normalizer(critic_obs)
        return self.critic(critic_obs)

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_normalization(self, obs):
        if self.actor_obs_normalization:
            current_obs, _ = self._process_policy_obs(obs)
            self.actor_obs_normalizer.update(current_obs)
        if self.critic_obs_normalization:
            critic_obs = self.get_group_obs(obs, self.obs_groups["critic"])
            self.critic_obs_normalizer.update(critic_obs)

    def load_state_dict(self, state_dict, strict=True):
        super().load_state_dict(state_dict, strict=strict)
        return True
