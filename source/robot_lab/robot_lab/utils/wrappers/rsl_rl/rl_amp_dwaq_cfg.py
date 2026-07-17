# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING
from typing import Literal

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlPpoAlgorithmCfg, RslRlPpoActorCriticCfg, RslRlOnPolicyRunnerCfg, RslRlSymmetryCfg

#############################
# AMP-DWAQ Policy Config    #
#############################

@configclass
class RslRlActorCriticAmpDwaqCfg(RslRlPpoActorCriticCfg):
    """
    AMP-DWAQ 策略网络配置。
    本质上使用 DWAQ 的架构（Actor-Critic + VAE Context Encoder）。
    """
    
    class_name: str = "ActorCriticDwaq"
    """指向 ActorCriticDwaq 类 (包含 VAE)。"""

    vae_hidden_dims: list[int] = [128, 64]
    """VAE 编码器和解码器的隐藏层维度。"""

    latent_dim: int = 16
    """环境隐含状态 z_t 的维度。"""

    velocity_dim: int = 3
    """预测速度 v_t 的维度 (线速度 xyz)。"""


#############################
# AMP-DWAQ Algorithm Config #
#############################

@configclass
class RslRlAlgorithmAmpDwaqCfg(RslRlPpoAlgorithmCfg):
    """
    AMP-DWAQ PPO 算法配置。
    包含 PPO 基础参数以及 DWAQ 特有的算法参数。
    (AMP 的算法参数通常由 Runner 处理并传递给算法构造函数，但也部分保留在此处以保持结构一致性)
    """

    class_name: str = "AMPDWAQPPO"
    """指向 AMPDWAQPPO 算法类。"""

    # --- DWAQ 参数 ---
    vae_beta: float = 1.0
    """beta-VAE 中 KL 散度的权重系数。"""

    obs_dim: int = 41
    """本体感受观察值的维度 (用于从特权观测中提取速度标签)。"""

    symmetry_cfg: RslRlSymmetryCfg | None = None
    """Configuration for symmetry mechanism (e.g. data augmentation, mirror loss)."""


#############################
# AMP-DWAQ Runner Config    #
#############################

@configclass
class RslRlOnPolicyRunnerAmpDwaqCfg(RslRlOnPolicyRunnerCfg):
    """
    AMP-DWAQ 运行器配置。
    负责聚合所有参数，特别是 AMP 的数据集加载和判别器超参。
    """

    class_name: str = "OnPolicyRunnerAmpDwaq"
    """指向 OnPolicyRunnerAmpDwaq 类。"""

    # --- 嵌套配置 ---
    policy: RslRlActorCriticAmpDwaqCfg = RslRlActorCriticAmpDwaqCfg()
    """使用 AMP-DWAQ 策略配置。"""

    algorithm: RslRlAlgorithmAmpDwaqCfg = RslRlAlgorithmAmpDwaqCfg()
    """使用 AMP-DWAQ 算法配置。"""

    # --- AMP 数据集相关 (必须配置) ---
    amp_motion_files: list[str] = MISSING
    """动作数据文件路径列表 (如 .npy 或 .motionlib 文件)。"""

    amp_num_preload_transitions: int = 200000
    """预加载到内存中的转换对数量。"""

    # --- AMP 判别器与奖励相关 ---
    amp_reward_coef: float = 0.3 
    """AMP 风格奖励的权重系数。"""

    amp_task_reward_lerp: float = 0.7 
    """任务奖励与风格奖励的混合比例（Lerp系数）。"""

    amp_discr_hidden_dims: list[int] = [1024, 512, 256]
    """判别器 MLP 的隐藏层维度。"""

    amp_discriminator_history_window: bool = True
    """判别器是否直接使用单个 AMP 历史窗口，而不是拼接相邻的两个窗口。"""

    amp_replay_buffer_size: int = 100000
    """判别器训练使用的回放池大小。"""

    disc_learning_rate: float = 1e-4
    """判别器的独立学习率。"""

    # --- 策略噪声约束 (AMP 常用设置) ---
    min_normalized_std: list[float] = [0.05] * 20 
    """强制执行的最小动作标准差。注意：List 长度必须等于机器人的动作维度"""
