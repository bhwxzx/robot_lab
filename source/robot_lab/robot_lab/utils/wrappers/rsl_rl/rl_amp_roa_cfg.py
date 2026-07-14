# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING
from typing import Literal

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlPpoAlgorithmCfg, RslRlPpoActorCriticCfg, RslRlOnPolicyRunnerCfg, RslRlSymmetryCfg

#############################
# AMP-ROA Policy Config     #
#############################

@configclass
class RslRlActorCriticAmpRoaCfg(RslRlPpoActorCriticCfg):
    """
    AMP-ROA 策略网络配置。
    本质上使用 ROA 的架构（特权编码器 + 历史编码器）。
    """
    
    class_name: str = "ActorCriticROA"
    """指向 ActorCriticROA 类。"""

    priv_encoder_dims: list[int] = [64, 20]
    """特权编码器的隐藏层和输出维度 [hidden_dim, latent_dim]。"""

    vel_offset: int | None = None
    """在 Critic 观测组中真实速度特征的起始索引。默认 None 表示自动推断 (等于本体观测维度)。"""


#############################
# AMP-ROA Algorithm Config  #
#############################

@configclass
class RslRlAlgorithmAmpRoaCfg(RslRlPpoAlgorithmCfg):
    """
    AMP-ROA PPO 算法配置。
    包含 PPO 基础参数以及 ROA 特有的算法参数（如 DAgger 和特权正则化）。
    (AMP 的算法参数通常由 Runner 处理并传递给算法构造函数)
    """

    class_name: str = "AMPROAPPO"
    """指向 AMP_ROA_PPO 算法类。"""

    # --- ROA 参数 ---
    priv_reg_coef_schedule: list[float] = [0, 0.1, 1000, 2000]
    """特权正则化损失的权重调度。"""

    priv_reg_coef_schedule_resume: list[float] = [0.0, 0.1, 0, 1]
    """恢复训练时的特权正则化损失权重调度。"""

    dagger_update_freq: int = 20
    """历史编码器与特权编码器的 DAgger 蒸馏发生频率 (表示每隔几次 PPO 迭代使用一次历史编码器)。"""

    vel_loss_coef: float = 1.0
    """显式速度估计的监督损失权重系数。"""

    symmetry_cfg: RslRlSymmetryCfg | None = None
    """Configuration for symmetry mechanism (e.g. data augmentation, mirror loss)."""


#############################
# AMP-ROA Runner Config     #
#############################

@configclass
class RslRlOnPolicyRunnerAmpRoaCfg(RslRlOnPolicyRunnerCfg):
    """
    AMP-ROA 运行器配置。
    负责聚合所有参数，特别是 AMP 的数据集加载、判别器超参以及 ROA 迭代逻辑。
    """

    class_name: str = "OnPolicyRunnerAmpROA"
    """指向 OnPolicyRunnerAmpROA 类。"""

    # --- 嵌套配置 ---
    policy: RslRlActorCriticAmpRoaCfg = RslRlActorCriticAmpRoaCfg()
    """使用 AMP-ROA 策略配置。"""

    algorithm: RslRlAlgorithmAmpRoaCfg = RslRlAlgorithmAmpRoaCfg()
    """使用 AMP-ROA 算法配置。"""

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

    amp_replay_buffer_size: int = 100000
    """判别器训练使用的回放池大小。"""

    disc_learning_rate: float = 1e-4
    """判别器的独立学习率。"""

    # --- 策略噪声约束 (AMP 常用设置) ---
    min_normalized_std: list[float] = [0.05] * 20 
    """强制执行的最小动作标准差。注意：List 长度必须等于机器人的动作维度"""
