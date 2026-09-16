# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING
from typing import Literal

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlPpoAlgorithmCfg, RslRlPpoActorCriticCfg, RslRlOnPolicyRunnerCfg

#############################
# ROA Policy Config         #
#############################

@configclass
class RslRlActorCriticRoaCfg(RslRlPpoActorCriticCfg):
    """
    ROA 策略网络配置。
    使用 ROA 的架构（特权编码器 + 历史编码器）。
    """
    class_name: str = "ActorCriticROA"
    use_velocity_estimation: bool = True
    """False 移除显式速度头、actor 速度输入及速度监督；critic 与 latent 保留。"""
    priv_encoder_dims: list[int] = [64, 20]
    """特权编码器的隐藏层和输出维度 [hidden_dim, latent_dim]。"""
    
    vel_offset: int | None = None
    """在 Critic 观测组中真实速度特征的起始索引。默认 None 表示自动推断 (等于本体观测维度)。"""


#############################
# ROA Algorithm Config      #
#############################

@configclass
class RslRlAlgorithmRoaCfg(RslRlPpoAlgorithmCfg):
    """
    ROA PPO 算法配置。
    包含 PPO 基础参数以及 ROA 特有的算法参数（如 DAgger 和特权正则化）。
    """
    class_name: str = "ROAPPO"

    # --- ROA 参数 ---
    priv_reg_coef_schedule: list[float] = [0, 0.1, 1000, 2000]
    """特权正则化损失的权重调度。"""
    priv_reg_coef_schedule_resume: list[float] | None = None
    """可选恢复调度覆盖；None 延续 checkpoint 调度，显式覆盖使用恢复后的全局计数。"""

    dagger_update_freq: int = 20
    """历史编码器与特权编码器的 DAgger 蒸馏发生频率 (表示每隔几次 PPO 迭代使用一次历史编码器)。"""

    vel_loss_coef: float = 1.0
    """显式速度估计的监督损失权重系数。"""

    estimated_velocity_schedule: list[float] | None = None
    """PPO 使用估计速度的概率：[初值, 终值, 起始迭代, 渐变迭代数]。

    None 保持原有真实速度输入；学生控制仍使用估计速度。
    关闭 use_velocity_estimation 时课程也关闭，不改变 DAgger 频率。
    """
    estimated_velocity_schedule_resume: list[float] | None = None
    """None 延续 checkpoint 的课程；列表显式覆盖，迭代计数不重置。"""


#############################
# ROA Runner Config         #
#############################

@configclass
class RslRlOnPolicyRunnerRoaCfg(RslRlOnPolicyRunnerCfg):
    """
    ROA 运行器配置。
    """
    class_name: str = "OnPolicyRunnerROA"

    velocity_diagnostics: dict | None = {
        "zero_command_epsilon": 1e-6,
        "low_command_xy": 0.1,
        "low_command_yaw": 0.1,
        "near_stationary_xy": 0.02,
        "near_stationary_yaw": 0.05,
        "chunk_size": 4096,
    }
    """分组测速诊断；None 关闭。阈值仅用于统计，不改变奖励或指令。"""

    # --- 嵌套配置 ---
    policy: RslRlActorCriticRoaCfg = RslRlActorCriticRoaCfg()
    algorithm: RslRlAlgorithmRoaCfg = RslRlAlgorithmRoaCfg()
