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
    priv_reg_coef_schedule_resume: list[float] = [0, 0.1, 0, 1]
    """恢复训练时，特权正则化损失的权重调度。"""

    dagger_update_freq: int = 20
    """历史编码器与特权编码器的 DAgger 蒸馏发生频率 (表示每隔几次 PPO 迭代使用一次历史编码器)。"""

    vel_loss_coef: float = 1.0
    """显式速度估计的监督损失权重系数。"""


#############################
# ROA Runner Config         #
#############################

@configclass
class RslRlOnPolicyRunnerRoaCfg(RslRlOnPolicyRunnerCfg):
    """
    ROA 运行器配置。
    """
    class_name: str = "OnPolicyRunnerROA"

    # --- 嵌套配置 ---
    policy: RslRlActorCriticRoaCfg = RslRlActorCriticRoaCfg()
    algorithm: RslRlAlgorithmRoaCfg = RslRlAlgorithmRoaCfg()

