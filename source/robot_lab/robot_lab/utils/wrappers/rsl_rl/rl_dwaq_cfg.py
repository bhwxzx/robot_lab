# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING
from typing import Literal

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlPpoAlgorithmCfg, RslRlPpoActorCriticCfg, RslRlOnPolicyRunnerCfg

#########################
# DWAQ Policy Config    #
#########################

@configclass
class RslRlActorCriticDwaqCfg(RslRlPpoActorCriticCfg):
    """DWAQ 策略网络配置。"""
    
    class_name: str = "ActorCriticDwaq"
    """指向你定义的 ActorCriticDwaq 类"""

    vae_hidden_dims: list[int] = [128, 64]
    """VAE 编码器和解码器的隐藏层维度"""

    latent_dim: int = 16
    """环境隐含状态 z_t 的维度"""

    velocity_dim: int = 3
    """预测速度 v_t 的维度 (线速度 xyz)"""

    history_encoding: str = "term-first"
    """用于训练的历史观测编码格式，如果是term-first，内部会重组为time-first再用于训练"""

#########################
# DWAQ Algorithm Config #
#########################

@configclass
class RslRlAlgorithmDwaqCfg(RslRlPpoAlgorithmCfg):
    """DWAQ PPO 算法配置。"""

    class_name: str = "DWAQPPO"
    """指向你定义的 DWAQPPO 类"""

    vae_beta: float = 1.0
    """beta-VAE 中 KL 散度的权重系数"""

    obs_dim: int = 41
    """本体感受观察值的维度 (用于从特权观测中提取速度标签)"""

#########################
# DWAQ Runner Config    #
#########################

@configclass
class RslRlOnPolicyRunnerDwaqCfg(RslRlOnPolicyRunnerCfg):
    """DWAQ 运行器配置。"""

    class_name: str = "OnPolicyRunnerDwaq"
    """指向你定义的 OnPolicyRunnerDwaq 类"""

    policy: RslRlActorCriticDwaqCfg = RslRlActorCriticDwaqCfg()
    """覆盖基类的 policy 为 DWAQ 特有配置"""

    algorithm: RslRlAlgorithmDwaqCfg = RslRlAlgorithmDwaqCfg()
    """覆盖基类的 algorithm 为 DWAQ 特有配置"""

