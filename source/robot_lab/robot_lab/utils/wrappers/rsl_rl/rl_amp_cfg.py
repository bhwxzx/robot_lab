# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING
from typing import Literal

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlPpoAlgorithmCfg, RslRlPpoActorCriticCfg, RslRlOnPolicyRunnerCfg

@configclass
class RslRlAlgorithmAmpCfg(RslRlPpoAlgorithmCfg):
    """Configuration for the AMPPPO algorithm."""

    class_name: str = "AMPPPO"
    """The algorithm class name."""

@configclass
class RslRlOnPolicyRunnerAmpCfg(RslRlOnPolicyRunnerCfg):
    """Configuration for the AMP On-Policy Runner."""

    class_name: str = "OnPolicyRunnerAmp"
    """The runner class name. Make sure this matches your class name."""

    # --- AMP 数据集相关 ---
    amp_motion_files: list[str] = MISSING
    """动作数据文件路径列表。"""

    amp_num_preload_transitions: int = 200000
    """预加载到内存中的转换对数量。"""

    # --- AMP 判别器与奖励相关 ---
    amp_reward_coef: float = 0.3
    """AMP 风格奖励的权重系数。"""

    amp_task_reward_lerp: float = 0.7
    """任务奖励与风格奖励的混合比例（输入为任务奖励权重）。"""

    amp_discr_hidden_dims: list[int] = [1024, 512, 256]
    """判别器 MLP 的隐藏层维度。"""

    amp_replay_buffer_size: int = 100000
    """判别器训练使用的回放池大小（存储机器人的历史动作）。"""

    # --- 策略噪声约束 ---
    min_normalized_std: list[float] = [0.05]*20
    """强制执行的最小动作标准差（List 长度需等于动作维度）。"""

    # --- 覆盖算法配置的类型限制 ---
    algorithm: RslRlAlgorithmAmpCfg = RslRlAlgorithmAmpCfg()