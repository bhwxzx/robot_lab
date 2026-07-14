# Copyright (c) 2024-2025
# SPDX-License-Identifier: Apache-2.0

"""Functions to specify the symmetry in the observation and action space for LW_Leg."""

from __future__ import annotations

import torch
from tensordict import TensorDict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv

__all__ = ["compute_symmetric_states"]


@torch.no_grad()
def compute_symmetric_states(
    env: ManagerBasedRLEnv,
    obs: TensorDict | None = None,
    actions: torch.Tensor | None = None,
):
    """Augments the given observations and actions by applying left-right symmetry transformations."""
    # observations
    if obs is not None:
        batch_size = obs.batch_size[0]
        # since we only have left-right symmetry, we augment batch size by 2
        obs_aug = obs.repeat(2)

        # policy observation group
        # -- original
        obs_aug["policy"][:batch_size] = obs["policy"][:]
        # -- left-right
        obs_aug["policy"][batch_size : 2 * batch_size] = _transform_policy_obs_left_right(env.unwrapped, obs["policy"])
    else:
        obs_aug = None

    # actions
    if actions is not None:
        batch_size = actions.shape[0]
        # since we only have left-right symmetry, we augment batch size by 2
        actions_aug = torch.zeros(batch_size * 2, actions.shape[1], device=actions.device)
        # -- original
        actions_aug[:batch_size] = actions[:]
        # -- left-right
        actions_aug[batch_size : 2 * batch_size] = _transform_actions_left_right(actions)
    else:
        actions_aug = None

    return obs_aug, actions_aug


def _transform_policy_obs_left_right(env: ManagerBasedRLEnv, obs: torch.Tensor) -> torch.Tensor:
    """Apply a left-right symmetry transformation to the observation tensor."""
    obs = obs.clone()
    device = obs.device
    
    # ang vel (Roll, Pitch, Yaw -> -Roll, Pitch, -Yaw)
    obs[..., 0:3] = obs[..., 0:3] * torch.tensor([-1, 1, -1], device=device)
    # projected gravity (X, Y, Z -> X, -Y, Z)
    obs[..., 3:6] = obs[..., 3:6] * torch.tensor([1, -1, 1], device=device)
    # velocity command (vx, vy, wz -> vx, -vy, -wz)
    obs[..., 6:9] = obs[..., 6:9] * torch.tensor([1, -1, -1], device=device)
    
    # joint pos
    obs[..., 9:19] = _switch_lw_joints_left_right(obs[..., 9:19])
    # joint vel
    obs[..., 19:29] = _switch_lw_joints_left_right(obs[..., 19:29])
    # last actions
    obs[..., 29:39] = _switch_lw_joints_left_right(obs[..., 29:39])
    
    # gait_phase (39:41) is a global time-based clock, remains unchanged

    return obs


def _transform_actions_left_right(actions: torch.Tensor) -> torch.Tensor:
    """Applies a left-right symmetry transformation to the actions tensor."""
    actions = actions.clone()
    actions[...] = _switch_lw_joints_left_right(actions[...])
    return actions


def _switch_lw_joints_left_right(joint_data: torch.Tensor) -> torch.Tensor:
    """
    Applies a left-right symmetry transformation to the joint data tensor for LW.
    
    Based on rough_env_cfg.py (preserve_order=True), the order is:
    0: right_hip_joint
    1: left_hip_joint
    2: right_thigh_joint
    3: left_thigh_joint
    4: right_shank_joint
    5: left_shank_joint
    6: right_foot_joint
    7: left_foot_joint
    8: right_wheel_joint
    9: left_wheel_joint
    """
    joint_data_switched = torch.zeros_like(joint_data)
    
    # Swap Left (1,3,5,7,9) and Right (0,2,4,6,8)
    joint_data_switched[..., [0, 2, 4, 6, 8]] = joint_data[..., [1, 3, 5, 7, 9]]
    joint_data_switched[..., [1, 3, 5, 7, 9]] = joint_data[..., [0, 2, 4, 6, 8]]
    
    # Based on LW URDF:
    # 1. Hip axes are identical ([-1, 0, 0]), so +rot means outward for left but inward for right.
    #    To maintain mirrored symmetry, they must be negated upon swap.
    # 2. Thigh/Shank/Foot/Wheel axes are perfectly opposite (e.g., [0,1,0] vs [0,-1,0]).
    #    However, to achieve the same physical posture (e.g. both leaning forward),
    #    their values must have opposite signs (as seen in LW.py default pose: +0.43 vs -0.43).
    #    Therefore, they must ALSO be negated upon swap!
    joint_data_switched *= -1.0
    
    return joint_data_switched
