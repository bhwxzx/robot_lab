# Copyright (c) 2024-2025
# SPDX-License-Identifier: Apache-2.0

"""Functions to specify the symmetry in the observation and action space for LW_Leg."""

from __future__ import annotations

import torch
from functools import lru_cache
from tensordict import TensorDict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv

__all__ = ["compute_symmetric_states", "compute_roa_symmetric_states", "compute_flat_amp_symmetric_states"]


@lru_cache(maxsize=4)
def _gain_mirror_order(joint_names: tuple[str, ...]) -> list[int]:
    """Gain observations follow articulation order, independently of the action order."""
    order = []
    for name in joint_names:
        side, joint = name.split("_", 1)
        other_side = {"left": "right", "right": "left"}[side]
        order.append(joint_names.index(f"{other_side}_{joint}"))
    return order


def _mirror_roa_privileged(obs: torch.Tensor, gain_order: list[int]) -> torch.Tensor:
    mirrored = obs.clone()
    # Mass, left/right foot material triples, stiffness, damping, base COM xyz.
    mirrored[..., 1:4] = obs[..., 4:7]
    mirrored[..., 4:7] = obs[..., 1:4]
    mirrored[..., 7:17] = obs[..., 7:17][..., gain_order]
    mirrored[..., 17:27] = obs[..., 17:27][..., gain_order]
    mirrored[..., 28] = -obs[..., 28]
    return mirrored


@torch.no_grad()
def compute_roa_symmetric_states(env, obs: TensorDict | None = None, actions: torch.Tensor | None = None):
    """Append a sagittal reflection of each Flat LW_Leg ROA sample."""
    obs_aug = None
    if obs is not None:
        policy, critic, privileged = obs["policy"], obs["critic"], obs["privileged"]
        if (policy.shape[-1], critic.shape[-1], privileged.shape[-1]) != (41, 74, 30):
            raise ValueError("Flat LW_Leg ROA symmetry expects 41 policy, 74 critic and 30 privileged features")
        unknown = set(obs.keys()) - {"policy", "critic", "privileged", "_roa_actor_velocity", "_roa_estimated_velocity_mask"}
        if unknown:
            raise ValueError(f"Undefined Flat LW_Leg ROA symmetry for observation groups {unknown}")
        gain_order = _gain_mirror_order(tuple(env.unwrapped.scene["robot"].joint_names))
        batch = obs.batch_size[0]
        obs_aug = obs.repeat(2)
        for key, original in (("policy", policy), ("critic", critic)):
            mirrored = _transform_policy_obs_left_right(env.unwrapped, original[..., :41])
            # Exchanging the stance leg advances the sin/cos clock by half a stride.
            mirrored[..., 39:41].neg_()
            obs_aug[key][batch:, ..., :41] = mirrored
        obs_aug["critic"][batch:, 42].neg_()  # body-frame lateral linear velocity
        obs_aug["critic"][batch:, 44:] = _mirror_roa_privileged(critic[..., 44:], gain_order)
        obs_aug["privileged"][batch:] = _mirror_roa_privileged(privileged, gain_order)
        if "_roa_actor_velocity" in obs:
            # Preserve the rollout's estimated/true selection for each reflected sample.
            obs_aug["_roa_actor_velocity"][batch:, 1].neg_()

    actions_aug = None
    if actions is not None:
        if actions.shape[-1] != 10:
            raise ValueError("Flat LW_Leg ROA symmetry expects 10 actions")
        actions_aug = torch.cat((actions, _switch_lw_joints_left_right(actions)), dim=0)
    return obs_aug, actions_aug


compute_roa_symmetric_states.supports_roa_velocity_curriculum = True


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


_LW_JOINT_MIRROR = ((1, 0, 3, 2, 5, 4, 7, 6, 9, 8), (-1,) * 10)
# Each term lists its source columns and signs for one frame.
_FLAT_AMP_OBS_MIRROR = (
    ((0, 1, 2), (-1, 1, -1)),  # angular velocity
    ((0, 1, 2), (1, -1, 1)),   # projected gravity
    ((0, 1, 2), (1, -1, -1)),  # velocity command
    _LW_JOINT_MIRROR,           # joint position
    _LW_JOINT_MIRROR,           # joint velocity
    _LW_JOINT_MIRROR,           # last action
    ((0, 1), (-1, -1)),         # half-stride shift of the sine/cosine clock
)
_FLAT_AMP_FRAME_WIDTH = sum(len(order) for order, _ in _FLAT_AMP_OBS_MIRROR)


@lru_cache(maxsize=32)
def _flat_amp_mirror_map(terms, history: int, device: torch.device, dtype: torch.dtype):
    columns, signs = [], []
    # Isaac Lab stores the complete history of each term before the next term.
    for order, term_signs in terms:
        for _ in range(history):
            offset = len(columns)
            columns.extend(offset + index for index in order)
            signs.extend(term_signs)
    return torch.tensor(columns, device=device, dtype=torch.long), torch.tensor(signs, device=device, dtype=dtype)


def _append_flat_amp_mirror(values: torch.Tensor, terms, history: int) -> torch.Tensor:
    columns, signs = _flat_amp_mirror_map(terms, history, values.device, values.dtype)
    mirrored = values.index_select(-1, columns).mul_(signs)
    return torch.cat((values, mirrored), dim=0)


@torch.no_grad()
def compute_flat_amp_symmetric_states(env, obs: TensorDict | None = None, actions: torch.Tensor | None = None):
    """Mirror the flat AMP actor and critic in Isaac Lab's flattened, term-major history layout."""
    obs_aug = None
    if obs is not None:
        groups = {}
        for key in ("policy", "critic"):
            flat = obs[key]
            if flat.ndim != 2 or flat.shape[1] == 0 or flat.shape[1] % _FLAT_AMP_FRAME_WIDTH:
                raise ValueError(f"Unexpected LW flat AMP {key} layout: {flat.shape}")
            history = flat.shape[1] // _FLAT_AMP_FRAME_WIDTH
            groups[key] = _append_flat_amp_mirror(flat, _FLAT_AMP_OBS_MIRROR, history)
        obs_aug = TensorDict(groups, batch_size=[2 * obs.batch_size[0]], device=obs.device, names=obs.names)

    actions_aug = None
    if actions is not None:
        actions_aug = _append_flat_amp_mirror(actions, (_LW_JOINT_MIRROR,), 1)
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
    
    # Task-specific symmetry functions handle the gait clock (39:41).

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
