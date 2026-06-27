# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
from typing import TYPE_CHECKING, List
from torch import distributions

import re
import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import mdp
from isaaclab.managers import ManagerTermBase
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor, RayCaster
from isaaclab.utils.math import quat_apply_inverse, yaw_quat, euler_xyz_from_quat, quat_apply, wrap_to_pi

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import RewardTermCfg

def track_lin_vel_xy_exp(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    # compute the error
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - asset.data.root_lin_vel_b[:, :2]),
        dim=1,
    )
    reward = torch.exp(-lin_vel_error / std**2)
    # reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def track_ang_vel_z_exp(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    # compute the error
    ang_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_b[:, 2])
    reward = torch.exp(-ang_vel_error / std**2)
    # reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def track_lin_vel_xy_yaw_frame_exp(
    env, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) in the gravity aligned robot frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    vel_yaw = quat_apply_inverse(yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3])
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - vel_yaw[:, :2]), dim=1
    )
    reward = torch.exp(-lin_vel_error / std**2)
    # reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def track_ang_vel_z_world_exp(
    env, command_name: str, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) in world frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    ang_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_w[:, 2])
    reward = torch.exp(-ang_vel_error / std**2)
    # reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def joint_power(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Reward joint_power"""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute the reward
    reward = torch.sum(
        torch.abs(asset.data.joint_vel[:, asset_cfg.joint_ids] * asset.data.applied_torque[:, asset_cfg.joint_ids]),
        dim=1,
    )
    return reward


def joint_torque_limit_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    torque_limit: float,
) -> torch.Tensor:
    """Penalize joint torques when they exceed a specified limit."""
    asset: Articulation = env.scene[asset_cfg.name]
    torques = asset.data.applied_torque[:, asset_cfg.joint_ids]
    # Calculate excess torque beyond the limit
    excess_torque = torch.clamp(torch.abs(torques) - torque_limit, min=0.0)
    # Penalize the square of the excess torque
    reward = torch.sum(torch.square(excess_torque), dim=1)
    return reward


def stand_still(
    env: ManagerBasedRLEnv,
    command_name: str,
    command_threshold: float = 0.06,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize offsets from the default joint positions when the command is very small."""
    # Penalize motion when command is nearly zero.
    reward = mdp.joint_deviation_l1(env, asset_cfg)
    reward *= torch.norm(env.command_manager.get_command(command_name), dim=1) < command_threshold
    # reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward

def stop_motion(
    env, lin_threshold: float = 0.05, ang_threshold: float = 0.05, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """
    penalizing linear and angular motion when command velocities are near zero.
    """

    asset = env.scene[asset_cfg.name]
    base_lin_vel = asset.data.root_lin_vel_w[:, :2]
    base_ang_vel = asset.data.root_ang_vel_w[:, -1]

    commands = env.command_manager.get_command("base_velocity")

    lin_commands = commands[:, :2]
    ang_commands = commands[:, 2]

    reward_lin = torch.sum(
        torch.abs(base_lin_vel) * (torch.norm(lin_commands, dim=1, keepdim=True) < lin_threshold), dim=-1
    )

    reward_ang = torch.abs(base_ang_vel) * (torch.abs(ang_commands) < ang_threshold)

    total_reward = reward_lin + reward_ang
    return total_reward


def joint_pos_penalty(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    stand_still_scale: float,
    velocity_threshold: float,
    command_threshold: float,
) -> torch.Tensor:
    """Penalize joint position error from default on the articulation."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = torch.linalg.norm(env.command_manager.get_command(command_name), dim=1)
    body_vel = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1)
    running_reward = torch.linalg.norm(
        (asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]), dim=1
    )
    reward = torch.where(
        torch.logical_or(cmd > command_threshold, body_vel > velocity_threshold),
        running_reward,
        stand_still_scale * running_reward,
    )
    # reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def wheel_vel_penalty(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    command_name: str,
    velocity_threshold: float,
    command_threshold: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = torch.linalg.norm(env.command_manager.get_command(command_name), dim=1)
    body_vel = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1)
    joint_vel = torch.abs(asset.data.joint_vel[:, asset_cfg.joint_ids])
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    in_air = contact_sensor.compute_first_air(env.step_dt)[:, sensor_cfg.body_ids]
    running_reward = torch.sum(in_air * joint_vel, dim=1)
    standing_reward = torch.sum(joint_vel, dim=1)
    reward = torch.where(
        torch.logical_or(cmd > command_threshold, body_vel > velocity_threshold),
        running_reward,
        standing_reward,
    )
    return reward


class GaitReward(ManagerTermBase):
    """Gait enforcing reward term for quadrupeds.

    This reward penalizes contact timing differences between selected foot pairs defined in :attr:`synced_feet_pair_names`
    to bias the policy towards a desired gait, i.e trotting, bounding, or pacing. Note that this reward is only for
    quadrupedal gaits with two pairs of synchronized feet.
    """

    def __init__(self, cfg: RewTerm, env: ManagerBasedRLEnv):
        """Initialize the term.

        Args:
            cfg: The configuration of the reward.
            env: The RL environment instance.
        """
        super().__init__(cfg, env)
        self.std: float = cfg.params["std"]
        self.command_name: str = cfg.params["command_name"]
        self.max_err: float = cfg.params["max_err"]
        self.velocity_threshold: float = cfg.params["velocity_threshold"]
        self.command_threshold: float = cfg.params["command_threshold"]
        self.contact_sensor: ContactSensor = env.scene.sensors[cfg.params["sensor_cfg"].name]
        self.asset: Articulation = env.scene[cfg.params["asset_cfg"].name]
        # match foot body names with corresponding foot body ids
        synced_feet_pair_names = cfg.params["synced_feet_pair_names"]
        if (
            len(synced_feet_pair_names) != 2
            or len(synced_feet_pair_names[0]) != 2
            or len(synced_feet_pair_names[1]) != 2
        ):
            raise ValueError("This reward only supports gaits with two pairs of synchronized feet, like trotting.")
        synced_feet_pair_0 = self.contact_sensor.find_bodies(synced_feet_pair_names[0])[0]
        synced_feet_pair_1 = self.contact_sensor.find_bodies(synced_feet_pair_names[1])[0]
        self.synced_feet_pairs = [synced_feet_pair_0, synced_feet_pair_1]

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        std: float,
        command_name: str,
        max_err: float,
        velocity_threshold: float,
        command_threshold: float,
        synced_feet_pair_names,
        asset_cfg: SceneEntityCfg,
        sensor_cfg: SceneEntityCfg,
    ) -> torch.Tensor:
        """Compute the reward.

        This reward is defined as a multiplication between six terms where two of them enforce pair feet
        being in sync and the other four rewards if all the other remaining pairs are out of sync

        Args:
            env: The RL environment instance.
        Returns:
            The reward value.
        """
        # for synchronous feet, the contact (air) times of two feet should match
        sync_reward_0 = self._sync_reward_func(self.synced_feet_pairs[0][0], self.synced_feet_pairs[0][1])
        sync_reward_1 = self._sync_reward_func(self.synced_feet_pairs[1][0], self.synced_feet_pairs[1][1])
        sync_reward = sync_reward_0 * sync_reward_1
        # for asynchronous feet, the contact time of one foot should match the air time of the other one
        async_reward_0 = self._async_reward_func(self.synced_feet_pairs[0][0], self.synced_feet_pairs[1][0])
        async_reward_1 = self._async_reward_func(self.synced_feet_pairs[0][1], self.synced_feet_pairs[1][1])
        async_reward_2 = self._async_reward_func(self.synced_feet_pairs[0][0], self.synced_feet_pairs[1][1])
        async_reward_3 = self._async_reward_func(self.synced_feet_pairs[1][0], self.synced_feet_pairs[0][1])
        async_reward = async_reward_0 * async_reward_1 * async_reward_2 * async_reward_3
        # only enforce gait if cmd > 0
        cmd = torch.linalg.norm(env.command_manager.get_command(self.command_name), dim=1)
        body_vel = torch.linalg.norm(self.asset.data.root_com_lin_vel_b[:, :2], dim=1)
        reward = torch.where(
            torch.logical_or(cmd > self.command_threshold, body_vel > self.velocity_threshold),
            sync_reward * async_reward,
            0.0,
        )
        # reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
        return reward

    """
    Helper functions.
    """

    def _sync_reward_func(self, foot_0: int, foot_1: int) -> torch.Tensor:
        """Reward synchronization of two feet."""
        air_time = self.contact_sensor.data.current_air_time
        contact_time = self.contact_sensor.data.current_contact_time
        # penalize the difference between the most recent air time and contact time of synced feet pairs.
        se_air = torch.clip(torch.square(air_time[:, foot_0] - air_time[:, foot_1]), max=self.max_err**2)
        se_contact = torch.clip(torch.square(contact_time[:, foot_0] - contact_time[:, foot_1]), max=self.max_err**2)
        return torch.exp(-(se_air + se_contact) / self.std)

    def _async_reward_func(self, foot_0: int, foot_1: int) -> torch.Tensor:
        """Reward anti-synchronization of two feet."""
        air_time = self.contact_sensor.data.current_air_time
        contact_time = self.contact_sensor.data.current_contact_time
        # penalize the difference between opposing contact modes air time of feet 1 to contact time of feet 2
        # and contact time of feet 1 to air time of feet 2) of feet pairs that are not in sync with each other.
        se_act_0 = torch.clip(torch.square(air_time[:, foot_0] - contact_time[:, foot_1]), max=self.max_err**2)
        se_act_1 = torch.clip(torch.square(contact_time[:, foot_0] - air_time[:, foot_1]), max=self.max_err**2)
        return torch.exp(-(se_act_0 + se_act_1) / self.std)


def joint_mirror(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, mirror_joints: list[list[str]]) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    if not hasattr(env, "joint_mirror_joints_cache") or env.joint_mirror_joints_cache is None:
        # Cache joint positions for all pairs
        env.joint_mirror_joints_cache = [
            [asset.find_joints(joint_name) for joint_name in joint_pair] for joint_pair in mirror_joints
        ]
    reward = torch.zeros(env.num_envs, device=env.device)
    # Iterate over all joint pairs
    for joint_pair in env.joint_mirror_joints_cache:
        # Calculate the difference for each pair and add to the total reward
        diff = torch.sum(
            torch.square(asset.data.joint_pos[:, joint_pair[0][0]] - asset.data.joint_pos[:, joint_pair[1][0]]),
            dim=-1,
        )
        reward += diff
    reward *= 1 / len(mirror_joints) if len(mirror_joints) > 0 else 0
    # reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def action_mirror(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, mirror_joints: list[list[str]]) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    if not hasattr(env, "action_mirror_joints_cache") or env.action_mirror_joints_cache is None:
        # Cache joint positions for all pairs
        env.action_mirror_joints_cache = [
            [asset.find_joints(joint_name) for joint_name in joint_pair] for joint_pair in mirror_joints
        ]
    reward = torch.zeros(env.num_envs, device=env.device)
    # Iterate over all joint pairs
    for joint_pair in env.action_mirror_joints_cache:
        # Calculate the difference for each pair and add to the total reward
        diff = torch.sum(
            torch.square(
                torch.abs(env.action_manager.action[:, joint_pair[0][0]])
                - torch.abs(env.action_manager.action[:, joint_pair[1][0]])
            ),
            dim=-1,
        )
        reward += diff
    reward *= 1 / len(mirror_joints) if len(mirror_joints) > 0 else 0
    # reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def action_sync(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, joint_groups: list[list[str]]) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]

    # Cache joint indices if not already done
    if not hasattr(env, "action_sync_joint_cache") or env.action_sync_joint_cache is None:
        env.action_sync_joint_cache = [
            [asset.find_joints(joint_name) for joint_name in joint_group] for joint_group in joint_groups
        ]

    reward = torch.zeros(env.num_envs, device=env.device)
    # Iterate over each joint group
    for joint_group in env.action_sync_joint_cache:
        if len(joint_group) < 2:
            continue  # need at least 2 joints to compare

        # Get absolute actions for all joints in this group
        actions = torch.stack(
            [torch.abs(env.action_manager.action[:, joint[0]]) for joint in joint_group], dim=1
        )  # shape: (num_envs, num_joints_in_group)

        # Calculate mean action for each environment
        mean_actions = torch.mean(actions, dim=1, keepdim=True)

        # Calculate variance from mean for each joint
        variance = torch.mean(torch.square(actions - mean_actions), dim=1)

        # Add to reward (we want to minimize this variance)
        reward += variance.squeeze()
    reward *= 1 / len(joint_groups) if len(joint_groups) > 0 else 0
    # reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_air_time(
    env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, threshold: float
) -> torch.Tensor:
    """Reward long steps taken by the feet using L2-kernel.

    This function rewards the agent for taking steps that are longer than a threshold. This helps ensure
    that the robot lifts its feet off the ground and takes steps. The reward is computed as the sum of
    the time for which the feet are in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name), dim=1) > 0.1
    # reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_air_time_positive_biped(env, command_name: str, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward long steps taken by the feet for bipeds.

    This function rewards the agent for taking steps up to a specified threshold and also keep one foot at
    a time in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    reward = torch.clamp(reward, max=threshold)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name), dim=1) > 0.1
    # reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_air_time_variance_penalty(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize variance in the amount of time each foot spends in the air/on the ground relative to each other"""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    last_contact_time = contact_sensor.data.last_contact_time[:, sensor_cfg.body_ids]
    reward = torch.var(torch.clip(last_air_time, max=0.5), dim=1) + torch.var(
        torch.clip(last_contact_time, max=0.5), dim=1
    )
    # reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_contact(
    env: ManagerBasedRLEnv, command_name: str, expect_contact_num: int, sensor_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Reward feet contact"""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    contact_num = torch.sum(contact, dim=1)
    reward = (contact_num != expect_contact_num).float()
    # no reward for zero command
    reward *= torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) > 0.1
    # reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_contact_without_cmd(env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward feet contact"""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    reward = torch.sum(contact, dim=-1).float()
    reward *= torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) < 0.1
    # reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward

def feet_standing_force_reward(
    env: ManagerBasedRLEnv, 
    command_name: str, 
    sensor_cfg: SceneEntityCfg, 
    force_threshold: float = 20.0  # 受力阈值（单位：牛顿）
) -> torch.Tensor:
    """根据足部受力大小奖励静止时的站立状态。
    
    参数:
        force_threshold: 只有当足部受力超过此值时，才认为该脚是有效支撑。
                         建议设为机器人总重(mg)的 10%~20%。
    """
    # 1. 获取传感器对象
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    
    # 2. 获取所有足部的合力大小 (Net Force Norm)
    # net_forces_w 形状为 (num_envs, num_bodies, 3)
    # 我们只取指定的 foot_link 的索引
    foot_forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]
    foot_force_norms = torch.norm(foot_forces, dim=-1) # (num_envs, num_feet)
    
    # 3. 计算符合受力阈值的脚的数量
    # 只有受力 > force_threshold 时，该脚计为 1.0
    standing_feet_count = torch.sum(foot_force_norms > force_threshold, dim=-1).float()
    
    # 4. 条件遮罩：只在没有速度指令时生效
    vel_cmd = env.command_manager.get_command(command_name)
    is_static = torch.norm(vel_cmd[:, :2], dim=1) < 0.1
    
    # 5. 姿态遮罩：确保机器人向上
    upright_mask = torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    
    # 返回奖励：静止且朝上时，有效支撑的脚越多，奖励越高
    return standing_feet_count * is_static

def feet_stumble(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces_z = torch.abs(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2])
    forces_xy = torch.linalg.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :2], dim=2)
    # Penalize feet hitting vertical surfaces
    reward = torch.any(forces_xy > 4 * forces_z, dim=1).float()
    # reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_distance_y_exp(
    env: ManagerBasedRLEnv, stance_width: float, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    cur_footsteps_translated = asset.data.body_link_pos_w[:, asset_cfg.body_ids, :] - asset.data.root_link_pos_w[
        :, :
    ].unsqueeze(1)
    n_feet = len(asset_cfg.body_ids)
    footsteps_in_body_frame = torch.zeros(env.num_envs, n_feet, 3, device=env.device)
    for i in range(n_feet):
        footsteps_in_body_frame[:, i, :] = math_utils.quat_apply(
            math_utils.quat_conjugate(asset.data.root_link_quat_w), cur_footsteps_translated[:, i, :]
        )
    side_sign = torch.tensor(
        [1.0 if i % 2 == 0 else -1.0 for i in range(n_feet)],
        device=env.device,
    )
    stance_width_tensor = stance_width * torch.ones([env.num_envs, 1], device=env.device)
    desired_ys = stance_width_tensor / 2 * side_sign.unsqueeze(0)
    stance_diff = torch.square(desired_ys - footsteps_in_body_frame[:, :, 1])
    reward = torch.exp(-torch.sum(stance_diff, dim=1) / (std**2))
    # reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_distance_xy_exp(
    env: ManagerBasedRLEnv,
    stance_width: float,
    stance_length: float,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]

    # Compute the current footstep positions relative to the root
    cur_footsteps_translated = asset.data.body_link_pos_w[:, asset_cfg.body_ids, :] - asset.data.root_link_pos_w[
        :, :
    ].unsqueeze(1)

    footsteps_in_body_frame = torch.zeros(env.num_envs, 4, 3, device=env.device)
    for i in range(4):
        footsteps_in_body_frame[:, i, :] = math_utils.quat_apply(
            math_utils.quat_conjugate(asset.data.root_link_quat_w), cur_footsteps_translated[:, i, :]
        )

    # Desired x and y positions for each foot
    stance_width_tensor = stance_width * torch.ones([env.num_envs, 1], device=env.device)
    stance_length_tensor = stance_length * torch.ones([env.num_envs, 1], device=env.device)

    desired_xs = torch.cat(
        [stance_length_tensor / 2, stance_length_tensor / 2, -stance_length_tensor / 2, -stance_length_tensor / 2],
        dim=1,
    )
    desired_ys = torch.cat(
        [stance_width_tensor / 2, -stance_width_tensor / 2, stance_width_tensor / 2, -stance_width_tensor / 2], dim=1
    )

    # Compute differences in x and y
    stance_diff_x = torch.square(desired_xs - footsteps_in_body_frame[:, :, 0])
    stance_diff_y = torch.square(desired_ys - footsteps_in_body_frame[:, :, 1])

    # Combine x and y differences and compute the exponential penalty
    stance_diff = stance_diff_x + stance_diff_y
    reward = torch.exp(-torch.sum(stance_diff, dim=1) / std**2)
    # reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_height(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    target_height: float,
    tanh_mult: float,
) -> torch.Tensor:
    """Reward the swinging feet for clearing a specified height off the ground"""
    asset: RigidObject = env.scene[asset_cfg.name]
    foot_z_target_error = torch.square(asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - target_height)
    foot_velocity_tanh = torch.tanh(
        tanh_mult * torch.linalg.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2)
    )
    reward = torch.sum(foot_z_target_error * foot_velocity_tanh, dim=1)
    # no reward for zero command
    reward *= torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) > 0.1
    # reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward

def feet_clearance_reward(
    env: ManagerBasedRLEnv, 
    command_name: str,
    asset_cfg: SceneEntityCfg, 
    target_height: float, 
    std: float, 
    tanh_mult: float
) -> torch.Tensor:
    """Reward the swinging feet for clearing a specified height off the ground"""
    asset: RigidObject = env.scene[asset_cfg.name]
    foot_z_target_error = torch.square(asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - target_height)
    foot_velocity_tanh = torch.tanh(
        tanh_mult * torch.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2)
    )
    error = foot_z_target_error * foot_velocity_tanh
    reward = torch.exp(-torch.sum(error, dim=1) / std)
    reward *= torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) > 0.1
    # reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward

def feet_height_body(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    target_height: float,
    tanh_mult: float,
) -> torch.Tensor:
    """Reward the swinging feet for clearing a specified height off the ground"""
    asset: RigidObject = env.scene[asset_cfg.name]
    cur_footpos_translated = asset.data.body_pos_w[:, asset_cfg.body_ids, :] - asset.data.root_pos_w[:, :].unsqueeze(1)
    footpos_in_body_frame = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=env.device)
    cur_footvel_translated = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :] - asset.data.root_lin_vel_w[
        :, :
    ].unsqueeze(1)
    footvel_in_body_frame = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=env.device)
    for i in range(len(asset_cfg.body_ids)):
        footpos_in_body_frame[:, i, :] = math_utils.quat_apply_inverse(
            asset.data.root_quat_w, cur_footpos_translated[:, i, :]
        )
        footvel_in_body_frame[:, i, :] = math_utils.quat_apply_inverse(
            asset.data.root_quat_w, cur_footvel_translated[:, i, :]
        )
    foot_z_target_error = torch.square(footpos_in_body_frame[:, :, 2] - target_height).view(env.num_envs, -1)
    foot_velocity_tanh = torch.tanh(tanh_mult * torch.norm(footvel_in_body_frame[:, :, :2], dim=2))
    reward = torch.sum(foot_z_target_error * foot_velocity_tanh, dim=1)
    reward *= torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) > 0.1
    # reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_slide(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize feet sliding.

    This function penalizes the agent for sliding its feet on the ground. The reward is computed as the
    norm of the linear velocity of the feet multiplied by a binary contact sensor. This ensures that the
    agent is penalized only when the feet are in contact with the ground.
    """
    # Penalize feet sliding
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    asset: RigidObject = env.scene[asset_cfg.name]

    # feet_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    # reward = torch.sum(feet_vel.norm(dim=-1) * contacts, dim=1)

    cur_footvel_translated = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :] - asset.data.root_lin_vel_w[
        :, :
    ].unsqueeze(1)
    footvel_in_body_frame = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=env.device)
    for i in range(len(asset_cfg.body_ids)):
        footvel_in_body_frame[:, i, :] = math_utils.quat_apply_inverse(
            asset.data.root_quat_w, cur_footvel_translated[:, i, :]
        )
    foot_leteral_vel = torch.sqrt(torch.sum(torch.square(footvel_in_body_frame[:, :, :2]), dim=2)).view(
        env.num_envs, -1
    )
    reward = torch.sum(foot_leteral_vel * contacts, dim=1)
    # reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


# def smoothness_1(env: ManagerBasedRLEnv) -> torch.Tensor:
#     # Penalize changes in actions
#     diff = torch.square(env.action_manager.action - env.action_manager.prev_action)
#     diff = diff * (env.action_manager.prev_action[:, :] != 0)  # ignore first step
#     return torch.sum(diff, dim=1)


# def smoothness_2(env: ManagerBasedRLEnv) -> torch.Tensor:
#     # Penalize changes in actions
#     diff = torch.square(env.action_manager.action - 2 * env.action_manager.prev_action + env.action_manager.prev_prev_action)
#     diff = diff * (env.action_manager.prev_action[:, :] != 0)  # ignore first step
#     diff = diff * (env.action_manager.prev_prev_action[:, :] != 0)  # ignore second step
#     return torch.sum(diff, dim=1)


def upward(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize z-axis base linear velocity using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    reward = torch.square(1 - asset.data.projected_gravity_b[:, 2])
    return reward


def base_height_l2(
    env: ManagerBasedRLEnv,
    target_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Penalize asset height from its target using L2 squared kernel.

    Note:
        For flat terrain, target height is in the world frame. For rough terrain,
        sensor readings can adjust the target height to account for the terrain.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    if sensor_cfg is not None:
        sensor: RayCaster = env.scene[sensor_cfg.name]
        # Adjust the target height using the sensor data
        ray_hits = sensor.data.ray_hits_w[..., 2]
        if torch.isnan(ray_hits).any() or torch.isinf(ray_hits).any() or torch.max(torch.abs(ray_hits)) > 1e6:
            adjusted_target_height = asset.data.root_link_pos_w[:, 2]
        else:
            adjusted_target_height = target_height + torch.mean(ray_hits, dim=1)
    else:
        # Use the provided target height directly for flat terrain
        adjusted_target_height = target_height
    # Compute the L2 squared penalty
    reward = torch.square(asset.data.root_pos_w[:, 2] - adjusted_target_height)
    # reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def lin_vel_z_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize z-axis base linear velocity using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    reward = torch.square(asset.data.root_lin_vel_b[:, 2])
    # reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def ang_vel_xy_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize xy-axis base angular velocity using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    reward = torch.sum(torch.square(asset.data.root_ang_vel_b[:, :2]), dim=1)
    # reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def undesired_contacts(env: ManagerBasedRLEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize undesired contacts as the number of violations that are above a threshold."""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # check if contact force is above threshold
    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold
    # sum over contacts for each environment
    reward = torch.sum(is_contact, dim=1).float()
    # reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def flat_orientation_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize non-flat base orientation using L2 squared kernel.

    This is computed by penalizing the xy-components of the projected gravity vector.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    reward = torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)
    # reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward

def keep_ankle_pitch_zero_in_air(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_sensor", body_names=[""]),
    left_ankle_joint_index: int = 7,
    right_ankle_joint_index: int = 6,
    force_threshold: float = 2.0,
    pitch_scale: float = 0.2
) -> torch.Tensor:
    """Reward for keeping ankle pitch angle close to zero when foot is in the air.
    
    Args:
        env: The environment object.
        asset_cfg: Configuration for the robot asset containing DOF positions.
        sensor_cfg: Configuration for the contact force sensor.
        force_threshold: Threshold value for contact detection (in Newtons).
        pitch_scale: Scaling factor for the exponential reward.
        
    Returns:
        The computed reward tensor.
    """
    asset = env.scene[asset_cfg.name]
    contact_forces_history = env.scene.sensors[sensor_cfg.name].data.net_forces_w_history[:, :, sensor_cfg.body_ids]
    current_contact = torch.norm(contact_forces_history[:, -1], dim=-1) > force_threshold
    last_contact = torch.norm(contact_forces_history[:, -2], dim=-1) > force_threshold
    contact_filt = torch.logical_or(current_contact, last_contact)
    ankle_pitch_left = torch.abs(asset.data.joint_pos[:, left_ankle_joint_index]) * ~contact_filt[:, 0]
    ankle_pitch_right = torch.abs(asset.data.joint_pos[:, right_ankle_joint_index]) * ~contact_filt[:, 1]
    weighted_ankle_pitch = ankle_pitch_left + ankle_pitch_right
    return torch.exp(-weighted_ankle_pitch / pitch_scale)

# def keep_foot_pitch_zero_in_world(
#     env: ManagerBasedRLEnv,
#     asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
#     sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_sensor", body_names=[""]),
#     foot_body_names: list[str] = [".*_foot_link"], # 需要指定脚部刚体的名字
#     force_threshold: float = 2.0,
#     pitch_scale: float = 0.2
# ) -> torch.Tensor:
    
#     asset = env.scene[asset_cfg.name]
    
#     # 获取接触历史
#     contact_forces_history = env.scene.sensors[sensor_cfg.name].data.net_forces_w_history[:, :, sensor_cfg.body_ids]
#     current_contact = torch.norm(contact_forces_history[:, -1], dim=-1) > force_threshold
#     last_contact = torch.norm(contact_forces_history[:, -2], dim=-1) > force_threshold
#     contact_filt = torch.logical_or(current_contact, last_contact) # [env_num, 2]
    
#     # 获取脚部刚体在世界坐标系下的四元数
#     foot_indices, _ = asset.find_bodies(foot_body_names)
#     foot_quats = asset.data.body_quat_w[:, foot_indices, :] # [env_num, 2, 4]
    
#     # 展平张量以适配 euler_xyz_from_quat
#     # 将 [env_num, 2, 4] -> [env_num * 2, 4]
#     B, F, _ = foot_quats.shape 
#     foot_quats_flat = foot_quats.view(B * F, 4)
    
#     # 计算欧拉角
#     roll_flat, pitch_flat, yaw_flat = euler_xyz_from_quat(foot_quats_flat)
    
#     # 将结果恢复维度 [env_num * 2] -> [env_num, 2]
#     pitch = pitch_flat.view(B, F)
    
#     # 计算奖励 
#     foot_pitch_error = torch.abs(pitch) * ~contact_filt
#     weighted_pitch_error = torch.sum(foot_pitch_error, dim=1) 
    
#     return torch.exp(-weighted_pitch_error / pitch_scale)

class BipedalGaitReward(ManagerTermBase):
    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        """初始化奖励项。"""
        super().__init__(cfg, env)

        # 提取必要的传感器和资产信息
        self.sensor_cfg = cfg.params["sensor_cfg"]
        self.asset_cfg = cfg.params["asset_cfg"]
        self.contact_sensor: ContactSensor = env.scene.sensors[self.sensor_cfg.name]
        self.asset: Articulation = env.scene[self.asset_cfg.name]

        # 奖励缩放与超参数
        self.force_scale = float(cfg.params["tracking_contacts_shaped_force"])
        self.vel_scale = float(cfg.params["tracking_contacts_shaped_vel"])
        self.force_sigma = cfg.params["gait_force_sigma"]
        self.vel_sigma = cfg.params["gait_vel_sigma"]
        self.kappa_gait_probs = cfg.params["kappa_gait_probs"]
        self.vel_command_name = cfg.params["vel_command_name"]
        
        self.dt = env.step_dt

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        tracking_contacts_shaped_force: float,
        tracking_contacts_shaped_vel: float,
        gait_force_sigma: float,
        gait_vel_sigma: float,
        kappa_gait_probs: float,
        vel_command_name: str,
        sensor_cfg: any,
        asset_cfg: any,
        gait_params: List[float],
    ) -> torch.Tensor:
        """计算奖励。"""
        
        # 将传入的 [1.2, 0.5, 0.5] 转换为 tensor 并扩展到所有环境
        # gait_params 形状: (num_envs, 3)
        gait_tensor = torch.tensor(gait_params, device=self.device, dtype=torch.float32).repeat(self.num_envs, 1)

        # 1. 计算目标接触状态 (Desired Contact States)
        # foot_indices: 每个脚当前的相位, desired_contact_states: 目标触地概率 [0, 1]
        foot_indices, desired_contact_states = self.compute_contact_targets(gait_tensor)

        # 2. 力量奖励 (鼓励在支撑相触地，摆动相离地)
        foot_forces = torch.norm(self.contact_sensor.data.net_forces_w[:, self.sensor_cfg.body_ids], dim=-1)
        force_reward = self._compute_force_reward(foot_forces, desired_contact_states)

        # 3. 速度奖励 (鼓励在摆动相移动，支撑相静止)
        foot_velocities = torch.norm(self.asset.data.body_lin_vel_w[:, self.asset_cfg.body_ids], dim=-1)
        velocity_reward = self._compute_velocity_reward(foot_velocities, desired_contact_states)

        # 4. 汇总奖励
        total_reward = force_reward + velocity_reward
        
        # 5. 条件遮罩 (Masking)
        # 只有当速度指令大于 0.1m/s 时才给步态奖励
        vel_cmd = env.command_manager.get_command(self.vel_command_name)
        moving_mask = torch.norm(vel_cmd, dim=1) > 0.1
        
        # 姿态遮罩：如果机器人摔倒或严重倾斜（重力投影 Z 分量），减小奖励
        upright_mask = torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
        
        return total_reward * moving_mask

    def compute_contact_targets(self, gait_params):
        """根据当前时间计算期望的触地状态。"""
        frequencies = gait_params[:, 0]
        offsets = gait_params[:, 1]
        # 假设双足是对称的，持续时间(duration)一致
        durations = gait_params[:, 2].view(self.num_envs, 1).expand(-1, 2)

        # 当前整体相位 [0, 1)
        gait_indices = torch.remainder(self._env.episode_length_buf * self.dt * frequencies, 1.0)

        # 计算两只脚各自的相位 (加上 offset 产生相位差)
        foot_indices = torch.stack([
            gait_indices, 
            torch.remainder(gait_indices + offsets, 1.0)
        ], dim=1)

        # 判断处于支撑相还是摆动相并归一化索引
        # 支撑相: [0, duration] -> 映射到 [0, 0.5]
        # 摆动相: [duration, 1] -> 映射到 [0.5, 1.0]
        stance_idxs = foot_indices < durations
        swing_idxs = ~stance_idxs

        # 映射逻辑
        foot_indices[stance_idxs] = foot_indices[stance_idxs] * (0.5 / durations[stance_idxs])
        foot_indices[swing_idxs] = 0.5 + (foot_indices[swing_idxs] - durations[swing_idxs]) * (
            0.5 / (1 - durations[swing_idxs])
        )

        # 使用 Von Mises 类似平滑分布计算目标触地状态
        # 这里的逻辑保持原样，用于产生平滑的 0-1 触地信号
        smoothing_cdf = distributions.normal.Normal(0, self.kappa_gait_probs).cdf
        desired_contact_states = smoothing_cdf(foot_indices) * (
            1 - smoothing_cdf(foot_indices - 0.5)
        ) + smoothing_cdf(foot_indices - 1) * (1 - smoothing_cdf(foot_indices - 1.5))

        return foot_indices, desired_contact_states

    def _compute_force_reward(self, forces: torch.Tensor, desired_contacts: torch.Tensor) -> torch.Tensor:
        reward = torch.zeros_like(forces[:, 0])
        if self.force_scale < 0:  # Negative scale means penalize unwanted contact
            for i in range(forces.shape[1]):
                reward += (1 - desired_contacts[:, i]) * (1 - torch.exp(-forces[:, i] ** 2 / self.force_sigma))
        else:  # Positive scale means reward desired contact
            for i in range(forces.shape[1]):
                reward += (1 - desired_contacts[:, i]) * torch.exp(-forces[:, i] ** 2 / self.force_sigma)

        return (reward / forces.shape[1]) * self.force_scale

    def _compute_velocity_reward(self, velocities: torch.Tensor, desired_contacts: torch.Tensor) -> torch.Tensor:
        reward = torch.zeros_like(velocities[:, 0])
        if self.vel_scale < 0:  # Negative scale means penalize movement during contact
            for i in range(velocities.shape[1]):
                reward += desired_contacts[:, i] * (1 - torch.exp(-velocities[:, i] ** 2 / self.vel_sigma))
        else:  # Positive scale means reward movement during swing
            for i in range(velocities.shape[1]):
                reward += desired_contacts[:, i] * torch.exp(-velocities[:, i] ** 2 / self.vel_sigma)

        return (reward / velocities.shape[1]) * self.vel_scale
    
def feet_distance_penalize(env: ManagerBasedRLEnv,
                  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
                  feet_links_name: list[str]=["foot_[RL]_Link"],
                  min_feet_distance: float = 0.1,
                  max_feet_distance: float = 1.5,)-> torch.Tensor:
    # Penalize base height away from target
    asset: Articulation = env.scene[asset_cfg.name]
    feet_links_idx = asset.find_bodies(feet_links_name)[0]
    feet_pos = asset.data.body_link_pos_w[:,feet_links_idx]
    # feet distance on x-y plane
    feet_distance = torch.norm(feet_pos[:, 0, :2] - feet_pos[:, 1, :2], dim=-1)
    reward = torch.clip(min_feet_distance - feet_distance, 0, 1)
    reward += torch.clip(feet_distance - max_feet_distance, 0, 1)
    # reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward

def leg_symmetry(env: ManagerBasedRLEnv,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),) -> torch.Tensor:
    """Reward regulate abad joint position."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    feet_pos_w = asset.data.body_link_pos_w[:, asset_cfg.body_ids]
    base_quat = asset.data.root_link_quat_w.unsqueeze(1).expand(-1, 2, -1)
    # assert (compute_rotation_distance(asset.data.root_com_quat_w, asset.data.root_link_quat_w) < 0.1).all()
    base_pos = asset.data.root_link_state_w[:, :3].unsqueeze(1).expand(-1, 2, -1)
    feet_pos_b = quat_apply_inverse(
        base_quat,
        feet_pos_w - base_pos,
    )
    leg_symmetry_err = torch.abs(feet_pos_b[:, 0, 1]) - torch.abs(feet_pos_b[:, 1, 1])

    return torch.exp(-leg_symmetry_err ** 2 / std**2)

def same_feet_x_position(env: ManagerBasedRLEnv,
                  asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward regulate abad joint position."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    feet_pos_w = asset.data.body_link_pos_w[:, asset_cfg.body_ids]
    base_quat = asset.data.root_link_quat_w.unsqueeze(1).expand(-1, 2, -1)
    # assert (compute_rotation_distance(asset.data.root_com_quat_w, asset.data.root_link_quat_w) < 0.1).all()
    base_pos = asset.data.root_link_state_w[:, :3].unsqueeze(1).expand(-1, 2, -1)
    feet_pos_b = quat_apply_inverse(
        base_quat,
        feet_pos_w - base_pos,
    )
    feet_x_distance = torch.abs(feet_pos_b[:, 0, 0] - feet_pos_b[:, 1, 0])
    # return torch.exp(-feet_x_distance / 0.2)
    return feet_x_distance

class ActionSmoothnessPenalty(ManagerTermBase):
    """
    A reward term for penalizing large instantaneous changes in the network action output.
    This penalty encourages smoother actions over time.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        """Initialize the term.

        Args:
            cfg: The configuration of the reward term.
            env: The RL environment instance.
        """
        super().__init__(cfg, env)
        self.dt = env.step_dt
        self.prev_prev_action = None
        self.prev_action = None
        # self.__name__ = "action_smoothness_penalty"

    def __call__(self, env: ManagerBasedRLEnv) -> torch.Tensor:
        """Compute the action smoothness penalty.

        Args:
            env: The RL environment instance.

        Returns:
            The penalty value based on the action smoothness.
        """
        # Get the current action from the environment's action manager
        current_action = env.action_manager.action.clone()

        # If this is the first call, initialize the previous actions
        if self.prev_action is None:
            self.prev_action = current_action
            return torch.zeros(current_action.shape[0], device=current_action.device)

        if self.prev_prev_action is None:
            self.prev_prev_action = self.prev_action
            self.prev_action = current_action
            return torch.zeros(current_action.shape[0], device=current_action.device)

        # Compute the smoothness penalty
        penalty = torch.sum(torch.square(current_action - 2 * self.prev_action + self.prev_prev_action), dim=1)

        # Update the previous actions for the next call
        self.prev_prev_action = self.prev_action
        self.prev_action = current_action

        # Apply a condition to ignore penalty during the first few episodes
        startup_env_mask = env.episode_length_buf < 3
        penalty[startup_env_mask] = 0

        # Return the penalty scaled by the configured weight
        return penalty
    
def foot_landing_vel(
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg,
        sensor_cfg: SceneEntityCfg,
        foot_radius: float,
        about_landing_threshold: float,
        height_scanner_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Penalize high foot landing velocities"""
    asset = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    z_vels = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, 2]
    contacts = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2] > 0.1

    terrain_h_under_foot = torch.zeros_like(z_vels)

    if height_scanner_cfg is not None:
        scanner = env.scene.sensors[height_scanner_cfg.name]
        ray_hits_w = scanner.data.ray_hits_w
        feet_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids, :]
        num_feet = feet_pos_w.shape[1]
        
        default_terrain_z = asset.data.root_pos_w[:, 2] - 0.69 # fallback standard height

        for i in range(num_feet):
            foot_xy = feet_pos_w[:, i, :2]
            # 计算脚底周围的射线距离
            rel_pos = ray_hits_w[..., :2] - foot_xy.unsqueeze(1)
            dist_sq = torch.sum(rel_pos ** 2, dim=-1)
            
            # 取脚周围 15cm 内的探测点
            near_mask = dist_sq < (0.15 ** 2)
            near_z = torch.where(near_mask, ray_hits_w[..., 2], torch.tensor(float('inf'), device=env.device))
            min_z, _ = torch.min(near_z, dim=-1)
            terrain_h = torch.where(torch.isinf(min_z), default_terrain_z, min_z)
            terrain_h_under_foot[:, i] = terrain_h

    foot_heights = torch.clip(
        asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - foot_radius - terrain_h_under_foot, 0, 1
    )

    about_to_land = (foot_heights < about_landing_threshold) & (~contacts) & (z_vels < 0.0)
    landing_z_vels = torch.where(about_to_land, z_vels, torch.zeros_like(z_vels))
    reward = torch.sum(torch.square(landing_z_vels), dim=1)
    return reward

def track_adaptive_swing_height(
    env,
    min_clearance: float = 0.1,
    obstacle_scan_range: tuple = (0.1, 0.45),
    gait_params: list = [1.2, 0.5, 0.5], # [频率, 偏移, 支撑相占比]
    scan_dot_threshold: float = 0.05,
    foot_height_offset: float = 0.07, 
    vel_command_name: str = "base_velocity",
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("height_scanner"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=[".*_foot_link"]),
) -> torch.Tensor:
    """
    自适应抬腿高度奖励 - 最终对齐版。
    
    逻辑：
    1. 提取指令掩码：静止时不产生奖励/惩罚。
    2. 提取姿态掩码：摔倒时不产生奖励。
    3. 同步相位：严格遵循 BipedalGaitReward 的支撑/摆动周期。
    4. 地形感知：动态搜索脚下地面高度，支持楼梯/斜坡。
    5. 障碍检测：计算运动方向前方的台阶高度。
    6. 轨迹跟踪：在摆动相跟踪一个受障碍物高度影响的正弦曲线。
    """

    # =========================================================================
    # 0. 准备掩码 (Masks) - 确保静止或摔倒时不强制抬腿
    # =========================================================================
    vel_cmd = env.command_manager.get_command(vel_command_name)
    # 只有当线速度或角速度指令大于 0.1 时才激活奖励
    moving_mask = torch.norm(vel_cmd[:, :3], dim=1) > 0.1
    
    # 姿态遮罩：鼓励机器人保持直立，重力投影 Z 越接近 -1 奖励权重越高
    upright_mask = torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    
    # 提取步态参数
    gait_freq, gait_offset, gait_duration = gait_params
    
    # =========================================================================
    # 1. 计算相位 (严格对齐 BipedalGaitReward)
    # =========================================================================
    current_time = env.episode_length_buf * env.step_dt
    base_phase = torch.remainder(current_time * gait_freq, 1.0)
    
    # Index 0: Base, Index 1: Offset
    phase = torch.stack([
        base_phase,                                    
        torch.remainder(base_phase + gait_offset, 1.0) 
    ], dim=1)

    # =========================================================================
    # 2. 获取资产与传感器
    # =========================================================================
    robot = env.scene[asset_cfg.name]
    foot_indices, _ = robot.find_bodies(asset_cfg.body_names)
    num_feet = len(foot_indices)
    feet_pos_w = robot.data.body_pos_w[:, foot_indices, :] 

    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    ray_hits_w = sensor.data.ray_hits_w
    
    # =========================================================================
    # 3. 确定运动方向 (世界坐标系)
    # =========================================================================
    cmd_vel_b = vel_cmd[:, :2]
    cmd_norm = torch.norm(cmd_vel_b, dim=-1, keepdim=True)
    move_dir_b = torch.where(
        cmd_norm > 0.01,
        cmd_vel_b / (cmd_norm + 1e-5),
        torch.tensor([1.0, 0.0], device=env.device).expand_as(cmd_vel_b)
    )
    root_quat = robot.data.root_quat_w
    move_dir_w = quat_apply(root_quat, torch.cat([move_dir_b, torch.zeros_like(move_dir_b[:, :1])], dim=-1))[:, :2]

    # =========================================================================
    # 4. 计算：地形参考高度 & 前方障碍高度
    # =========================================================================
    terrain_h_under_foot = torch.zeros(env.num_envs, num_feet, device=env.device)
    obstacle_h = torch.zeros(env.num_envs, num_feet, device=env.device)
    default_terrain_z = robot.data.root_pos_w[:, 2] - 0.69 # 假设标准站姿高度

    for i in range(num_feet):
        foot_xy = feet_pos_w[:, i, :2]
        rel_pos = ray_hits_w[..., :2] - foot_xy.unsqueeze(1)
        dist_sq = torch.sum(rel_pos ** 2, dim=-1)
        
        # A. 脚下地形参考 (取脚周围 15cm 内的最矮点)
        near_mask = dist_sq < (0.15 ** 2)
        near_z = torch.where(near_mask, ray_hits_w[..., 2], torch.tensor(float('inf'), device=env.device))
        min_z, _ = torch.min(near_z, dim=-1)
        terrain_h = torch.where(torch.isinf(min_z), default_terrain_z, min_z)
        terrain_h_under_foot[:, i] = terrain_h
        
        # B. 运动方向前方障碍高度
        forward_dist = (rel_pos * move_dir_w.unsqueeze(1)).sum(dim=-1)
        dist = torch.sqrt(dist_sq)
        scan_mask = (forward_dist > scan_dot_threshold) & \
                    (dist > obstacle_scan_range[0]) & \
                    (dist < obstacle_scan_range[1])
        
        # 相对于该脚地面的高度差
        rel_z = ray_hits_w[..., 2] - terrain_h.unsqueeze(1)
        valid_hits = torch.where(scan_mask, rel_z, torch.tensor(-1.0, device=env.device))
        max_h, _ = torch.max(valid_hits, dim=-1)
        obstacle_h[:, i] = max_h.clamp(min=0.0)

    # =========================================================================
    # 5. 计算目标轨迹与奖励 (对齐摆动相)
    # =========================================================================
    swing_start = gait_duration # 支撑相结束即摆动相开始
    swing_end = 1.0
    
    # 摆动相掩码 (num_envs, 2)
    in_swing = (phase >= swing_start) & (phase < swing_end)
    
    # 计算摆动进度 [0, 1]
    swing_duration_len = max(swing_end - swing_start, 1e-3)
    swing_progress = (phase - swing_start) / swing_duration_len
    
    # 目标：(基础高度 + 障碍物高度) * 正弦波
    target_peak = min_clearance + obstacle_h
    target_traj = target_peak * torch.sin(torch.pi * swing_progress)
    
    # 实际足底 Z = foot_link_z - foot_height_offset
    # 实际离地间隙 = 实际足底 Z - 地面参考 Z
    actual_sole_z = feet_pos_w[:, :, 2] - foot_height_offset
    current_clearance = actual_sole_z - terrain_h_under_foot
    
    error_sq = torch.square(current_clearance - target_traj)
    
    # 汇总奖励：只有在摆动相、且正在移动、且姿态直立时才计算
    # unsqueeze(1) 用于将 (num_envs,) 广播到 (num_envs, num_feet)
    reward = -error_sq * in_swing * moving_mask.unsqueeze(1)
    
    return torch.sum(reward, dim=1)

def idle_when_commanded(
    env: ManagerBasedRLEnv,
    command_name: str,
    cmd_threshold: float = 0.2,
    vel_threshold: float = 0.1,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """惩罚机器人“懒惰”行为：当收到移动或旋转指令时却保持静止。
    
    检测条件：(线性指令或旋转指令超过阈值) 且 (线性速度和旋转速度都低于阈值)。

    Args:
        env: 环境实例。
        command_name: 指令名称。通常指令为 [vx, vy, yaw_rate]。
        cmd_threshold: 指令幅值阈值。
        vel_threshold: 实际运动幅值阈值。
        asset_cfg: 机器人资产配置。
    """
    asset: Articulation = env.scene[asset_cfg.name]
    
    # 1. 获取指令 (包含 vx, vy 和 yaw_rate)
    # command 形状通常为 (num_envs, 3) -> [vx, vy, omega]
    cmd = env.command_manager.get_command(command_name)[:, :3]
    # 计算综合指令强度 (线性速度和旋转速度的 L2 范数)
    cmd_magnitude = torch.norm(cmd, dim=-1)
    
    # 2. 获取实际线性速度 (投影到 Yaw 坐标系)
    root_vel_w = asset.data.root_lin_vel_w[:, :3]
    root_quat_w = asset.data.root_quat_w
    yaw_quat = math_utils.yaw_quat(root_quat_w)
    
    vel_yaw = math_utils.quat_apply_inverse(yaw_quat, root_vel_w)
    lin_vel_mag_sq = torch.sum(torch.square(vel_yaw[:, :2]), dim=-1) # vx^2 + vy^2
    
    # 3. 获取实际旋转速度 (Base 坐标系的角速度 z 分量)
    # root_ang_vel_w 是世界坐标系下的角速度
    # 我们将其转到机器人局部坐标系下，看 z 轴旋转
    ang_vel_b = math_utils.quat_apply_inverse(root_quat_w, asset.data.root_ang_vel_w)
    ang_vel_z_sq = torch.square(ang_vel_b[:, 2]) # omega_z^2
    
    # 4. 计算综合实际运动强度
    # 综合速度 = sqrt(vx^2 + vy^2 + omega_z^2)
    vel_magnitude = torch.sqrt(lin_vel_mag_sq + ang_vel_z_sq)
    
    # 5. 逻辑判定
    # 是否下达了移动/旋转指令
    is_commanded = cmd_magnitude > cmd_threshold
    # 是否实际表现为静止 (既没有线位移也没有旋转)
    is_idle = vel_magnitude < vel_threshold
    
    return (is_commanded & is_idle).float()

def is_flying(env: ManagerBasedRLEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """惩罚机器人全身腾空的状态。

    逻辑：如果在指定的时间历史内，所有指定的部位（如足部）的最大受力都低于阈值，则认为处于腾空状态。

    Args:
        env: 环境实例。
        threshold: 力的阈值（单位：牛顿）。低于此值认为没接触地面。
        sensor_cfg: 传感器配置。需指定传感器名称和要检测的 body_ids（通常是足部）。

    Returns:
        torch.Tensor: 形状为 (num_envs,)。1.0 表示全身腾空，0.0 表示有接触。
    """
    # 获取接触传感器对象
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    
    # 获取受力历史数据
    net_contact_forces = contact_sensor.data.net_forces_w_history

    # 计算力模长并取历史最大值
    target_forces = net_contact_forces[:, :, sensor_cfg.body_ids, :]
    
    # 计算 XYZ 合力大小: (num_envs, history, num_bodies)
    force_norms = torch.norm(target_forces, dim=-1)
    
    # 在历史维度（dim=1）上找最大值: (num_envs, num_bodies)
    # 这确保了如果在最近几帧内只要接触过地面，就不算“飞行”
    max_forces_in_history, _ = torch.max(force_norms, dim=1)
    
    # 判断每只脚是否在接触状态
    is_contact = max_forces_in_history > threshold
    
    # 计算触地脚的总数
    contact_count = torch.sum(is_contact, dim=-1)
    
    # 返回判定结果
    # 如果触地数 < 0.5 (即等于0)，表示正在飞行，返回 1.0
    return (contact_count < 0.5).float()

def stay_alive(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Reward for staying alive."""
    return torch.ones(env.num_envs, device=env.device)

def body_orientation_l2(
    env: ManagerBasedRLEnv, 
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """惩罚躯干或指定部位偏离垂直方向（L2 范数）。
    
    该函数衡量重力向量在局部坐标系 XY 平面上的投影。
    当机器人完全竖直时，返回 0.0；倾斜程度越大，返回数值越大。
    """

    asset: Articulation = env.scene[asset_cfg.name]
    
    # 获取目标部位的重力投影
    projected_gravity = asset.data.projected_gravity_b

    # 计算 X 和 Y 分量的平方和 (L2 范数)
    # 取前两维 [gx, gy]，计算 gx^2 + gy^2
    return torch.sum(torch.square(projected_gravity[:, :2]), dim=1)

def specific_joint_action_penalty(env: ManagerBasedRLEnv, action_term_name: str, joint_regex: str) -> torch.Tensor:
    """
    专门针对被拆分的动作空间，惩罚特定 Action Term 下的特定关节动作。
    """
    # 动态获取配置的特定 Action 组（比如 "joint_pos"）
    action_term = env.action_manager.get_term(action_term_name)
    
    # 从该组负责控制的关节名字列表中，找到符合正则表达式的关节的【相对索引】
    pattern = re.compile(joint_regex)
    target_indices =[
        i for i, name in enumerate(action_term._joint_names) if pattern.search(name)
    ]
    
    if len(target_indices) == 0:
        # 如果没匹配到，返回 0 避免报错
        return torch.zeros(env.num_envs, device=env.device)
        
    # 获取策略网络输出给这些关节的【原始动作】(raw_actions)
    target_actions = action_term.raw_actions[:, target_indices]
    
    # 计算 L1 惩罚并返回 (求绝对值之和)
    return torch.sum(torch.abs(target_actions), dim=1)

def foot_impact_reduction(
    env: ManagerBasedRLEnv,
    max_delta_v_sq: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """
    足部柔和落地奖励 (Impact Reduction Reward)
    惩罚重力方向（Z轴）上连续两步之间的速度突变，并使用最大阈值进行截断以稳定训练。
    
    公式: sum( min( (v_{z,t} - v_{z,t-1})^2, max_delta_v_sq ) )
    注意：此函数返回的是正的惩罚值，在配置中需将 weight 设为负数。
    """
    # 获取机器人的 asset 数据
    asset = env.scene[asset_cfg.name]
    
    # 获取指定刚体（如左右脚）在世界坐标系下 Z 轴（索引2）的线速度
    # body_lin_vel_w 的 shape 为[num_envs, num_bodies, 3]
    # current_vel_z 的 shape 为 [num_envs, num_target_bodies]
    current_vel_z = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, 2]
    
    # 巧妙利用 asset 挂载自定义属性，来保存上一步的速度 (处理状态依赖)
    if not hasattr(asset, "prev_body_vel_z"):
        asset.prev_body_vel_z = torch.zeros_like(current_vel_z)
        asset.prev_body_vel_z.copy_(current_vel_z)
        
    # 【关键处理】处理环境重置 (Reset)
    # 当某些环境发生重置时，上一步的速度必须与当前速度对齐，否则会产生因为“瞬移”导致的巨大惩罚
    reset_env_ids = env.reset_buf.nonzero(as_tuple=False).squeeze(-1)
    if len(reset_env_ids) > 0:
        asset.prev_body_vel_z[reset_env_ids] = current_vel_z[reset_env_ids]
        
    # 计算当前步与上一步在 Z 轴上的速度变化量
    delta_v_z = current_vel_z - asset.prev_body_vel_z
    
    # 计算速度变化的平方
    delta_v_z_sq = torch.square(delta_v_z)
    
    # 模拟公式中的 min(Delta_v^2, Delta_v_max^2)，使用 clamp 截断上限，防止物理引擎结算时的突变毁掉 Critic 网络
    penalty_per_body = torch.clamp(delta_v_z_sq, max=max_delta_v_sq)
    
    # 将两只脚的惩罚值相加，shape 变为 [num_envs]
    total_penalty = torch.sum(penalty_per_body, dim=1)
    
    # 就地更新速度缓存，供下一个时间步使用
    asset.prev_body_vel_z.copy_(current_vel_z)
    
    return total_penalty

def centrifugal_compensation_reward(
    env: ManagerBasedRLEnv, 
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """
    鼓励机器人在转弯时产生适当的身体倾斜，以补偿离心力。
    """
    # 提取机器人的 asset 数据
    asset = env.scene[asset_cfg.name]
    
    # 1. 获取机身坐标系下的前向速度 v_x (X轴)
    # 形状: (num_envs,)
    v_x = asset.data.root_lin_vel_b[:, 0]
    
    # 2. 获取机身坐标系下的偏航角速度 omega_yaw (Z轴)
    # 形状: (num_envs,)
    omega_yaw = asset.data.root_ang_vel_b[:, 2]
    
    # 3. 计算期望的质心倾斜角 theta_des
    g = 9.81
    theta_des = torch.atan((v_x * omega_yaw) / g)
    
    # 4. 计算目标项: min(0.3, sin(theta_des))
    # 注意：论文公式写的是 min(0.3, ...)，这在正向转弯时是对的。
    # 为了保证左右转弯时的对称性，实际工程中建议使用对称截断 clamp(-0.3, 0.3)。
    # 这里严格按照您的公式，但为了兼顾左右转，使用了对称的 torch.clamp。
    target_val = torch.clamp(torch.sin(theta_des), min=-0.3, max=0.3)
    
    # 5. 获取观测到的侧向加速度 a_obs_y
    # 在 Isaac Lab 中，IMU 的姿态观测通常用重力在机身系下的投影来表示。
    # projected_gravity_b 的 Y 轴分量即相当于归一化后的侧向倾斜加速度 (近似 sin(roll))
    # 形状: (num_envs,)
    a_obs_y = asset.data.projected_gravity_b[:, 1]
    
    # 6. 计算奖励
    reward = -torch.square(a_obs_y - target_val)
    
    return reward