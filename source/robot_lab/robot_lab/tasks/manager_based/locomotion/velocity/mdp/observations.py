# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Union

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply_inverse

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv

def joint_pos_rel_without_wheel(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    wheel_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """The joint positions of the asset w.r.t. the default joint positions.(Without the wheel joints)"""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos_rel = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    joint_pos_rel[:, wheel_asset_cfg.joint_ids] = 0
    return joint_pos_rel


def phase(env: ManagerBasedRLEnv, cycle_time: float) -> torch.Tensor:
    if not hasattr(env, "episode_length_buf") or env.episode_length_buf is None:
        env.episode_length_buf = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
    phase = env.episode_length_buf[:, None] * env.step_dt / cycle_time
    phase_tensor = torch.cat([torch.sin(2 * torch.pi * phase), torch.cos(2 * torch.pi * phase)], dim=-1)
    return phase_tensor

def robot_joint_torque(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """joint torque of the robot"""
    asset: Articulation = env.scene[asset_cfg.name]
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    return asset.data.applied_torque.to(device)


def robot_joint_acc(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """joint acc of the robot"""
    asset: Articulation = env.scene[asset_cfg.name]
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    return asset.data.joint_acc.to(device)


def robot_feet_contact_force(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg):
    """contact force of the robot feet"""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    contact_force_tensor = contact_sensor.data.net_forces_w_history.to(device)
    return contact_force_tensor.view(contact_force_tensor.shape[0], -1)


def robot_mass(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """mass of the robot"""
    asset: Articulation = env.scene[asset_cfg.name]
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    return asset.data.default_mass.to(device)


def robot_inertia(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """inertia of the robot"""
    asset: Articulation = env.scene[asset_cfg.name]
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    inertia_tensor = asset.data.default_inertia.to(device)
    return inertia_tensor.view(inertia_tensor.shape[0], -1)


def robot_joint_pos(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """joint positions of the robot"""
    asset: Articulation = env.scene[asset_cfg.name]
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    return asset.data.default_joint_pos.to(device)


def robot_joint_stiffness(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """joint stiffness of the robot"""
    asset: Articulation = env.scene[asset_cfg.name]
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    return asset.data.default_joint_stiffness.to(device)


def robot_joint_damping(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """joint damping of the robot"""
    asset: Articulation = env.scene[asset_cfg.name]
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    return asset.data.default_joint_damping.to(device)


def robot_pos(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """pose of the robot"""
    asset: Articulation = env.scene[asset_cfg.name]
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    return asset.data.root_pos_w.to(device)

def robot_pose_z_world(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """pose z of the robot in world"""
    asset: Articulation = env.scene[asset_cfg.name]
    # device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    return asset.data.root_pos_w[:, 2:3]

def robot_vel(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """velocity of the robot"""
    asset: Articulation = env.scene[asset_cfg.name]
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    return asset.data.root_vel_w.to(device)


def robot_material_properties(
    env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """material properties of the robot"""
    asset: Articulation = env.scene[asset_cfg.name]
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    material_tensor = asset.root_physx_view.get_material_properties().to(device)
    return material_tensor.view(material_tensor.shape[0], -1)


def robot_center_of_mass(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """center of mass of the robot"""
    asset: Articulation = env.scene[asset_cfg.name]
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    com_tensor = asset.root_physx_view.get_coms().clone().to(device)
    return com_tensor.view(com_tensor.shape[0], -1)


def robot_contact_force(env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """The contact forces of the body."""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    body_contact_force= contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids]

    return body_contact_force.reshape(body_contact_force.shape[0], -1)


def get_gait_phase_from_command(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Get the current gait phase as observation.

    The gait phase is represented by [sin(phase), cos(phase)] to ensure continuity.
    The phase is calculated based on the episode length and gait frequency.

    Returns:
        torch.Tensor: The gait phase observation. Shape: (num_envs, 2).
    """
    # check if episode_length_buf is available
    if not hasattr(env, "episode_length_buf"):
        return torch.zeros(env.num_envs, 2, device=env.device)

    # Get the gait command from command manager
    command_term = env.command_manager.get_term("gait_command")
    # Calculate gait indices based on episode length
    gait_indices = torch.remainder(env.episode_length_buf * env.step_dt * command_term.command[:, 0], 1.0)
    # Reshape gait_indices to (num_envs, 1)
    gait_indices = gait_indices.unsqueeze(-1)
    # Convert to sin/cos representation
    sin_phase = torch.sin(2 * torch.pi * gait_indices)
    cos_phase = torch.cos(2 * torch.pi * gait_indices)

    return torch.cat([sin_phase, cos_phase], dim=-1)

def get_gait_phase_from_param(env: ManagerBasedRLEnv, gait_freq: Union[float, torch.Tensor]) -> torch.Tensor:
    """获取当前的步态相位。

    步态相位通过 [sin(phase), cos(phase)] 表示以确保连续性。
    相位根据情节步数（episode length）和传入的步频（gait frequency）计算。

    参数:
        env: 环境实例。
        gait_freq: 步频参数。可以是标量（float）或形状为 (num_envs,) 的 Tensor。

    返回:
        torch.Tensor: 步态相位观测。形状为 (num_envs, 2)。
    """
    # 检查 episode_length_buf 是否可用
    if not hasattr(env, "episode_length_buf"):
        return torch.zeros(env.num_envs, 2, device=env.device)

    # 如果 gait_freq 是标量，确保它能与 env.num_envs 匹配
    if isinstance(gait_freq, (float, int)):
        gait_freq = torch.tensor(gait_freq, device=env.device)

    # 计算步态指数 (phase = time * frequency % 1.0)
    # env.episode_length_buf * env.step_dt 得到的是当前情节持续的时间（秒）
    gait_indices = torch.remainder(env.episode_length_buf * env.step_dt * gait_freq, 1.0)
    
    # 将形状调整为 (num_envs, 1) 以进行后续计算
    gait_indices = gait_indices.view(env.num_envs, 1)
    
    # 转换为正余弦表示
    sin_phase = torch.sin(2 * torch.pi * gait_indices)
    cos_phase = torch.cos(2 * torch.pi * gait_indices)

    return torch.cat([sin_phase, cos_phase], dim=-1)


def get_gait_command(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Get the current gait command parameters as observation.

    Returns:
        torch.Tensor: The gait command parameters [frequency, offset, duration].
                     Shape: (num_envs, 3).
    """
    return env.command_manager.get_command(command_name)


def robot_base_pose(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """pose of the robot base"""
    asset: Articulation = env.scene[asset_cfg.name]
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    return asset.data.root_pos_w.to(device)

def feet_lin_vel(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Root linear velocity in the asset's root frame."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.body_lin_vel_w[:, asset_cfg.body_ids].flatten(start_dim=1)

def feet_lin_vel_in_body(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """获取足部相对于机身的速度，并转换到机身坐标系下。"""
    
    asset: Articulation = env.scene[asset_cfg.name]
    
    # 获取足部在世界系下的线性速度 (Shape: [num_envs, num_feet, 3])
    foot_vel_w = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :]
    
    base_vel_w = asset.data.root_lin_vel_w
    
    # 使用 unsqueeze(1) 将 base_vel_w 从 [N, 3] 变为 [N, 1, 3]，以便对所有足部进行广播相减
    rel_vel_w = foot_vel_w - base_vel_w.unsqueeze(1)

    base_quat_w = asset.data.root_quat_w  # [N, 4]
    
    # 因为 rel_vel_w 是 [N, num_feet, 3]，而 base_quat_w 是 [N, 4]
    # 我们需要将四元数扩展到与足部数量一致，或者利用该函数内部的 reshape 特性
    num_feet = rel_vel_w.shape[1]
    # [N, num_feet, 4]
    base_quat_w_expanded = base_quat_w.unsqueeze(1).expand(-1, num_feet, -1)
    
    rel_vel_b = quat_apply_inverse(base_quat_w_expanded, rel_vel_w)
    
    # 返回展平后的观测 [num_envs, num_feet * 3]
    return rel_vel_b.view(env.num_envs, -1)

def feet_contact_bool(env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """获取足部触地状态
    
    Args:
        sensor_cfg: 触地传感器的配置，需指定传感器名称和关联的 body_ids（足部索引）。
    """
    # 这里是从 env.scene.sensors 中获取，而不是 env.scene
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    
    # 获取传感器记录的合力大小 (Shape: [num_envs, num_bodies_in_sensor, 3])
    # 通常关注的是合力的模长（或者 Z 轴分量）
    net_forces = sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]
    
    # 计算力的模长 [num_envs, num_feet]
    force_magnitudes = torch.norm(net_forces, dim=-1)
    
    # 判断是否触地。设置一个阈值（如 1.0 牛顿）以过滤传感器的数值噪声
    # 返回的是布尔张量 (True 为触地，False 为悬空)
    contact_bool = force_magnitudes > 1.0
    
    return contact_bool.float()

def feet_pos_in_body(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """获取足端相对于机身的位置（在机身坐标系下表示）。"""
    
    asset: Articulation = env.scene[asset_cfg.name]
    
    # 获取足部在世界系下的位置 (Shape: [num_envs, num_feet, 3])
    foot_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids, :]
    
    # 获取机身在世界系下的位置 (Shape: [num_envs, 3])
    base_pos_w = asset.data.root_pos_w.unsqueeze(1) # 增加维度以便广播相减
    
    rel_pos_w = foot_pos_w - base_pos_w
    
    # 获取机身姿态四元数 [N, 4]
    base_quat_w = asset.data.root_quat_w
    
    # 准备四元数以匹配足部数量 [N, num_feet, 4]
    num_feet = rel_pos_w.shape[1]
    base_quat_w_expanded = base_quat_w.unsqueeze(1).expand(-1, num_feet, -1)
    
    rel_pos_b = quat_apply_inverse(base_quat_w_expanded, rel_pos_w)
    
    # 返回展平后的观测 [num_envs, num_feet * 3]
    return rel_pos_b.reshape(env.num_envs, -1)

def feet_contact_forces_in_body(env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """获取足端在机身坐标系下的三维接触力。"""
    
    # 获取传感器和机器人资源对象
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    asset: Articulation = env.scene["robot"] 
    
    # 获取世界系下的接触力 [num_envs, num_feet, 3]
    if sensor_cfg.body_ids is not None:
        forces_w = sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]
    else:
        forces_w = sensor.data.net_forces_w
        
    # 获取机身姿态四元数 [num_envs, 4]
    base_quat_w = asset.data.root_quat_w
    
    # 准备四元数以匹配足部数量
    num_feet = forces_w.shape[1]
    base_quat_w_expanded = base_quat_w.unsqueeze(1).expand(-1, num_feet, -1)
    
    # 将力从世界系旋转到机身系
    forces_b = quat_apply_inverse(base_quat_w_expanded, forces_w)
    
    # 返回展平后的观测 [num_envs, num_feet * 3]
    return forces_b.reshape(env.num_envs, -1)

def generated_commands(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """The generated command from command term in the command manager with the given name."""
    return env.command_manager.get_command(command_name)

def joint_pos_rel_exclude_wheel(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
                                wheel_joints_name: list[str] = ["wheel_[RL]_Joint"] 
                                ) -> torch.Tensor:
    """The joint positions of the asset w.r.t. the default joint positions.

    Note: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their positions returned.
    """
    # extract the used quantities (to enable type-hinting)

    asset: Articulation = env.scene[asset_cfg.name]
    wheel_joints_idx = asset.find_joints(wheel_joints_name)[0]
    all_joints_idx = range(asset.num_joints)
    pos_idx_exclude_wheel = [i for i in all_joints_idx if i not in wheel_joints_idx]
    return asset.data.joint_pos[:, pos_idx_exclude_wheel] - asset.data.default_joint_pos[:, pos_idx_exclude_wheel]



