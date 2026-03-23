from __future__ import annotations

import torch
from typing import TYPE_CHECKING, List
from torch import distributions

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import mdp
from isaaclab.managers import ManagerTermBase
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_apply_inverse, yaw_quat, euler_xyz_from_quat, quat_apply, wrap_to_pi

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import RewardTermCfg

def base_illegal_contact(env: ManagerBasedRLEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """只有base和地面碰撞才会终止"""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    
    # 获取总受力矢量 (Shape: [env, history, body, 3])
    net_forces_vec = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
    
    # 获取自碰撞受力矢量 (Shape: [env, history, body, filtered_bodies, 3])
    self_forces_vec_matrix = contact_sensor.data.force_matrix_w_history[:, :, sensor_cfg.body_ids, :, :]
    self_forces_vec_sum = torch.sum(self_forces_vec_matrix, dim=3)
    terrain_forces_vec = net_forces_vec - self_forces_vec_sum
    terrain_force_mag = torch.norm(terrain_forces_vec, dim=-1) # Shape: [env, history, body]
    
    return torch.any(torch.max(terrain_force_mag, dim=1)[0] > threshold, dim=1)