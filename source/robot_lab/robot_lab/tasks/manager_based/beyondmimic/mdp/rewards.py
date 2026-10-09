from __future__ import annotations

import math
import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_apply_inverse, quat_error_magnitude
from isaaclab.managers import ManagerTermBase
from isaaclab.assets import Articulation

from robot_lab.tasks.manager_based.beyondmimic.mdp.commands import MotionCommand
from robot_lab.tasks.manager_based.locomotion.velocity.mdp.rewards import (
    leg_symmetry as _locomotion_leg_symmetry,
    same_feet_x_position as _locomotion_same_feet_x_position,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _get_body_indexes(command: MotionCommand, body_names: list[str] | None) -> list[int]:
    return [i for i, name in enumerate(command.cfg.body_names) if (body_names is None) or (name in body_names)]


def motion_global_anchor_position_error_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = torch.sum(torch.square(command.anchor_pos_w - command.robot_anchor_pos_w), dim=-1)
    return torch.exp(-error / std**2)


def motion_global_anchor_orientation_error_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = quat_error_magnitude(command.anchor_quat_w, command.robot_anchor_quat_w) ** 2
    return torch.exp(-error / std**2)


def motion_anchor_roll_horizontal_error_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    """Reward zero world-frame anchor roll throughout the motion with a positive weight.

    Use the robot's normalized wxyz quaternion, independently of reference roll
    and reference progress. The XYZ Euler roll and ``std`` are in radians;
    pitch and yaw are free. The roll convention is singular at pitch +/- pi/2.
    """
    if not math.isfinite(std) or std <= 0.0:
        raise ValueError("Horizontal roll reward requires a finite, positive standard deviation.")
    command: MotionCommand = env.command_manager.get_term(command_name)
    qw, qx, qy, qz = command.robot_anchor_quat_w.unbind(dim=-1)
    roll = torch.atan2(2.0 * (qw * qx + qy * qz), 1.0 - 2.0 * (qx.square() + qy.square()))
    return torch.exp(-roll.square() / std**2)


def _motion_phase_window_weight(
    env: ManagerBasedRLEnv,
    command_name: str,
    start_time_s: float,
    end_time_s: float,
    ramp_time_s: float,
) -> torch.Tensor:
    """Blend a bounded reference-time window with smooth entry and exit."""
    if not math.isfinite(start_time_s) or start_time_s < 0.0:
        raise ValueError("Motion phase rewards require a finite, nonnegative start time.")
    if not math.isfinite(end_time_s) or end_time_s <= start_time_s:
        raise ValueError("Motion phase rewards require a finite end time after the start.")
    if not math.isfinite(ramp_time_s) or ramp_time_s <= 0.0:
        raise ValueError("Motion phase rewards require a finite, positive ramp duration.")
    if 2.0 * ramp_time_s > end_time_s - start_time_s:
        raise ValueError("Motion phase entry and exit ramps must fit within the window.")

    command: MotionCommand = env.command_manager.get_term(command_name)
    fps = float(command.motion.fps.item())
    if not math.isfinite(fps) or fps <= 0.0:
        raise ValueError("Motion phase rewards require a finite, positive reference frame rate.")

    reference_time = command.time_steps / fps
    entry = torch.clamp((reference_time - start_time_s) / ramp_time_s, min=0.0, max=1.0)
    exit = torch.clamp((end_time_s - reference_time) / ramp_time_s, min=0.0, max=1.0)
    return entry.square() * (3.0 - 2.0 * entry) * exit.square() * (3.0 - 2.0 * exit)


def motion_phase_anchor_pitch_angular_velocity_error_l2(
    env: ManagerBasedRLEnv,
    command_name: str,
    start_time_s: float,
    end_time_s: float,
    ramp_time_s: float,
) -> torch.Tensor:
    """Penalize anchor pitch-rate error with a negative reward weight.

    Project the world-frame angular-velocity difference onto the reference
    anchor's local y axis. Tracking the reference preserves intended takeoff
    rotation, without averaging the anchor error over the other tracked bodies.
    """
    blend = _motion_phase_window_weight(env, command_name, start_time_s, end_time_s, ramp_time_s)
    command: MotionCommand = env.command_manager.get_term(command_name)
    error_w = command.robot_anchor_ang_vel_w - command.anchor_ang_vel_w
    error_ref = quat_apply_inverse(command.anchor_quat_w, error_w)
    return blend * error_ref[:, 1].square()


def motion_phase_anchor_orientation_error_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
    start_time_s: float,
    end_time_s: float,
    ramp_time_s: float,
) -> torch.Tensor:
    """Add reference anchor orientation tracking in a bounded motion phase."""
    if not math.isfinite(std) or std <= 0.0:
        raise ValueError("Motion phase orientation reward requires a finite, positive standard deviation.")
    blend = _motion_phase_window_weight(env, command_name, start_time_s, end_time_s, ramp_time_s)
    return blend * motion_global_anchor_orientation_error_exp(env, command_name, std)


def motion_relative_body_position_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.body_pos_relative_w[:, body_indexes] - command.robot_body_pos_w[:, body_indexes]), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_relative_body_orientation_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = (
        quat_error_magnitude(command.body_quat_relative_w[:, body_indexes], command.robot_body_quat_w[:, body_indexes])
        ** 2
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_global_body_linear_velocity_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.body_lin_vel_w[:, body_indexes] - command.robot_body_lin_vel_w[:, body_indexes]), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_global_body_angular_velocity_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.body_ang_vel_w[:, body_indexes] - command.robot_body_ang_vel_w[:, body_indexes]), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)


def feet_contact_time(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    first_air = contact_sensor.compute_first_air(env.step_dt, env.physics_dt)[:, sensor_cfg.body_ids]
    last_contact_time = contact_sensor.data.last_contact_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_contact_time < threshold) * first_air, dim=-1)
    return reward


def wheel_contact_continuous(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    start_time_s: float,
    min_contact_force: float,
    stable_contact_time: float,
) -> torch.Tensor:
    """Reward sustained bilateral wheel support in the reference's wheel phase.

    Reference frames handle adaptive motion starts independently of episode time.
    The sensor's contact timers reset on loss of contact, and current positive
    world-z forces must also exceed the threshold on both wheels. The shorter
    contact duration ramps to one; larger forces and wheel speed earn no bonus.
    Requires exactly two sensor bodies and ``track_air_time=True``.
    """
    if start_time_s < 0.0 or min_contact_force < 0.0 or stable_contact_time <= 0.0:
        raise ValueError("Wheel contact reward requires nonnegative start/force and positive stable contact time.")

    command: MotionCommand = env.command_manager.get_term(command_name)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    sensor_data = contact_sensor.data
    if sensor_data.current_contact_time is None:
        raise RuntimeError("Wheel contact reward requires the contact sensor to enable track_air_time.")
    contact_time = sensor_data.current_contact_time[:, sensor_cfg.body_ids]
    if contact_time.shape[1] != 2:
        raise ValueError("Wheel contact reward requires exactly two wheel bodies in sensor_cfg.")

    vertical_force = sensor_data.net_forces_w[:, sensor_cfg.body_ids, 2]
    both_supported = (vertical_force > min_contact_force).all(dim=-1)
    continuity = torch.clamp(contact_time.min(dim=-1).values / stable_contact_time, min=0.0, max=1.0)
    start_frame = math.ceil(start_time_s * command.motion.fps.item())
    wheel_phase = command.time_steps >= start_frame
    return continuity * both_supported * wheel_phase


def _wheel_phase_geometry_weight(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    start_time_s: float,
    ramp_time_s: float,
) -> torch.Tensor:
    """Blend wheel geometry terms using reference progress, including after contact loss."""
    if not math.isfinite(start_time_s) or start_time_s < 0.0:
        raise ValueError("Wheel geometry rewards require a finite, nonnegative start time.")
    if not math.isfinite(ramp_time_s) or ramp_time_s <= 0.0:
        raise ValueError("Wheel geometry rewards require a finite, positive ramp duration.")

    asset: Articulation = env.scene[asset_cfg.name]
    if asset.data.body_link_pos_w[:, asset_cfg.body_ids].shape[1] != 2:
        raise ValueError("Wheel geometry rewards require exactly two wheel bodies in asset_cfg.")

    command: MotionCommand = env.command_manager.get_term(command_name)
    fps = float(command.motion.fps.item())
    if not math.isfinite(fps) or fps <= 0.0:
        raise ValueError("Wheel geometry rewards require a finite, positive reference frame rate.")
    reference_time = command.time_steps / fps
    blend = torch.clamp((reference_time - start_time_s) / ramp_time_s, min=0.0, max=1.0)
    return blend.square() * (3.0 - 2.0 * blend)


def wheel_phase_leg_symmetry(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    std: float,
    start_time_s: float,
    ramp_time_s: float,
) -> torch.Tensor:
    """Reward lateral wheel symmetry in the base frame after the reference landing."""
    if not math.isfinite(std) or std <= 0.0:
        raise ValueError("Wheel symmetry reward requires a finite, positive standard deviation.")
    blend = _wheel_phase_geometry_weight(env, command_name, asset_cfg, start_time_s, ramp_time_s)
    return blend * _locomotion_leg_symmetry(env, std=std, asset_cfg=asset_cfg)


def wheel_phase_same_feet_x_position(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    start_time_s: float,
    ramp_time_s: float,
) -> torch.Tensor:
    """Penalize base-frame wheel fore-aft separation with a negative reward weight."""
    blend = _wheel_phase_geometry_weight(env, command_name, asset_cfg, start_time_s, ramp_time_s)
    return blend * _locomotion_same_feet_x_position(env, asset_cfg=asset_cfg)


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

def soft_torque_limit_penalty(
    env: ManagerBasedRLEnv, 
    ratio: float, 
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """
    惩罚超过指定比例 (ratio) 的关节力矩。
    
    例如：ratio=0.9 时，只有当力矩超过最大额定力矩的 90% 时，才会触发惩罚。
    超出越多，惩罚越大。
    """
    asset: Articulation = env.scene[asset_cfg.name]
    
    # Explicit actuators use a large solver limit to avoid double-clipping.
    # Resolve their actual limits in native joint order before selecting joints.
    effort_limits = asset.data.joint_effort_limits.clone()
    for actuator in asset.actuators.values():
        effort_limits[:, actuator.joint_indices] = actuator.effort_limit
    
    threshold = effort_limits[:, asset_cfg.joint_ids] * ratio
    
    torques = asset.data.applied_torque[:, asset_cfg.joint_ids]
    
    out_of_limits = torch.relu(torch.abs(torques) - threshold)
    
    return torch.sum(out_of_limits, dim=1)


def joint_torque_margin_penalty(
    env: ManagerBasedRLEnv,
    ratio: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize applied torque near its limit and unclipped requests above it.

    Normalize by the actual actuator limits, rather than explicit actuators'
    large solver limits. The applied-torque cost rises from zero at ``ratio``
    to one at the limit; the request cost is squared relative excess above the
    limit. Sum both dimensionless costs over the selected joints.

    This term reads the current control-step state; it does not accumulate
    physics-step peaks or change actuator clipping.
    """
    if not math.isfinite(ratio) or not 0.0 <= ratio < 1.0:
        raise ValueError("Torque margin ratio must be finite and in [0, 1).")

    asset: Articulation = env.scene[asset_cfg.name]
    effort_limits = asset.data.joint_effort_limits.clone()
    for actuator in asset.actuators.values():
        effort_limits[:, actuator.joint_indices] = actuator.effort_limit
    limits = effort_limits[:, asset_cfg.joint_ids]

    applied_ratio = torch.abs(asset.data.applied_torque[:, asset_cfg.joint_ids]) / limits
    requested_ratio = torch.abs(asset.data.computed_torque[:, asset_cfg.joint_ids]) / limits
    margin_cost = (torch.relu(applied_ratio - ratio) / (1.0 - ratio)).square()
    request_cost = torch.relu(requested_ratio - 1.0).square()
    return torch.sum(margin_cost + request_cost, dim=1)


def joint_power(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Sum absolute mechanical joint power over the selected joints."""
    asset: Articulation = env.scene[asset_cfg.name]
    torque = asset.data.applied_torque[:, asset_cfg.joint_ids]
    velocity = asset.data.joint_vel[:, asset_cfg.joint_ids]
    return torch.sum(torch.abs(torque * velocity), dim=1)


def joint_pos_zero_l2(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Sum squared joint angles in radians relative to physical zero."""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.joint_pos[:, asset_cfg.joint_ids]), dim=1)
