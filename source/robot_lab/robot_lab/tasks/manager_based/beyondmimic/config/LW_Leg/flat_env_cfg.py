# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

import os

from isaaclab.utils import configclass
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg

from robot_lab.assets.LW import LW_LEG_CFG  
from robot_lab.tasks.manager_based.beyondmimic.tracking_env_cfg import RewardsCfg, TrackingEnvCfg
import robot_lab.tasks.manager_based.beyondmimic.mdp as mdp
from robot_lab.tasks.manager_based.locomotion.velocity.mdp.noise import UniformJointPositionNoiseCfg

@configclass
class LWActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=[".*"], scale=0.25, use_default_offset=True, clip=None, preserve_order=True
    )
    joint_vel = mdp.JointVelocityActionCfg(
        asset_name="robot", joint_names=[""], scale=1.0, use_default_offset=True, clip=None, preserve_order=True
    )

@configclass
class LWLegRewardsCfg(RewardsCfg):
    """Separate effort penalties for the LW Leg foot and shank joints."""

    hip_pos_zero_l2 = RewTerm(
        func=mdp.joint_pos_zero_l2,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["right_hip_joint", "left_hip_joint"])},
    )

    joint_torques_l2 = None
    joint_torques_foot_l2 = RewTerm(
        func=mdp.joint_torques_l2,
        weight=0.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["right_foot_joint", "left_foot_joint"])},
    )
    joint_torques_shank_l2 = RewTerm(
        func=mdp.joint_torques_l2,
        weight=0.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["right_shank_joint", "left_shank_joint"])},
    )
    torque_limit_foot = RewTerm(
        func=mdp.joint_torque_margin_penalty,
        weight=0.0,
        params={
            "ratio": 0.90,
            "asset_cfg": SceneEntityCfg("robot", joint_names=["right_foot_joint", "left_foot_joint"]),
        },
    )
    torque_limit_shank = RewTerm(
        func=mdp.joint_torque_margin_penalty,
        weight=0.0,
        params={
            "ratio": 0.95,
            "asset_cfg": SceneEntityCfg("robot", joint_names=["right_shank_joint", "left_shank_joint"]),
        },
    )


@configclass
class LWLegBeyondMimicFlatEnvCfg(TrackingEnvCfg):

    actions: LWActionsCfg = LWActionsCfg()
    rewards: LWLegRewardsCfg = LWLegRewardsCfg()

    base_link_name = "base_link"
    foot_link_name = ".*_foot_link"
    joint_names_without_wheels = [
        "right_hip_joint",
        "left_hip_joint",
        "right_thigh_joint",
        "left_thigh_joint",
        "right_shank_joint",
        "left_shank_joint",
        "right_foot_joint",
        "left_foot_joint",
    ]
    wheel_joint_names = [
        "right_wheel_joint",
        "left_wheel_joint",
    ]
    joint_names = joint_names_without_wheels + wheel_joint_names

    def __post_init__(self):
        super().__post_init__()
        # scene
        self.scene.robot = LW_LEG_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # observations
        self.observations.policy.history_length = 10
        self.observations.policy.flatten_history_dim = True
        self.observations.policy.motion_anchor_pos_b = None
        self.observations.policy.base_lin_vel = None
        self.observations.policy.joint_pos.func = mdp.joint_pos_rel_without_wheel
        self.observations.policy.joint_pos.params["wheel_asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=self.wheel_joint_names
        )
        self.observations.critic.joint_pos.func = mdp.joint_pos_rel_without_wheel
        self.observations.critic.joint_pos.params["wheel_asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=self.wheel_joint_names
        )
        self.observations.policy.joint_pos.params["asset_cfg"].joint_names = self.joint_names
        self.observations.policy.joint_pos.noise = UniformJointPositionNoiseCfg(
            n_min=-0.01, n_max=0.01,
            joint_names=self.joint_names,
            wheel_joint_names=self.wheel_joint_names,
        )
        self.observations.policy.joint_vel.params["asset_cfg"].joint_names = self.joint_names
        self.observations.critic.joint_pos.params["asset_cfg"].joint_names = self.joint_names
        self.observations.critic.joint_vel.params["asset_cfg"].joint_names = self.joint_names
        # actions
        self.actions.joint_pos.scale = 0.25
        self.actions.joint_vel.scale = 1.0
        self.actions.joint_pos.clip = {".*": (-100.0, 100.0)}
        self.actions.joint_vel.clip = {".*": (-100.0, 100.0)}
        self.actions.joint_pos.joint_names = self.joint_names_without_wheels
        self.actions.joint_vel.joint_names = self.wheel_joint_names
        # events
        self.events.add_joint_default_pos.params["asset_cfg"].joint_names = self.joint_names_without_wheels
        self.events.base_com.params["asset_cfg"].body_names = self.base_link_name
        self.events.randomize_rigid_body_mass_base.params["asset_cfg"].body_names = [self.base_link_name]
        # Keep link mass scaling from overwriting the base payload randomization.
        self.events.randomize_rigid_body_mass_others.params["asset_cfg"].body_names = [
            "right_hip_link",
            "left_hip_link",
            "right_thigh_link",
            "left_thigh_link",
            "right_shank_link",
            "left_shank_link",
            "right_foot_link",
            "left_foot_link",
            "right_wheel_link",
            "left_wheel_link",
        ]
        self.events.push_robot = None
        # self.events.randomize_rigid_body_mass_others = None
        # self.events.randomize_actuator_gains = None
        # rewards
        # Limit excess takeoff rotation before liftoff and during flight.
        # Keep the old 50 ms post-landing offset for the new 1.6 s reference landing.
        # Reference-time ramps: 1.0-1.1 s entry, 1.1-1.55 s full, 1.55-1.65 s exit.
        self.rewards.motion_takeoff_pitch_ang_vel.weight = -0.03
        self.rewards.motion_takeoff_pitch_ang_vel.params["start_time_s"] = 1.0
        self.rewards.motion_takeoff_pitch_ang_vel.params["end_time_s"] = 1.65
        self.rewards.motion_takeoff_pitch_ang_vel.params["ramp_time_s"] = 0.1
        self.rewards.motion_takeoff_anchor_ori.weight = 0.5
        self.rewards.motion_takeoff_anchor_ori.params["std"] = 0.3
        self.rewards.motion_takeoff_anchor_ori.params["start_time_s"] = 1.0
        self.rewards.motion_takeoff_anchor_ori.params["end_time_s"] = 1.65
        self.rewards.motion_takeoff_anchor_ori.params["ramp_time_s"] = 0.1
        self.rewards.action_rate_l2.weight = -0.2
        self.rewards.action_smoothness.weight = -0.075
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = ["base_link", ".*hip_link", ".*thigh_link",".*shank_link"]
        self.rewards.joint_limit.params["asset_cfg"].joint_names = self.joint_names_without_wheels
        self.rewards.torque_limit.weight = -0.0
        self.rewards.joint_power.weight = 0.0
        self.rewards.joint_power.params["asset_cfg"].joint_names = [
            "right_foot_joint", "left_foot_joint", "right_shank_joint", "left_shank_joint"
        ]
        self.rewards.joint_vel_wheel_l2.weight = 0.0
        self.rewards.joint_vel_wheel_l2.params["asset_cfg"].joint_names = self.wheel_joint_names
        self.rewards.joint_acc_wheel_l2.weight = 0.0
        self.rewards.joint_acc_wheel_l2.params["asset_cfg"].joint_names = self.wheel_joint_names
        # Start wheel rewards 50 ms after reference landing; at 50 Hz this first activates at 1.66 s.
        self.rewards.wheel_contact_continuous.weight = 0.5
        self.rewards.wheel_contact_continuous.params["sensor_cfg"].body_names = ["right_wheel_link", "left_wheel_link"]
        self.rewards.wheel_contact_continuous.params["start_time_s"] = 1.65
        self.rewards.wheel_contact_continuous.params["min_contact_force"] = 10.0
        self.rewards.wheel_contact_continuous.params["stable_contact_time"] = 0.1
        # Blend wheel geometry constraints in after landing, independently of current contact.
        self.rewards.leg_symmetry.weight = 0.0
        self.rewards.leg_symmetry.params["asset_cfg"].body_names = ["right_wheel_link", "left_wheel_link"]
        self.rewards.leg_symmetry.params["std"] = 0.05
        self.rewards.leg_symmetry.params["start_time_s"] = 1.65
        self.rewards.leg_symmetry.params["ramp_time_s"] = 0.2
        self.rewards.same_foot_x_position.weight = -10.0
        self.rewards.same_foot_x_position.params["asset_cfg"].body_names = ["right_wheel_link", "left_wheel_link"]
        self.rewards.same_foot_x_position.params["start_time_s"] = 1.65
        self.rewards.same_foot_x_position.params["ramp_time_s"] = 0.2
        # terminations
        self.terminations.ee_body_pos.params["body_names"] = ["right_foot_link", "left_foot_link"]
        # commands
        self.commands.motion.initialize_reset_targets = True
        self.commands.motion.motion_file = "source/robot_lab/robot_lab/datasets/LW/motion_beyondmimic/se3_trajopt/2026-10-08/leg_to_wheel_transform_50hz.npz"
        self.commands.motion.anchor_body_name = "base_link"
        self.commands.motion.body_names = [ # 需要追踪的连杆
            "base_link",
            "right_hip_link",
            "left_hip_link",
            "right_thigh_link",
            "left_thigh_link",
            "right_shank_link",
            "left_shank_link",
            "right_foot_link",
            "left_foot_link"
        ]
        self.commands.motion.joint_names = self.joint_names # 指定command的关节顺序，但不是都必须要追踪的关节

        self.episode_length_s = 10.0
