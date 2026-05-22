# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

import os

from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg

from robot_lab.assets.LW import LW_WHEEL_CFG  
from robot_lab.tasks.manager_based.beyondmimic.tracking_env_cfg import TrackingEnvCfg
import robot_lab.tasks.manager_based.beyondmimic.mdp as mdp

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
class LWWheelBeyondMimicFlatEnvCfg(TrackingEnvCfg):

    actions: LWActionsCfg = LWActionsCfg()

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
        self.scene.robot = LW_WHEEL_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # observations
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
        # rewards
        self.rewards.action_rate_l2.weight = -0.02
        self.rewards.action_smoothness.weight = -0.02
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = ["base_link", ".*hip_link", ".*thigh_link",".*shank_link"]
        self.rewards.joint_limit.params["asset_cfg"].joint_names = self.joint_names_without_wheels
        self.rewards.torque_limit.weight = -0.1
        self.rewards.joint_vel_wheel_l2.weight = -5e-2 # -5e-3
        self.rewards.joint_vel_wheel_l2.params["asset_cfg"].joint_names = self.wheel_joint_names
        self.rewards.joint_acc_wheel_l2.weight = -1.5e-5
        self.rewards.joint_acc_wheel_l2.params["asset_cfg"].joint_names = self.wheel_joint_names
        # terminations
        self.terminations.ee_body_pos.params["body_names"] = [self.foot_link_name]
        # commands
        self.commands.motion.motion_file = "source/robot_lab/robot_lab/datasets/LW/motion_beyondmimic/wheel_to_leg_transform_60hz.npz"
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
