# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0
import math
from isaaclab.utils import configclass

from .rough_env_cfg import LWLegRoughNormalPPOEnvCfg, LWLegRoughDwaqEnvCfg, LWLegRoughAmpEnvCfg, LWLegRoughAmpDwaqEnvCfg
from robot_lab.tasks.manager_based.locomotion.velocity.mdp.terrains.terrains_cfg import (
    DWAQ_FLAT_TERRAINS_CFG,
)

@configclass
class LWLegFlatNormalPPOEnvCfg(LWLegRoughNormalPPOEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # change terrain to flat
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        # no height scan
        self.scene.height_scanner = None
        # self.scene.height_scanner_base = None
        self.observations.policy.height_scan = None
        self.observations.critic.height_scan = None
        # no terrain curriculum
        self.curriculum.terrain_levels = None

        # events
        self.events.randomize_reset_joints.params["position_range"] = (-0.2, 0.2)
        self.events.randomize_reset_joints.params["velocity_range"] = (-0.3, 0.3)
        self.events.randomize_push_robot.params["velocity_range"] = {"x": (-1.5, 1.5), "y": (-1.5, 1.5)}
        self.events.randomize_rigid_body_mass_base.params["mass_distribution_params"] = (-1.0, 3.0)
        self.events.randomize_com_positions.params["com_range"] = {"x": (-0.075, 0.075), "y": (-0.075, 0.075), "z": (-0.075, 0.075)}

        # Rewards
        self.rewards.base_height_l2.weight = -50.0
        self.rewards.track_lin_vel_xy_exp.weight = 3.5
        self.rewards.track_ang_vel_z_exp.weight = 4.5 
        self.rewards.ang_vel_xy_l2.weight = -0.1 # -0.05
        self.rewards.flat_orientation_l2.weight = -8.0  # -5.0
        self.rewards.stand_still.weight = 0.0
        self.rewards.joint_pos_penalty.weight = -1.0
        self.rewards.joint_pos_penalty.params["stand_still_scale"] = 6.0
        self.rewards.stop_motion.weight = -5.0 # -3.0
        self.rewards.action_rate_l2.weight = -0.05 
        self.rewards.action_smoothness.weight = -0.02
        # self.rewards.rew_keep_ankle_pitch_zero_in_air.weight = 0.5
        self.rewards.feet_air_time.weight = 2.0
        self.rewards.feet_air_time.params["threshold"] = 0.4
        self.rewards.feet_height_body.weight = 0.0
        self.rewards.bipedal_gait_reward.weight = 5.0
        self.rewards.feet_clearance.weight = 1.0
        self.rewards.feet_clearance.params["target_height"] = 0.15 + 0.071 # foot_radius
        self.rewards.feet_clearance.params["asset_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_landing_vel.weight = -0.5 # -0.15
        self.rewards.feet_landing_vel.params["asset_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_landing_vel.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_landing_vel.params["foot_radius"] = 0.071
        self.rewards.feet_landing_vel.params["about_landing_threshold"] = 0.08
        self.rewards.feet_stumble.weight = 0.0

        # If the weight of rewards is 0, set rewards to None
        if self.__class__.__name__ == "LWLegFlatNormalPPOEnvCfg":
            self.disable_zero_weight_rewards()

@configclass
class LWLegFlatNormalPPOEnvCfg_Play(LWLegFlatNormalPPOEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # self.curriculum.terrain_levels = None
        self.commands.base_velocity.ranges.lin_vel_x = (-1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.5, 0.5)
        self.events.randomize_actuator_gains = None
        self.events.randomize_apply_external_force_torque = None
        self.events.push_robot_hard = None
        self.events.randomize_push_robot = None

        # If the weight of rewards is 0, set rewards to None
        if self.__class__.__name__ == "LWLegFlatNormalPPOEnvCfg_Play":
            self.disable_zero_weight_rewards()

@configclass
class LWLegFlatDwaqEnvCfg(LWLegRoughDwaqEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # change terrain to flat
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        # no height scan
        self.scene.height_scanner = None
        # self.scene.height_scanner_base = None
        self.observations.policy.height_scan = None
        self.observations.critic.height_scan = None
        # no terrain curriculum
        self.curriculum.terrain_levels = None

        # events
        self.events.randomize_reset_joints.params["position_range"] = (-0.2, 0.2)
        self.events.randomize_reset_joints.params["velocity_range"] = (-0.3, 0.3)
        self.events.randomize_push_robot.params["velocity_range"] = {"x": (-1.5, 1.5), "y": (-1.5, 1.5)}
        self.events.randomize_rigid_body_mass_base.params["mass_distribution_params"] = (-1.0, 3.0)
        self.events.randomize_com_positions.params["com_range"] = {"x": (-0.075, 0.075), "y": (-0.075, 0.075), "z": (-0.075, 0.075)}
        self.events.randomize_actuator_gains.params["stiffness_distribution_params"] = (0.7, 1.3)
        self.events.randomize_actuator_gains.params["damping_distribution_params"] = (0.7, 1.3)
        self.events.randomize_rigid_body_mass_others.params["mass_distribution_params"] = (0.7, 1.3)

        # Rewards
        self.rewards.base_height_l2.weight = -10.0
        self.rewards.lin_vel_z_l2.weight = -1.0
        self.rewards.track_lin_vel_xy_exp.weight = 3.5
        self.rewards.track_ang_vel_z_exp.weight = 2.5
        self.rewards.lazy_penalty.weight = -2.0
        self.rewards.ang_vel_xy_l2.weight = -0.1 # -0.05
        self.rewards.undesired_contacts.weight = -5.0
        self.rewards.flat_orientation_l2.weight = -2.0  # -5.0
        self.rewards.body_orientation_l2.weight = -3.0
        self.rewards.stand_still.weight = -1.0
        self.rewards.joint_pos_penalty.weight = -0.0
        self.rewards.joint_pos_penalty.params["stand_still_scale"] = 1.0
        self.rewards.stop_motion.weight = -3.0 # -3.0
        self.rewards.action_rate_l2.weight = -0.10 
        self.rewards.action_smoothness.weight = -0.03
        self.rewards.rew_keep_ankle_pitch_zero_in_air.weight = 0.5
        self.rewards.feet_standing_force_without_cmd.weight = 0.0
        self.rewards.feet_air_time.weight = 2.0
        self.rewards.feet_air_time.params["threshold"] = 0.4
        self.rewards.feet_height_body.weight = 0.0
        self.rewards.bipedal_gait_reward.weight = 2.0
        self.rewards.feet_clearance.weight = 2.0
        self.rewards.feet_clearance.params["target_height"] = 0.15 + 0.071 # foot_radius
        self.rewards.feet_clearance.params["asset_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_landing_vel.weight = -0.25 # -0.15
        self.rewards.feet_landing_vel.params["asset_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_landing_vel.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_landing_vel.params["foot_radius"] = 0.071
        self.rewards.feet_landing_vel.params["about_landing_threshold"] = 0.08
        self.rewards.feet_stumble.weight = 0.0
        self.rewards.feet_height_body.weight = 0.0
        self.rewards.feet_distance_y_exp.weight = 6.0
        self.rewards.feet_distance_penalize.weight = -100.0
        self.rewards.ankle_torque_limit.weight = -0.2
        self.rewards.feet_impact_reduction.weight = -2.5e-2
        self.rewards.feet_impact_reduction.params["max_delta_v_sq"] = 2.0

        # If the weight of rewards is 0, set rewards to None
        if self.__class__.__name__ == "LWLegFlatDwaqEnvCfg":
            self.disable_zero_weight_rewards()

@configclass
class LWLegFlatDwaqEnvCfg_Play(LWLegFlatDwaqEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # self.curriculum.terrain_levels = None
        self.commands.base_velocity.ranges.lin_vel_x = (-1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.5, 0.5)
        self.events.randomize_actuator_gains = None
        self.events.randomize_apply_external_force_torque = None
        self.events.push_robot_hard = None
        self.events.randomize_push_robot = None

        # If the weight of rewards is 0, set rewards to None
        if self.__class__.__name__ == "LWLegFlatDwaqEnvCfg_Play":
            self.disable_zero_weight_rewards()

@configclass
class LWLegFlatAmpEnvCfg(LWLegRoughAmpEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # change terrain to flat
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        # no height scan
        self.scene.height_scanner = None
        # self.scene.height_scanner_base = None
        self.observations.policy.height_scan = None
        self.observations.critic.height_scan = None
        # no terrain curriculum
        self.curriculum.terrain_levels = None

        # events
        self.events.randomize_reset_joints.params["position_range"] = (-0.2, 0.2)
        self.events.randomize_reset_joints.params["velocity_range"] = (-0.3, 0.3)
        self.events.randomize_push_robot.params["velocity_range"] = {"x": (-1.5, 1.5), "y": (-1.5, 1.5)}
        self.events.randomize_rigid_body_mass_base.params["mass_distribution_params"] = (-1.0, 3.0)
        self.events.randomize_com_positions.params["com_range"] = {"x": (-0.075, 0.075), "y": (-0.075, 0.075), "z": (-0.075, 0.075)}

        # Rewards
        self.rewards.base_height_l2.weight = 0.0
        self.rewards.track_lin_vel_xy_exp.weight = 3.5
        self.rewards.track_ang_vel_z_exp.weight = 2.5
        self.rewards.ang_vel_xy_l2.weight = -0.1 # -0.05
        self.rewards.flat_orientation_l2.weight = -5.0  # -5.0
        self.rewards.stand_still.weight = 0.0
        self.rewards.joint_pos_penalty.weight = -1.0
        self.rewards.joint_pos_penalty.params["stand_still_scale"] = 1.0
        self.rewards.stop_motion.weight = -3.0 # -3.0
        self.rewards.action_rate_l2.weight = -0.02 
        self.rewards.action_smoothness.weight = -0.02
        self.rewards.undesired_contacts.weight = -1.0
        self.rewards.rew_keep_ankle_pitch_zero_in_air.weight = 0.0
        self.rewards.feet_standing_force_without_cmd.weight = 0.0
        self.rewards.feet_air_time.weight = 1.0
        self.rewards.feet_air_time.params["threshold"] = 0.4
        self.rewards.feet_height_body.weight = 0.0
        self.rewards.bipedal_gait_reward.weight = 0.0
        self.rewards.feet_clearance.weight = 0.0
        self.rewards.feet_clearance.params["target_height"] = 0.10 + 0.071 # foot_radius
        self.rewards.feet_clearance.params["asset_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_landing_vel.weight = 0.0 # -0.15
        self.rewards.feet_landing_vel.params["asset_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_landing_vel.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_landing_vel.params["foot_radius"] = 0.071
        self.rewards.feet_landing_vel.params["about_landing_threshold"] = 0.08
        self.rewards.feet_stumble.weight = 0.0
        self.rewards.lazy_penalty.weight = 0.0
        self.rewards.feet_distance_y_exp.weight = 0.0
        self.rewards.feet_distance_penalize.weight = 0.0

        # If the weight of rewards is 0, set rewards to None
        if self.__class__.__name__ == "LWLegFlatAmpEnvCfg":
            self.disable_zero_weight_rewards()

@configclass
class LWLegFlatAmpEnvCfg_Play(LWLegFlatAmpEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # self.curriculum.terrain_levels = None
        self.commands.base_velocity.ranges.lin_vel_x = (-1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.5, 0.5)
        self.events.randomize_actuator_gains = None
        self.events.randomize_apply_external_force_torque = None
        self.events.push_robot_hard = None
        self.events.randomize_push_robot = None

        # If the weight of rewards is 0, set rewards to None
        if self.__class__.__name__ == "LWLegFlatAmpEnvCfg_Play":
            self.disable_zero_weight_rewards()

@configclass
class LWLegFlatAmpDwaqEnvCfg(LWLegRoughAmpDwaqEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # change terrain to flat
        # self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = DWAQ_FLAT_TERRAINS_CFG
        # no height scan
        self.scene.height_scanner = None
        # self.scene.height_scanner_base = None
        self.observations.policy.height_scan = None
        self.observations.critic.height_scan = None
        # no terrain curriculum
        self.curriculum.terrain_levels = None

        # events
        self.events.randomize_reset_joints.params["position_range"] = (-0.2, 0.2)
        self.events.randomize_reset_joints.params["velocity_range"] = (-0.3, 0.3)
        self.events.randomize_push_robot.params["velocity_range"] = {"x": (-1.0, 1.0), "y": (-1.0, 1.0)}
        self.events.randomize_rigid_body_mass_base.params["mass_distribution_params"] = (-1.0, 3.0)
        # self.events.randomize_com_positions.params["com_range"] = {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (-0.05, 0.05)}
        # self.events.randomize_actuator_gains.params["distribution"] = "log_uniform"
        # self.events.randomize_actuator_gains.params["stiffness_distribution_params"] = (0.5, 2.0)
        # self.events.randomize_actuator_gains.params["damping_distribution_params"] = (0.5, 2.0)
        self.events.randomize_com_positions = None
        self.events.randomize_actuator_gains = None
        self.events.randomize_apply_external_force_torque = None
        self.events.randomize_rigid_body_mass_others.params["mass_distribution_params"] = (0.7, 1.3)

        # Rewards
        self.rewards.base_height_l2.weight = -10.0
        self.rewards.lin_vel_z_l2.weight = -2.0
        self.rewards.track_lin_vel_xy_exp.weight = 5.0
        self.rewards.track_lin_vel_xy_exp.params["std"] = math.sqrt(0.25)
        self.rewards.track_ang_vel_z_exp.weight = 3.0
        self.rewards.track_ang_vel_z_exp.params["std"] = math.sqrt(0.25)
        self.rewards.ang_vel_xy_l2.weight = -0.1 # -0.05
        self.rewards.flat_orientation_l2.weight = -5.0  # -5.0
        self.rewards.body_orientation_l2.weight = -3.0
        self.rewards.stand_still.weight = -1.0
        self.rewards.joint_pos_penalty.weight = -0.0
        self.rewards.joint_pos_penalty.params["stand_still_scale"] = 1.0
        self.rewards.stop_motion.weight = -3.0 # -3.0
        self.rewards.action_rate_l2.weight = -0.15  # -0.02
        self.rewards.action_smoothness.weight = -0.075 # -0.02
        self.rewards.undesired_contacts.weight = -5.0
        self.rewards.rew_keep_ankle_pitch_zero_in_air.weight = 0.0
        self.rewards.feet_standing_force_without_cmd.weight = 0.0
        self.rewards.feet_air_time.weight = 2.0
        self.rewards.feet_air_time.params["threshold"] = 0.4
        self.rewards.feet_air_time_variance.weight = -0.0
        self.rewards.feet_height_body.weight = 0.0
        self.rewards.bipedal_gait_reward.weight = 2.0
        self.rewards.feet_clearance.weight = 2.0
        self.rewards.feet_clearance.params["target_height"] = 0.125 + 0.071 # 0.125 + 0.071 foot_radius 
        self.rewards.feet_clearance.params["asset_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_landing_vel.weight = -0.25 # -0.15
        self.rewards.feet_landing_vel.params["asset_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_landing_vel.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_landing_vel.params["foot_radius"] = 0.071
        self.rewards.feet_landing_vel.params["about_landing_threshold"] = 0.08
        self.rewards.feet_stumble.weight = 0.0
        self.rewards.lazy_penalty.weight = -1.0
        self.rewards.feet_distance_y_exp.weight = 2.0
        self.rewards.feet_distance_penalize.weight = -100.0
        self.rewards.fly_penalty.weight = -0.0
        self.rewards.ankle_torque_limit.weight = -0.2
        self.rewards.penalize_hip_roll_action.weight = -1.0
        self.rewards.feet_impact_reduction.weight = -2.5e-3
        self.rewards.feet_impact_reduction.params["max_delta_v_sq"] = 2.0

        # If the weight of rewards is 0, set rewards to None
        if self.__class__.__name__ == "LWLegFlatAmpDwaqEnvCfg":
            self.disable_zero_weight_rewards()

@configclass
class LWLegFlatAmpDwaqEnvCfg_Play(LWLegFlatAmpDwaqEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.curriculum.terrain_levels = None
        self.commands.base_velocity.ranges.lin_vel_x = (-1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.5, 0.5)
        self.events.randomize_actuator_gains = None
        self.events.randomize_apply_external_force_torque = None
        self.events.push_robot_hard = None
        self.events.randomize_push_robot = None

        # If the weight of rewards is 0, set rewards to None
        if self.__class__.__name__ == "LWLegFlatAmpDwaqEnvCfg_Play":
            self.disable_zero_weight_rewards()