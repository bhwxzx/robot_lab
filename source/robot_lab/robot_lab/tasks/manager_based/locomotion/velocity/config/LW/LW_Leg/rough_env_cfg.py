import math

from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab.utils.noise import AdditiveGaussianNoiseCfg as GaussianNoise
from isaaclab.sensors import patterns

import robot_lab.tasks.manager_based.locomotion.velocity.mdp as mdp
from robot_lab.tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg

##
# Pre-defined configs
##
from robot_lab.assets.LW import LW_LEG_CFG
from robot_lab.tasks.manager_based.locomotion.velocity.mdp.terrains.terrains_cfg import BLIND_HARD_ROUGH_TERRAINS_CFG, BLIND_ROUGH_AND_STAIRS_TERRAINS_CFG
import robot_lab.tasks.manager_based.locomotion.velocity.mdp as mdp
from robot_lab.tasks.manager_based.locomotion.velocity.velocity_env_cfg import (
    ActionsCfg,
    LocomotionVelocityRoughEnvCfg,
    CommandsCfg,
    ObservationsCfg,
    RewardsCfg,
    CurriculumCfg
)

@configclass
class LWLegObservationsCfg(ObservationsCfg):
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            noise=Unoise(n_min=-0.2, n_max=0.2),
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"},
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*", preserve_order=True)},
            noise=Unoise(n_min=-0.01, n_max=0.01),
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*", preserve_order=True)},
            noise=Unoise(n_min=-1.5, n_max=1.5),
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        actions = ObsTerm(
            func=mdp.last_action,
            clip=(-100.0, 100.0),
            scale=1.0,
        )

        gait_phase = ObsTerm(func=mdp.get_gait_phase)
        gait_command = ObsTerm(func=mdp.get_gait_command, params={"command_name": "gait_command"})

        # Privileged observation
        base_lin_vel = ObsTerm(
            func=mdp.base_lin_vel,
            scale=1.0,
            clip=(-100.0, 100.0),
        )
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            clip=(-1.0, 1.0),
            scale=1.0,
        )
        robot_joint_torque = ObsTerm(func=mdp.robot_joint_torque)
        robot_joint_acc = ObsTerm(func=mdp.robot_joint_acc)
        feet_lin_vel = ObsTerm(
            func=mdp.feet_lin_vel, params={"asset_cfg": SceneEntityCfg("robot", body_names=".*_foot_link")}
        )
        robot_mass = ObsTerm(func=mdp.robot_mass)
        robot_inertia = ObsTerm(func=mdp.robot_inertia)
        robot_joint_pos = ObsTerm(func=mdp.robot_joint_pos)
        robot_joint_stiffness = ObsTerm(func=mdp.robot_joint_stiffness)
        robot_joint_damping = ObsTerm(func=mdp.robot_joint_damping)
        robot_pos = ObsTerm(func=mdp.robot_pos)
        robot_vel = ObsTerm(func=mdp.robot_vel)
        robot_material_properties = ObsTerm(func=mdp.robot_material_properties)
        feet_contact_force = ObsTerm(
            func=mdp.robot_contact_force, params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot_link")}
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        """Observations for critic group."""

        # observation terms (order preserved)
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"},
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*", preserve_order=True)},
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*", preserve_order=True)},
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        actions = ObsTerm(
            func=mdp.last_action,
            clip=(-100.0, 100.0),
            scale=1.0,
        )

        gait_phase = ObsTerm(func=mdp.get_gait_phase)
        gait_command = ObsTerm(func=mdp.get_gait_command, params={"command_name": "gait_command"})

        # Privileged observation
        base_lin_vel = ObsTerm(
            func=mdp.base_lin_vel,
            scale=1.0,
            clip=(-100.0, 100.0),
        )
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            clip=(-1.0, 1.0),
            scale=1.0,
        )
        robot_joint_torque = ObsTerm(func=mdp.robot_joint_torque)
        robot_joint_acc = ObsTerm(func=mdp.robot_joint_acc)
        feet_lin_vel = ObsTerm(
            func=mdp.feet_lin_vel, params={"asset_cfg": SceneEntityCfg("robot", body_names=".*_foot_link")}
        )
        robot_mass = ObsTerm(func=mdp.robot_mass)
        robot_inertia = ObsTerm(func=mdp.robot_inertia)
        robot_joint_pos = ObsTerm(func=mdp.robot_joint_pos)
        robot_joint_stiffness = ObsTerm(func=mdp.robot_joint_stiffness)
        robot_joint_damping = ObsTerm(func=mdp.robot_joint_damping)
        robot_pos = ObsTerm(func=mdp.robot_pos)
        robot_vel = ObsTerm(func=mdp.robot_vel)
        robot_material_properties = ObsTerm(func=mdp.robot_material_properties)
        feet_contact_force = ObsTerm(
            func=mdp.robot_contact_force, params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot_link")}
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()

@configclass
class LWLegCommandsCfg(CommandsCfg):

    gait_command = mdp.UniformGaitCommandCfg(
        resampling_time_range=(5.0, 5.0),  # Fixed resampling time of 5 seconds
        debug_vis=False,  # No debug visualization needed
        ranges=mdp.UniformGaitCommandCfg.Ranges(
            frequencies=(0.8, 1.6), # (1.5, 2.5),  # Gait frequency range [Hz]
            offsets=(0.5, 0.5),  # Phase offset range [0-1]
            durations=(0.5, 0.5),  # Contact duration range [0-1]
            swing_height=(0.2, 0.2)
        ),
    )

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(3.0, 15.0), # (3.0, 15.0)
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=1.0,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0), lin_vel_y=(-0.5, 0.5), ang_vel_z=(-1.0, 1.0), heading=(-math.pi, math.pi)
        ),
    )

@configclass
class LWLegActionsCfg(ActionsCfg):

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=[".*"], scale=0.25, use_default_offset=True, clip=None, preserve_order=True
    )
    joint_vel = mdp.JointVelocityActionCfg(
        asset_name="robot", joint_names=[""], scale=1.0, use_default_offset=True, clip=None, preserve_order=True
    )

@configclass
class LWLegRewardsCfg(RewardsCfg):

    joint_vel_wheel_l2 = RewTerm(
        func=mdp.joint_vel_l2, weight=0.0, params={"asset_cfg": SceneEntityCfg("robot", joint_names="")}
    )

    joint_acc_wheel_l2 = RewTerm(
        func=mdp.joint_acc_l2, weight=0.0, params={"asset_cfg": SceneEntityCfg("robot", joint_names="")}
    )

    joint_torques_wheel_l2 = RewTerm(
        func=mdp.joint_torques_l2, weight=0.0, params={"asset_cfg": SceneEntityCfg("robot", joint_names="")}
    )

    rew_keep_ankle_pitch_zero_in_air = RewTerm(
        func=mdp.keep_ankle_pitch_zero_in_air, weight=0.0, params={"asset_cfg": SceneEntityCfg("robot"),
                                                                    "sensor_cfg": SceneEntityCfg("contact_forces", 
                                                                                  body_names=["right_foot_link", "left_foot_link"])}
    )

    bipedal_gait_reward = RewTerm(
        func=mdp.BipedalGaitReward,
        weight=0.0,
        params={
            "tracking_contacts_shaped_force": -2.0,
            "tracking_contacts_shaped_vel": -2.0,
            "gait_force_sigma": 25.0,
            "gait_vel_sigma": 0.25,
            "kappa_gait_probs": 0.05,
            "vel_command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot_link"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot_link"),
            "height_sensor_cfg": SceneEntityCfg("height_scanner_base"),
            "base_target_height": 0.6, # 0.65
            # "gait_param_cfg": [2.5, 0.5, 0.5]
            "command_name": "gait_command",
        },
    )

    feet_distance_penalize = RewTerm(
        func=mdp.feet_distance_penalize,
        weight=0.0,
        params={"min_feet_distance": 0.115,"feet_links_name": [".*_foot_link"]}
    )

    stop_motion = RewTerm(
        func=mdp.stop_motion,
        weight=0.0
    )

    action_smoothness = RewTerm(func=mdp.ActionSmoothnessPenalty, weight=0.0)

# @configclass
# class LWLegCurriculumCfg(CurriculumCfg):

#     action_rate = CurrTerm(
#         func=mdp.modify_reward_weight, params={"term_name": "action_smoothness", "weight": -0.1, "num_steps": 24*30000}
#     )

@configclass
class LWLegRoughTeacherEnvCfg(LocomotionVelocityRoughEnvCfg):

    observations: LWLegObservationsCfg = LWLegObservationsCfg()
    commands: LWLegCommandsCfg = LWLegCommandsCfg()
    actions: LWLegActionsCfg = LWLegActionsCfg()
    rewards: LWLegRewardsCfg = LWLegRewardsCfg()
    # curriculum: LWLegCurriculumCfg = LWLegCurriculumCfg()

    base_link_name = "base_link"
    foot_link_name = ".*_foot_link"
    # fmt: off
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
    # fmt: on

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # ------------------------------Sence------------------------------
        self.scene.robot = LW_LEG_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name
        # self.scene.height_scanner.pattern_cfg = patterns.GridPatternCfg(resolution=0.05, size=(0.8, 0.5)),
        self.scene.height_scanner_base.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name
        self.scene.terrain.terrain_generator = BLIND_HARD_ROUGH_TERRAINS_CFG
        # self.scene.terrain.max_init_terrain_level = 0

        # ------------------------------Observations------------------------------
        self.observations.policy.joint_pos.func = mdp.joint_pos_rel_without_wheel
        self.observations.policy.joint_pos.params["wheel_asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=self.wheel_joint_names
        )
        self.observations.critic.joint_pos.func = mdp.joint_pos_rel_without_wheel
        self.observations.critic.joint_pos.params["wheel_asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=self.wheel_joint_names
        )
        self.observations.policy.base_ang_vel.scale = 0.25
        self.observations.policy.joint_pos.scale = 1.0
        self.observations.policy.joint_vel.scale = 0.05
        self.observations.policy.joint_pos.params["asset_cfg"].joint_names = self.joint_names
        self.observations.policy.joint_vel.params["asset_cfg"].joint_names = self.joint_names
        self.observations.critic.base_ang_vel.scale = 0.25
        self.observations.critic.joint_pos.scale = 1.0
        self.observations.critic.joint_vel.scale = 0.05
        self.observations.critic.joint_pos.params["asset_cfg"].joint_names = self.joint_names
        self.observations.critic.joint_vel.params["asset_cfg"].joint_names = self.joint_names

        # for blind
        self.observations.policy.height_scan = None
        self.observations.critic.height_scan = None

        # ------------------------------Actions------------------------------
        # reduce action scale
        self.actions.joint_pos.scale = {".*_hip_joint": 0.125, "^(?!.*_hip_joint).*": 0.25}
        self.actions.joint_vel.scale = 1.0
        self.actions.joint_pos.clip = {".*": (-100.0, 100.0)}
        self.actions.joint_vel.clip = {".*": (-100.0, 100.0)}
        self.actions.joint_pos.joint_names = self.joint_names_without_wheels
        self.actions.joint_vel.joint_names = self.wheel_joint_names

        # ------------------------------Events------------------------------
        # self.events.randomize_rigid_body_material.params["static_friction_range"] = (0.4, 1.0)
        # self.events.randomize_rigid_body_material.params["dynamic_friction_range"] = (0.4, 0.8)
        # self.events.randomize_rigid_body_material.params["restitution_range"] = (0.0, 1.0)
        # self.events.randomize_rigid_body_material.params["num_buckets"] = 48

        self.events.randomize_rigid_body_mass_base.params["asset_cfg"].body_names = [self.base_link_name]
        self.events.randomize_rigid_body_mass_others.params["asset_cfg"].body_names = [
            "right_.*", "left_.*",
        ]

        self.events.randomize_com_positions.params["asset_cfg"].body_names = [self.base_link_name]
        self.events.randomize_reset_joints.func = mdp.reset_joints_by_offset
        self.events.randomize_reset_joints.params["position_range"] = (-0.1, 0.1)
        self.events.randomize_reset_joints.params["velocity_range"] = (-0.2, 0.2)
        self.events.randomize_push_robot.params["velocity_range"] = {"x": (-1.0, 1.0), "y": (-1.0, 1.0)}
        
        self.events.push_robot_hard = None
        self.events.randomize_apply_external_force_torque = None 

        # ------------------------------Rewards------------------------------
        # General
        # self.rewards.is_terminated.weight = -200.0
        # self.rewards.keep_alive = 1.0
        self.rewards.upward.weight = 1.0 # 1.0

        # Root penalties
        self.rewards.lin_vel_z_l2.weight = -1.0 #-1.0
        self.rewards.ang_vel_xy_l2.weight = -0.05 # -0.05
        self.rewards.flat_orientation_l2.weight = -2.5 # -5.0
        self.rewards.base_height_l2.weight = -50.0 # -50.0 
        self.rewards.base_height_l2.params["target_height"] = 0.69 # 0.647
        self.rewards.base_height_l2.params["asset_cfg"].body_names = [self.base_link_name]

        # Joint penalties
        self.rewards.joint_torques_l2.weight = -8e-5 # 1.25e-5
        self.rewards.joint_torques_l2.params["asset_cfg"].joint_names = self.joint_names_without_wheels
        self.rewards.joint_torques_wheel_l2.weight = -1.6e-4
        self.rewards.joint_torques_wheel_l2.params["asset_cfg"].joint_names = self.wheel_joint_names
        
        self.rewards.joint_acc_l2.weight = -1e-6 # -1.25e-7
        self.rewards.joint_acc_l2.params["asset_cfg"].joint_names = self.joint_names_without_wheels
        self.rewards.joint_acc_wheel_l2.weight = -1.5e-7 # -1.25e-7
        self.rewards.joint_acc_wheel_l2.params["asset_cfg"].joint_names = self.wheel_joint_names

        self.rewards.joint_vel_l2.weight = -5e-5
        self.rewards.joint_vel_l2.params["asset_cfg"].joint_names = self.joint_names_without_wheels
        self.rewards.joint_vel_wheel_l2.weight = -5e-3
        self.rewards.joint_vel_wheel_l2.params["asset_cfg"].joint_names = self.wheel_joint_names

        # self.rewards.joint_vel_limits.weight = -1.0
        # self.rewards.joint_vel_limits.params["asset_cfg"].joint_names = self.wheel_joint_names

        self.rewards.joint_pos_limits.weight = -2.0
        self.rewards.joint_pos_limits.params["asset_cfg"].joint_names = self.joint_names_without_wheels

        self.rewards.joint_power.weight = -2e-5

        self.rewards.stop_motion.weight = -5.0

        # self.rewards.stand_still.weight = -3.0
        # self.rewards.stop_motion.params["asset_cfg"].joint_names = self.joint_names_without_wheels

        self.rewards.joint_pos_penalty.weight = -0.2 # -2.0
        self.rewards.joint_pos_penalty.params["asset_cfg"].joint_names = self.joint_names_without_wheels
        self.rewards.joint_pos_penalty.params["stand_still_scale"] = 1.0

        # Action penalties
        self.rewards.action_rate_l2.weight = -0.01 # -0.01 
        self.rewards.action_smoothness.weight = -0.15 # -0.03 

        # Contact sensorstand_still
        self.rewards.undesired_contacts.weight = -1.0
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = ["base_link", ".*hip_link", ".*thigh_link",".*shank_link", ".*wheel_link"]
        # self.rewards.contact_forces.weight = 0
        # self.rewards.contact_forces.params["sensor_cfg"].body_names = [self.foot_link_name]

        # Velocity-tracking rewards
        self.rewards.track_lin_vel_xy_exp.weight = 5.0 # 3.0
        self.rewards.track_lin_vel_xy_exp.func = mdp.track_lin_vel_xy_yaw_frame_exp
        self.rewards.track_lin_vel_xy_exp.params["std"] = math.sqrt(0.2)
        self.rewards.track_ang_vel_z_exp.weight = 5.0
        self.rewards.track_ang_vel_z_exp.func = mdp.track_ang_vel_z_world_exp

        # Others
        # self.rewards.rew_keep_ankle_pitch_zero_in_air.weight = 0.5

        self.rewards.bipedal_gait_reward.weight = 2.0
        self.rewards.feet_air_time.weight = 1.0
        self.rewards.feet_air_time.func = mdp.feet_air_time_positive_biped
        self.rewards.feet_air_time.params["threshold"] = 0.4
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = [self.foot_link_name]
        # self.rewards.feet_air_only_one.weight = -40.0

        # self.rewards.feet_air_time_variance.weight = -30.0
        # self.rewards.feet_contact.weight = 0
        # self.rewards.feet_contact.params["sensor_cfg"].body_names = [self.foot_link_name]
        # self.rewards.feet_contact_without_cmd.weight = 0
        # self.rewards.feet_contact_without_cmd.params["sensor_c fg"].body_names = [self.foot_link_name]
        self.rewards.feet_stumble.weight = -1.0
        self.rewards.feet_stumble.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_slide.weight = -0.25
        self.rewards.feet_slide.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_slide.params["asset_cfg"].body_names = [self.foot_link_name]
        # self.rewards.feet_height.weight = 0
        # self.rewards.feet_height.params["target_height"] = 0.05
        # self.rewards.feet_height.params["asset_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_height_body.weight = -1.0
        self.rewards.feet_height_body.params["target_height"] = -0.4
        self.rewards.feet_height_body.params["asset_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_distance_y_exp.weight = 10.0
        self.rewards.feet_distance_y_exp.params["stance_width"] = 0.42 # 0.42
        self.rewards.feet_distance_y_exp.params["asset_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_distance_penalize.weight = -100.0
        self.rewards.feet_distance_penalize.params["min_feet_distance"] = 0.2

        # If the weight of rewards is 0, set rewards to None
        if self.__class__.__name__ == "LWLegRoughTeacherEnvCfg":
            self.disable_zero_weight_rewards()

        # ------------------------------Terminations------------------------------
        self.terminations.illegal_contact.params["sensor_cfg"].body_names = [self.base_link_name]

        # ------------------------------Curriculums------------------------------
        # self.curriculum.command_levels.params["range_multiplier"] = (0.2, 1.0)
        self.curriculum.command_levels_ang_vel = None
        self.curriculum.command_levels_lin_vel = None

        # ------------------------------Commands------------------------------
        self.commands.base_velocity.ranges.lin_vel_x = (-1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.5, 0.5)

@configclass
class LWLegRoughTeacherEnvCfg_Play(LWLegRoughTeacherEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # self.curriculum.terrain_levels = None
        self.commands.base_velocity.ranges.lin_vel_x = (-1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.5, 0.5)
        self.commands.gait_command.ranges.frequencies = (1.2, 1.2)
        self.commands.gait_command.ranges.swing_height = (0.2, 0.2)
        self.events.randomize_actuator_gains = None
        self.events.randomize_apply_external_force_torque = None
        self.events.push_robot_hard = None
        self.events.randomize_push_robot = None

        # If the weight of rewards is 0, set rewards to None
        if self.__class__.__name__ == "LWLegRoughTeacherEnvCfg_Play":
            self.disable_zero_weight_rewards()

@configclass
class LWLegRoughStudentObservationsCfg:

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            noise=Unoise(n_min=-0.2, n_max=0.2),
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"},
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*", preserve_order=True)},
            noise=Unoise(n_min=-0.01, n_max=0.01),
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*", preserve_order=True)},
            noise=Unoise(n_min=-1.5, n_max=1.5),
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        actions = ObsTerm(
            func=mdp.last_action,
            clip=(-100.0, 100.0),
            scale=1.0,
        )

        gait_phase = ObsTerm(func=mdp.get_gait_phase)
        gait_command = ObsTerm(func=mdp.get_gait_command, params={"command_name": "gait_command"})

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            self.history_length = 10
            self.flatten_history_dim = True

    @configclass
    class TeacherCfg(ObsGroup):

        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            clip=(-100.0, 100.0),
            noise=Unoise(n_min=-0.2, n_max=0.2),
            scale=1.0,
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"},
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*", preserve_order=True)},
            noise=Unoise(n_min=-0.01, n_max=0.01),
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*", preserve_order=True)},
            noise=Unoise(n_min=-1.5, n_max=1.5),
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        actions = ObsTerm(
            func=mdp.last_action,
            clip=(-100.0, 100.0),
            scale=1.0,
        )

        gait_phase = ObsTerm(func=mdp.get_gait_phase)
        gait_command = ObsTerm(func=mdp.get_gait_command, params={"command_name": "gait_command"})

        # Privileged observation
        base_lin_vel = ObsTerm(
            func=mdp.base_lin_vel,
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            clip=(-1.0, 1.0),
            scale=1.0,
        )
        robot_joint_torque = ObsTerm(func=mdp.robot_joint_torque)
        robot_joint_acc = ObsTerm(func=mdp.robot_joint_acc)
        feet_lin_vel = ObsTerm(
            func=mdp.feet_lin_vel, params={"asset_cfg": SceneEntityCfg("robot", body_names=".*_foot_link")}
        )
        robot_mass = ObsTerm(func=mdp.robot_mass)
        robot_inertia = ObsTerm(func=mdp.robot_inertia)
        robot_joint_pos = ObsTerm(func=mdp.robot_joint_pos)
        robot_joint_stiffness = ObsTerm(func=mdp.robot_joint_stiffness)
        robot_joint_damping = ObsTerm(func=mdp.robot_joint_damping)
        robot_pos = ObsTerm(func=mdp.robot_pos)
        robot_vel = ObsTerm(func=mdp.robot_vel)
        robot_material_properties = ObsTerm(func=mdp.robot_material_properties)
        feet_contact_force = ObsTerm(
            func=mdp.robot_contact_force, params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot_link")}
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True


    policy: PolicyCfg = PolicyCfg()
    teacher: TeacherCfg = TeacherCfg()


@configclass
class LWLegRoughStudentEnvCfg(LocomotionVelocityRoughEnvCfg):
    
    observations: LWLegRoughStudentObservationsCfg = LWLegRoughStudentObservationsCfg()
    commands: LWLegCommandsCfg = LWLegCommandsCfg()
    actions: LWLegActionsCfg = LWLegActionsCfg()
    rewards: LWLegRewardsCfg = LWLegRewardsCfg()

    base_link_name = "base_link"
    foot_link_name = ".*_foot_link"
    # fmt: off
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
    # fmt: on

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # ------------------------------Sence------------------------------
        self.scene.robot = LW_LEG_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name
        # self.scene.height_scanner.pattern_cfg = patterns.GridPatternCfg(resolution=0.05, size=(0.8, 0.5)),
        self.scene.height_scanner_base.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name
        self.scene.terrain.terrain_generator = BLIND_HARD_ROUGH_TERRAINS_CFG
        # self.scene.terrain.max_init_terrain_level = 0

        # ------------------------------Observations------------------------------
        self.observations.policy.joint_pos.func = mdp.joint_pos_rel_without_wheel
        self.observations.policy.joint_pos.params["wheel_asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=self.wheel_joint_names
        )
        self.observations.teacher.joint_pos.func = mdp.joint_pos_rel_without_wheel
        self.observations.teacher.joint_pos.params["wheel_asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=self.wheel_joint_names
        )
        self.observations.policy.base_ang_vel.scale = 0.25
        self.observations.policy.joint_pos.scale = 1.0
        self.observations.policy.joint_vel.scale = 0.05
        self.observations.policy.joint_pos.params["asset_cfg"].joint_names = self.joint_names
        self.observations.policy.joint_vel.params["asset_cfg"].joint_names = self.joint_names
        self.observations.teacher.base_ang_vel.scale = 0.25
        self.observations.teacher.joint_pos.scale = 1.0
        self.observations.teacher.joint_vel.scale = 0.05
        self.observations.teacher.joint_pos.params["asset_cfg"].joint_names = self.joint_names
        self.observations.teacher.joint_vel.params["asset_cfg"].joint_names = self.joint_names

        # for blind
        self.observations.teacher.height_scan = None

        # ------------------------------Actions------------------------------
        # reduce action scale
        self.actions.joint_pos.scale = {".*_hip_joint": 0.125, "^(?!.*_hip_joint).*": 0.25}
        self.actions.joint_vel.scale = 1.0
        self.actions.joint_pos.clip = {".*": (-100.0, 100.0)}
        self.actions.joint_vel.clip = {".*": (-100.0, 100.0)}
        self.actions.joint_pos.joint_names = self.joint_names_without_wheels
        self.actions.joint_vel.joint_names = self.wheel_joint_names

        # ------------------------------Events------------------------------
        # self.events.randomize_rigid_body_material.params["static_friction_range"] = (0.4, 1.0)
        # self.events.randomize_rigid_body_material.params["dynamic_friction_range"] = (0.4, 0.8)
        # self.events.randomize_rigid_body_material.params["restitution_range"] = (0.0, 1.0)
        # self.events.randomize_rigid_body_material.params["num_buckets"] = 48

        self.events.randomize_rigid_body_mass_base.params["asset_cfg"].body_names = [self.base_link_name]
        self.events.randomize_rigid_body_mass_others.params["asset_cfg"].body_names = [
            "right_.*", "left_.*",
        ]

        self.events.randomize_com_positions.params["asset_cfg"].body_names = [self.base_link_name]
        self.events.randomize_reset_joints.func = mdp.reset_joints_by_offset
        self.events.randomize_reset_joints.params["position_range"] = (-0.1, 0.1)
        self.events.randomize_reset_joints.params["velocity_range"] = (-0.2, 0.2)
        self.events.randomize_push_robot.params["velocity_range"] = {"x": (-1.0, 1.0), "y": (-1.0, 1.0)}
        self.events.push_robot_hard = None
        self.events.randomize_apply_external_force_torque = None

        # If the weight of rewards is 0, set rewards to None
        if self.__class__.__name__ == "LWLegRoughStudentEnvCfg":
            self.disable_zero_weight_rewards()

        # ------------------------------Terminations------------------------------
        self.terminations.illegal_contact.params["sensor_cfg"].body_names = [self.base_link_name]

        # ------------------------------Curriculums------------------------------
        # self.curriculum.command_levels.params["range_multiplier"] = (0.2, 1.0)
        self.curriculum.command_levels_ang_vel = None
        self.curriculum.command_levels_lin_vel = None

        # ------------------------------Commands------------------------------
        self.commands.base_velocity.ranges.lin_vel_x = (-1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.5, 0.5)

@configclass
class LWLegRoughStudentEnvCfg_Play(LWLegRoughStudentEnvCfg):

    def __post_init__(self):
        super().__post_init__()

        self.curriculum.terrain_levels = None
        self.commands.base_velocity.ranges.lin_vel_x = (-0.7, 0.7)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.5, 0.5)
        self.commands.gait_command.ranges.frequencies = (1.2, 1.2)
        self.commands.gait_command.ranges.swing_height = (0.2, 0.2)
        self.events.randomize_actuator_gains = None
        self.events.randomize_apply_external_force_torque = None

        # If the weight of rewards is 0, set rewards to None
        if self.__class__.__name__ == "LWLegRoughStudentEnvCfg_Play":
            self.disable_zero_weight_rewards()
