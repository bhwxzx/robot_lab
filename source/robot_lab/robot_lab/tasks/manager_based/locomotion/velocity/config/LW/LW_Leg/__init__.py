# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="RobotLab-Isaac-Velocity-Rough-LW-leg-teacher-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:LWLegRoughTeacherEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_distil_cfg:LWRoughPPORunnerCfg",
    },
)

gym.register(
    id="RobotLab-Isaac-Velocity-Rough-LW-leg-teacher-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:LWLegRoughTeacherEnvCfg_Play",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_distil_cfg:LWRoughPPORunnerCfg",
    },
)

gym.register(
    id="RobotLab-Isaac-Velocity-Rough-LW-leg-student-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:LWLegRoughStudentEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_distil_cfg:LWRoughDistillationRunnerCfg",
    },
)

gym.register(
    id="RobotLab-Isaac-Velocity-Rough-LW-leg-student-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:LWLegRoughStudentEnvCfg_Play",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_distil_cfg:LWRoughDistillationRunnerCfg",
    },
)

gym.register(
    id="RobotLab-Isaac-Velocity-Rough-LW-leg-normal-ppo-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:LWLegRoughNormalPPOEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LWRoughNormalPPORunnerCfg",
    },
)

gym.register(
    id="RobotLab-Isaac-Velocity-Rough-LW-leg-normal-ppo-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:LWLegRoughNormalPPOEnvCfg_Play",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LWRoughNormalPPORunnerCfg",
    },
)

gym.register(
    id="RobotLab-Isaac-Velocity-Flat-LW-leg-normal-ppo-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:LWLegFlatNormalPPOEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LWFlatNormalPPORunnerCfg",
    },
)

gym.register(
    id="RobotLab-Isaac-Velocity-Flat-LW-leg-normal-ppo-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:LWLegFlatNormalPPOEnvCfg_Play",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LWFlatNormalPPORunnerCfg",
    },
)

gym.register(
    id="RobotLab-Isaac-Velocity-Rough-LW-leg-Dwaq-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:LWLegRoughDwaqEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_dwaq_cfg:LWRoughDwaqRunnerCfg",
    },
)

gym.register(
    id="RobotLab-Isaac-Velocity-Rough-LW-leg-Dwaq-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:LWLegRoughDwaqEnvCfg_Play",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_dwaq_cfg:LWRoughDwaqRunnerCfg",
    },
)

gym.register(
    id="RobotLab-Isaac-Velocity-Flat-LW-leg-Dwaq-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:LWLegFlatDwaqEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_dwaq_cfg:LWFlatDwaqRunnerCfg",
    },
)

gym.register(
    id="RobotLab-Isaac-Velocity-Flat-LW-leg-Dwaq-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:LWLegFlatDwaqEnvCfg_Play",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_dwaq_cfg:LWFlatDwaqRunnerCfg",
    },
)

gym.register(
    id="RobotLab-Isaac-Velocity-Rough-LW-leg-Amp-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:LWLegRoughAmpEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_amp_cfg:LWRoughAmpRunnerCfg",
    },
)

gym.register(
    id="RobotLab-Isaac-Velocity-Rough-LW-leg-Amp-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:LWLegRoughAmpEnvCfg_Play",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_amp_cfg:LWRoughAmpRunnerCfg",
    },
)

gym.register(
    id="RobotLab-Isaac-Velocity-Flat-LW-leg-Amp-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:LWLegFlatAmpEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_amp_cfg:LWFlatAmpRunnerCfg",
    },
)

gym.register(
    id="RobotLab-Isaac-Velocity-Flat-LW-leg-Amp-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:LWLegFlatAmpEnvCfg_Play",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_amp_cfg:LWFlatAmpRunnerCfg",
    },
)

gym.register(
    id="RobotLab-Isaac-Velocity-Rough-LW-leg-Amp-Dwaq-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:LWLegRoughAmpDwaqEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_amp_dwaq_cfg:LWRoughAmpDwaqRunnerCfg",
    },
)

gym.register(
    id="RobotLab-Isaac-Velocity-Rough-LW-leg-Amp-Dwaq-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:LWLegRoughAmpDwaqEnvCfg_Play",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_amp_dwaq_cfg:LWRoughAmpDwaqRunnerCfg",
    },
)

gym.register(
    id="RobotLab-Isaac-Velocity-Flat-LW-leg-Amp-Dwaq-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:LWLegFlatAmpDwaqEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_amp_dwaq_cfg:LWFlatAmpDwaqRunnerCfg",
    },
)

gym.register(
    id="RobotLab-Isaac-Velocity-Flat-LW-leg-Amp-Dwaq-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:LWLegFlatAmpDwaqEnvCfg_Play",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_amp_dwaq_cfg:LWFlatAmpDwaqRunnerCfg",
    },
)



