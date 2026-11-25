# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

import gymnasium as gym

from . import agents, dynamic_env_cfg, flat_env_cfg

##
# Register Gym environments.
##

gym.register(
    id="RobotLab-Isaac-BeyondMimic-Flat-LW-wheel-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:LWWheelBeyondMimicFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LWBeyondMimicFlatPPORunnerCfg",
    },
)

gym.register(
    id="RobotLab-Isaac-BeyondMimic-Flat-LW-wheel-dynamic-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.dynamic_env_cfg:LWWheelBeyondMimicDynamicEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LWBeyondMimicDynamicPPORunnerCfg",
    },
)
