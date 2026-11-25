# SPDX-License-Identifier: Apache-2.0

"""Baseline environment for developing dynamic wheel-to-leg transitions."""

from isaaclab.utils import configclass

from .flat_env_cfg import LWWheelBeyondMimicFlatEnvCfg


@configclass
class LWWheelBeyondMimicDynamicEnvCfg(LWWheelBeyondMimicFlatEnvCfg):
    """Start from the existing reference-tracking task under a separate task ID.

    The initial baseline inherits the motion, observations, actions, rewards,
    resets and terminations of the flat wheel task. Velocity-conditioned
    transitions and the handoff to legged walking will be added here later.
    """
