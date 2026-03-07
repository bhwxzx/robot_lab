# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Implementation of different RL agents."""

from .distillation import Distillation
from .ppo import PPO
from .dwaq_ppo import DWAQPPO
from .amp_ppo import AMPPPO
from .amp_dwaq_ppo import AMPDWAQPPO

__all__ = ["PPO", "Distillation", "DWAQPPO", "AMPPPO", "AMPDWAQPPO"]
