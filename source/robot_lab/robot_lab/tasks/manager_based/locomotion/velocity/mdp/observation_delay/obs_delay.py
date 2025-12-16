import torch
from dataclasses import MISSING
from typing import Sequence

from omni.isaac.lab.managers import ModifierBase, ModifierTermCfg
from omni.isaac.lab.utils import configclass

@configclass
class RandomizedDelayModifierCfg(ModifierTermCfg):
    """Configuration for the randomized delay modifier."""
    class_type = None  # Will be assigned to RandomizedDelayModifier below
    
    min_delay: int = MISSING
    """The minimum delay in env steps (inclusive)."""
    
    max_delay: int = MISSING
    """The maximum delay in env steps (inclusive)."""


class RandomizedDelayModifier(ModifierBase):
    """Modifier that applies a randomized delay to the observations.
    
    It maintains a history buffer of observations and returns the observation
    corresponding to a randomly selected delay index for each environment.
    The delay is randomized per environment upon reset.
    """

    cfg: RandomizedDelayModifierCfg

    def __init__(self, cfg: RandomizedDelayModifierCfg, env, **kwargs):
        super().__init__(cfg, env, **kwargs)
        self.min_delay = cfg.min_delay
        self.max_delay = cfg.max_delay
        
        # Buffer to store history: [num_envs, max_delay + 1, obs_dim]
        # index 0 is the latest observation (t), index N is (t-N)
        self._obs_buffer = None
        
        # Store the current delay step for each environment: [num_envs]
        self._current_delays = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)

    def reset(self, env_ids: Sequence[int] = None):
        """Resets the modifier state for the specified environments."""
        if env_ids is None:
            env_ids = slice(None)
            num_reset = self._num_envs
        else:
            num_reset = len(env_ids)

        # 1. Randomize delay steps for the resetting environments
        # Range: [min_delay, max_delay]
        self._current_delays[env_ids] = torch.randint(
            self.min_delay, 
            self.max_delay + 1, 
            (num_reset,), 
            device=self._device
        )

        # 2. Clear history buffer for resetting environments to prevent ghosting
        # Note: We fill with zeros. In the first step after reset, the buffer 
        # will effectively have 0s for history. This is acceptable for RL robustness.
        if self._obs_buffer is not None:
            self._obs_buffer[env_ids] = 0.0

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        """Apply delay to the input data.
        
        Args:
            data: Current observation tensor of shape (num_envs, obs_dim).
        """
        # Initialize buffer on first call (since we need obs_dim)
        if self._obs_buffer is None:
            obs_dim = data.shape[-1]
            self._obs_buffer = torch.zeros(
                (self._num_envs, self.max_delay + 1, obs_dim),
                dtype=data.dtype,
                device=self._device
            )
            # Initialize buffer with current data to avoid cold start issues
            # Broadcasting: (N, 1, D) -> (N, T, D)
            self._obs_buffer[:] = data.unsqueeze(1)

        # 1. Update Buffer: Shift history to the right
        # Move [0...N-1] to [1...N]
        # This keeps index 0 free for the new data
        self._obs_buffer = torch.roll(self._obs_buffer, shifts=1, dims=1)
        
        # 2. Insert new data at index 0 (Newest)
        self._obs_buffer[:, 0, :] = data

        # 3. Gather data based on random delays
        # _current_delays shape: (N,)
        # We need to expand it to (N, 1, D) to gather along dim 1
        delay_indices = self._current_delays.view(-1, 1, 1).expand(-1, 1, data.shape[-1])
        
        # Gather: out[i][0][k] = input[i][delay_indices[i][0][k]][k]
        delayed_obs = torch.gather(self._obs_buffer, 1, delay_indices).squeeze(1)

        return delayed_obs

# Link the implementation to the config
RandomizedDelayModifierCfg.class_type = RandomizedDelayModifier