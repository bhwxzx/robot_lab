# Copyright (c) 2021-2024, The RSL-RL Project Developers.
# All rights reserved.
# Original code is licensed under the BSD-3-Clause license.
#
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# Copyright (c) 2025-2026, The Legged Lab Project Developers.
# All rights reserved.
#
# Copyright (c) 2025-2026, The TienKung-Lab Project Developers.
# All rights reserved.
# Modifications are licensed under the BSD-3-Clause license.
#
# This file contains code derived from the RSL-RL, Isaac Lab, and Legged Lab Projects,
# with additional modifications by the TienKung-Lab Project,
# and is distributed under the BSD-3-Clause license.

import torch


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, obs_dim, buffer_size, device, command_dim=0):
        """Initialize a ReplayBuffer object.
        Arguments:
            buffer_size (int): maximum size of buffer
        """
        if buffer_size < 1:
            raise ValueError("Replay buffer capacity must be positive.")
        self.states = torch.zeros(buffer_size, obs_dim, device=device)
        self.next_states = torch.zeros_like(self.states)
        self.commands = torch.zeros(buffer_size, command_dim, device=device) if command_dim else None
        self.buffer_size = buffer_size
        self.device = device

        self.step = 0
        self.num_samples = 0

    def insert(self, states, next_states, commands=None):
        """Add new states to memory."""

        n = states.shape[0]
        if states.shape != next_states.shape or states.shape[1:] != self.states.shape[1:]:
            raise ValueError("AMP replay requires equally shaped current and next observations.")
        fields = [(self.states, states), (self.next_states, next_states)]
        if self.commands is not None:
            if commands is None or commands.shape != (n, self.commands.shape[1]):
                raise ValueError("Expected one correctly sized command per AMP transition.")
            fields.append((self.commands, commands))
        elif commands is not None:
            raise ValueError("This replay buffer does not store commands.")

        # A single insert may exceed capacity. The most recent rows occupy the ring.
        count = min(n, self.buffer_size)
        start = (self.step + n - count) % self.buffer_size
        first = min(count, self.buffer_size - start)
        for buffer, values in fields:
            values = values[n - count :]
            buffer[start : start + first].copy_(values[:first])
            if first < count:
                buffer[: count - first].copy_(values[first:])
        self.num_samples = min(self.num_samples + n, self.buffer_size)
        self.step = (self.step + n) % self.buffer_size

    def feed_forward_generator(self, num_mini_batch, mini_batch_size):
        for _ in range(num_mini_batch):
            sample_idxs = torch.randint(
                self.num_samples,
                (mini_batch_size,),
                device=self.device,
            )
            pair = (self.states[sample_idxs], self.next_states[sample_idxs])
            if self.commands is None:
                yield pair
            else:
                yield (*pair, self.commands[sample_idxs])
