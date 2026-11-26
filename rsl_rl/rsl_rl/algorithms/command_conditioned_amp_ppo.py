# SPDX-License-Identifier: BSD-3-Clause
"""AMP with one velocity-conditioned discriminator and locally sampled experts."""

import torch

from .amp_ppo import AMPPPO


class CommandConditionedAMPPPO(AMPPPO):
    amp_command_dim = 2

    def init_storage(self, training_type, num_envs, num_transitions_per_env, obs, actions_shape):
        history = self.amp_data.history_length
        if obs["amp"].shape != (num_envs, history, self.amp_data.frame_dim):
            raise ValueError("Expected oldest-first AMP observations shaped [env, history, frame].")
        if "amp_command" not in obs or obs["amp_command"].shape != (num_envs, 3):
            raise ValueError("Expected an unscaled [env, 3] base velocity in the amp_command observation group.")
        super().init_storage(training_type, num_envs, num_transitions_per_env, obs, actions_shape)
        self._step_command = torch.zeros(num_envs, self.amp_command_dim, device=self.device)
        self._command_steps = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self._valid_transitions = 0

    def act(self, obs, amp_obs=None):
        raw = obs["amp_command"]
        command = raw[:, (0, 2)]
        if not ((raw[:, 1] == 0) & self.amp_data.covers(command)).all():
            raise ValueError("The environment command must lie in the finite expert vx/wz grid with vy=0.")
        changed = (command != self._step_command).any(dim=1)
        self._command_steps.masked_fill_(changed, 0).add_(1)
        # Isaac Lab may resample a command or reset the environment during step().
        self._step_command.copy_(command)
        return super().act(obs, amp_obs)

    def _record_amp_step(self, rewards, amp_obs, valid, dones):
        # Both windows span H+1 physical frames. Discard mixed or reset-padded histories.
        valid = valid & (self._command_steps >= self.amp_data.history_length + 1)
        self._command_steps.masked_fill_(dones.reshape(-1).bool(), 0)
        total = rewards * self.discriminator.task_reward_lerp
        if valid.any():
            state = self.amp_transition.observations[valid]
            next_state = amp_obs[valid]
            command = self._step_command[valid]
            self.amp_storage.insert(state, next_state, command)
            self._valid_transitions += len(command)
            mixed, _ = self.discriminator.predict_amp_reward(
                state, next_state, rewards[valid], normalizer=self.amp_normalizer, command=command
            )
            total[valid] = mixed
        return total

    def _amp_batches(self, count, size):
        for state, next_state, command in self.amp_storage.feed_forward_generator(count, size):
            expert = self.amp_data.sample_for_commands(command)
            # Both labels are the query, including when the chosen expert is at a neighboring grid point.
            yield (state, next_state), expert, command

    def update(self):
        transitions = self.storage.step * self.storage.num_envs
        stats = super().update()
        stats["amp/valid_window_fraction"] = self._valid_transitions / transitions
        stats["amp/replay_samples"] = self.amp_storage.num_samples
        self._valid_transitions = 0
        return stats
