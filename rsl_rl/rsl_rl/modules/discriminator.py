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
import torch.nn as nn
from torch import autograd


class Discriminator(nn.Module):
    """
    Discriminator neural network for adversarial motion priors (AMP) reward prediction.

    Args:
        input_dim (int): Dimension of the discriminator input feature vector.
        amp_reward_coef (float): Coefficient to scale the AMP reward.
        hidden_layer_sizes (list[int]): Sizes of hidden layers in the MLP trunk.
        device (torch.device): Device to run the model on (CPU or GPU).
        task_reward_lerp (float, optional): Interpolation factor between AMP reward and task reward.
            Defaults to 0.0 (only AMP reward).

    Attributes:
        trunk (nn.Sequential): MLP layers processing input features.
        amp_linear (nn.Linear): Final linear layer producing discriminator output.
        task_reward_lerp (float): Interpolation factor for combining rewards.
    """

    def __init__(
        self,
        input_dim,
        amp_reward_coef,
        hidden_layer_sizes,
        device,
        task_reward_lerp=0.0,
        dt=0.02,
        use_history_window=False,
    ):
        super().__init__()

        self.device = device
        self.dt = dt
        self.input_dim = input_dim
        self.use_history_window = use_history_window

        self.amp_reward_coef = amp_reward_coef
        amp_layers = []
        curr_in_dim = input_dim
        for hidden_dim in hidden_layer_sizes:
            amp_layers.append(nn.Linear(curr_in_dim, hidden_dim))
            amp_layers.append(nn.ReLU())
            curr_in_dim = hidden_dim
        self.trunk = nn.Sequential(*amp_layers).to(device)
        self.amp_linear = nn.Linear(hidden_layer_sizes[-1], 1).to(device)

        self.trunk.train()
        self.amp_linear.train()

        self.task_reward_lerp = task_reward_lerp

    def forward(self, x):
        """
        Forward pass through the discriminator network.

        Args:
            x (torch.Tensor): Input tensor with shape (batch_size, input_dim).

        Returns:
            torch.Tensor: Discriminator output logits with shape (batch_size, 1).
        """
        h = self.trunk(x)
        d = self.amp_linear(h)
        return d

    def prepare_input(self, state, next_state):
        """Build the discriminator input from an AMP transition.

        History-window mode uses the post-step window directly. Legacy AMP mode
        keeps the original concatenated state-transition representation.
        """
        if self.use_history_window:
            return next_state
        return torch.cat([state, next_state], dim=-1)

    def normalize_amp_observation(self, observation, normalizer):
        """Normalize a flattened AMP history with shared per-frame statistics.

        The normalizer dimension determines the frame size. This also keeps old
        checkpoints with a full-window normalizer compatible.
        """
        normalizer_dim = int(normalizer.mean.shape[0])
        if observation.shape[-1] % normalizer_dim != 0:
            raise ValueError(
                "AMP observation dimension must be divisible by the normalizer dimension: "
                f"got observation={observation.shape[-1]}, normalizer={normalizer_dim}."
            )
        original_shape = observation.shape
        frames = observation.reshape(-1, normalizer_dim)
        normalized_frames = normalizer.normalize_torch(frames, self.device)
        return normalized_frames.reshape(original_shape)

    def update_amp_normalizer(self, normalizer, observation):
        """Update shared per-frame normalization statistics from flattened histories."""
        normalizer_dim = int(normalizer.mean.shape[0])
        if observation.shape[-1] % normalizer_dim != 0:
            raise ValueError(
                "AMP observation dimension must be divisible by the normalizer dimension: "
                f"got observation={observation.shape[-1]}, normalizer={normalizer_dim}."
            )
        frames = observation.detach().reshape(-1, normalizer_dim)
        normalizer.update(frames.cpu().numpy())

    def compute_grad_pen(self, expert_state, expert_next_state, lambda_=10):
        """
        Compute gradient penalty for the expert data, used to regularize the discriminator.

        Args:
            expert_state (torch.Tensor): Batch of expert states.
            expert_next_state (torch.Tensor): Batch of expert next states.
            lambda_ (float, optional): Gradient penalty coefficient. Defaults to 10.

        Returns:
            torch.Tensor: Scalar gradient penalty loss.
        """
        expert_data = self.prepare_input(expert_state, expert_next_state).detach().clone()
        expert_data.requires_grad = True

        disc = self.amp_linear(self.trunk(expert_data))
        ones = torch.ones(disc.size(), device=disc.device)
        grad = autograd.grad(
            outputs=disc, inputs=expert_data, grad_outputs=ones, create_graph=True, retain_graph=True, only_inputs=True
        )[0]

        # Enforce that the grad norm approaches 0.
        grad_pen = lambda_ * (grad.norm(2, dim=1) - 0).pow(2).mean()
        return grad_pen

    def predict_amp_reward(self, state, next_state, task_reward, normalizer=None):
        """
        Predict the AMP reward given current and next states, optionally interpolated with a task reward.

        Args:
            state (torch.Tensor): Current state tensor.
            next_state (torch.Tensor): Next state tensor.
            task_reward (torch.Tensor): Task-specific reward tensor.
            normalizer (optional): Normalizer object to normalize input states before prediction.

        Returns:
            tuple:
                - reward (torch.Tensor): Predicted AMP reward (optionally interpolated) with shape (batch_size,).
                - d (torch.Tensor): Raw discriminator output logits with shape (batch_size, 1).
        """
        with torch.no_grad():
            self.eval()
            if normalizer is not None:
                next_state = self.normalize_amp_observation(next_state, normalizer)
                if not self.use_history_window:
                    state = self.normalize_amp_observation(state, normalizer)

            d = self.amp_linear(self.trunk(self.prepare_input(state, next_state)))
            reward = self.dt * self.amp_reward_coef * torch.clamp(1 - (1 / 4) * torch.square(d - 1), min=0)
            if self.task_reward_lerp > 0:
                reward = self._lerp_reward(reward, task_reward.unsqueeze(-1))
            self.train()
        return reward.squeeze(), d

    def _lerp_reward(self, disc_r, task_r):
        """
        Linearly interpolate between discriminator reward and task reward.

        Args:
            disc_r (torch.Tensor): Discriminator reward.
            task_r (torch.Tensor): Task reward.

        Returns:
            torch.Tensor: Interpolated reward.
        """
        r = (1.0 - self.task_reward_lerp) * disc_r + self.task_reward_lerp * task_r
        return r
