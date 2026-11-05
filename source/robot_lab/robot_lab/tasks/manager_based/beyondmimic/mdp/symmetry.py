"""Pure tensor sagittal reflection for the LW BeyondMimic observation contract.

No simulator imports. Policy histories are concatenated
per term, oldest first; they are NOT 10 concatenated 59-dimensional frames.
Joint values use the explicit task order below, not articulation storage order.
Relative joint positions/actions assume the default pose is reflected along with
the robot. For randomized defaults this describes a mirrored environment, not
an action conversion within the same asymmetric environment.
"""

from __future__ import annotations

from functools import lru_cache
from weakref import WeakSet

import torch
from tensordict import TensorDict


JOINT_NAMES = tuple(f"{side}_{part}_joint" for part in ("hip", "thigh", "shank", "foot", "wheel")
                    for side in ("right", "left"))
BODY_NAMES = ("base_link",) + tuple(f"{side}_{part}_link" for part in ("hip", "thigh", "shank", "foot")
                                  for side in ("right", "left"))
POLICY_TERMS = (("command", 20), ("motion_anchor_ori_b", 6), ("base_ang_vel", 3),
                ("joint_pos", 10), ("joint_vel", 10), ("actions", 10))
CRITIC_TERMS = (("command", 20), ("motion_anchor_pos_b", 3), ("motion_anchor_ori_b", 6),
                ("body_pos", 27), ("body_ori", 54), ("base_lin_vel", 3), ("base_ang_vel", 3),
                ("joint_pos", 10), ("joint_vel", 10), ("actions", 10))


def _check(x: torch.Tensor, width: int) -> None:
    if not isinstance(x, torch.Tensor) or x.ndim < 1 or x.shape[-1] != width:
        raise ValueError(f"Expected a tensor with final dimension {width}.")
    if not x.is_floating_point() or not torch.isfinite(x).all():
        raise ValueError("Reflection requires finite floating-point inputs.")


def mirror_joints(x: torch.Tensor, names: tuple[str, ...] = JOINT_NAMES) -> torch.Tensor:
    """Swap each named left/right pair and negate, as required by LW's axes."""
    if len(names) != 10 or len(set(names)) != 10 or set(names) != set(JOINT_NAMES):
        raise ValueError("Expected all ten distinct LW joint names.")
    _check(x, len(names))
    other = ["left_" + n[6:] if n.startswith("right_") else "right_" + n[5:] for n in names]
    indices = torch.tensor([names.index(n) for n in other], device=x.device)
    return -x.index_select(-1, indices)


def mirror_actions(actions: torch.Tensor) -> torch.Tensor:
    """Reflect eight position actions and two wheel-velocity actions."""
    return mirror_joints(actions)


def mirror_rotation_6d(x: torch.Tensor) -> torch.Tensor:
    """Reflect R by S R S, S=diag(1,-1,1), for R[:,:2].reshape(6)."""
    _check(x, 6)
    return x * x.new_tensor((1, -1, -1, 1, 1, -1))


def _mirror_term(name: str, x: torch.Tensor) -> torch.Tensor:
    if name == "command":
        return torch.cat((mirror_joints(x[..., :10]), mirror_joints(x[..., 10:])), dim=-1)
    if name in ("joint_pos", "joint_vel", "actions"):
        return mirror_joints(x)
    if name == "motion_anchor_ori_b":
        return mirror_rotation_6d(x)
    if name in ("motion_anchor_pos_b", "base_lin_vel"):
        return x * x.new_tensor((1, -1, 1))
    if name == "base_ang_vel":
        # Angular velocity is an axial vector: det(S) S omega.
        return x * x.new_tensor((-1, 1, -1))
    if name in ("body_pos", "body_ori"):
        width = 3 if name == "body_pos" else 6
        bodies = x.reshape(*x.shape[:-1], len(BODY_NAMES), width)
        bodies = bodies[..., (0, 2, 1, 4, 3, 6, 5, 8, 7), :]
        result = (bodies * x.new_tensor((1, -1, 1)) if width == 3 else mirror_rotation_6d(bodies))
        return result.reshape_as(x)
    raise ValueError(f"Unsupported observation term: {name}")


def _mirror_group(obs: torch.Tensor, terms: tuple, history_length: int) -> torch.Tensor:
    if type(history_length) is not int or history_length < 1:
        raise ValueError("history_length must be a positive integer.")
    _check(obs, sum(width for _, width in terms) * history_length)
    offset = 0
    parts = []
    for name, width in terms:
        count = history_length * width
        term = obs[..., offset:offset + count].reshape(*obs.shape[:-1], history_length, width)
        parts.append(_mirror_term(name, term).reshape(*obs.shape[:-1], count))
        offset += count
    return torch.cat(parts, dim=-1)


def mirror_policy_observations(obs: torch.Tensor, history_length: int = 10) -> torch.Tensor:
    """Reflect every history slot without changing its time or environment."""
    return _mirror_group(obs, POLICY_TERMS, history_length)


def mirror_critic_observations(obs: torch.Tensor) -> torch.Tensor:
    """Reflect the task's 146-dimensional privileged, history-free observation."""
    return _mirror_group(obs, CRITIC_TERMS, 1)


_VALIDATED_TRAINING_ENVS = WeakSet()


def _validate_training_env(env) -> None:
    """Check the resolved layout once per environment, before using fixed maps."""
    env = getattr(env, "unwrapped", env)
    if env in _VALIDATED_TRAINING_ENVS:
        return
    manager = env.observation_manager
    for group, terms, history in (("policy", POLICY_TERMS, 10), ("critic", CRITIC_TERMS, 1)):
        if (
            tuple(manager.active_terms[group]) != tuple(name for name, _ in terms)
            or not manager.group_obs_concatenate[group]
            or tuple(manager.group_obs_dim[group]) != (sum(width for _, width in terms) * history,)
            or tuple(manager.group_obs_term_dim[group]) != tuple((width * history,) for _, width in terms)
        ):
            raise ValueError(f"LW symmetry requires the declared {group} observation/history layout.")
        group_cfg = getattr(env.cfg.observations, group)
        for name in ("joint_pos", "joint_vel"):
            selector = getattr(group_cfg, name).params["asset_cfg"]
            if tuple(selector.joint_names) != JOINT_NAMES or not selector.preserve_order:
                raise ValueError(f"LW symmetry requires ordered joint names in {group}.{name}.")
    motion = env.cfg.commands.motion
    if tuple(motion.joint_names) != JOINT_NAMES or tuple(motion.body_names) != BODY_NAMES:
        raise ValueError("LW symmetry requires the declared reference joint and body order.")
    if tuple(env.action_manager.active_terms) != ("joint_pos", "joint_vel"):
        raise ValueError("LW symmetry requires position actions followed by wheel velocity actions.")
    for name, names, scale in (("joint_pos", JOINT_NAMES[:8], 0.25), ("joint_vel", JOINT_NAMES[8:], 1.0)):
        action = getattr(env.cfg.actions, name)
        if (
            tuple(action.joint_names) != names
            or not action.preserve_order
            or action.scale != scale
            or not action.use_default_offset
        ):
            raise ValueError(f"LW symmetry requires the declared {name} action mapping.")
    _VALIDATED_TRAINING_ENVS.add(env)


@lru_cache(maxsize=16)
def _training_map(group: str, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    """Compile the tested signed permutations once; avoid GPU checks per PPO batch."""
    width, reflect = {
        "policy": (590, mirror_policy_observations),
        "critic": (146, mirror_critic_observations),
        "actions": (10, mirror_actions),
    }[group]
    # Positive, unique labels recover both index and sign from the reference mapping.
    reflected = reflect(torch.arange(1, width + 1, dtype=torch.float64))
    indices = reflected.abs().to(dtype=torch.long) - 1
    signs = reflected.sign()
    return indices.to(device=device), signs.to(device=device, dtype=dtype)


def _reflect_training_tensor(data: torch.Tensor, group: str) -> torch.Tensor:
    width = {"policy": 590, "critic": 146, "actions": 10}[group]
    if not isinstance(data, torch.Tensor) or data.ndim != 2 or data.shape[1] != width or not data.is_floating_point():
        raise ValueError(f"LW symmetry expects a floating-point [batch, {width}] {group} tensor.")
    indices, signs = _training_map(group, data.device, data.dtype)
    return data.index_select(-1, indices) * signs


def compute_symmetric_states(env, obs: TensorDict | None = None, actions: torch.Tensor | None = None):
    """RSL-RL callback returning original samples followed by their left-right mirrors.

    Supports observation-only and action-only calls used by PPO's mirror loss.
    Randomized default poses and other latent physical parameters are interpreted
    as reflected together with the observations; this does not change the live env.
    ``env=None`` permits simulator-free tensor tests. Gradients are preserved; PPO
    itself detaches the mirrored mean-action targets when computing its loss.
    """
    if env is not None:
        _validate_training_env(env)
    obs_aug = None
    if obs is not None:
        if not isinstance(obs, TensorDict) or len(obs.batch_size) != 1 or set(obs.keys()) != {"policy", "critic"}:
            raise ValueError("LW symmetry expects flat policy/critic TensorDict groups with one batch dimension.")
        mirrored = TensorDict(
            {group: _reflect_training_tensor(obs[group], group) for group in ("policy", "critic")},
            batch_size=obs.batch_size,
            device=obs.device,
        )
        obs_aug = torch.cat((obs, mirrored), dim=0)
    actions_aug = None
    if actions is not None:
        reflected_actions = _reflect_training_tensor(actions, "actions")
        if obs is not None and actions.shape[0] != obs.batch_size[0]:
            raise ValueError("LW symmetry requires matching observation and action batch sizes.")
        actions_aug = torch.cat((actions, reflected_actions), dim=0)
    return obs_aug, actions_aug
