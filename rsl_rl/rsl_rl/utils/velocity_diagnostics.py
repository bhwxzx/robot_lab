"""Physical-unit velocity diagnostics; never part of an optimization objective.

Command and state thresholds are inclusive. Clip-boundary samples are excluded
conservatively: their underlying physical values cannot be recovered from storage.
"""
from dataclasses import dataclass
import math

import torch


@dataclass(frozen=True)
class VelocityDiagnosticConfig:
    zero_command_epsilon: float = 1e-6
    low_command_xy: float = 0.1
    low_command_yaw: float = 0.1
    near_stationary_xy: float = 0.02
    near_stationary_yaw: float = 0.05
    chunk_size: int = 4096

    def __post_init__(self):
        for name in ("zero_command_epsilon", "low_command_xy", "low_command_yaw",
                     "near_stationary_xy", "near_stationary_yaw"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
        if self.zero_command_epsilon >= min(self.low_command_xy, self.low_command_yaw):
            raise ValueError("zero command epsilon must be smaller than low command thresholds")
        if type(self.chunk_size) is not int or self.chunk_size <= 0:
            raise ValueError("chunk_size must be a positive integer")


class GroupedVelocityDiagnostics:
    """Accumulate each pre-action sample once, separately by command and motion.

    Command groups are mutually exclusive. State groups are a second partition.
    Near-stationary refers to planar speed and yaw rate, not vertical/roll/pitch
    motion or a reward gate. Invalid samples (nonfinite or clip-boundary) are
    counted, never treated as zero. clipped_sample_count is a subset of invalid
    samples. Group fractions use the total valid sample count as denominator.
    """
    GROUPS = ("all", "command_zero", "command_low", "command_other",
              "state_near_stationary", "state_moving")

    def __init__(self, config=None, *, device="cpu"):
        self.config = config or VelocityDiagnosticConfig()
        self.stats = torch.zeros(len(self.GROUPS), 14, device=device, dtype=torch.float64)
        self.counts = torch.zeros(3, device=device, dtype=torch.float64)

    @torch.no_grad()
    def add(self, command, actual_velocity, estimated_velocity, actual_yaw_rate, *, clipped=None):
        n = command.shape[0]
        if (command.shape != (n, 3) or actual_velocity.shape != (n, 3)
                or estimated_velocity.shape != (n, 3) or actual_yaw_rate.shape != (n,)):
            raise ValueError("expected commands/velocities [N,3] and yaw rate [N]")
        cmd, actual, predicted, yaw = [v.detach().to(device=self.stats.device, dtype=torch.float64)
                                       for v in (command, actual_velocity, estimated_velocity, actual_yaw_rate)]
        valid = (torch.isfinite(cmd).all(-1) & torch.isfinite(actual).all(-1)
                 & torch.isfinite(predicted).all(-1) & torch.isfinite(yaw))
        if clipped is not None:
            if clipped.shape != (n,) or clipped.dtype != torch.bool:
                raise ValueError("expected clipped mask [N] with boolean dtype")
            clipped = clipped.to(self.stats.device)
            valid &= ~clipped
            self.counts[2] += clipped.sum()
        self.counts[0] += n
        self.counts[1] += (~valid).sum()
        cmd, actual, predicted, yaw = cmd[valid], actual[valid], predicted[valid], yaw[valid]
        cfg = self.config
        zero = (cmd.abs() <= cfg.zero_command_epsilon).all(-1)
        low = (~zero & (cmd[:, :2].norm(dim=-1) <= cfg.low_command_xy)
               & (cmd[:, 2].abs() <= cfg.low_command_yaw))
        stationary = ((actual[:, :2].norm(dim=-1) <= cfg.near_stationary_xy)
                      & (yaw.abs() <= cfg.near_stationary_yaw))
        masks = torch.stack((torch.ones_like(zero), zero, low, ~(zero | low), stationary, ~stationary))
        error = predicted - actual
        values = torch.cat((torch.ones_like(yaw[:, None]), stationary[:, None], actual,
                            predicted, error, error.square()), dim=-1)
        self.stats += masks.to(torch.float64) @ values

    @torch.no_grad()
    def report(self, *, distributed=False):
        # Sum sufficient statistics, not rank means; groups may have different sizes.
        packed = torch.cat((self.counts, self.stats.flatten()))
        if distributed:
            torch.distributed.all_reduce(packed, op=torch.distributed.ReduceOp.SUM)
        counts = packed[:3].cpu().tolist()
        rows = packed[3:].reshape(len(self.GROUPS), 14).cpu().tolist()
        valid = counts[0] - counts[1]
        result = {"sample_count": counts[0], "invalid_sample_count": counts[1],
                  "clipped_sample_count": counts[2], "valid_sample_count": valid}
        if counts[0]:
            result["valid_fraction"] = valid / counts[0]
        for group, row in zip(self.GROUPS, rows):
            n = row[0]
            result[f"{group}/sample_count"] = n
            if valid:
                result[f"{group}/fraction_valid"] = n / valid
            if not n:
                continue  # An empty group has no bias/RMSE, rather than perfect estimation.
            result[f"{group}/near_stationary_fraction"] = row[1] / n
            for i, axis in enumerate(("vx", "vy", "vz")):
                result[f"{group}/actual_{axis}_mean_m_s"] = row[2 + i] / n
                result[f"{group}/estimated_{axis}_mean_m_s"] = row[5 + i] / n
                result[f"{group}/bias_{axis}_m_s"] = row[8 + i] / n
                result[f"{group}/rmse_{axis}_m_s"] = math.sqrt(row[11 + i] / n)
        return result


@dataclass(frozen=True)
class CriticDiagnosticTerm:
    offset: int
    scale: torch.Tensor
    clip: tuple[float, float] | None

    def decode(self, critic):
        observed = critic[:, self.offset:self.offset + 3]
        scale = self.scale.to(device=observed.device, dtype=observed.dtype)
        clipped = torch.zeros(len(observed), device=observed.device, dtype=torch.bool)
        if self.clip is not None:
            # Compare in storage precision: dividing first could round a saturated
            # observation just inside the original physical clipping bounds.
            clipped = ((observed <= self.clip[0] * scale) | (observed >= self.clip[1] * scale)).any(-1)
        return observed.to(torch.float64) / self.scale.to(observed.device), clipped


def critic_diagnostic_layout(env, policy):
    """Resolve offsets and scales from IsaacLab metadata, never wheel constants."""
    unwrapped = env.unwrapped
    manager = unwrapped.observation_manager
    names = manager.active_terms["critic"]
    dimensions = manager.group_obs_term_dim["critic"]
    if len(names) != len(dimensions):
        raise ValueError("critic observation metadata length mismatch")
    # IsaacLab deep-copies the environment config, then resolves scales and
    # group-level history overrides in the manager's own configuration.
    group = manager.cfg["critic"] if isinstance(manager.cfg, dict) else manager.cfg.critic
    if group.enable_corruption:
        raise ValueError("velocity diagnostics require uncorrupted critic observations")
    if not group.concatenate_terms or group.concatenate_dim not in (-1, 0):
        raise ValueError("velocity diagnostics require concatenated critic vectors")
    if group.history_length not in (None, 0):
        raise ValueError("velocity diagnostics require current-frame critic observations")
    layout, offset = {}, 0
    required = {"base_lin_vel", "base_ang_vel", "velocity_commands"}
    for name, shape in zip(names, dimensions):
        if len(shape) != 1 or shape[0] <= 0:
            raise ValueError("velocity diagnostics require one-dimensional critic terms")
        size = math.prod(shape)
        if name in required:
            term = getattr(group, name)
            if tuple(shape) != (3,) or getattr(term, "history_length", 0) not in (None, 0):
                raise ValueError(f"diagnostics require a current-frame 3-vector: {name}")
            if getattr(term, "modifiers", None):
                raise ValueError(f"cannot undo observation modifiers for {name}")
            scale = torch.as_tensor(1.0 if term.scale is None else term.scale, dtype=torch.float64).reshape(-1)
            if scale.numel() not in (1, 3) or not torch.isfinite(scale).all() or (scale <= 0).any():
                raise ValueError(f"invalid observation scale for {name}")
            clip = getattr(term, "clip", None)
            if clip is not None:
                if len(clip) != 2 or not clip[0] < clip[1]:
                    raise ValueError(f"invalid observation clip for {name}")
                clip = tuple(clip)
            layout[name] = CriticDiagnosticTerm(offset, scale, clip)
        offset += size
    if set(layout) != required or layout["base_lin_vel"].offset != policy.vel_offset:
        raise ValueError("velocity diagnostic terms missing or policy velocity offset mismatched")
    return layout
