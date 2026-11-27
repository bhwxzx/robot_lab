# SPDX-License-Identifier: BSD-3-Clause
"""Sample AMP windows from a rectangular grid of signed forward and yaw commands."""

import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from .motion_loader import AMPLoader, resolve_motion_files


class CommandConditionedAMPLoader:
    """Match the online AMP frame width, or keep the full expert layout if no width is requested."""

    _joint_foot_dim = AMPLoader.END_POS_END_IDX
    _joint_foot_layout = ["joint_position_offset", "joint_velocity", "foot_position_base"]
    _heading_velocity_layout = ["base_linear_velocity_heading", "base_angular_velocity_heading", *_joint_foot_layout]

    def __init__(
        self,
        device,
        time_between_frames,
        motion_files,
        history_length,
        stand_threshold=0.0,
        *,
        frame_dim: int | None = None,
    ):
        if history_length < 1 or not math.isfinite(time_between_frames) or time_between_frames <= 0:
            raise ValueError("AMP history length and control timestep must be positive.")
        if not math.isfinite(stand_threshold) or stand_threshold < 0:
            raise ValueError("AMP standing command threshold must be finite and non-negative.")
        self.stand_threshold = stand_threshold
        self.history_length = history_length
        self.motion_files = list(dict.fromkeys(resolve_motion_files(motion_files)))
        groups = defaultdict(list)
        segments = []
        digest = hashlib.sha256()
        for clip_id, path in enumerate(self.motion_files):
            data = Path(path).read_bytes()
            digest.update(hashlib.sha256(data).digest())
            try:
                motion = json.loads(data)
                vx, vy, wz = motion["Command"]
                weight = float(motion["MotionWeight"])
                if vy != 0 or not all(map(math.isfinite, (vx, wz, weight))) or weight <= 0:
                    raise ValueError("expected finite (vx, 0, wz) and a positive MotionWeight")
                stand = (vx, wz) == (0, 0)
                if motion["ClipType"] != ("stand" if stand else "episode"):
                    raise ValueError("zero command must denote a stand, moving commands an episode")
                segment = self._sampled_segment(motion, stand, time_between_frames, frame_dim)
                if segments and segment.shape[1] != segments[0].shape[1]:
                    raise ValueError("expert clips must share the same AMP observation layout")
                if len(segment) <= history_length:
                    raise ValueError(f"segment needs at least {history_length + 1} control frames")
            except (KeyError, ValueError, TypeError) as exc:
                raise ValueError(f"Invalid command-conditioned AMP expert {path}: {exc}") from exc
            groups[vx, wz].append((clip_id, weight))
            segments.append(segment)

        vx_axis = sorted({vx for vx, _ in groups})
        wz_axis = sorted({wz for _, wz in groups})
        if (0, 0) not in groups or len(groups) != len(vx_axis) * len(wz_axis):
            raise ValueError("Expert commands must cover a full vx/wz grid including the stationary point.")
        self.axes = tuple(torch.tensor(axis, dtype=torch.float32, device=device) for axis in (vx_axis, wz_axis))
        self.command_scale = tuple(max(map(abs, axis)) or 1.0 for axis in (vx_axis, wz_axis))
        # Each grid cell has its own categorical distribution over clips.
        shape = (len(vx_axis), len(wz_axis), max(map(len, groups.values())))
        clip_ids = np.zeros(shape, dtype=np.int64)
        weights = np.zeros(shape, dtype=np.float32)
        for i, vx in enumerate(vx_axis):
            for j, wz in enumerate(wz_axis):
                for k, (clip_id, weight) in enumerate(groups[vx, wz]):
                    clip_ids[i, j, k] = clip_id
                    weights[i, j, k] = weight
        self.clip_ids = torch.as_tensor(clip_ids, device=device)
        self.clip_weights = torch.as_tensor(weights, device=device)
        lengths = np.array([len(segment) for segment in segments], dtype=np.int64)
        self.starts = torch.as_tensor(lengths.cumsum() - lengths, device=device)
        self.window_counts = torch.as_tensor(lengths - history_length, device=device)
        self.frames = torch.as_tensor(np.concatenate(segments), device=device)
        self.frame_dim = self.frames.shape[1]
        self.observation_dim = history_length * self.frame_dim
        self.offsets = torch.arange(history_length + 1, device=device)
        self.dataset_sha256 = digest.hexdigest()

    @classmethod
    def _sampled_segment(cls, motion, stand, step_dt, frame_dim=None):
        frames = np.asarray(motion["Frames"], dtype=np.float32)
        frame_dt = float(motion["FrameDuration"])
        layout = motion.get("FrameLayout")
        source_width = cls._joint_foot_dim  # Legacy joint and foot frames have no layout header.
        if layout == cls._heading_velocity_layout:
            if motion.get("RootVelocityFrame") != "heading" or motion.get("RootVelocityPoint") != "base_link":
                raise ValueError("expected heading-frame velocities at the base_link origin")
            source_width += 6
        elif layout is not None and layout != cls._joint_foot_layout:
            raise ValueError(f"unsupported AMP FrameLayout: {layout!r}")
        if frames.ndim != 2 or frames.shape[1] != source_width or not np.isfinite(frames).all():
            raise ValueError(f"expected finite frames of width {source_width} for this AMP layout")
        if frame_dim == cls._joint_foot_dim:
            # Legacy online observations contain only the trailing joint and foot features.
            frames = frames[:, -frame_dim:]
        elif frame_dim is not None and frame_dim != source_width:
            raise ValueError(f"online AMP requests {frame_dim} features per frame, expert provides {source_width}")
        if not math.isfinite(frame_dt) or frame_dt <= 0:
            raise ValueError("FrameDuration must be positive and finite")
        if not stand:
            start, stop = motion["CruiseStart"], motion["BrakeStart"]
            if not 0 <= start < stop <= (len(frames) - 1) * frame_dt + 1e-8:
                raise ValueError("cruise interval outside trajectory")
            # Include CruiseStart and exclude BrakeStart, allowing for decimal timestamp roundoff.
            frames = frames[math.ceil(start / frame_dt - 1e-8) : math.ceil(stop / frame_dt - 1e-8)]
        if len(frames) < 2 or math.isclose(step_dt, frame_dt):
            return frames
        count = math.floor((len(frames) - 1) * frame_dt / step_dt + 1e-8) + 1
        position = np.minimum(np.arange(count) * step_dt / frame_dt, len(frames) - 1)
        low = position.astype(np.int64)
        fraction = (position - low).astype(np.float32)[:, None]
        return frames[low] + fraction * (frames[np.minimum(low + 1, len(frames) - 1)] - frames[low])

    @staticmethod
    def _bracket(axis, values):
        high = torch.searchsorted(axis, values.contiguous()).clamp(max=len(axis) - 1)
        low = (high - 1).clamp_min(0)
        span = axis[high] - axis[low]
        width = torch.where(span > 0, span, 1.0)
        low_weight = torch.where(span > 0, (axis[high] - values) / width, 1.0)
        high_weight = (values - axis[low]) / width
        return low, high, low_weight, high_weight

    def covers(self, commands):
        """Return one validity flag per physical command, including the grid endpoints."""
        vx, wz = commands.unbind(dim=1)
        xs, zs = self.axes
        return (xs[0] <= vx) & (vx <= xs[-1]) & (zs[0] <= wz) & (wz <= zs[-1])

    def sample_clip_ids(self, commands):
        """Select the standing grid cell within the raw command radius, otherwise a moving neighbor."""
        if commands.ndim != 2 or commands.shape[1] != 2:
            raise ValueError("Expected a batch of (vx, wz) commands.")
        xs, zs = self.axes
        if not self.covers(commands).all():
            raise ValueError("AMP query outside the expert vx/wz grid or non-finite.")
        lookup_commands = commands
        if self.stand_threshold > 0.0:
            norm_dtype = torch.promote_types(commands.dtype, xs.dtype)
            standing = torch.linalg.vector_norm(commands.to(norm_dtype), dim=1) <= self.stand_threshold
            # Only the expert lookup uses the origin. The discriminator still receives the raw query.
            lookup_commands = commands.masked_fill(standing[:, None], 0)
        vx, wz = lookup_commands.unbind(dim=1)
        x0, x1, a0, a1 = self._bracket(xs, vx)
        z0, z1, b0, b1 = self._bracket(zs, wz)
        x = torch.stack((x0, x0, x1, x1), dim=1)
        z = torch.stack((z0, z1, z0, z1), dim=1)
        weights = torch.stack((a0 * b0, a0 * b1, a1 * b0, a1 * b1), dim=1)
        # Moving queries can bracket the origin but must always select a moving clip.
        moving = (vx != 0) | (wz != 0)
        weights.masked_fill_(moving[:, None] & (xs[x] == 0) & (zs[z] == 0), 0)
        corner = torch.multinomial(weights, 1)
        i, j = x.gather(1, corner).squeeze(1), z.gather(1, corner).squeeze(1)
        clips = self.clip_ids[i, j]
        if clips.shape[1] == 1:
            return clips[:, 0]
        return clips.gather(1, torch.multinomial(self.clip_weights[i, j], 1)).squeeze(1)

    def sample_for_commands(self, commands):
        """Return two overlapping, oldest-first windows, wholly within one expert segment."""
        clips = self.sample_clip_ids(commands)
        offsets = (torch.rand(len(clips), device=clips.device) * self.window_counts[clips]).long()
        first = self.starts[clips] + offsets
        frames = self.frames[first[:, None] + self.offsets]
        return frames[:, :-1].flatten(1), frames[:, 1:].flatten(1)
