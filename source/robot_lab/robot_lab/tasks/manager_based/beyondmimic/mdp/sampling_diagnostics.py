"""Optional motion-sampling counters, independent of IsaacSim and of policy RNG.

The command calls observe() before resetting references and before advancing
frames. Each common control step is counted once, for all environments. Counts
are accumulated on the simulation device; only window writes transfer to CPU.
Failure rates here are failures per exposed environment step, not per episode.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch


class MotionSamplingDiagnostics:
    """Write disjoint windows to a new directory; never reopen an existing run.

    ``reset_starts`` counts selected reset frames, which can differ from the first
    frame actually simulated because MotionCommand advances after reset. Exposure
    and failure use the frame at the termination check. Failure ages count actual
    physics/control steps since the latest motion resample, independent of the
    runner's possibly randomized episode_length_buf.
    """

    AGE_UPPER_STEPS = (1, 2, 3, 4, 5, 10, 25, 50, 100)

    def __init__(
        self, *, output_dir: str, num_envs: int, frame_count: int, fps: float,
        bin_count: int, term_timeouts: dict[str, bool], device: str,
        window_steps: int = 2400,
    ):
        for name, value in (("num_envs", num_envs), ("frame_count", frame_count),
                            ("bin_count", bin_count), ("window_steps", window_steps)):
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("fps must be finite and positive")
        if not output_dir or not term_timeouts:
            raise ValueError("An explicit new output directory and termination terms are required")
        self.frame_count = frame_count
        self.num_envs = num_envs
        self.bin_count = bin_count
        self.window_steps = window_steps
        self.term_names = tuple(term_timeouts)
        self.last_step = 0
        self.window_first_step: int | None = None
        self.window_control_steps = 0
        self.window_index = 0
        self.last_probabilities: torch.Tensor | None = None
        self.start_frames = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.known_starts = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.ages = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.age_bounds = torch.tensor(self.AGE_UPPER_STEPS, device=device)
        self.ones = torch.ones(num_envs, dtype=torch.long, device=device)
        self.counts = {
            name: torch.zeros(frame_count, dtype=torch.long, device=device)
            for name in (
                "exposure_steps", "failures", "timeout_only", "motion_finished_without_failure",
                "reset_starts", "failures_by_start_frame",
            )
        }
        self.term_counts = torch.zeros(frame_count, len(self.term_names), dtype=torch.long, device=device)
        self.age_counts = torch.zeros(frame_count, len(self.AGE_UPPER_STEPS) + 1, dtype=torch.long, device=device)
        self.unknown_start_failures = torch.zeros((), dtype=torch.long, device=device)
        self.bin_ids = [frame * bin_count // frame_count for frame in range(frame_count)]
        self.phase_ids = [math.floor(frame / fps / 0.1 + 1e-9) for frame in range(frame_count)]
        self.phase_count = max(self.phase_ids) + 1
        directory = Path(output_dir).expanduser().resolve()
        directory.mkdir(parents=True, exist_ok=False)
        self.path = directory / "motion_sampling_diagnostics.jsonl"
        metadata = {
            "type": "metadata", "schema_version": 1, "num_envs": num_envs,
            "frame_count": frame_count, "fps": fps, "bin_count": bin_count,
            "window_control_steps": window_steps, "term_timeouts": term_timeouts,
            "failure_bin_by_frame": self.bin_ids, "phase_0p1s_bin_by_frame": self.phase_ids,
            "age_bucket_upper_steps": [*self.AGE_UPPER_STEPS, None],
            "failure_rate_unit": "failures_per_10000_exposed_environment_steps",
            "term_counts_are_nonexclusive": True,
            "capture_phase": "post_physics_post_termination_pre_reference_reset_or_advance",
            "reset_starts_semantics": "selected reference frames, before subsequent command advance",
            "failure_bin_rule": "floor(frame * bin_count / frame_count)",
            "sampler_frame_rule": "floor((sampled_bin + uniform) / bin_count * (frame_count - 1))",
            "age_semantics": "control steps since latest motion resample; first step has age 1",
            "reset_window_semantics": "resamples since previous flush, including starts preceding this window's first step",
            "partial_tail": "not persisted unless flush_sampling_diagnostics() is called",
        }
        with self.path.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(metadata, allow_nan=False) + "\n")

    def record_starts(self, env_ids, frames: torch.Tensor, probabilities: torch.Tensor):
        self.start_frames[env_ids] = frames
        self.known_starts[env_ids] = True
        self.ages[env_ids] = 0
        self.counts["reset_starts"].scatter_add_(0, frames, torch.ones_like(frames))
        self.last_probabilities = probabilities.detach().clone()

    def observe(self, *, step: int, frames: torch.Tensor, terminated: torch.Tensor,
                term_values: dict[str, torch.Tensor]):
        if step == self.last_step:
            return  # Includes the initial reset at step zero; no physics exposure.
        if step != self.last_step + 1:
            raise ValueError("Diagnostic control steps must be consecutive and monotonic")
        if tuple(term_values) != self.term_names:
            raise ValueError("Termination terms changed during diagnostic collection")
        self.last_step = step
        if self.window_first_step is None:
            self.window_first_step = step
        self.window_control_steps += 1
        self.ages += 1
        flags = torch.stack(tuple(term_values.values()), dim=1)
        self.counts["exposure_steps"].scatter_add_(0, frames, self.ones)
        self.counts["failures"].scatter_add_(0, frames, terminated.long())
        self.counts["timeout_only"].scatter_add_(0, frames, (flags.any(dim=1) & ~terminated).long())
        if "motion_finished" in term_values:
            self.counts["motion_finished_without_failure"].scatter_add_(
                0, frames, (term_values["motion_finished"] & ~terminated).long(),
            )
        self.term_counts.scatter_add_(0, frames[:, None].expand_as(flags), flags.long())
        ages = torch.bucketize(self.ages, self.age_bounds, right=False)
        self.age_counts.view(-1).scatter_add_(
            0, frames * self.age_counts.shape[1] + ages, terminated.long(),
        )
        self.counts["failures_by_start_frame"].scatter_add_(
            0, self.start_frames, (terminated & self.known_starts).long(),
        )
        self.unknown_start_failures += (terminated & ~self.known_starts).long().sum()
        if self.window_control_steps == self.window_steps:
            self.flush()

    def _aggregate(self, raw: dict, ids: list[int], size: int) -> dict:
        counts = {name: [0] * size for name in self.counts}
        terms = {name: [0] * size for name in self.term_names}
        ages = [[0] * self.age_counts.shape[1] for _ in range(size)]
        for frame, group in enumerate(ids):
            for name in counts:
                counts[name][group] += raw[name][frame]
            for i, name in enumerate(self.term_names):
                terms[name][group] += raw["termination_counts"][frame][i]
            for i in range(len(ages[group])):
                ages[group][i] += raw["failures_by_age_steps"][frame][i]
        return {
            **counts, "termination_counts": terms, "failures_by_age_steps": ages,
            "termination_triggers_per_10000_steps": {
                name: [10000.0 * count / exposure if exposure else None
                       for count, exposure in zip(values, counts["exposure_steps"])]
                for name, values in terms.items()
            },
            "failures_per_10000_steps": [
                10000.0 * failures / exposure if exposure else None
                for failures, exposure in zip(counts["failures"], counts["exposure_steps"])
            ],
        }

    def flush(self):
        """Persist a nonempty window, then clear only diagnostic counters."""
        if not self.window_control_steps:
            return
        raw = {name: value.cpu().tolist() for name, value in self.counts.items()}
        raw["termination_counts"] = self.term_counts.cpu().tolist()
        raw["failures_by_age_steps"] = self.age_counts.cpu().tolist()
        record = {
            "type": "window", "window_index": self.window_index,
            "first_control_step": self.window_first_step, "last_control_step": self.last_step,
            "control_steps": self.window_control_steps,
            "partial": self.window_control_steps != self.window_steps,
            "raw_by_frame": raw,
            "by_sampling_failure_bin": self._aggregate(raw, self.bin_ids, self.bin_count),
            "by_phase_0p1s": self._aggregate(raw, self.phase_ids, self.phase_count),
            "unknown_start_failures": self.unknown_start_failures.item(),
            "sampling_probabilities_at_last_resample": (
                None if self.last_probabilities is None else self.last_probabilities.cpu().tolist()
            ),
        }
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, allow_nan=False) + "\n")
        for value in self.counts.values():
            value.zero_()
        self.term_counts.zero_()
        self.age_counts.zero_()
        self.unknown_start_failures.zero_()
        self.window_control_steps = 0
        self.window_first_step = None
        self.window_index += 1
