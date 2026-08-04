#!/usr/bin/env python3
"""Track policy-evaluation signal availability without hiding capture errors."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping
from typing import Any


CAPTURE_ERRORS = (AttributeError, KeyError, IndexError, TypeError, ValueError, RuntimeError)
AMP_ROA_RUNNER = "OnPolicyRunnerAmpROA"

BASE_REQUIRED_SIGNALS = frozenset(
    {
        "command",
        "reward",
        "done",
        "timeout",
        "action",
    }
)
AMP_ROA_REQUIRED_SIGNALS = frozenset(
    {
        *BASE_REQUIRED_SIGNALS,
        "joint_names",
        "root_linear_velocity_b",
        "root_angular_velocity_b",
        "projected_gravity_b",
        "joint_position",
        "joint_velocity",
        "applied_torque",
        "joint_effort_limits",
        "joint_velocity_limits",
    }
)
TELEMETRY_SIGNALS = (
    "command",
    "reward",
    "done",
    "timeout",
    "root_position_w",
    "root_quaternion_w",
    "root_linear_velocity_b",
    "root_angular_velocity_b",
    "projected_gravity_b",
    "joint_names",
    "joint_position",
    "joint_velocity",
    "applied_torque",
    "joint_effort_limits",
    "joint_velocity_limits",
    "action",
)


def required_signals_for_runner(runner: str) -> frozenset[str]:
    if runner == AMP_ROA_RUNNER:
        return AMP_ROA_REQUIRED_SIGNALS
    return BASE_REQUIRED_SIGNALS


def runner_requires_complete_telemetry(runner: str) -> bool:
    return runner == AMP_ROA_RUNNER


def _bounded_error(exc: BaseException) -> str:
    message = " ".join(str(exc).split())
    rendered = f"{type(exc).__name__}: {message}" if message else type(exc).__name__
    return rendered[:500]


def _finite_matrix(
    values: Iterable[Iterable[float]],
    *,
    label: str,
    columns: int,
) -> list[list[float]]:
    rows: list[list[float]] = []
    for raw_row in values:
        row = [float(value) for value in raw_row]
        if len(row) != columns:
            raise ValueError(f"{label} row width does not match joint_names")
        if not all(math.isfinite(value) for value in row):
            raise ValueError(f"{label} contains non-finite values")
        rows.append(row)
    if not rows:
        raise ValueError(f"{label} must contain at least one environment row")
    return rows


class JointLimitTracker:
    """Aggregate per-joint runtime limit utilization across environments and steps."""

    def __init__(
        self,
        joint_names: Iterable[str],
        joint_effort_limits: Iterable[Iterable[float]],
        joint_velocity_limits: Iterable[Iterable[float]],
    ) -> None:
        self.joint_names = [str(name) for name in joint_names]
        if not self.joint_names or any(not name for name in self.joint_names):
            raise ValueError("joint_names must contain non-empty strings")
        if len(set(self.joint_names)) != len(self.joint_names):
            raise ValueError("joint_names must be unique")
        columns = len(self.joint_names)
        self.effort_limits = _finite_matrix(
            joint_effort_limits,
            label="joint_effort_limits",
            columns=columns,
        )
        self.velocity_limits = _finite_matrix(
            joint_velocity_limits,
            label="joint_velocity_limits",
            columns=columns,
        )
        if len(self.effort_limits) != len(self.velocity_limits):
            raise ValueError("joint limit environment counts do not match")
        if any(value <= 0.0 for row in self.effort_limits for value in row):
            raise ValueError("joint_effort_limits must be positive")
        if any(value <= 0.0 for row in self.velocity_limits for value in row):
            raise ValueError("joint_velocity_limits must be positive")

        self._environment_count = len(self.effort_limits)
        self._sample_count = 0
        self._joint_sample_counts = [0] * columns
        self._max_abs_effort = [0.0] * columns
        self._max_abs_velocity = [0.0] * columns
        self._max_effort_utilization = [0.0] * columns
        self._max_velocity_utilization = [0.0] * columns
        self._effort_peak_steps: list[int | None] = [None] * columns
        self._velocity_peak_steps: list[int | None] = [None] * columns
        self._effort_violation_counts = [0] * columns
        self._velocity_violation_counts = [0] * columns

    def observe(
        self,
        applied_torque: Iterable[Iterable[float]],
        joint_velocity: Iterable[Iterable[float]],
        *,
        step: int,
    ) -> None:
        if isinstance(step, bool) or not isinstance(step, int) or step < 0:
            raise ValueError("step must be a non-negative integer")
        columns = len(self.joint_names)
        efforts = _finite_matrix(
            applied_torque,
            label="applied_torque",
            columns=columns,
        )
        velocities = _finite_matrix(
            joint_velocity,
            label="joint_velocity",
            columns=columns,
        )
        if len(efforts) != self._environment_count:
            raise ValueError("applied_torque environment count does not match limits")
        if len(velocities) != self._environment_count:
            raise ValueError("joint_velocity environment count does not match limits")

        for env_index, (effort_row, velocity_row) in enumerate(
            zip(efforts, velocities, strict=True)
        ):
            for joint_index, (effort, velocity) in enumerate(
                zip(effort_row, velocity_row, strict=True)
            ):
                abs_effort = abs(effort)
                abs_velocity = abs(velocity)
                effort_limit = self.effort_limits[env_index][joint_index]
                velocity_limit = self.velocity_limits[env_index][joint_index]
                effort_utilization = abs_effort / effort_limit
                velocity_utilization = abs_velocity / velocity_limit
                if effort_utilization > self._max_effort_utilization[joint_index]:
                    self._max_effort_utilization[joint_index] = effort_utilization
                    self._effort_peak_steps[joint_index] = step
                if velocity_utilization > self._max_velocity_utilization[joint_index]:
                    self._max_velocity_utilization[joint_index] = velocity_utilization
                    self._velocity_peak_steps[joint_index] = step
                self._max_abs_effort[joint_index] = max(
                    self._max_abs_effort[joint_index], abs_effort
                )
                self._max_abs_velocity[joint_index] = max(
                    self._max_abs_velocity[joint_index], abs_velocity
                )
                if abs_effort > effort_limit:
                    self._effort_violation_counts[joint_index] += 1
                if abs_velocity > velocity_limit:
                    self._velocity_violation_counts[joint_index] += 1
                self._joint_sample_counts[joint_index] += 1
                self._sample_count += 1

    def report(self) -> dict[str, Any]:
        max_effort = max(self._max_effort_utilization)
        max_velocity = max(self._max_velocity_utilization)
        effort_peak_joint = self._max_effort_utilization.index(max_effort)
        velocity_peak_joint = self._max_velocity_utilization.index(max_velocity)
        effort_violations = sum(self._effort_violation_counts)
        velocity_violations = sum(self._velocity_violation_counts)
        joint_metrics: list[dict[str, Any]] = []
        for index, name in enumerate(self.joint_names):
            samples = self._joint_sample_counts[index]
            joint_metrics.append(
                {
                    "joint_name": name,
                    "effort_limit_min": min(row[index] for row in self.effort_limits),
                    "effort_limit_max": max(row[index] for row in self.effort_limits),
                    "velocity_limit_min": min(
                        row[index] for row in self.velocity_limits
                    ),
                    "velocity_limit_max": max(
                        row[index] for row in self.velocity_limits
                    ),
                    "max_abs_applied_torque": self._max_abs_effort[index],
                    "max_abs_joint_velocity": self._max_abs_velocity[index],
                    "max_effort_utilization": self._max_effort_utilization[index],
                    "max_velocity_utilization": self._max_velocity_utilization[index],
                    "effort_peak_step": self._effort_peak_steps[index],
                    "velocity_peak_step": self._velocity_peak_steps[index],
                    "effort_violation_count": self._effort_violation_counts[index],
                    "velocity_violation_count": self._velocity_violation_counts[index],
                    "effort_violation_rate": self._effort_violation_counts[index]
                    / max(samples, 1),
                    "velocity_violation_rate": self._velocity_violation_counts[index]
                    / max(samples, 1),
                    "sample_count": samples,
                }
            )
        return {
            "metrics": {
                "max_joint_effort_utilization": max_effort,
                "max_joint_velocity_utilization": max_velocity,
                "joint_effort_limit_violation_rate": effort_violations
                / max(self._sample_count, 1),
                "joint_velocity_limit_violation_rate": velocity_violations
                / max(self._sample_count, 1),
            },
            "peak_steps": {
                "max_joint_effort_utilization": self._effort_peak_steps[
                    effort_peak_joint
                ],
                "max_joint_velocity_utilization": self._velocity_peak_steps[
                    velocity_peak_joint
                ],
            },
            "peak_joints": {
                "max_joint_effort_utilization": self.joint_names[effort_peak_joint],
                "max_joint_velocity_utilization": self.joint_names[velocity_peak_joint],
            },
            "joint_metrics": joint_metrics,
            "sample_count": self._sample_count,
        }


class SignalLedger:
    """Record success and failure counts independently for named signals."""

    def __init__(
        self,
        expected_counts: Mapping[str, int],
        *,
        required_signals: Iterable[str] = (),
    ) -> None:
        if not expected_counts:
            raise ValueError("expected_counts must not be empty")
        self._expected: dict[str, int] = {}
        for name, count in expected_counts.items():
            if not isinstance(name, str) or not name:
                raise ValueError("signal names must be non-empty strings")
            if isinstance(count, bool) or not isinstance(count, int) or count < 1:
                raise ValueError(f"expected count for {name} must be positive")
            self._expected[name] = count
        self._required = frozenset(required_signals)
        unknown = sorted(self._required - self._expected.keys())
        if unknown:
            raise ValueError("required signals are not tracked: " + ", ".join(unknown))
        self._successes = {name: 0 for name in self._expected}
        self._failures = {name: 0 for name in self._expected}
        self._errors: dict[str, str | None] = {name: None for name in self._expected}

    def capture(self, name: str, getter: Callable[[], Any]) -> Any | None:
        """Capture one signal and retain a bounded diagnostic on expected errors."""
        self._require_name(name)
        try:
            value = getter()
            if value is None:
                raise ValueError("capture returned no value")
        except CAPTURE_ERRORS as exc:
            self.record_error(name, exc)
            return None
        self.record_success(name)
        return value

    def record_success(self, name: str) -> None:
        self._require_name(name)
        self._successes[name] += 1

    def record_error(self, name: str, error: BaseException | str) -> None:
        self._require_name(name)
        self._failures[name] += 1
        if self._errors[name] is None:
            if isinstance(error, BaseException):
                self._errors[name] = _bounded_error(error)
            else:
                self._errors[name] = str(error)[:500]

    def _require_name(self, name: str) -> None:
        if name not in self._expected:
            raise KeyError(f"untracked signal: {name}")

    def report(self) -> dict[str, Any]:
        signals: dict[str, dict[str, Any]] = {}
        missing_required: list[str] = []
        for name in self._expected:
            sample_count = self._successes[name]
            expected_count = self._expected[name]
            required = name in self._required
            complete = sample_count == expected_count and self._failures[name] == 0
            if required and not complete:
                missing_required.append(name)
            signals[name] = {
                "required": required,
                "available": sample_count > 0,
                "complete": complete,
                "sample_count": sample_count,
                "expected_sample_count": expected_count,
                "error_count": self._failures[name],
                "error": self._errors[name],
            }
        required_counts = [
            self._successes[name] for name in self._required
        ]
        if missing_required:
            status = "partial" if any(required_counts) else "unavailable"
        else:
            status = "complete"
        return {
            "status": status,
            "missing_required_signals": sorted(missing_required),
            "signals": signals,
        }


def telemetry_report(
    *,
    requested: bool,
    runner: str,
    ledger: SignalLedger | None,
) -> dict[str, Any]:
    required = sorted(required_signals_for_runner(runner))
    if not requested:
        return {
            "telemetry_status": "not_requested",
            "telemetry_required_for_complete_assessment": runner_requires_complete_telemetry(
                runner
            ),
            "required_signals": required,
            "missing_required_signals": required,
            "signal_status": {},
        }
    if ledger is None:
        raise ValueError("requested telemetry requires a signal ledger")
    report = ledger.report()
    return {
        "telemetry_status": report["status"],
        "telemetry_required_for_complete_assessment": runner_requires_complete_telemetry(
            runner
        ),
        "required_signals": required,
        "missing_required_signals": report["missing_required_signals"],
        "signal_status": report["signals"],
    }


def metric_availability_report(
    signal_status: Mapping[str, Mapping[str, Any]],
    metric_sources: Mapping[str, Iterable[str]],
) -> dict[str, dict[str, Any]]:
    """Describe whether every source needed for each derived metric is complete."""
    report: dict[str, dict[str, Any]] = {}
    for metric, raw_sources in metric_sources.items():
        source_names = tuple(raw_sources)
        if not source_names:
            raise ValueError(f"metric {metric} has no source signals")
        try:
            sources = [signal_status[name] for name in source_names]
        except KeyError as exc:
            raise ValueError(f"metric {metric} references an unknown signal") from exc
        complete = all(source.get("complete") is True for source in sources)
        errors = {
            name: signal_status[name].get("error")
            for name in source_names
            if signal_status[name].get("error") is not None
        }
        report[metric] = {
            "available": complete,
            "complete": complete,
            "partial": not complete
            and any(source.get("available") is True for source in sources),
            "source_signals": list(source_names),
            "errors": errors,
        }
    return report


def record_complete_metric(
    metrics: dict[str, float],
    availability: Mapping[str, Mapping[str, Any]],
    name: str,
    value: Callable[[], float],
) -> None:
    """Record a derived metric only when all of its sources are complete."""
    if availability.get(name, {}).get("complete") is True:
        metrics[name] = float(value())
