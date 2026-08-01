#!/usr/bin/env python3
"""Track policy-evaluation signal availability without hiding capture errors."""

from __future__ import annotations

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
