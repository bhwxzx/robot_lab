#!/usr/bin/env python3
"""Produce non-executable continue/stop and convergence advice for one run."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import fmean
from typing import Any, Callable


class AssessmentError(ValueError):
    """Raised when advisor evidence is malformed or insufficiently specified."""


OPERATORS: dict[str, Callable[[float, float], bool]] = {
    "<=": lambda actual, limit: actual <= limit,
    ">=": lambda actual, limit: actual >= limit,
    "<": lambda actual, limit: actual < limit,
    ">": lambda actual, limit: actual > limit,
}


def load_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise AssessmentError(f"{label} does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise AssessmentError(
            f"invalid {label} JSON at line {exc.lineno}: {exc.msg}"
        ) from exc
    if not isinstance(value, dict):
        raise AssessmentError(f"{label} must be a JSON object")
    return value


def _finite_values(records: list[dict[str, Any]], metric: str) -> list[float]:
    values: list[float] = []
    for record in records:
        value = record.get(metric)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        numeric = float(value)
        if math.isfinite(numeric):
            values.append(numeric)
    return values


def _validate_criteria(criteria: dict[str, Any]) -> tuple[int, int, int, dict[str, Any]]:
    if criteria.get("version") != 1:
        raise AssessmentError("criteria.version must be 1")
    window_size = criteria.get("window_size")
    minimum_records = criteria.get("minimum_records")
    plateau_required = criteria.get("plateau_required_metrics")
    metrics = criteria.get("metrics")
    if isinstance(window_size, bool) or not isinstance(window_size, int) or window_size < 2:
        raise AssessmentError("criteria.window_size must be an integer >= 2")
    if (
        isinstance(minimum_records, bool)
        or not isinstance(minimum_records, int)
        or minimum_records < window_size * 2
    ):
        raise AssessmentError("criteria.minimum_records must be >= two windows")
    if (
        isinstance(plateau_required, bool)
        or not isinstance(plateau_required, int)
        or plateau_required < 1
    ):
        raise AssessmentError("plateau_required_metrics must be a positive integer")
    if not isinstance(metrics, dict) or not metrics:
        raise AssessmentError("criteria.metrics must be a non-empty object")
    required_count = 0
    for name, rule in metrics.items():
        if not isinstance(name, str) or not name or not isinstance(rule, dict):
            raise AssessmentError("each metric criterion must be a named object")
        if rule.get("direction") not in {"maximize", "minimize"}:
            raise AssessmentError(f"metric {name} has invalid direction")
        tolerance = rule.get("plateau_relative_tolerance")
        if (
            isinstance(tolerance, bool)
            or not isinstance(tolerance, (int, float))
            or not math.isfinite(float(tolerance))
            or float(tolerance) < 0
        ):
            raise AssessmentError(f"metric {name} has invalid plateau tolerance")
        if rule.get("required", False):
            required_count += 1
        for limit_name in ("hard_min", "hard_max"):
            if limit_name in rule:
                limit = rule[limit_name]
                if (
                    isinstance(limit, bool)
                    or not isinstance(limit, (int, float))
                    or not math.isfinite(float(limit))
                ):
                    raise AssessmentError(f"metric {name} has invalid {limit_name}")
    if required_count < plateau_required:
        raise AssessmentError(
            "plateau_required_metrics exceeds the number of required metrics"
        )
    return window_size, minimum_records, plateau_required, metrics


def _metric_trend(
    name: str,
    rule: dict[str, Any],
    prior: list[dict[str, Any]],
    recent: list[dict[str, Any]],
) -> dict[str, Any]:
    prior_values = _finite_values(prior, name)
    recent_values = _finite_values(recent, name)
    if len(prior_values) != len(prior) or len(recent_values) != len(recent):
        return {
            "metric": name,
            "required": bool(rule.get("required", False)),
            "status": "insufficient",
            "prior_samples": len(prior_values),
            "recent_samples": len(recent_values),
        }
    prior_mean = fmean(prior_values)
    recent_mean = fmean(recent_values)
    scale = max(abs(prior_mean), 1.0e-12)
    raw_change = (recent_mean - prior_mean) / scale
    relative_improvement = (
        raw_change if rule["direction"] == "maximize" else -raw_change
    )
    tolerance = float(rule["plateau_relative_tolerance"])
    hard_failures: list[str] = []
    if "hard_min" in rule and recent_mean < float(rule["hard_min"]):
        hard_failures.append("below_hard_min")
    if "hard_max" in rule and recent_mean > float(rule["hard_max"]):
        hard_failures.append("above_hard_max")
    return {
        "metric": name,
        "required": bool(rule.get("required", False)),
        "status": "hard_failure" if hard_failures else "available",
        "direction": rule["direction"],
        "prior_mean": prior_mean,
        "recent_mean": recent_mean,
        "relative_improvement": relative_improvement,
        "plateau_tolerance": tolerance,
        "plateaued": abs(relative_improvement) <= tolerance,
        "meaningfully_improving": relative_improvement > tolerance,
        "meaningfully_degrading": relative_improvement < -tolerance,
        "hard_failures": hard_failures,
    }


def _play_gate_results(
    play_results: list[dict[str, Any]],
    criteria: dict[str, Any],
) -> dict[str, Any]:
    gates = criteria.get("play_gates", {})
    if not isinstance(gates, dict):
        raise AssessmentError("criteria.play_gates must be an object")
    if not play_results:
        return {"available": False, "passed": None, "checks": []}
    if not gates:
        return {"available": False, "passed": None, "checks": []}
    checks: list[dict[str, Any]] = []
    for result_index, result in enumerate(play_results):
        if result.get("status") != "completed" or not isinstance(result.get("metrics"), dict):
            checks.append(
                {
                    "result_index": result_index,
                    "status": "invalid_result",
                    "passed": False,
                }
            )
            continue
        metrics = result["metrics"]
        for metric, gate in gates.items():
            if not isinstance(gate, dict) or gate.get("op") not in OPERATORS:
                raise AssessmentError(f"invalid Play gate for {metric}")
            limit = gate.get("value")
            actual = metrics.get(metric)
            if (
                isinstance(limit, bool)
                or not isinstance(limit, (int, float))
                or isinstance(actual, bool)
                or not isinstance(actual, (int, float))
                or not math.isfinite(float(actual))
            ):
                checks.append(
                    {
                        "result_index": result_index,
                        "metric": metric,
                        "status": "missing_or_nonfinite",
                        "passed": False,
                    }
                )
                continue
            passed = OPERATORS[gate["op"]](float(actual), float(limit))
            checks.append(
                {
                    "result_index": result_index,
                    "metric": metric,
                    "actual": float(actual),
                    "op": gate["op"],
                    "limit": float(limit),
                    "status": "passed" if passed else "failed",
                    "passed": passed,
                }
            )
    return {
        "available": True,
        "passed": all(check.get("passed") is True for check in checks),
        "checks": checks,
    }


def assess_training(
    summary: dict[str, Any],
    criteria: dict[str, Any],
    *,
    health: dict[str, Any] | None = None,
    play_results: list[dict[str, Any]] | None = None,
    training_finished: bool = False,
) -> dict[str, Any]:
    """Assess evidence without mutating or signaling the training process."""
    window_size, minimum_records, plateau_required, metric_rules = _validate_criteria(criteria)
    records = summary.get("records")
    if not isinstance(records, list) or any(not isinstance(item, dict) for item in records):
        raise AssessmentError("summary.records must be an array of objects")
    non_finite = summary.get("non_finite_metrics", [])
    if not isinstance(non_finite, list):
        raise AssessmentError("summary.non_finite_metrics must be an array")
    enough_records = len(records) >= minimum_records
    trends: list[dict[str, Any]] = []
    if len(records) >= window_size * 2:
        prior = records[-2 * window_size : -window_size]
        recent = records[-window_size:]
        trends = [
            _metric_trend(name, rule, prior, recent)
            for name, rule in metric_rules.items()
        ]
    required_available = [
        item
        for item in trends
        if item.get("required") and item.get("status") != "insufficient"
    ]
    plateau_count = sum(item.get("plateaued") is True for item in required_available)
    improving_count = sum(
        item.get("meaningfully_improving") is True for item in required_available
    )
    degrading_count = sum(
        item.get("meaningfully_degrading") is True for item in required_available
    )
    hard_metric_failures = [
        item for item in trends if item.get("status") == "hard_failure"
    ]
    play = _play_gate_results(play_results or [], criteria)
    health_state = health.get("state") if isinstance(health, dict) else None
    invalid = bool(non_finite) or bool(hard_metric_failures) or play.get("passed") is False
    if health_state == "stalled":
        invalid = True

    if invalid:
        recommendation = "recommend_stop_invalid"
        reason = "nonfinite_stall_or_hard_gate_failure"
    elif not training_finished:
        if health_state in {"observing", "suspect"}:
            recommendation = "continue_and_recheck"
            reason = "training_progress_requires_another_snapshot"
        elif health_state != "healthy":
            recommendation = "insufficient_evidence"
            reason = "running_process_health_not_confirmed"
        elif not enough_records or len(required_available) < plateau_required:
            recommendation = "continue_and_recheck"
            reason = "insufficient_adjacent_metric_windows"
        elif plateau_count >= plateau_required and improving_count == 0:
            recommendation = "consider_stop_plateau"
            reason = "required_metrics_plateaued"
        elif improving_count > 0 and degrading_count == 0:
            recommendation = "continue"
            reason = "meaningful_improvement_without_hard_failure"
        else:
            recommendation = "continue_and_recheck"
            reason = "mixed_metric_trends"
    elif not enough_records or len(required_available) < plateau_required:
        recommendation = "insufficient_evidence"
        reason = "insufficient_adjacent_metric_windows"
    elif plateau_count >= plateau_required and improving_count == 0:
        recommendation = "consider_stop_plateau"
        reason = "required_metrics_plateaued"
    else:
        recommendation = "continue_and_recheck"
        reason = "training_finished_no_continue_advice"

    if not training_finished:
        convergence = "not_assessed_while_running"
    elif invalid:
        convergence = "not_converged"
    elif not enough_records or len(required_available) < plateau_required or not play["available"]:
        convergence = "indeterminate"
    elif plateau_count >= plateau_required and play["passed"] is True:
        convergence = "converged"
    elif plateau_count >= plateau_required:
        convergence = "plateaued_with_defects"
    else:
        convergence = "not_converged"

    return {
        "version": 1,
        "advisory_only": True,
        "recommendation": recommendation,
        "reason": reason,
        "convergence": convergence,
        "training_finished": training_finished,
        "profile_id": summary.get("profile_id"),
        "record_count": len(records),
        "window_size": window_size,
        "health_state": health_state,
        "non_finite_metrics": non_finite,
        "trends": trends,
        "counts": {
            "required_available": len(required_available),
            "plateaued": plateau_count,
            "improving": improving_count,
            "degrading": degrading_count,
            "hard_failures": len(hard_metric_failures),
        },
        "play": play,
        "pending_user_decision": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary")
    parser.add_argument("--criteria", required=True)
    parser.add_argument("--health")
    parser.add_argument("--play-result", action="append", default=[])
    parser.add_argument("--training-finished", action="store_true")
    parser.add_argument("--output")
    args = parser.parse_args()
    try:
        result = assess_training(
            load_object(Path(args.summary), "summary"),
            load_object(Path(args.criteria), "criteria"),
            health=load_object(Path(args.health), "health") if args.health else None,
            play_results=[
                load_object(Path(path), "Play result") for path in args.play_result
            ],
            training_finished=args.training_finished,
        )
    except AssessmentError as exc:
        parser.error(str(exc))
    encoded = json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
    if args.output:
        output = Path(args.output)
        if not output.is_absolute():
            parser.error("--output must be absolute")
        output.write_text(encoded + "\n", encoding="utf-8")
    else:
        print(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
