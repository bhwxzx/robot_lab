#!/usr/bin/env python3
"""Apply approved metric-window rules to a structured training-log summary."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Callable

from validate_session_spec import SpecError, load_and_validate


OPERATORS: dict[str, Callable[[float, float], bool]] = {
    "<=": lambda actual, limit: actual <= limit,
    ">=": lambda actual, limit: actual >= limit,
    "<": lambda actual, limit: actual < limit,
    ">": lambda actual, limit: actual > limit,
}


def _load_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise SpecError(f"{label} file does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise SpecError(
            f"invalid {label} JSON at line {exc.lineno}: {exc.msg}"
        ) from exc
    if not isinstance(value, dict):
        raise SpecError(f"{label} must be a JSON object")
    return value


def detect_anomalies(
    spec: dict[str, Any],
    summary: dict[str, Any],
) -> dict[str, Any]:
    """Return a deterministic stop/suspect decision without signaling a process."""
    if spec.get("version") != 6 or spec.get("mode") != "tune":
        raise SpecError("quality anomaly detection requires version-6 tune session")
    execution = spec["execution"]
    records = summary.get("records")
    if not isinstance(records, list):
        raise SpecError("training summary must contain a records array")
    non_finite = summary.get("non_finite_metrics")
    if not isinstance(non_finite, list):
        raise SpecError("training summary must contain non_finite_metrics")
    if any(not isinstance(record, dict) for record in records):
        raise SpecError("training summary records must be objects")

    triggered: list[dict[str, Any]] = []
    insufficient: list[dict[str, Any]] = []
    for rule in execution["quality_rules"]:
        count = rule["consecutive_windows"]
        minimum_progress = rule.get("minimum_progress")
        eligible_records = (
            [
                record
                for record in records
                if isinstance(record.get("progress"), int)
                and record["progress"] >= minimum_progress
            ]
            if minimum_progress is not None
            else records
        )
        latest_progress = next(
            (
                record.get("progress")
                for record in reversed(records)
                if isinstance(record, dict)
                and isinstance(record.get("progress"), int)
            ),
            None,
        )
        metric_values: list[float] = []
        for record in eligible_records[-count:]:
            value = record.get(rule["metric"])
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                continue
            metric_values.append(float(value))
        if len(metric_values) < count:
            insufficient.append(
                {
                    "rule_id": rule["id"],
                    "required_windows": count,
                    "available_windows": len(metric_values),
                    "minimum_progress": minimum_progress,
                    "latest_progress": latest_progress,
                }
            )
            continue
        anomalous = all(
            OPERATORS[rule["op"]](actual, float(rule["value"]))
            for actual in metric_values
        )
        if anomalous:
            triggered.append(
                {
                    "rule_id": rule["id"],
                    "metric": rule["metric"],
                    "op": rule["op"],
                    "limit": float(rule["value"]),
                    "values": metric_values,
                    "action": rule["action"],
                }
            )

    non_finite_stop = bool(non_finite)
    stop_rules = [
        item for item in triggered if item["action"] == "stop_trial"
    ]
    suspect_rules = [
        item for item in triggered if item["action"] == "mark_suspect"
    ]
    if non_finite_stop or stop_rules:
        status = "stop_approved"
    elif suspect_rules:
        status = "suspect"
    else:
        status = "healthy"
    return {
        "version": 1,
        "status": status,
        "stop_trial": status == "stop_approved",
        "non_finite_metrics": non_finite,
        "triggered_rules": triggered,
        "insufficient_data_rules": insufficient,
        "decision_basis": (
            "approved_nonfinite_or_consecutive_metric_rule"
            if status == "stop_approved"
            else "approved_metric_rules"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session", help="Validated version-6 tune session")
    parser.add_argument("summary", help="Structured training-log summary JSON")
    parser.add_argument("--output", help="Optional anomaly report JSON")
    args = parser.parse_args()
    try:
        report = detect_anomalies(
            load_and_validate(args.session),
            _load_object(Path(args.summary), "training summary"),
        )
    except SpecError as exc:
        parser.error(str(exc))
    encoded = json.dumps(
        report,
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
    )
    if args.output:
        Path(args.output).write_text(encoded + "\n", encoding="utf-8")
    else:
        print(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
