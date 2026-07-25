#!/usr/bin/env python3
"""Build a deterministic, bounded trial plan from an authorized tune session."""

from __future__ import annotations

import argparse
import json
from decimal import Decimal
from pathlib import Path
from typing import Any

from validate_session_spec import SpecError, load_and_validate


def _expand_parameter(parameter: dict[str, Any]) -> list[Any]:
    if "values" in parameter:
        return parameter["values"]
    range_spec = parameter["range"]
    minimum = Decimal(str(range_spec["min"]))
    maximum = Decimal(str(range_spec["max"]))
    step = Decimal(str(range_spec["step"]))
    values: list[Any] = []
    current = minimum
    while current <= maximum:
        value = float(current)
        values.append(int(value) if current == current.to_integral_value() else value)
        current += step
    return values


def _select_evenly_indexes(length: int, count: int) -> list[int]:
    if count >= length:
        return list(range(length))
    if count == 1:
        return [0]
    return [round(index * (length - 1) / (count - 1)) for index in range(count)]


def _combination_at(index: int, value_lists: list[list[Any]]) -> tuple[Any, ...]:
    values: list[Any] = [None] * len(value_lists)
    for position in range(len(value_lists) - 1, -1, -1):
        index, value_index = divmod(index, len(value_lists[position]))
        values[position] = value_lists[position][value_index]
    return tuple(values)


def _baseline_index(parameters: list[dict[str, Any]], value_lists: list[list[Any]]) -> int | None:
    positions: list[int] = []
    for parameter, values in zip(parameters, value_lists, strict=True):
        if "baseline" not in parameter or parameter["baseline"] not in values:
            return None
        positions.append(values.index(parameter["baseline"]))
    index = 0
    for position, values in zip(positions, value_lists, strict=True):
        index = index * len(values) + position
    return index


def build_plan(spec: dict[str, Any]) -> dict[str, Any]:
    """Build baseline plus a deterministic subset of the authorized grid."""
    if spec["mode"] != "tune":
        raise SpecError("trial plans are only valid in tune mode")
    tuning = spec["tuning"]
    parameters = tuning["allowed_parameters"]
    value_lists = [_expand_parameter(parameter) for parameter in parameters]
    combination_count = 1
    for values in value_lists:
        combination_count *= len(values)
    if combination_count > 1_000_000_000:
        raise SpecError("authorized grid exceeds 1,000,000,000 combinations; narrow the parameter domains")

    baseline_index = _baseline_index(parameters, value_lists)
    candidate_count = combination_count - (1 if baseline_index is not None else 0)
    tuning_slots = min(tuning["max_trials"] - 1, candidate_count)
    selected_positions = _select_evenly_indexes(candidate_count, tuning_slots)
    selected: list[dict[str, Any]] = []
    for candidate_position in selected_positions:
        grid_index = (
            candidate_position
            if baseline_index is None or candidate_position < baseline_index
            else candidate_position + 1
        )
        combination = _combination_at(grid_index, value_lists)
        overrides = {parameter["path"]: value for parameter, value in zip(parameters, combination, strict=True)}
        selected.append(overrides)

    trials = [{"trial_id": "baseline", "overrides": {}, "seeds": tuning["seeds"]}]
    trials.extend(
        {
            "trial_id": f"trial-{index:03d}",
            "overrides": overrides,
            "seeds": tuning["seeds"],
        }
        for index, overrides in enumerate(selected, start=1)
    )
    return {
        "version": 3,
        "mode": "tune",
        "run_id": spec["training"]["run_id"],
        "algorithm": spec["algorithm"],
        "mutation_scope": tuning["mutation_scope"],
        "authorized_parameter_paths": [parameter["path"] for parameter in parameters],
        "full_grid_size": combination_count,
        "planned_trial_count": len(trials),
        "planned_run_count": len(trials) * len(tuning["seeds"]),
        "trials": trials,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session", help="Validated tune session JSON")
    parser.add_argument("--output", help="Optional JSON output path")
    args = parser.parse_args()
    try:
        spec = load_and_validate(args.session)
        plan = build_plan(spec)
    except SpecError as exc:
        parser.error(str(exc))
    encoded = json.dumps(plan, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
    if args.output:
        Path(args.output).write_text(encoded + "\n", encoding="utf-8")
    else:
        print(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
