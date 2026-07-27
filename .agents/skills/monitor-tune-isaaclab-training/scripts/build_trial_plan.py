#!/usr/bin/env python3
"""Build a deterministic, bounded trial plan from an authorized tune session."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
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


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise SpecError("adaptive search input must be finite JSON") from exc


def _object_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _signature(overrides: dict[str, Any]) -> str:
    return _object_sha256(overrides)


def _all_grid(
    parameters: list[dict[str, Any]],
    value_lists: list[list[Any]],
    combination_count: int,
) -> list[dict[str, Any]]:
    if combination_count > 100_000:
        raise SpecError(
            "adaptive search requires an authorized grid of at most 100,000 "
            "combinations"
        )
    return [
        {
            parameter["path"]: value
            for parameter, value in zip(
                parameters,
                _combination_at(index, value_lists),
                strict=True,
            )
        }
        for index in range(combination_count)
    ]


def _grid_position(
    overrides: dict[str, Any],
    parameters: list[dict[str, Any]],
    value_lists: list[list[Any]],
) -> tuple[int, ...]:
    positions: list[int] = []
    for parameter, values in zip(parameters, value_lists, strict=True):
        value = overrides[parameter["path"]]
        matches = [
            index
            for index, candidate in enumerate(values)
            if type(candidate) is type(value) and candidate == value
            or (
                not isinstance(candidate, bool)
                and not isinstance(value, bool)
                and isinstance(candidate, (int, float))
                and isinstance(value, (int, float))
                and math.isclose(
                    float(candidate),
                    float(value),
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                )
            )
        ]
        if len(matches) != 1:
            raise SpecError(
                f"adaptive override is outside the approved grid: "
                f"{parameter['path']}"
            )
        positions.append(matches[0])
    return tuple(positions)


def _distance(
    left: dict[str, Any],
    right: dict[str, Any],
    parameters: list[dict[str, Any]],
    value_lists: list[list[Any]],
) -> float:
    left_position = _grid_position(left, parameters, value_lists)
    right_position = _grid_position(right, parameters, value_lists)
    return sum(
        abs(a - b) / max(1, len(values) - 1)
        for a, b, values in zip(
            left_position,
            right_position,
            value_lists,
            strict=True,
        )
    )


def _constraint_passes(
    metrics: dict[str, Any],
    constraint: dict[str, Any],
) -> bool:
    value = float(metrics[constraint["metric"]])
    threshold = float(constraint["value"])
    return {
        "<=": value <= threshold,
        ">=": value >= threshold,
        "<": value < threshold,
        ">": value > threshold,
    }[constraint["op"]]


def _rank_metric_records(
    spec: dict[str, Any],
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    required = {
        item["metric"] for item in spec["tuning"]["objectives"]
    } | {
        item["metric"] for item in spec["tuning"]["constraints"]
    }
    valid = []
    for record in records:
        metrics = record.get("metrics")
        if (
            isinstance(metrics, dict)
            and set(metrics) >= required
            and all(
                not isinstance(metrics[key], bool)
                and isinstance(metrics[key], (int, float))
                and math.isfinite(float(metrics[key]))
                for key in required
            )
            and all(
                _constraint_passes(metrics, constraint)
                for constraint in spec["tuning"]["constraints"]
            )
        ):
            valid.append(record)
    if not valid:
        return []
    objectives = spec["tuning"]["objectives"]
    ranges = {
        objective["metric"]: (
            min(float(item["metrics"][objective["metric"]]) for item in valid),
            max(float(item["metrics"][objective["metric"]]) for item in valid),
        )
        for objective in objectives
    }
    total_weight = sum(float(item["weight"]) for item in objectives)
    ranked = []
    for record in valid:
        score = 0.0
        for objective in objectives:
            metric = objective["metric"]
            low, high = ranges[metric]
            value = float(record["metrics"][metric])
            normalized = (
                0.5
                if math.isclose(low, high)
                else (
                    (value - low) / (high - low)
                    if objective["goal"] == "maximize"
                    else (high - value) / (high - low)
                )
            )
            score += normalized * float(objective["weight"])
        ranked.append({**record, "_adaptive_score": score / total_weight})
    ranked.sort(
        key=lambda item: (
            -item["_adaptive_score"],
            str(item.get("run_id", item.get("trial_id", ""))),
        )
    )
    return ranked


def _select_near_anchors(
    candidates: list[dict[str, Any]],
    anchors: list[dict[str, Any]],
    count: int,
    parameters: list[dict[str, Any]],
    value_lists: list[list[Any]],
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    remaining = list(candidates)
    if not anchors or count <= 0:
        return selected
    anchor_index = 0
    while remaining and len(selected) < count:
        anchor = anchors[anchor_index % len(anchors)]
        choice = min(
            remaining,
            key=lambda item: (
                _distance(item, anchor, parameters, value_lists),
                _signature(item),
            ),
        )
        selected.append(choice)
        remaining.remove(choice)
        anchor_index += 1
    return selected


def _select_diverse(
    candidates: list[dict[str, Any]],
    references: list[dict[str, Any]],
    count: int,
    parameters: list[dict[str, Any]],
    value_lists: list[list[Any]],
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    remaining = list(candidates)
    while remaining and len(selected) < count:
        active_references = [*references, *selected]
        choice = max(
            remaining,
            key=lambda item: (
                min(
                    (
                        _distance(item, reference, parameters, value_lists)
                        for reference in active_references
                    ),
                    default=0.0,
                ),
                _signature(item),
            ),
        )
        selected.append(choice)
        remaining.remove(choice)
    return selected


def _validated_prior(
    spec: dict[str, Any],
    prior: dict[str, Any],
    parameters: list[dict[str, Any]],
    value_lists: list[list[Any]],
) -> dict[str, Any]:
    unsigned = dict(prior)
    claimed_hash = unsigned.pop("prior_sha256", None)
    history = spec["history_prior"]
    runs = prior.get("selected_runs")
    source_hashes = prior.get("source_index_sha256")
    if (
        prior.get("schema_version") != 1
        or prior.get("event") != "historical_prior_merged"
        or prior.get("session_sha256") != _object_sha256(spec)
        or prior.get("wandb_project") != history["wandb_project"]
        or prior.get("max_selected_runs") != history["max_selected_runs"]
        or prior.get("max_points_per_run") != history["max_points_per_run"]
        or claimed_hash != _object_sha256(unsigned)
        or not isinstance(runs, list)
        or len(runs) > history["max_selected_runs"]
        or prior.get("selected_run_count") != len(runs)
        or not isinstance(source_hashes, list)
        or len(source_hashes) != len(set(source_hashes))
        or any(
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
            for value in source_hashes
        )
        or prior.get("source_git_mismatch_count")
        != sum(
            not bool(run.get("source_git_match"))
            for run in runs
            if isinstance(run, dict)
        )
    ):
        raise SpecError("historical prior fails session, limit, or hash binding")
    expected_parameters = {parameter["path"] for parameter in parameters}
    expected_metrics = set(history["metric_key_map"])
    seen_run_ids: set[str] = set()
    seen_overrides: set[str] = set()
    for run in runs:
        overrides = run.get("overrides") if isinstance(run, dict) else None
        metrics = run.get("metrics") if isinstance(run, dict) else None
        retained = run.get("retained_points") if isinstance(run, dict) else None
        run_id = run.get("run_id") if isinstance(run, dict) else None
        if (
            not isinstance(run, dict)
            or not isinstance(run_id, str)
            or not run_id
            or run_id in seen_run_ids
            or run.get("project") != history["wandb_project"]
            or run.get("status") not in {"completed", "failed"}
            or not isinstance(run.get("source_git_match"), bool)
            or run.get("worker_id") not in history["worker_roots"]
            or not isinstance(overrides, dict)
            or set(overrides) != expected_parameters
            or run.get("overrides_sha256") != _object_sha256(overrides)
            or run.get("overrides_sha256") in seen_overrides
            or not isinstance(metrics, dict)
            or set(metrics) != expected_metrics
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                for value in metrics.values()
            )
            or not isinstance(retained, dict)
            or set(retained) != expected_metrics
            or any(
                not isinstance(count, int)
                or isinstance(count, bool)
                or not 1 <= count <= history["max_points_per_run"]
                for count in retained.values()
            )
        ):
            raise SpecError("historical prior contains an invalid run")
        if run["status"] != "completed" and not history["include_failed_runs"]:
            raise SpecError("historical prior contains an unauthorized failed run")
        _grid_position(overrides, parameters, value_lists)
        seen_run_ids.add(run_id)
        seen_overrides.add(run["overrides_sha256"])
    return prior


def _trial_runs(
    spec: dict[str, Any],
    trials: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    seed = spec["tuning"]["seed_strategy"]["screening_seeds"][0]
    return [
        {
            "run_id": (
                f"{spec['training']['run_id']}--screening--"
                f"{trial['trial_id']}--seed-{seed}"
            ),
            "stage": "screening",
            "trial_id": trial["trial_id"],
            "seed": seed,
            "overrides": trial["overrides"],
        }
        for trial in trials
    ]


def build_plan(
    spec: dict[str, Any],
    history_prior: dict[str, Any] | None = None,
) -> dict[str, Any]:
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
    adaptive = spec.get("adaptive_search")
    if isinstance(adaptive, dict) and adaptive.get("enabled"):
        if history_prior is None:
            raise SpecError(
                "adaptive search requires a merged historical prior"
            )
        prior = _validated_prior(
            spec,
            history_prior,
            parameters,
            value_lists,
        )
        grid = _all_grid(
            parameters,
            value_lists,
            combination_count,
        )
        baseline_overrides = (
            _combination_at(baseline_index, value_lists)
            if baseline_index is not None
            else None
        )
        baseline_full = (
            {
                parameter["path"]: value
                for parameter, value in zip(
                    parameters,
                    baseline_overrides,
                    strict=True,
                )
            }
            if baseline_overrides is not None
            else {}
        )
        excluded_signatures = {
            _signature(run["overrides"])
            for run in prior["selected_runs"]
        }
        excluded_signatures.add(_signature(baseline_full))
        candidates = [
            item for item in grid if _signature(item) not in excluded_signatures
        ]
        slots = min(
            adaptive["trials_per_round"],
            tuning["max_trials"] - 1,
            len(candidates),
        )
        ranked_history = _rank_metric_records(
            spec,
            [
                run
                for run in prior["selected_runs"]
                if run["status"] == "completed"
            ],
        )
        historical_slots = min(
            math.floor(
                slots * spec["history_prior"]["max_first_round_fraction"]
            ),
            len(ranked_history),
        )
        near = _select_near_anchors(
            candidates,
            [item["overrides"] for item in ranked_history],
            historical_slots,
            parameters,
            value_lists,
        )
        near_signatures = {_signature(item) for item in near}
        diverse = _select_diverse(
            [
                item
                for item in candidates
                if _signature(item) not in near_signatures
            ],
            [
                baseline_full,
                *[run["overrides"] for run in prior["selected_runs"]],
                *near,
            ],
            slots - len(near),
            parameters,
            value_lists,
        )
        selected = [*near, *diverse]
        trials = [
            {"trial_id": "baseline", "overrides": {}, "seeds": tuning["seeds"]}
        ]
        trials.extend(
            {
                "trial_id": f"trial-{index:03d}",
                "overrides": overrides,
                "seeds": tuning["seeds"],
            }
            for index, overrides in enumerate(selected, start=1)
        )
        runs = _trial_runs(spec, trials)
        return {
            "version": 5,
            "mode": "tune",
            "run_id": spec["training"]["run_id"],
            "algorithm": spec["algorithm"],
            "mutation_scope": tuning["mutation_scope"],
            "authorized_parameter_paths": [
                parameter["path"] for parameter in parameters
            ],
            "full_grid_size": combination_count,
            "planned_trial_count": len(trials),
            "planned_run_count": len(runs),
            "trials": trials,
            "seed_strategy": tuning["seed_strategy"],
            "stages": {
                "screening": {
                    "status": "planned",
                    "seeds": tuning["seed_strategy"]["screening_seeds"],
                    "runs": copy.deepcopy(runs),
                },
                "confirmation": {
                    "status": "selection_only_after_screening",
                    "seeds": tuning["seed_strategy"]["confirmation_seeds"],
                    "remaining_seeds": [],
                    "confirmation_top_k": tuning["seed_strategy"][
                        "confirmation_top_k"
                    ],
                    "selected_trial_ids": [],
                    "runs": [],
                },
            },
            "runs": runs,
            "adaptive": {
                "history_prior": prior,
                "history_prior_sha256": prior["prior_sha256"],
                "max_rounds": adaptive["max_rounds"],
                "trials_per_round": adaptive["trials_per_round"],
                "exploration_fraction": adaptive["exploration_fraction"],
                "rounds": [
                    {
                        "round": 1,
                        "source": "bounded_historical_prior",
                        "history_influenced_trial_count": len(near),
                        "diverse_trial_count": len(diverse),
                        "trial_ids": [
                            trial["trial_id"] for trial in trials[1:]
                        ],
                    }
                ],
            },
        }

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
    base = {
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
    if spec["version"] < 6:
        return base

    strategy = tuning["seed_strategy"]
    strategy_mode = strategy.get("mode", "robust_multi_seed")
    if len(trials) - 1 < strategy["confirmation_top_k"]:
        raise SpecError(
            "authorized grid contains fewer non-baseline trials than "
            "confirmation_top_k"
        )
    screening_seeds = strategy["screening_seeds"]
    confirmation_seeds = strategy["confirmation_seeds"]
    screening_runs = [
        {
            "run_id": (
                f"{spec['training']['run_id']}--screening--"
                f"{trial['trial_id']}--seed-{seed}"
            ),
            "stage": "screening",
            "trial_id": trial["trial_id"],
            "seed": seed,
            "overrides": trial["overrides"],
        }
        for trial in trials
        for seed in screening_seeds
    ]
    remaining_confirmation = [
        seed for seed in confirmation_seeds if seed not in screening_seeds
    ]
    maximum_confirmation_runs = (
        strategy["confirmation_top_k"] + 1
    ) * len(remaining_confirmation)
    return {
        **base,
        "version": 4,
        "seed_strategy": strategy,
        "stages": {
            "screening": {
                "status": "planned",
                "seeds": screening_seeds,
                "runs": screening_runs,
            },
            "confirmation": {
                "status": (
                    "selection_only_after_screening"
                    if strategy_mode == "fixed_single_seed"
                    else "awaiting_screening_selection"
                ),
                "seeds": confirmation_seeds,
                "remaining_seeds": remaining_confirmation,
                "confirmation_top_k": strategy["confirmation_top_k"],
                "selected_trial_ids": [],
                "runs": [],
            },
        },
        "runs": screening_runs,
        "planned_run_count": len(screening_runs) + maximum_confirmation_runs,
    }


def build_confirmation_runs(
    spec: dict[str, Any],
    plan: dict[str, Any],
    selected_trial_ids: list[str],
) -> list[dict[str, Any]]:
    """Build remaining-seed runs, or none for fixed-single-seed selection."""
    if spec.get("version", 0) < 6 or plan.get("version") not in {4, 5}:
        raise SpecError(
            "confirmation staging requires version-6-or-newer session and version-4 plan"
        )
    top_k = spec["tuning"]["seed_strategy"]["confirmation_top_k"]
    if (
        len(selected_trial_ids) != top_k
        or "baseline" in selected_trial_ids
        or len(set(selected_trial_ids)) != len(selected_trial_ids)
    ):
        raise SpecError("confirmation selection must contain exact unique non-baseline top_k")
    trials = {trial["trial_id"]: trial for trial in plan["trials"]}
    unknown = sorted(set(selected_trial_ids) - set(trials))
    if unknown:
        raise SpecError(f"confirmation selection contains unknown trials: {unknown}")
    chosen = ["baseline", *selected_trial_ids]
    remaining = plan["stages"]["confirmation"]["remaining_seeds"]
    return [
        {
            "run_id": (
                f"{spec['training']['run_id']}--confirmation--"
                f"{trial_id}--seed-{seed}"
            ),
            "stage": "confirmation",
            "trial_id": trial_id,
            "seed": seed,
            "overrides": trials[trial_id]["overrides"],
        }
        for trial_id in chosen
        for seed in remaining
    ]


def extend_adaptive_plan(
    spec: dict[str, Any],
    plan: dict[str, Any],
    completed_results: list[dict[str, Any]],
    *,
    validate_existing: bool = True,
) -> dict[str, Any]:
    """Append one deterministic round without changing prior plan entries."""
    if validate_existing:
        validate_trial_plan(spec, plan)
    adaptive = plan.get("adaptive")
    if plan.get("version") != 5 or not isinstance(adaptive, dict):
        raise SpecError("adaptive round extension requires a version-5 plan")
    rounds = adaptive["rounds"]
    if len(rounds) >= adaptive["max_rounds"]:
        raise SpecError("adaptive search already reached max_rounds")
    expected_trial_ids = {trial["trial_id"] for trial in plan["trials"]}
    by_trial: dict[str, dict[str, Any]] = {}
    for result in completed_results:
        if (
            not isinstance(result, dict)
            or result.get("status") != "completed"
            or result.get("seed")
            != spec["tuning"]["seed_strategy"]["screening_seeds"][0]
            or result.get("trial_id") not in expected_trial_ids
            or not isinstance(result.get("metrics"), dict)
        ):
            raise SpecError(
                "adaptive results must be completed fixed-seed plan results"
            )
        trial_id = result["trial_id"]
        if trial_id in by_trial:
            raise SpecError("adaptive results contain duplicate trial IDs")
        by_trial[trial_id] = result
    if set(by_trial) != expected_trial_ids:
        raise SpecError(
            "adaptive round requires one completed result for every existing trial"
        )
    parameters = spec["tuning"]["allowed_parameters"]
    value_lists = [_expand_parameter(parameter) for parameter in parameters]
    combination_count = math.prod(len(values) for values in value_lists)
    grid = _all_grid(parameters, value_lists, combination_count)
    baseline_full = {
        parameter["path"]: parameter.get("baseline")
        for parameter in parameters
    }
    trial_overrides = {
        trial["trial_id"]: (
            baseline_full if trial["trial_id"] == "baseline" else trial["overrides"]
        )
        for trial in plan["trials"]
    }
    records = [
        {
            "trial_id": trial_id,
            "run_id": result.get("run_id", trial_id),
            "overrides": trial_overrides[trial_id],
            "metrics": result["metrics"],
        }
        for trial_id, result in by_trial.items()
    ]
    ranked = _rank_metric_records(spec, records)
    if not ranked:
        raise SpecError("adaptive round has no constraint-satisfying anchor")
    excluded = {
        _signature(overrides) for overrides in trial_overrides.values()
    } | {
        _signature(run["overrides"])
        for run in adaptive["history_prior"]["selected_runs"]
    }
    candidates = [
        item for item in grid if _signature(item) not in excluded
    ]
    remaining_budget = spec["tuning"]["max_trials"] - len(plan["trials"])
    slots = min(
        adaptive["trials_per_round"],
        remaining_budget,
        len(candidates),
    )
    if slots <= 0:
        raise SpecError("adaptive search has no remaining trial budget or grid")
    exploration_count = min(
        slots,
        math.ceil(slots * adaptive["exploration_fraction"]),
    )
    exploitation_count = slots - exploration_count
    near = _select_near_anchors(
        candidates,
        [item["overrides"] for item in ranked],
        exploitation_count,
        parameters,
        value_lists,
    )
    near_signatures = {_signature(item) for item in near}
    diverse = _select_diverse(
        [
            item
            for item in candidates
            if _signature(item) not in near_signatures
        ],
        [*trial_overrides.values(), *near],
        exploration_count,
        parameters,
        value_lists,
    )
    chosen = [*near, *diverse]
    expanded = copy.deepcopy(plan)
    start = len(expanded["trials"])
    new_trials = [
        {
            "trial_id": f"trial-{index:03d}",
            "overrides": overrides,
            "seeds": spec["tuning"]["seeds"],
        }
        for index, overrides in enumerate(chosen, start=start)
    ]
    new_runs = _trial_runs(spec, new_trials)
    expanded["trials"].extend(new_trials)
    expanded["runs"].extend(new_runs)
    expanded["stages"]["screening"]["runs"].extend(new_runs)
    expanded["planned_trial_count"] = len(expanded["trials"])
    expanded["planned_run_count"] = len(expanded["runs"])
    result_snapshot = [
        by_trial[trial_id] for trial_id in sorted(by_trial)
    ]
    expanded["adaptive"]["rounds"].append(
        {
            "round": len(rounds) + 1,
            "source": "completed_round_results",
            "input_results": result_snapshot,
            "input_results_sha256": _object_sha256(result_snapshot),
            "exploitation_trial_count": len(near),
            "diverse_trial_count": len(diverse),
            "trial_ids": [trial["trial_id"] for trial in new_trials],
        }
    )
    return expanded


def validate_trial_plan(
    spec: dict[str, Any],
    plan: dict[str, Any],
) -> None:
    """Rebuild a static or adaptive plan from its immutable inputs."""
    if spec.get("adaptive_search") is None:
        if plan != build_plan(spec):
            raise SpecError(
                "trial plan does not exactly match the validated session"
            )
        return
    adaptive = plan.get("adaptive")
    if plan.get("version") != 5 or not isinstance(adaptive, dict):
        raise SpecError("adaptive session requires a version-5 trial plan")
    prior = adaptive.get("history_prior")
    if not isinstance(prior, dict):
        raise SpecError("adaptive plan is missing its historical prior")
    expected = build_plan(spec, prior)
    original_rounds = adaptive.get("rounds")
    if not isinstance(original_rounds, list) or not original_rounds:
        raise SpecError("adaptive plan rounds are invalid")
    for round_record in original_rounds[1:]:
        results = round_record.get("input_results")
        if (
            not isinstance(results, list)
            or round_record.get("input_results_sha256")
            != _object_sha256(results)
        ):
            raise SpecError("adaptive round result snapshot hash is invalid")
        expected = extend_adaptive_plan(
            spec,
            expected,
            results,
            validate_existing=False,
        )
    if plan != expected:
        raise SpecError(
            "adaptive trial plan is not the deterministic authorized expansion"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session", help="Validated tune session JSON")
    parser.add_argument(
        "--history-prior",
        help="Merged bounded historical prior required by adaptive search",
    )
    parser.add_argument("--output", help="Optional JSON output path")
    args = parser.parse_args()
    try:
        spec = load_and_validate(args.session)
        history_prior = None
        if args.history_prior is not None:
            history_prior = json.loads(
                Path(args.history_prior).read_text(encoding="utf-8")
            )
            if not isinstance(history_prior, dict):
                raise SpecError("historical prior must be a JSON object")
        plan = build_plan(spec, history_prior)
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
