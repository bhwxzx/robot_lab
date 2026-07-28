#!/usr/bin/env python3
"""Build a deterministic, bounded trial plan from an authorized tune session."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import re
from decimal import Decimal
from pathlib import Path
from typing import Any

from validate_session_spec import SpecError, load_and_validate


SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


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


def _select_near_with_provenance(
    candidates: list[dict[str, Any]],
    anchors: list[dict[str, Any]],
    count: int,
    parameters: list[dict[str, Any]],
    value_lists: list[list[Any]],
    *,
    source: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    selected: list[dict[str, Any]] = []
    provenance: list[dict[str, Any]] = []
    remaining = list(candidates)
    if not anchors or count <= 0:
        return selected, provenance
    anchor_index = 0
    while remaining and len(selected) < count:
        anchor = anchors[anchor_index % len(anchors)]
        choice = min(
            remaining,
            key=lambda item: (
                _distance(
                    item,
                    anchor["overrides"],
                    parameters,
                    value_lists,
                ),
                _signature(item),
            ),
        )
        distance = _distance(
            choice,
            anchor["overrides"],
            parameters,
            value_lists,
        )
        selected.append(choice)
        provenance.append(
            {
                "selection_type": source,
                "anchor_id": anchor["anchor_id"],
                "anchor_distance": distance,
                "overrides_sha256": _signature(choice),
                "historical_duplicate": False,
                "previous_trial_duplicate": False,
            }
        )
        remaining.remove(choice)
        anchor_index += 1
    return selected, provenance


def _select_diverse_with_provenance(
    candidates: list[dict[str, Any]],
    references: list[dict[str, Any]],
    count: int,
    parameters: list[dict[str, Any]],
    value_lists: list[list[Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    selected: list[dict[str, Any]] = []
    provenance: list[dict[str, Any]] = []
    remaining = list(candidates)
    while remaining and len(selected) < count:
        active_references = [*references, *selected]

        def minimum_distance(item: dict[str, Any]) -> float:
            return min(
                (
                    _distance(
                        item,
                        reference,
                        parameters,
                        value_lists,
                    )
                    for reference in active_references
                ),
                default=0.0,
            )

        choice = max(
            remaining,
            key=lambda item: (
                minimum_distance(item),
                _signature(item),
            ),
        )
        distance = minimum_distance(choice)
        selected.append(choice)
        provenance.append(
            {
                "selection_type": "diverse_exploration",
                "anchor_id": None,
                "anchor_distance": distance,
                "overrides_sha256": _signature(choice),
                "historical_duplicate": False,
                "previous_trial_duplicate": False,
            }
        )
        remaining.remove(choice)
    return selected, provenance


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
        prior.get("schema_version") != 2
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
        or prior.get("guidance_eligible_count")
        != sum(
            bool(run.get("guidance_eligible"))
            for run in runs
            if isinstance(run, dict)
        )
    ):
        raise SpecError("historical prior fails session, limit, or hash binding")
    expected_parameters = {parameter["path"] for parameter in parameters}
    expected_metrics = set(history["metric_key_map"])
    expected_context = history["compatibility"]["expected_context"]
    source_policy = history["compatibility"]["source_policy"]
    quality_gates = history["quality_gates"]
    seen_run_ids: set[str] = set()
    seen_overrides: set[str] = set()
    for run in runs:
        overrides = run.get("overrides") if isinstance(run, dict) else None
        metrics = run.get("metrics") if isinstance(run, dict) else None
        retained = run.get("retained_points") if isinstance(run, dict) else None
        quality = run.get("quality") if isinstance(run, dict) else None
        statistics = (
            quality.get("metric_statistics")
            if isinstance(quality, dict)
            else None
        )
        observed_context = (
            run.get("observed_context")
            if isinstance(run, dict)
            else None
        )
        run_id = run.get("run_id") if isinstance(run, dict) else None
        context_match = (
            isinstance(observed_context, dict)
            and set(observed_context) == set(expected_context)
            and all(
                type(observed_context[key]) is type(expected_context[key])
                and observed_context[key] == expected_context[key]
                for key in expected_context
            )
        )
        approved_source_commit = spec["training"].get("source_git_commit")
        source_git_match = (
            isinstance(approved_source_commit, str)
            and isinstance(run, dict)
            and run.get("source_git_commit") == approved_source_commit
        )
        guidance_eligible = (
            source_git_match
            or context_match
            if isinstance(run, dict)
            else False
        )
        if (
            not isinstance(run, dict)
            or not isinstance(run_id, str)
            or not run_id
            or run_id in seen_run_ids
            or run.get("project") != history["wandb_project"]
            or run.get("status") not in {"completed", "failed"}
            or not isinstance(run.get("source_git_match"), bool)
            or run.get("source_git_match") is not source_git_match
            or run.get("source_policy") != source_policy
            or not isinstance(observed_context, dict)
            or set(observed_context) != set(expected_context)
            or run.get("context_match") is not context_match
            or run.get("guidance_eligible") is not guidance_eligible
            or (source_policy == "exact" and not source_git_match)
            or (source_policy == "compatible" and not guidance_eligible)
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
            or not isinstance(quality, dict)
            or set(quality)
            != {"passed", "final_progress", "metric_statistics"}
            or quality.get("passed") is not True
            or isinstance(quality.get("final_progress"), bool)
            or not isinstance(quality.get("final_progress"), (int, float))
            or not math.isfinite(float(quality["final_progress"]))
            or quality["final_progress"]
            < quality_gates["minimum_final_progress"]
            or not isinstance(statistics, dict)
            or set(statistics) != expected_metrics
        ):
            raise SpecError("historical prior contains an invalid run")
        for metric, metric_statistics in statistics.items():
            if (
                not isinstance(metric_statistics, dict)
                or set(metric_statistics)
                != {
                    "count",
                    "mean",
                    "standard_deviation",
                    "slope",
                }
                or metric_statistics.get("count") != retained[metric]
                or any(
                    isinstance(metric_statistics.get(field), bool)
                    or not isinstance(
                        metric_statistics.get(field),
                        (int, float),
                    )
                    or not math.isfinite(
                        float(metric_statistics[field])
                    )
                    for field in (
                        "mean",
                        "standard_deviation",
                        "slope",
                    )
                )
                or not math.isclose(
                    float(metric_statistics["mean"]),
                    float(metrics[metric]),
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                )
            ):
                raise SpecError(
                    "historical prior contains invalid metric statistics"
                )
        stability = quality_gates["stability"]
        stability_statistics = statistics[stability["metric"]]
        if (
            any(
                count < quality_gates["minimum_points_per_metric"]
                for count in retained.values()
            )
            or stability_statistics["standard_deviation"]
            > stability["max_standard_deviation"]
            or abs(float(stability_statistics["slope"]))
            > stability["max_abs_slope"]
        ):
            raise SpecError("historical prior run fails quality gates")
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


def _multifidelity_run(
    spec: dict[str, Any],
    trial: dict[str, Any],
    rung: int,
    target_budget: int,
    resume_from: dict[str, Any] | None,
) -> dict[str, Any]:
    seed = spec["tuning"]["seed_strategy"]["screening_seeds"][0]
    return {
        "run_id": (
            f"{spec['training']['run_id']}--fidelity-rung-{rung:03d}--"
            f"{trial['trial_id']}--seed-{seed}"
        ),
        "stage": f"fidelity-rung-{rung:03d}",
        "trial_id": trial["trial_id"],
        "seed": seed,
        "overrides": trial["overrides"],
        "rung": rung,
        "target_budget": target_budget,
        "resume_from": resume_from,
    }


def _build_multifidelity_plan(
    spec: dict[str, Any],
    base: dict[str, Any],
    trials: list[dict[str, Any]],
) -> dict[str, Any]:
    contract = spec["multi_fidelity"]
    first_budget = contract["rungs"][0]["budget"]
    runs = [
        _multifidelity_run(spec, trial, 1, first_budget, None)
        for trial in trials
    ]
    seed = spec["tuning"]["seed_strategy"]["screening_seeds"][0]
    return {
        **base,
        "version": 6,
        "seed_strategy": spec["tuning"]["seed_strategy"],
        "planned_run_count": len(runs),
        "stages": {
            "screening": {
                "status": "multi_fidelity_rung_1",
                "seeds": [seed],
                "runs": copy.deepcopy(runs),
            },
            "confirmation": {
                "status": "selection_only_after_multi_fidelity",
                "seeds": [seed],
                "remaining_seeds": [],
                "confirmation_top_k": spec["tuning"]["seed_strategy"][
                    "confirmation_top_k"
                ],
                "selected_trial_ids": [],
                "runs": [],
            },
        },
        "runs": runs,
        "multi_fidelity": {
            "status": "running",
            "stop_reason": None,
            "metric": contract["metric"],
            "minimum_margin": contract["minimum_margin"],
            "minimum_rungs_before_performance_pruning": contract[
                "minimum_rungs_before_performance_pruning"
            ],
            "required_consecutive_underperformance": contract[
                "required_consecutive_underperformance"
            ],
            "resume_same_worker": contract["resume_same_worker"],
            "schedule": copy.deepcopy(contract["rungs"]),
            "decisions": [],
            "rungs": [
                {
                    "rung": 1,
                    "budget": first_budget,
                    "source": "initial_authorized_trials",
                    "trial_ids": [trial["trial_id"] for trial in trials],
                    "run_ids": [run["run_id"] for run in runs],
                }
            ],
        },
    }


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
        guidance_runs = [
            run
            for run in prior["selected_runs"]
            if run["guidance_eligible"]
        ]
        excluded_signatures = {
            _signature(run["overrides"])
            for run in guidance_runs
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
                for run in guidance_runs
                if run["status"] == "completed"
            ],
        )
        historical_slots = min(
            math.floor(
                slots * spec["history_prior"]["max_first_round_fraction"]
            ),
            len(ranked_history),
        )
        near, near_provenance = _select_near_with_provenance(
            candidates,
            [
                {
                    "anchor_id": item["run_id"],
                    "overrides": item["overrides"],
                }
                for item in ranked_history
            ],
            historical_slots,
            parameters,
            value_lists,
            source="historical_exploitation",
        )
        near_signatures = {_signature(item) for item in near}
        diverse, diverse_provenance = _select_diverse_with_provenance(
            [
                item
                for item in candidates
                if _signature(item) not in near_signatures
            ],
            [
                baseline_full,
                *[run["overrides"] for run in guidance_runs],
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
        selection_provenance = [
            {
                **record,
                "trial_id": trial["trial_id"],
            }
            for trial, record in zip(
                trials[1:],
                [*near_provenance, *diverse_provenance],
                strict=True,
            )
        ]
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
                "status": "running",
                "stop_reason": None,
                "history_prior": prior,
                "history_prior_sha256": prior["prior_sha256"],
                "max_rounds": adaptive["max_rounds"],
                "trials_per_round": adaptive["trials_per_round"],
                "exploration_fraction": adaptive["exploration_fraction"],
                "stop_policy": adaptive["stop_policy"],
                "decisions": [],
                "rounds": [
                    {
                        "round": 1,
                        "source": "bounded_historical_prior",
                        "history_influenced_trial_count": len(near),
                        "diverse_trial_count": len(diverse),
                        "trial_ids": [
                            trial["trial_id"] for trial in trials[1:]
                        ],
                        "selection_provenance": selection_provenance,
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

    if spec.get("multi_fidelity") is not None:
        return _build_multifidelity_plan(spec, base, trials)

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


def _adaptive_stop_decision(
    spec: dict[str, Any],
    plan: dict[str, Any],
    records: list[dict[str, Any]],
    result_snapshot: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    adaptive = plan["adaptive"]
    stop_policy = adaptive["stop_policy"]
    metric = stop_policy["metric"]
    objective = next(
        item
        for item in spec["tuning"]["objectives"]
        if item["metric"] == metric
    )
    goal = objective["goal"]
    ranked = _rank_metric_records(spec, records)
    current_trial_ids = set(adaptive["rounds"][-1]["trial_ids"])
    current = [
        item for item in ranked if item["trial_id"] in current_trial_ids
    ]
    previous = [
        item for item in ranked if item["trial_id"] not in current_trial_ids
    ]

    def best(items: list[dict[str, Any]]) -> dict[str, Any] | None:
        if not items:
            return None
        record = (
            max(items, key=lambda item: float(item["metrics"][metric]))
            if goal == "maximize"
            else min(items, key=lambda item: float(item["metrics"][metric]))
        )
        return {
            "trial_id": record["trial_id"],
            "value": float(record["metrics"][metric]),
        }

    current_best = best(current)
    previous_best = best(previous)
    if current_best is None:
        improvement: float | None = None
        improvement_met = False
    elif previous_best is None:
        improvement = None
        improvement_met = True
    else:
        improvement = (
            current_best["value"] - previous_best["value"]
            if goal == "maximize"
            else previous_best["value"] - current_best["value"]
        )
        improvement_met = (
            improvement >= stop_policy["minimum_improvement"]
        )
    prior_decisions = adaptive["decisions"]
    previous_trailing = (
        prior_decisions[-1]["trailing_no_improvement_rounds"]
        if prior_decisions
        else 0
    )
    trailing = 0 if improvement_met else previous_trailing + 1
    remaining_budget = (
        spec["tuning"]["max_trials"] - len(plan["trials"])
    )
    reason = "continue"
    action = "continue"
    if not ranked:
        action, reason = "stop", "no_constraint_satisfying_trial"
    elif len(ranked) < stop_policy["minimum_feasible_trials"]:
        action, reason = "stop", "insufficient_feasible_trials"
    elif len(adaptive["rounds"]) >= adaptive["max_rounds"]:
        action, reason = "stop", "max_rounds_reached"
    elif remaining_budget <= 0:
        action, reason = "stop", "trial_budget_exhausted"
    elif not candidates:
        action, reason = "stop", "authorized_grid_exhausted"
    elif trailing >= stop_policy["patience_rounds"]:
        action, reason = "stop", "no_improvement_patience_reached"
    return {
        "decision": len(prior_decisions) + 1,
        "evaluated_round": len(adaptive["rounds"]),
        "input_results": result_snapshot,
        "input_results_sha256": _object_sha256(result_snapshot),
        "metric": metric,
        "goal": goal,
        "minimum_improvement": stop_policy["minimum_improvement"],
        "current_round_best": current_best,
        "previous_best": previous_best,
        "improvement": improvement,
        "improvement_met": improvement_met,
        "trailing_no_improvement_rounds": trailing,
        "feasible_trial_count": len(ranked),
        "remaining_candidate_count": len(candidates),
        "action": action,
        "reason": reason,
    }


def extend_adaptive_plan(
    spec: dict[str, Any],
    plan: dict[str, Any],
    completed_results: list[dict[str, Any]],
    *,
    validate_existing: bool = True,
) -> dict[str, Any]:
    """Evaluate one round, then append candidates or a deterministic stop."""
    if validate_existing:
        validate_trial_plan(spec, plan)
    adaptive = plan.get("adaptive")
    if plan.get("version") != 5 or not isinstance(adaptive, dict):
        raise SpecError("adaptive round extension requires a version-5 plan")
    if adaptive.get("status") != "running":
        raise SpecError("adaptive search is already stopped")
    rounds = adaptive["rounds"]
    expected_trial_ids = {trial["trial_id"] for trial in plan["trials"]}
    required_metrics = {
        item["metric"] for item in spec["tuning"]["objectives"]
    } | {
        item["metric"] for item in spec["tuning"]["constraints"]
    }
    by_trial: dict[str, dict[str, Any]] = {}
    for result in completed_results:
        metrics = result.get("metrics") if isinstance(result, dict) else None
        if (
            not isinstance(result, dict)
            or result.get("status") != "completed"
            or result.get("seed")
            != spec["tuning"]["seed_strategy"]["screening_seeds"][0]
            or result.get("trial_id") not in expected_trial_ids
            or not isinstance(metrics, dict)
            or not set(metrics) >= required_metrics
            or any(
                isinstance(metrics[metric], bool)
                or not isinstance(metrics[metric], (int, float))
                or not math.isfinite(float(metrics[metric]))
                for metric in required_metrics
            )
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
    excluded = {
        _signature(overrides) for overrides in trial_overrides.values()
    } | {
        _signature(run["overrides"])
        for run in adaptive["history_prior"]["selected_runs"]
        if run["guidance_eligible"]
    }
    candidates = [
        item for item in grid if _signature(item) not in excluded
    ]
    result_snapshot = [
        by_trial[trial_id] for trial_id in sorted(by_trial)
    ]
    decision = _adaptive_stop_decision(
        spec,
        plan,
        records,
        result_snapshot,
        candidates,
    )
    expanded = copy.deepcopy(plan)
    expanded["adaptive"]["decisions"].append(decision)
    if decision["action"] == "stop":
        expanded["adaptive"]["status"] = "stopped"
        expanded["adaptive"]["stop_reason"] = decision["reason"]
        return expanded

    remaining_budget = spec["tuning"]["max_trials"] - len(plan["trials"])
    slots = min(
        adaptive["trials_per_round"],
        remaining_budget,
        len(candidates),
    )
    if slots <= 0 or not ranked:
        raise SpecError(
            "adaptive continue decision has no anchor, budget, or candidate"
        )
    exploration_count = min(
        slots,
        math.ceil(slots * adaptive["exploration_fraction"]),
    )
    exploitation_count = slots - exploration_count
    near, near_provenance = _select_near_with_provenance(
        candidates,
        [
            {
                "anchor_id": item["trial_id"],
                "overrides": item["overrides"],
            }
            for item in ranked
        ],
        exploitation_count,
        parameters,
        value_lists,
        source="current_trial_exploitation",
    )
    near_signatures = {_signature(item) for item in near}
    diverse, diverse_provenance = _select_diverse_with_provenance(
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
    selection_provenance = [
        {
            **record,
            "trial_id": trial["trial_id"],
        }
        for trial, record in zip(
            new_trials,
            [*near_provenance, *diverse_provenance],
            strict=True,
        )
    ]
    expanded["trials"].extend(new_trials)
    expanded["runs"].extend(new_runs)
    expanded["stages"]["screening"]["runs"].extend(new_runs)
    expanded["planned_trial_count"] = len(expanded["trials"])
    expanded["planned_run_count"] = len(expanded["runs"])
    expanded["adaptive"]["rounds"].append(
        {
            "round": len(rounds) + 1,
            "source": "completed_round_results",
            "decision": decision["decision"],
            "decision_sha256": _object_sha256(decision),
            "exploitation_trial_count": len(near),
            "diverse_trial_count": len(diverse),
            "trial_ids": [trial["trial_id"] for trial in new_trials],
            "selection_provenance": selection_provenance,
        }
    )
    return expanded


def _validated_fidelity_results(
    spec: dict[str, Any],
    plan: dict[str, Any],
    completed_results: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    fidelity = plan["multi_fidelity"]
    current_rung = len(fidelity["rungs"])
    expected_runs = {
        run["run_id"]: run
        for run in plan["runs"]
        if run.get("rung") == current_rung
    }
    required_metrics = {
        item["metric"] for item in spec["tuning"]["objectives"]
    } | {
        item["metric"] for item in spec["tuning"]["constraints"]
    }
    by_run: dict[str, dict[str, Any]] = {}
    for result in completed_results:
        run_id = result.get("run_id") if isinstance(result, dict) else None
        run = expected_runs.get(run_id)
        metrics = result.get("metrics") if isinstance(result, dict) else None
        checkpoint = (
            result.get("checkpoint") if isinstance(result, dict) else None
        )
        if (
            not isinstance(result, dict)
            or set(result)
            != {
                "run_id",
                "trial_id",
                "seed",
                "status",
                "metrics",
                "checkpoint",
                "rung",
                "target_budget",
            }
            or not isinstance(run, dict)
            or run_id in by_run
            or result.get("trial_id") != run["trial_id"]
            or result.get("seed") != run["seed"]
            or result.get("status") != "completed"
            or result.get("rung") != current_rung
            or result.get("target_budget") != run["target_budget"]
            or not isinstance(metrics, dict)
            or not set(metrics) >= required_metrics
            or any(
                isinstance(metrics[metric], bool)
                or not isinstance(metrics[metric], (int, float))
                or not math.isfinite(float(metrics[metric]))
                for metric in required_metrics
            )
            or not isinstance(checkpoint, dict)
            or set(checkpoint)
            != {"path", "sha256", "step", "rsl_rl_run_dir"}
            or not isinstance(checkpoint.get("path"), str)
            or not Path(checkpoint["path"]).is_absolute()
            or not isinstance(checkpoint.get("sha256"), str)
            or SHA256_RE.fullmatch(checkpoint["sha256"]) is None
            or isinstance(checkpoint.get("step"), bool)
            or not isinstance(checkpoint.get("step"), int)
            or checkpoint["step"] < 0
            or not isinstance(checkpoint.get("rsl_rl_run_dir"), str)
            or not Path(checkpoint["rsl_rl_run_dir"]).is_absolute()
        ):
            raise SpecError(
                "multi-fidelity results must exactly cover the current rung "
                "with finite metrics and checkpoint evidence"
            )
        resume = run.get("resume_from")
        if (
            isinstance(resume, dict)
            and checkpoint["step"] <= resume["checkpoint_step"]
        ):
            raise SpecError(
                "multi-fidelity checkpoint progress must increase after resume"
            )
        by_run[run_id] = copy.deepcopy(result)
    if set(by_run) != set(expected_runs):
        raise SpecError(
            "multi-fidelity advancement requires one result for every current "
            "rung run"
        )
    snapshot = [by_run[run_id] for run_id in sorted(by_run)]
    return snapshot, expected_runs


def _fidelity_gap(
    value: float,
    best: float,
    goal: str,
) -> float:
    return best - value if goal == "maximize" else value - best


def _all_constraints_pass(
    spec: dict[str, Any],
    metrics: dict[str, Any],
) -> bool:
    return all(
        _constraint_passes(metrics, constraint)
        for constraint in spec["tuning"]["constraints"]
    )


def advance_multifidelity_plan(
    spec: dict[str, Any],
    plan: dict[str, Any],
    completed_results: list[dict[str, Any]],
    *,
    validate_existing: bool = True,
) -> dict[str, Any]:
    """Append one synchronized rung or an immutable terminal decision."""
    if validate_existing:
        validate_trial_plan(spec, plan)
    fidelity = plan.get("multi_fidelity")
    if plan.get("version") != 6 or not isinstance(fidelity, dict):
        raise SpecError(
            "multi-fidelity advancement requires a version-6 trial plan"
        )
    if fidelity.get("status") != "running":
        raise SpecError("multi-fidelity training is already terminal")
    snapshot, expected_runs = _validated_fidelity_results(
        spec,
        plan,
        completed_results,
    )
    current_rung = len(fidelity["rungs"])
    schedule = fidelity["schedule"]
    results_by_trial = {
        result["trial_id"]: result for result in snapshot
    }
    records = [
        {
            "trial_id": result["trial_id"],
            "run_id": result["run_id"],
            "overrides": expected_runs[result["run_id"]]["overrides"],
            "metrics": result["metrics"],
        }
        for result in snapshot
    ]
    ranked = _rank_metric_records(spec, records)
    ranked_ids = [record["trial_id"] for record in ranked]
    baseline_result = results_by_trial.get("baseline")
    if baseline_result is None:
        raise SpecError("multi-fidelity rung is missing the baseline")
    baseline_feasible = _all_constraints_pass(
        spec,
        baseline_result["metrics"],
    )
    metric = fidelity["metric"]
    objective = next(
        item
        for item in spec["tuning"]["objectives"]
        if item["metric"] == metric
    )
    feasible_values = [
        float(result["metrics"][metric])
        for result in snapshot
        if _all_constraints_pass(spec, result["metrics"])
    ]
    best_value = (
        (
            max(feasible_values)
            if objective["goal"] == "maximize"
            else min(feasible_values)
        )
        if feasible_values
        else None
    )
    previous_streaks = (
        {
            record["trial_id"]: record[
                "consecutive_underperformance"
            ]
            for record in fidelity["decisions"][-1]["trial_records"]
        }
        if fidelity["decisions"]
        else {}
    )
    pruning_allowed = (
        current_rung
        >= fidelity["minimum_rungs_before_performance_pruning"]
    )
    trial_records: list[dict[str, Any]] = []
    for result in snapshot:
        trial_id = result["trial_id"]
        feasible = _all_constraints_pass(spec, result["metrics"])
        gap = (
            _fidelity_gap(
                float(result["metrics"][metric]),
                float(best_value),
                objective["goal"],
            )
            if feasible and best_value is not None
            else None
        )
        underperforming = (
            isinstance(gap, float)
            and gap > float(fidelity["minimum_margin"])
        )
        streak = (
            previous_streaks.get(trial_id, 0) + 1
            if underperforming
            else 0
        )
        trial_records.append(
            {
                "trial_id": trial_id,
                "run_id": result["run_id"],
                "feasible": feasible,
                "metric_value": float(result["metrics"][metric]),
                "gap_from_best": gap,
                "underperforming": underperforming,
                "consecutive_underperformance": streak,
                "protected_from_performance_pruning": (
                    trial_id == "baseline"
                    or (
                        feasible
                        and (
                            not pruning_allowed
                            or streak
                            < fidelity[
                                "required_consecutive_underperformance"
                            ]
                        )
                    )
                ),
                "promoted": False,
                "disposition": None,
            }
        )
    by_trial_record = {
        record["trial_id"]: record for record in trial_records
    }
    target = schedule[current_rung - 1]["target_promoted_candidates"]
    candidate_ids = [
        trial_id for trial_id in ranked_ids if trial_id != "baseline"
    ]
    feasible_candidates = [
        trial_id
        for trial_id in candidate_ids
        if by_trial_record[trial_id]["feasible"]
    ]
    terminal_action: str | None = None
    reason = "continue"
    selected_trial_ids: list[str] = []
    promoted_candidates: list[str] = []
    if not baseline_feasible:
        terminal_action, reason = "stop", "baseline_constraint_failure"
    elif current_rung == len(schedule):
        if feasible_candidates:
            terminal_action, reason = "complete", "final_rung_completed"
            selected_trial_ids = feasible_candidates[
                : spec["tuning"]["seed_strategy"]["confirmation_top_k"]
            ]
        else:
            terminal_action, reason = "stop", "no_feasible_candidate"
    elif not feasible_candidates:
        terminal_action, reason = "stop", "no_feasible_candidate"
    else:
        protected = [
            trial_id
            for trial_id in feasible_candidates
            if by_trial_record[trial_id][
                "protected_from_performance_pruning"
            ]
        ]
        promoted_candidates = list(protected)
        for trial_id in feasible_candidates:
            if (
                len(promoted_candidates) >= target
                or trial_id in promoted_candidates
            ):
                continue
            promoted_candidates.append(trial_id)
    promoted_ids = (
        ["baseline", *promoted_candidates]
        if terminal_action is None
        else []
    )
    for record in trial_records:
        trial_id = record["trial_id"]
        record["promoted"] = trial_id in promoted_ids
        if trial_id in promoted_ids:
            record["disposition"] = "promoted"
        elif not record["feasible"]:
            record["disposition"] = "hard_constraint_eliminated"
        elif terminal_action == "complete":
            record["disposition"] = (
                "final_selected"
                if trial_id in selected_trial_ids
                else "final_not_selected"
            )
        elif terminal_action == "stop":
            record["disposition"] = "campaign_stopped"
        else:
            record["disposition"] = "performance_eliminated"
    decision = {
        "decision": len(fidelity["decisions"]) + 1,
        "evaluated_rung": current_rung,
        "input_results": snapshot,
        "input_results_sha256": _object_sha256(snapshot),
        "metric": metric,
        "goal": objective["goal"],
        "best_value": best_value,
        "minimum_margin": fidelity["minimum_margin"],
        "pruning_allowed": pruning_allowed,
        "target_promoted_candidates": target,
        "promoted_trial_ids": promoted_ids,
        "selected_trial_ids": selected_trial_ids,
        "protected_candidate_count": sum(
            record["protected_from_performance_pruning"]
            and record["trial_id"] != "baseline"
            for record in trial_records
        ),
        "target_exceeded_for_safety": (
            terminal_action is None
            and len(promoted_candidates) > target
        ),
        "trial_records": trial_records,
        "action": terminal_action or "continue",
        "reason": reason,
    }
    expanded = copy.deepcopy(plan)
    expanded["multi_fidelity"]["decisions"].append(decision)
    if terminal_action is not None:
        expanded["multi_fidelity"]["status"] = (
            "completed" if terminal_action == "complete" else "stopped"
        )
        expanded["multi_fidelity"]["stop_reason"] = (
            None if terminal_action == "complete" else reason
        )
        expanded["stages"]["confirmation"]["selected_trial_ids"] = (
            selected_trial_ids
        )
        expanded["stages"]["confirmation"]["status"] = (
            "single_seed_selection_complete"
            if terminal_action == "complete"
            else "multi_fidelity_stopped"
        )
        return expanded

    trials = {trial["trial_id"]: trial for trial in plan["trials"]}
    next_rung = current_rung + 1
    next_budget = schedule[next_rung - 1]["budget"]
    new_runs: list[dict[str, Any]] = []
    for trial_id in promoted_ids:
        parent = results_by_trial[trial_id]
        checkpoint = parent["checkpoint"]
        resume_from = {
            "parent_run_id": parent["run_id"],
            "checkpoint_path": checkpoint["path"],
            "checkpoint_sha256": checkpoint["sha256"],
            "checkpoint_step": checkpoint["step"],
            "rsl_rl_run_dir": checkpoint["rsl_rl_run_dir"],
        }
        new_runs.append(
            _multifidelity_run(
                spec,
                trials[trial_id],
                next_rung,
                next_budget,
                resume_from,
            )
        )
    expanded["runs"].extend(new_runs)
    expanded["planned_run_count"] = len(expanded["runs"])
    expanded["stages"]["screening"]["status"] = (
        f"multi_fidelity_rung_{next_rung}"
    )
    expanded["stages"]["screening"]["runs"].extend(copy.deepcopy(new_runs))
    expanded["multi_fidelity"]["rungs"].append(
        {
            "rung": next_rung,
            "budget": next_budget,
            "source": "synchronized_rung_promotion",
            "decision": decision["decision"],
            "decision_sha256": _object_sha256(decision),
            "trial_ids": promoted_ids,
            "run_ids": [run["run_id"] for run in new_runs],
        }
    )
    return expanded


def validate_trial_plan(
    spec: dict[str, Any],
    plan: dict[str, Any],
) -> None:
    """Rebuild a static or adaptive plan from its immutable inputs."""
    if spec.get("multi_fidelity") is not None:
        fidelity = plan.get("multi_fidelity")
        if plan.get("version") != 6 or not isinstance(fidelity, dict):
            raise SpecError(
                "multi-fidelity session requires a version-6 trial plan"
            )
        expected = build_plan(spec)
        decisions = fidelity.get("decisions")
        if not isinstance(decisions, list):
            raise SpecError("multi-fidelity decisions must be an array")
        for decision in decisions:
            results = (
                decision.get("input_results")
                if isinstance(decision, dict)
                else None
            )
            if (
                not isinstance(results, list)
                or decision.get("input_results_sha256")
                != _object_sha256(results)
            ):
                raise SpecError(
                    "multi-fidelity rung result snapshot hash is invalid"
                )
            expected = advance_multifidelity_plan(
                spec,
                expected,
                results,
                validate_existing=False,
            )
        if plan != expected:
            raise SpecError(
                "multi-fidelity plan is not the deterministic authorized "
                "expansion"
            )
        return
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
    original_decisions = adaptive.get("decisions")
    if (
        not isinstance(original_rounds, list)
        or not original_rounds
        or not isinstance(original_decisions, list)
    ):
        raise SpecError("adaptive plan rounds are invalid")
    for decision in original_decisions:
        results = decision.get("input_results")
        if (
            not isinstance(results, list)
            or decision.get("input_results_sha256")
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
