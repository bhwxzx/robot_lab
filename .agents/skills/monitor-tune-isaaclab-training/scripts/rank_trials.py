#!/usr/bin/env python3
"""Rank paired trials under robust-multi-seed or fixed-single-seed evidence."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import fmean, stdev
from typing import Any, Callable

from validate_session_spec import SpecError, load_and_validate


OPERATORS: dict[str, Callable[[float, float], bool]] = {
    "<=": lambda actual, limit: actual <= limit,
    ">=": lambda actual, limit: actual >= limit,
    "<": lambda actual, limit: actual < limit,
    ">": lambda actual, limit: actual > limit,
}
T_CRITICAL_95 = {
    2: 12.706,
    3: 4.303,
    4: 3.182,
    5: 2.776,
    6: 2.571,
    7: 2.447,
    8: 2.365,
    9: 2.306,
    10: 2.262,
    11: 2.228,
    12: 2.201,
    13: 2.179,
    14: 2.160,
    15: 2.145,
    16: 2.131,
}


def _load_results(path: Path) -> list[dict[str, Any]]:
    try:
        root = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise SpecError(f"results file does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise SpecError(
            f"invalid results JSON at line {exc.lineno}: {exc.msg}"
        ) from exc
    if not isinstance(root, dict) or not isinstance(root.get("runs"), list):
        raise SpecError("results must be an object containing a runs array")
    return root["runs"]


def _normalize_runs(
    runs: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen: set[tuple[str, int]] = set()
    for index, run in enumerate(runs):
        if not isinstance(run, dict):
            raise SpecError(f"runs[{index}] must be an object")
        trial_id = run.get("trial_id")
        seed = run.get("seed")
        status = run.get("status")
        metrics = run.get("metrics")
        if not isinstance(trial_id, str) or not trial_id:
            raise SpecError(f"runs[{index}].trial_id must be a non-empty string")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise SpecError(f"runs[{index}].seed must be an integer")
        key = (trial_id, seed)
        if key in seen:
            raise SpecError(f"duplicate result for trial {trial_id} seed {seed}")
        seen.add(key)
        if status != "completed":
            grouped[trial_id].append(
                {"seed": seed, "status": status, "metrics": {}}
            )
            continue
        if not isinstance(metrics, dict):
            raise SpecError(f"runs[{index}].metrics must be an object")
        normalized: dict[str, float] = {}
        for name, value in metrics.items():
            if (
                not isinstance(name, str)
                or not name
                or isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                raise SpecError(
                    f"runs[{index}].metrics.{name} must be a finite number"
                )
            normalized[name] = float(value)
        grouped[trial_id].append(
            {"seed": seed, "status": status, "metrics": normalized}
        )
    return grouped


def _stats(values: list[float]) -> dict[str, int | float | None]:
    mean = fmean(values)
    if len(values) < 2:
        deviation = 0.0
        low = None
        high = None
    else:
        deviation = stdev(values)
        margin = T_CRITICAL_95[len(values)] * deviation / math.sqrt(len(values))
        low = mean - margin
        high = mean + margin
    return {
        "count": len(values),
        "mean": mean,
        "std": deviation,
        "min": min(values),
        "max": max(values),
        "ci95_low": low,
        "ci95_high": high,
    }


def _objective_value(item: dict[str, Any], objective: dict[str, Any]) -> float:
    return float(item["metric_statistics"][objective["metric"]]["mean"])


def _pareto_front(
    eligible: list[dict[str, Any]],
    objectives: list[dict[str, Any]],
) -> set[str]:
    front: set[str] = set()
    for candidate in eligible:
        dominated = False
        for other in eligible:
            if other is candidate:
                continue
            no_worse = True
            strictly_better = False
            for objective in objectives:
                candidate_value = _objective_value(candidate, objective)
                other_value = _objective_value(other, objective)
                if objective["goal"] == "maximize":
                    no_worse &= other_value >= candidate_value
                    strictly_better |= other_value > candidate_value
                else:
                    no_worse &= other_value <= candidate_value
                    strictly_better |= other_value < candidate_value
            if no_worse and strictly_better:
                dominated = True
                break
        if not dominated:
            front.add(candidate["trial_id"])
    return front


def _aggregate(
    spec: dict[str, Any],
    runs: list[dict[str, Any]],
    expected_seeds: list[int],
    enforce_improvement: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    tuning = spec["tuning"]
    objectives = tuning["objectives"]
    constraints = tuning["constraints"]
    required_metrics = {
        objective["metric"] for objective in objectives
    } | {constraint["metric"] for constraint in constraints}
    grouped = _normalize_runs(runs)
    expected = set(expected_seeds)
    eligible: list[dict[str, Any]] = []
    ineligible: list[dict[str, Any]] = []
    for trial_id, trial_runs in sorted(grouped.items()):
        reasons: list[str] = []
        completed = [run for run in trial_runs if run["status"] == "completed"]
        by_seed = {run["seed"]: run for run in completed}
        if set(by_seed) != expected or len(completed) != len(expected):
            reasons.append(
                f"completed seeds {sorted(by_seed)} do not match "
                f"required seeds {sorted(expected)}"
            )
        for run in completed:
            missing = sorted(required_metrics - set(run["metrics"]))
            if missing:
                reasons.append(
                    f"seed {run['seed']} missing metrics: {', '.join(missing)}"
                )
        if reasons:
            ineligible.append({"trial_id": trial_id, "reasons": reasons})
            continue
        metric_statistics = {
            metric: _stats(
                [by_seed[seed]["metrics"][metric] for seed in expected_seeds]
            )
            for metric in sorted(required_metrics)
        }
        constraint_failures: list[dict[str, Any]] = []
        for constraint in constraints:
            metric = constraint["metric"]
            scope = constraint.get(
                "scope",
                "each_seed" if spec["version"] >= 6 else "mean",
            )
            if scope == "each_seed":
                failed = [
                    {
                        "seed": seed,
                        "actual": by_seed[seed]["metrics"][metric],
                    }
                    for seed in expected_seeds
                    if not OPERATORS[constraint["op"]](
                        by_seed[seed]["metrics"][metric],
                        float(constraint["value"]),
                    )
                ]
                if failed:
                    constraint_failures.append(
                        {
                            "metric": metric,
                            "scope": "each_seed",
                            "failures": failed,
                            "op": constraint["op"],
                            "limit": float(constraint["value"]),
                        }
                    )
            else:
                actual = float(metric_statistics[metric]["mean"])
                if not OPERATORS[constraint["op"]](
                    actual,
                    float(constraint["value"]),
                ):
                    constraint_failures.append(
                        {
                            "metric": metric,
                            "scope": "mean",
                            "actual": actual,
                            "op": constraint["op"],
                            "limit": float(constraint["value"]),
                        }
                    )
        if constraint_failures:
            ineligible.append(
                {
                    "trial_id": trial_id,
                    "reasons": ["constraint failure"],
                    "constraint_failures": constraint_failures,
                    "metric_statistics": metric_statistics,
                }
            )
            continue
        eligible.append(
            {
                "trial_id": trial_id,
                "metric_statistics": metric_statistics,
                "mean_metrics": {
                    metric: statistics["mean"]
                    for metric, statistics in metric_statistics.items()
                },
                "per_seed_metrics": {
                    str(seed): by_seed[seed]["metrics"]
                    for seed in expected_seeds
                },
            }
        )

    baseline = next(
        (item for item in eligible if item["trial_id"] == "baseline"),
        None,
    )
    if baseline is None:
        raise SpecError(
            "a complete constraint-satisfying baseline is required for ranking"
        )
    baseline_by_seed = baseline["per_seed_metrics"]
    for item in list(eligible):
        paired: dict[str, dict[str, float | None]] = {}
        failures: list[dict[str, Any]] = []
        for objective in objectives:
            metric = objective["metric"]
            improvements = []
            for seed in expected_seeds:
                baseline_value = baseline_by_seed[str(seed)][metric]
                candidate_value = item["per_seed_metrics"][str(seed)][metric]
                improvements.append(
                    candidate_value - baseline_value
                    if objective["goal"] == "maximize"
                    else baseline_value - candidate_value
                )
            paired[metric] = _stats(improvements)
            minimum = float(objective.get("minimum_improvement", 0.0))
            if (
                enforce_improvement
                and item["trial_id"] != "baseline"
                and "minimum_improvement" in objective
                and (
                    float(paired[metric]["mean"]) < minimum
                    or float(paired[metric]["min"]) < 0.0
                )
            ):
                failures.append(
                    {
                        "metric": metric,
                        "minimum_mean_improvement": minimum,
                        "actual_mean_improvement": paired[metric]["mean"],
                        "worst_seed_improvement": paired[metric]["min"],
                    }
                )
        item["paired_improvement"] = paired
        if failures:
            eligible.remove(item)
            ineligible.append(
                {
                    **item,
                    "reasons": ["minimum paired improvement failure"],
                    "improvement_failures": failures,
                }
            )
    if not eligible:
        raise SpecError("no eligible trials remain after robustness gates")
    return eligible, ineligible


def _score(
    eligible: list[dict[str, Any]],
    objectives: list[dict[str, Any]],
) -> None:
    total_weight = sum(float(item["weight"]) for item in objectives)
    ranges = {
        objective["metric"]: (
            min(_objective_value(item, objective) for item in eligible),
            max(_objective_value(item, objective) for item in eligible),
        )
        for objective in objectives
    }
    front = _pareto_front(eligible, objectives)
    for item in eligible:
        score = 0.0
        components: dict[str, float] = {}
        for objective in objectives:
            metric = objective["metric"]
            actual = _objective_value(item, objective)
            minimum, maximum = ranges[metric]
            if minimum == maximum:
                normalized = 0.5
            elif objective["goal"] == "maximize":
                normalized = (actual - minimum) / (maximum - minimum)
            else:
                normalized = (maximum - actual) / (maximum - minimum)
            components[metric] = normalized
            score += normalized * float(objective["weight"])
        item["score"] = score / total_weight
        item["score_components"] = components
        item["pareto_optimal"] = item["trial_id"] in front
    eligible.sort(
        key=lambda item: (
            not item["pareto_optimal"],
            -item["score"],
            item["trial_id"],
        )
    )


def select_confirmation_candidates(
    spec: dict[str, Any],
    runs: list[dict[str, Any]],
) -> list[str]:
    """Select exact top-k non-baseline trials after the approved screening seeds."""
    if spec.get("version", 0) < 6:
        raise SpecError(
            "staged confirmation selection requires session version 6 or newer"
        )
    eligible, _ = _aggregate(
        spec,
        runs,
        spec["tuning"]["seed_strategy"]["screening_seeds"],
        enforce_improvement=False,
    )
    _score(eligible, spec["tuning"]["objectives"])
    candidates = [
        item["trial_id"] for item in eligible if item["trial_id"] != "baseline"
    ]
    top_k = spec["tuning"]["seed_strategy"]["confirmation_top_k"]
    if len(candidates) < top_k:
        raise SpecError("screening produced fewer eligible candidates than top_k")
    return candidates[:top_k]


def rank(
    spec: dict[str, Any],
    runs: list[dict[str, Any]],
    evaluation_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Aggregate approved seeds, apply gates, and score eligible trials."""
    if spec["mode"] != "tune":
        raise SpecError("ranking is only valid in tune mode")
    expected_seeds = (
        spec["tuning"]["seed_strategy"]["confirmation_seeds"]
        if spec["version"] >= 6
        else spec["tuning"]["seeds"]
    )
    if spec["version"] >= 6:
        minimum = spec["tuning"]["ranking"]["minimum_final_training_seeds"]
        if len(expected_seeds) < minimum:
            raise SpecError("confirmation seed count is below final minimum")
    eligible, ineligible = _aggregate(
        spec,
        runs,
        expected_seeds,
        enforce_improvement=spec["version"] >= 6,
    )
    _score(eligible, spec["tuning"]["objectives"])
    baseline = next(item for item in eligible if item["trial_id"] == "baseline")
    result: dict[str, Any] = {
        "version": 4 if spec["version"] >= 6 else 3,
        "algorithm": spec["algorithm"],
        "expected_seeds": expected_seeds,
        "eligible_trial_count": len(eligible),
        "ineligible_trial_count": len(ineligible),
        "baseline_metric_statistics": baseline["metric_statistics"],
        "baseline_mean_metrics": baseline["mean_metrics"],
        "pareto_front": [
            item["trial_id"] for item in eligible if item["pareto_optimal"]
        ],
        "ranking": eligible,
        "ineligible": ineligible,
        "final_selection": None,
        "hardware_ready": False,
    }
    seed_mode = (
        spec["tuning"]["seed_strategy"].get("mode", "robust_multi_seed")
        if spec["version"] >= 6
        else "legacy"
    )
    result["seed_strategy_mode"] = seed_mode
    result["training_evidence"] = (
        "single_seed_selected"
        if seed_mode == "fixed_single_seed"
        else "robust_multi_seed_ranked"
    )
    if seed_mode == "fixed_single_seed":
        result["final_authority"] = "supervised_hardware"
        result["generalization_claim"] = False
        result["policy_acceptance_status"] = "awaiting_supervised_hardware"
    evaluation = spec.get("evaluation")
    if not isinstance(evaluation, dict) or not evaluation.get("enabled"):
        result["selection_status"] = "training_ranking_only"
        result["recommended_for_evaluation"] = eligible[0]["trial_id"]
        return result
    if evaluation_report is None:
        result["selection_status"] = "awaiting_policy_evaluation"
        result["recommended_for_evaluation"] = eligible[0]["trial_id"]
        return result
    if evaluation_report.get("algorithm") != spec["algorithm"]:
        raise SpecError("policy evaluation report algorithm does not match session")
    qualified = set(
        evaluation_report.get("simulation_qualified_candidates", [])
    )
    evaluated = [
        item
        for item in eligible
        if item["pareto_optimal"] and item["trial_id"] in qualified
    ]
    result["policy_evaluation"] = evaluation_report
    if not evaluated:
        result["selection_status"] = "no_simulation_qualified_candidate"
        return result
    result["selection_status"] = "simulation_qualified_hardware_candidate"
    result["final_selection"] = evaluated[0]
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session", help="Validated tune session JSON")
    parser.add_argument("results", help="Per-trial, per-seed results JSON")
    parser.add_argument("--evaluation-plan", help="Optional policy evaluation plan")
    parser.add_argument(
        "--evaluation-results",
        help="Optional consolidated policy evaluation results",
    )
    parser.add_argument("--output", help="Optional ranked JSON output path")
    args = parser.parse_args()
    try:
        spec = load_and_validate(args.session)
        if bool(args.evaluation_plan) != bool(args.evaluation_results):
            raise SpecError(
                "--evaluation-plan and --evaluation-results must be provided together"
            )
        evaluation_report = None
        if args.evaluation_plan:
            from validate_policy_evaluation import (
                evaluate_results,
                load_evaluation_plan,
                load_evaluation_results,
            )

            evaluation_report = evaluate_results(
                spec,
                load_evaluation_plan(Path(args.evaluation_plan)),
                load_evaluation_results(Path(args.evaluation_results)),
            )
        ranked = rank(
            spec,
            _load_results(Path(args.results)),
            evaluation_report=evaluation_report,
        )
    except SpecError as exc:
        parser.error(str(exc))
    encoded = json.dumps(
        ranked,
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
