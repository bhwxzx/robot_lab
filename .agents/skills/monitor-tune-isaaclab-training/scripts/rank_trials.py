#!/usr/bin/env python3
"""Rank completed multi-seed trials using authorized objectives and constraints."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import fmean
from typing import Any

from validate_session_spec import SpecError, load_and_validate


OPERATORS = {
    "<=": lambda actual, limit: actual <= limit,
    ">=": lambda actual, limit: actual >= limit,
    "<": lambda actual, limit: actual < limit,
    ">": lambda actual, limit: actual > limit,
}


def _load_results(path: Path) -> list[dict[str, Any]]:
    try:
        root = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise SpecError(f"results file does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise SpecError(f"invalid results JSON at line {exc.lineno}: {exc.msg}") from exc
    if not isinstance(root, dict) or not isinstance(root.get("runs"), list):
        raise SpecError("results must be an object containing a runs array")
    return root["runs"]


def rank(
    spec: dict[str, Any],
    runs: list[dict[str, Any]],
    evaluation_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Aggregate seeds, enforce constraints, and score eligible trials."""
    if spec["mode"] != "tune":
        raise SpecError("ranking is only valid in tune mode")
    tuning = spec["tuning"]
    expected_seeds = set(tuning["seeds"])
    required_metrics = {
        objective["metric"] for objective in tuning["objectives"]
    } | {
        constraint["metric"] for constraint in tuning["constraints"]
    }

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
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
        if status != "completed":
            grouped[trial_id].append({"seed": seed, "status": status, "metrics": {}})
            continue
        if not isinstance(metrics, dict):
            raise SpecError(f"runs[{index}].metrics must be an object")
        normalized_metrics: dict[str, float] = {}
        for name, value in metrics.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise SpecError(f"runs[{index}].metrics.{name} must be a finite number")
            normalized_metrics[name] = float(value)
        grouped[trial_id].append({"seed": seed, "status": status, "metrics": normalized_metrics})

    eligible: list[dict[str, Any]] = []
    ineligible: list[dict[str, Any]] = []
    for trial_id, trial_runs in sorted(grouped.items()):
        reasons: list[str] = []
        completed_runs = [run for run in trial_runs if run["status"] == "completed"]
        completed_seeds = {run["seed"] for run in completed_runs}
        if completed_seeds != expected_seeds or len(completed_runs) != len(expected_seeds):
            reasons.append(
                f"completed seeds {sorted(completed_seeds)} do not match required seeds {sorted(expected_seeds)}"
            )
        for run in completed_runs:
            missing = sorted(required_metrics - set(run["metrics"]))
            if missing:
                reasons.append(f"seed {run['seed']} missing metrics: {', '.join(missing)}")
        if reasons:
            ineligible.append({"trial_id": trial_id, "reasons": reasons})
            continue

        aggregate = {
            metric: fmean(run["metrics"][metric] for run in completed_runs)
            for metric in sorted(required_metrics)
        }
        constraint_failures = []
        for constraint in tuning["constraints"]:
            actual = aggregate[constraint["metric"]]
            if not OPERATORS[constraint["op"]](actual, float(constraint["value"])):
                constraint_failures.append(
                    {
                        "metric": constraint["metric"],
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
                    "mean_metrics": aggregate,
                }
            )
            continue
        eligible.append({"trial_id": trial_id, "mean_metrics": aggregate})

    if not any(item["trial_id"] == "baseline" for item in eligible):
        raise SpecError("a complete constraint-satisfying baseline is required for ranking")

    total_weight = sum(float(objective["weight"]) for objective in tuning["objectives"])
    metric_ranges: dict[str, tuple[float, float]] = {}
    for objective in tuning["objectives"]:
        values = [item["mean_metrics"][objective["metric"]] for item in eligible]
        metric_ranges[objective["metric"]] = (min(values), max(values))

    baseline = next(item for item in eligible if item["trial_id"] == "baseline")
    for item in eligible:
        score = 0.0
        components: dict[str, float] = {}
        deltas: dict[str, float] = {}
        for objective in tuning["objectives"]:
            metric = objective["metric"]
            actual = item["mean_metrics"][metric]
            minimum, maximum = metric_ranges[metric]
            if maximum == minimum:
                normalized = 0.5
            elif objective["goal"] == "maximize":
                normalized = (actual - minimum) / (maximum - minimum)
            else:
                normalized = (maximum - actual) / (maximum - minimum)
            components[metric] = normalized
            deltas[metric] = actual - baseline["mean_metrics"][metric]
            score += normalized * float(objective["weight"])
        item["score"] = score / total_weight
        item["score_components"] = components
        item["delta_from_baseline"] = deltas

    eligible.sort(key=lambda item: (-item["score"], item["trial_id"]))
    result: dict[str, Any] = {
        "version": 3,
        "algorithm": spec["algorithm"],
        "eligible_trial_count": len(eligible),
        "ineligible_trial_count": len(ineligible),
        "baseline_mean_metrics": baseline["mean_metrics"],
        "ranking": eligible,
        "ineligible": ineligible,
        "final_selection": None,
        "hardware_ready": False,
    }
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
    qualified = set(evaluation_report.get("simulation_qualified_candidates", []))
    evaluated_ranking = [
        item for item in eligible if item["trial_id"] in qualified
    ]
    result["policy_evaluation"] = evaluation_report
    if not evaluated_ranking:
        result["selection_status"] = "no_simulation_qualified_candidate"
        return result
    result["selection_status"] = "simulation_qualified_hardware_candidate"
    result["final_selection"] = evaluated_ranking[0]
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
    encoded = json.dumps(ranked, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
    if args.output:
        Path(args.output).write_text(encoded + "\n", encoding="utf-8")
    else:
        print(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
