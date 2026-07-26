#!/usr/bin/env python3
"""Validate closed-loop policy evaluation runs and simulation promotion gates."""

from __future__ import annotations

import argparse
import json
import math
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


def load_evaluation_plan(path: Path) -> dict[str, Any]:
    """Load a version-1 evaluation plan."""
    plan = _load_object(path, "evaluation plan")
    if plan.get("version") != 1 or not isinstance(plan.get("runs"), list):
        raise SpecError("evaluation plan must be version 1 with a runs array")
    if not isinstance(plan.get("candidate_ids"), list) or not plan["candidate_ids"]:
        raise SpecError("evaluation plan candidate_ids must be a non-empty array")
    return plan


def load_evaluation_results(path: Path) -> dict[str, Any]:
    """Load consolidated version-1 run results and visual reviews."""
    results = _load_object(path, "evaluation results")
    if results.get("version") != 1:
        raise SpecError("evaluation results version must be 1")
    if not isinstance(results.get("runs"), list):
        raise SpecError("evaluation results must contain a runs array")
    if not isinstance(results.get("visual_reviews"), list):
        raise SpecError("evaluation results must contain a visual_reviews array")
    return results


def _normalize_metrics(value: Any, path: str) -> dict[str, float]:
    if not isinstance(value, dict):
        raise SpecError(f"{path} must be an object")
    normalized: dict[str, float] = {}
    for name, metric_value in value.items():
        if not isinstance(name, str) or not name:
            raise SpecError(f"{path} contains an invalid metric name")
        if (
            isinstance(metric_value, bool)
            or not isinstance(metric_value, (int, float))
            or not math.isfinite(float(metric_value))
        ):
            raise SpecError(f"{path}.{name} must be a finite number")
        normalized[name] = float(metric_value)
    return normalized


def _selected(selector: list[str], value: str) -> bool:
    return "*" in selector or value in selector


def _aggregate(values: list[float], aggregation: str) -> float:
    if aggregation == "max":
        return max(values)
    if aggregation == "min":
        return min(values)
    return fmean(values)


def _validate_visual_reviews(
    spec: dict[str, Any],
    plan: dict[str, Any],
    results: dict[str, Any],
) -> dict[str, list[str]]:
    evaluation = spec["evaluation"]
    visual_spec = evaluation["visual_review"]
    required_by_candidate: dict[str, set[str]] = {
        candidate_id: set() for candidate_id in plan["candidate_ids"]
    }
    for run in plan["runs"]:
        if run.get("video_required"):
            required_by_candidate[run["candidate_id"]].add(run["video_path"])

    reviews_by_candidate: dict[str, list[dict[str, Any]]] = {}
    for index, review in enumerate(results["visual_reviews"]):
        path = f"visual_reviews[{index}]"
        if not isinstance(review, dict):
            raise SpecError(f"{path} must be an object")
        unknown = sorted(
            set(review)
            - {
                "candidate_id",
                "status",
                "reviewer",
                "reviewed_video_paths",
                "notes",
            }
        )
        if unknown:
            raise SpecError(f"{path} contains unknown field(s): {', '.join(unknown)}")
        candidate_id = review.get("candidate_id")
        if candidate_id not in required_by_candidate:
            raise SpecError(f"{path}.candidate_id is not in the evaluation plan")
        if review.get("status") not in {"pass", "fail"}:
            raise SpecError(f"{path}.status must be pass or fail")
        reviewer = review.get("reviewer")
        if not isinstance(reviewer, str) or not reviewer.strip():
            raise SpecError(f"{path}.reviewer must be a non-empty string")
        reviewed_paths = review.get("reviewed_video_paths")
        if not isinstance(reviewed_paths, list):
            raise SpecError(f"{path}.reviewed_video_paths must be an array")
        if len(reviewed_paths) != len(set(reviewed_paths)):
            raise SpecError(f"{path}.reviewed_video_paths must be unique")
        for reviewed_path in reviewed_paths:
            if reviewed_path not in required_by_candidate[candidate_id]:
                raise SpecError(
                    f"{path}.reviewed_video_paths contains an unexpected path"
                )
        notes = review.get("notes")
        if not isinstance(notes, str):
            raise SpecError(f"{path}.notes must be a string")
        if visual_spec["require_notes"] and not notes.strip():
            raise SpecError(f"{path}.notes must not be empty")
        reviews_by_candidate.setdefault(candidate_id, []).append(review)

    failures: dict[str, list[str]] = {
        candidate_id: [] for candidate_id in plan["candidate_ids"]
    }
    minimum = visual_spec["minimum_reviewed_videos"]
    for candidate_id in plan["candidate_ids"]:
        reviews = reviews_by_candidate.get(candidate_id, [])
        if len(reviews) != 1:
            failures[candidate_id].append(
                "exactly one visual review is required"
            )
            continue
        review = reviews[0]
        if review["status"] != "pass":
            failures[candidate_id].append("visual review failed")
        if len(review["reviewed_video_paths"]) < minimum:
            failures[candidate_id].append(
                f"visual review covered fewer than {minimum} required videos"
            )
    return failures


def evaluate_results(
    spec: dict[str, Any],
    plan: dict[str, Any],
    results: dict[str, Any],
) -> dict[str, Any]:
    """Return per-candidate promotion decisions; never claim hardware readiness."""
    evaluation = spec.get("evaluation")
    if not isinstance(evaluation, dict) or not evaluation.get("enabled"):
        raise SpecError("session evaluation must be enabled")
    if plan.get("algorithm") != spec["algorithm"]:
        raise SpecError("evaluation plan algorithm does not match the session")
    if plan.get("training_run_id") != spec["training"]["run_id"]:
        raise SpecError("evaluation plan training_run_id does not match the session")
    if plan.get("gates") != evaluation["gates"]:
        raise SpecError("evaluation plan gates do not match the session")
    if plan.get("minimum_reviewed_videos") != evaluation["visual_review"][
        "minimum_reviewed_videos"
    ]:
        raise SpecError("evaluation plan visual-review threshold does not match")

    expected_runs: dict[str, dict[str, Any]] = {}
    for index, run in enumerate(plan["runs"]):
        if not isinstance(run, dict) or not isinstance(run.get("run_id"), str):
            raise SpecError(f"evaluation plan runs[{index}] is invalid")
        if run["run_id"] in expected_runs:
            raise SpecError(f"duplicate plan run_id: {run['run_id']}")
        expected_runs[run["run_id"]] = run

    actual_runs: dict[str, dict[str, Any]] = {}
    for index, run in enumerate(results["runs"]):
        path = f"runs[{index}]"
        if not isinstance(run, dict):
            raise SpecError(f"{path} must be an object")
        run_id = run.get("run_id")
        if not isinstance(run_id, str) or not run_id:
            raise SpecError(f"{path}.run_id must be a non-empty string")
        if run_id not in expected_runs:
            raise SpecError(f"{path}.run_id is not in the evaluation plan")
        if run_id in actual_runs:
            raise SpecError(f"duplicate evaluation result run_id: {run_id}")
        expected = expected_runs[run_id]
        for key in ("candidate_id", "artifact", "scenario_id", "seed"):
            if run.get(key) != expected[key]:
                raise SpecError(f"{path}.{key} does not match the evaluation plan")
        status = run.get("status")
        if not isinstance(status, str) or not status:
            raise SpecError(f"{path}.status must be a non-empty string")
        metrics = (
            _normalize_metrics(run.get("metrics"), f"{path}.metrics")
            if status == "completed"
            else {}
        )
        video_path = run.get("video_path")
        if not isinstance(video_path, str):
            raise SpecError(f"{path}.video_path must be a string")
        actual_runs[run_id] = {
            **run,
            "metrics": metrics,
            "video_path": video_path,
        }

    visual_failures = _validate_visual_reviews(spec, plan, results)
    candidate_reports: list[dict[str, Any]] = []
    for candidate_id in plan["candidate_ids"]:
        reasons = list(visual_failures[candidate_id])
        gate_failures: list[dict[str, Any]] = []
        parity_failures: list[dict[str, Any]] = []
        required_plan_runs = [
            run
            for run in plan["runs"]
            if run["candidate_id"] == candidate_id
            and run["artifact_required"]
            and run["scenario_required"]
        ]
        completed: list[tuple[dict[str, Any], dict[str, Any]]] = []
        for expected in required_plan_runs:
            actual = actual_runs.get(expected["run_id"])
            if actual is None:
                reasons.append(f"missing required run {expected['run_id']}")
                continue
            if actual["status"] != "completed":
                reasons.append(
                    f"required run {expected['run_id']} has status "
                    f"{actual['status']}"
                )
                continue
            if expected["video_required"] and actual["video_path"] != expected[
                "video_path"
            ]:
                reasons.append(
                    f"required run {expected['run_id']} is missing its exact video"
                )
            elif expected["video_required"]:
                video_path = Path(actual["video_path"])
                if not video_path.is_file() or video_path.stat().st_size == 0:
                    reasons.append(
                        f"required run {expected['run_id']} video is absent or empty"
                    )
            completed.append((expected, actual))

        for gate in evaluation["gates"]:
            selected_runs = [
                actual
                for expected, actual in completed
                if _selected(gate["artifacts"], expected["artifact"])
                and _selected(gate["scenarios"], expected["scenario_id"])
            ]
            values: list[float] = []
            missing_metric_runs: list[str] = []
            for actual in selected_runs:
                if gate["metric"] not in actual["metrics"]:
                    missing_metric_runs.append(actual["run_id"])
                else:
                    values.append(actual["metrics"][gate["metric"]])
            if missing_metric_runs or not values:
                gate_failures.append(
                    {
                        "metric": gate["metric"],
                        "reason": "missing metric",
                        "runs": missing_metric_runs,
                    }
                )
                continue
            aggregated = _aggregate(values, gate["aggregation"])
            if not OPERATORS[gate["op"]](aggregated, float(gate["value"])):
                gate_failures.append(
                    {
                        "metric": gate["metric"],
                        "aggregation": gate["aggregation"],
                        "actual": aggregated,
                        "op": gate["op"],
                        "limit": float(gate["value"]),
                    }
                )

        if evaluation["parity"]["required"]:
            reference = evaluation["parity"]["reference_artifact"]
            limit = float(evaluation["parity"]["max_abs_action_error"])
            required_artifacts = {
                artifact["kind"]
                for artifact in evaluation["artifacts"]
                if artifact["required"] and artifact["kind"] != reference
            }
            for artifact in sorted(required_artifacts):
                artifact_runs = [
                    (expected, actual)
                    for expected, actual in completed
                    if expected["artifact"] == artifact
                ]
                values = [
                    actual["metrics"]["max_abs_action_error"]
                    for _, actual in artifact_runs
                    if "max_abs_action_error" in actual["metrics"]
                ]
                missing_parity_runs = [
                    actual["run_id"]
                    for _, actual in artifact_runs
                    if "max_abs_action_error" not in actual["metrics"]
                ]
                if not values or missing_parity_runs:
                    parity_failures.append(
                        {
                            "artifact": artifact,
                            "reason": "missing parity metric",
                            "runs": missing_parity_runs,
                        }
                    )
                elif max(values) > limit:
                    parity_failures.append(
                        {
                            "artifact": artifact,
                            "max_abs_action_error": max(values),
                            "limit": limit,
                        }
                    )
                reference_runs = {
                    (expected["scenario_id"], expected["seed"]): actual
                    for expected, actual in completed
                    if expected["artifact"] == reference
                }
                for metric_contract in evaluation["parity"].get(
                    "closed_loop_metrics",
                    [],
                ):
                    metric_name = metric_contract["metric"]
                    deltas: list[float] = []
                    missing_pairs: list[str] = []
                    for expected, actual in artifact_runs:
                        key = (expected["scenario_id"], expected["seed"])
                        reference_actual = reference_runs.get(key)
                        if (
                            reference_actual is None
                            or metric_name not in actual["metrics"]
                            or metric_name not in reference_actual["metrics"]
                        ):
                            missing_pairs.append(actual["run_id"])
                            continue
                        deltas.append(
                            abs(
                                actual["metrics"][metric_name]
                                - reference_actual["metrics"][metric_name]
                            )
                        )
                    if missing_pairs or not deltas:
                        parity_failures.append(
                            {
                                "artifact": artifact,
                                "metric": metric_name,
                                "reason": "missing closed-loop parity pair",
                                "runs": missing_pairs,
                            }
                        )
                        continue
                    aggregation = metric_contract["aggregation"]
                    aggregated_delta = _aggregate(deltas, aggregation)
                    maximum_delta = float(
                        metric_contract["max_abs_delta"]
                    )
                    if aggregated_delta > maximum_delta:
                        parity_failures.append(
                            {
                                "artifact": artifact,
                                "metric": metric_name,
                                "aggregation": aggregation,
                                "absolute_delta": aggregated_delta,
                                "limit": maximum_delta,
                            }
                        )

        if gate_failures:
            reasons.append("one or more metric gates failed")
        if parity_failures:
            reasons.append("one or more artifact parity gates failed")
        passed = not reasons
        candidate_reports.append(
            {
                "candidate_id": candidate_id,
                "status": (
                    "simulation_qualified_hardware_candidate"
                    if passed
                    else "simulation_rejected"
                ),
                "passed": passed,
                "reasons": reasons,
                "gate_failures": gate_failures,
                "parity_failures": parity_failures,
                "hardware_ready": False,
            }
        )

    return {
        "version": 1,
        "algorithm": spec["algorithm"],
        "candidate_results": candidate_reports,
        "simulation_qualified_candidates": [
            report["candidate_id"]
            for report in candidate_reports
            if report["passed"]
        ],
        "hardware_ready": False,
        "hardware_status": (
            "supervised_real_robot_testing_required"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session", help="Validated version-3-through-6 session JSON")
    parser.add_argument("plan", help="Evaluation plan JSON")
    parser.add_argument("results", help="Consolidated evaluation results JSON")
    parser.add_argument("--output", help="Optional JSON output path")
    args = parser.parse_args()
    try:
        report = evaluate_results(
            load_and_validate(args.session),
            load_evaluation_plan(Path(args.plan)),
            load_evaluation_results(Path(args.results)),
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
