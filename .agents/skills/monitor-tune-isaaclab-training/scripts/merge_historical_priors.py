#!/usr/bin/env python3
"""Merge bounded worker history indexes into one metadata-only prior."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

from validate_session_spec import SpecError, load_and_validate


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
        raise SpecError("history prior contains non-finite JSON") from exc


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _load_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise SpecError(f"history index does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise SpecError(
            f"invalid history index JSON at line {exc.lineno}: {exc.msg}"
        ) from exc
    if not isinstance(value, dict):
        raise SpecError("history index must be a JSON object")
    return value


def _validate_index(
    spec: dict[str, Any],
    index: dict[str, Any],
) -> dict[str, Any]:
    unsigned = dict(index)
    claimed_hash = unsigned.pop("index_sha256", None)
    worker_id = index.get("worker_id")
    history = spec["history_prior"]
    if (
        index.get("schema_version") != 2
        or index.get("event") != "local_wandb_history_indexed"
        or index.get("session_sha256") != _sha256(spec)
        or not isinstance(worker_id, str)
        or worker_id not in history["worker_roots"]
        or index.get("wandb_project") != history["wandb_project"]
        or index.get("root") != str(
            Path(history["worker_roots"][worker_id]).resolve()
        )
        or index.get("lookback_days") != history["lookback_days"]
        or index.get("max_selected_runs") != history["max_selected_runs"]
        or index.get("max_points_per_run") != history["max_points_per_run"]
        or index.get("candidate_read_limit")
        != history["max_selected_runs"] * 2
        or claimed_hash != _sha256(unsigned)
        or not isinstance(index.get("selected_runs"), list)
        or not isinstance(index.get("excluded_runs"), list)
    ):
        raise SpecError("worker history index fails session or hash binding")
    if len(index["selected_runs"]) > history["max_selected_runs"]:
        raise SpecError("worker history index exceeds the approved run limit")
    seen_runs: set[str] = set()
    seen_overrides: set[str] = set()
    expected_parameters = set(history["config_path_map"])
    expected_metrics = set(history["metric_key_map"])
    compatibility = history["compatibility"]
    expected_context = compatibility["expected_context"]
    quality_gates = history["quality_gates"]
    for run in index["selected_runs"]:
        if not isinstance(run, dict):
            raise SpecError("worker history selected run must be an object")
        run_id = run.get("run_id")
        overrides = run.get("overrides")
        metrics = run.get("metrics")
        retained = run.get("retained_points")
        observed_context = run.get("observed_context")
        quality = run.get("quality")
        metric_statistics = (
            quality.get("metric_statistics")
            if isinstance(quality, dict)
            else None
        )
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
            and run.get("source_git_commit") == approved_source_commit
        )
        guidance_eligible = source_git_match or context_match
        source_policy = compatibility["source_policy"]
        if (
            not isinstance(run_id, str)
            or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", run_id) is None
            or run_id in seen_runs
            or run.get("project") != history["wandb_project"]
            or run.get("status") not in {"completed", "failed"}
            or (
                run.get("status") == "failed"
                and not history["include_failed_runs"]
            )
            or not isinstance(run.get("source_git_match"), bool)
            or run.get("source_git_match") is not source_git_match
            or run.get("source_policy") != source_policy
            or not isinstance(observed_context, dict)
            or set(observed_context) != set(expected_context)
            or run.get("context_match") is not context_match
            or not isinstance(run.get("guidance_eligible"), bool)
            or run.get("guidance_eligible") is not guidance_eligible
            or (source_policy == "exact" and not source_git_match)
            or (source_policy == "compatible" and not guidance_eligible)
            or not isinstance(overrides, dict)
            or set(overrides) != expected_parameters
            or run.get("overrides_sha256") != _sha256(overrides)
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
            or set(quality) != {
                "passed",
                "final_progress",
                "metric_statistics",
            }
            or quality.get("passed") is not True
            or isinstance(quality.get("final_progress"), bool)
            or not isinstance(quality.get("final_progress"), (int, float))
            or not math.isfinite(float(quality["final_progress"]))
            or quality["final_progress"]
            < quality_gates["minimum_final_progress"]
            or not isinstance(metric_statistics, dict)
            or set(metric_statistics) != expected_metrics
            or any(
                not isinstance(statistics, dict)
                or set(statistics)
                != {
                    "count",
                    "mean",
                    "standard_deviation",
                    "slope",
                }
                or statistics.get("count") != retained[metric]
                or any(
                    isinstance(statistics.get(field), bool)
                    or not isinstance(
                        statistics.get(field),
                        (int, float),
                    )
                    or not math.isfinite(float(statistics[field]))
                    for field in (
                        "mean",
                        "standard_deviation",
                        "slope",
                    )
                )
                or not math.isclose(
                    float(statistics["mean"]),
                    float(metrics[metric]),
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                )
                for metric, statistics in (
                    metric_statistics.items()
                    if isinstance(metric_statistics, dict)
                    else []
                )
            )
        ):
            raise SpecError("worker history selected run fails bounded schema")
        stability = quality_gates["stability"]
        stability_statistics = metric_statistics[stability["metric"]]
        if (
            stability_statistics["standard_deviation"]
            > stability["max_standard_deviation"]
            or abs(float(stability_statistics["slope"]))
            > stability["max_abs_slope"]
            or any(
                count < quality_gates["minimum_points_per_metric"]
                for count in retained.values()
            )
        ):
            raise SpecError("worker history selected run fails quality gates")
        seen_runs.add(run_id)
        seen_overrides.add(run["overrides_sha256"])
    return index


def merge_history_indexes(
    spec: dict[str, Any],
    indexes: list[dict[str, Any]],
) -> dict[str, Any]:
    history = spec.get("history_prior")
    if not isinstance(history, dict) or not history.get("enabled"):
        raise SpecError("session does not enable history prior merging")
    if not indexes:
        raise SpecError("at least one worker history index is required")
    validated = [_validate_index(spec, index) for index in indexes]
    workers = [index["worker_id"] for index in validated]
    if len(workers) != len(set(workers)):
        raise SpecError("history merge received duplicate worker indexes")

    candidates: list[dict[str, Any]] = []
    for index in validated:
        for run in index["selected_runs"]:
            if not isinstance(run, dict):
                raise SpecError("history selected run must be an object")
            candidates.append({**run, "worker_id": index["worker_id"]})
    candidates.sort(
        key=lambda run: (
            bool(run.get("guidance_eligible")),
            bool(run.get("source_git_match")),
            run["observed_at"],
            run["run_id"],
        ),
        reverse=True,
    )
    selected: list[dict[str, Any]] = []
    seen_runs: set[str] = set()
    seen_overrides: set[str] = set()
    for run in candidates:
        run_id = run.get("run_id")
        overrides_hash = run.get("overrides_sha256")
        if (
            not isinstance(run_id, str)
            or not isinstance(overrides_hash, str)
            or run_id in seen_runs
            or overrides_hash in seen_overrides
        ):
            continue
        seen_runs.add(run_id)
        seen_overrides.add(overrides_hash)
        selected.append(run)
        if len(selected) >= history["max_selected_runs"]:
            break
    base = {
        "schema_version": 2,
        "event": "historical_prior_merged",
        "session_sha256": _sha256(spec),
        "wandb_project": history["wandb_project"],
        "max_selected_runs": history["max_selected_runs"],
        "max_points_per_run": history["max_points_per_run"],
        "source_index_sha256": sorted(
            index["index_sha256"] for index in validated
        ),
        "selected_runs": selected,
        "selected_run_count": len(selected),
        "source_git_mismatch_count": sum(
            not bool(run.get("source_git_match")) for run in selected
        ),
        "guidance_eligible_count": sum(
            bool(run.get("guidance_eligible")) for run in selected
        ),
    }
    return {**base, "prior_sha256": _sha256(base)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session")
    parser.add_argument("indexes", nargs="+")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    try:
        spec = load_and_validate(args.session)
        prior = merge_history_indexes(
            spec,
            [_load_object(Path(path)) for path in args.indexes],
        )
        output = Path(args.output)
        if (
            not output.is_absolute()
            or output.exists()
            or not output.parent.is_dir()
            or output.parent.is_symlink()
        ):
            raise SpecError(
                "history prior output must be a new absolute file under an "
                "existing regular parent"
            )
        output.write_bytes(
            json.dumps(
                prior,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            ).encode("utf-8")
            + b"\n"
        )
    except (OSError, SpecError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
