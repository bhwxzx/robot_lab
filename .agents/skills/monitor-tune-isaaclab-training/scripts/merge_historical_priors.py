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
        index.get("schema_version") != 1
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
    for run in index["selected_runs"]:
        if not isinstance(run, dict):
            raise SpecError("worker history selected run must be an object")
        run_id = run.get("run_id")
        overrides = run.get("overrides")
        metrics = run.get("metrics")
        retained = run.get("retained_points")
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
        ):
            raise SpecError("worker history selected run fails bounded schema")
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
            not bool(run.get("source_git_match")),
            str(run.get("observed_at", "")),
            str(run.get("run_id", "")),
        ),
        reverse=False,
    )
    exact_source = [
        run for run in candidates if bool(run.get("source_git_match"))
    ]
    older_source = [
        run for run in candidates if not bool(run.get("source_git_match"))
    ]
    exact_source.sort(
        key=lambda run: (run["observed_at"], run["run_id"]),
        reverse=True,
    )
    older_source.sort(
        key=lambda run: (run["observed_at"], run["run_id"]),
        reverse=True,
    )
    selected: list[dict[str, Any]] = []
    seen_runs: set[str] = set()
    seen_overrides: set[str] = set()
    for run in [*exact_source, *older_source]:
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
        "schema_version": 1,
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
