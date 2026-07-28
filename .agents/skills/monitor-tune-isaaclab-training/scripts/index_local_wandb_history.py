#!/usr/bin/env python3
"""Build a small, read-only history prior from selected local W&B runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from collections import deque
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from build_trial_plan import _expand_parameter
from validate_session_spec import SpecError, load_and_validate


RUN_DIR_RE = re.compile(
    r"^(?:offline-)?run-(?P<date>\d{8})_(?P<time>\d{6})-"
    r"(?P<run_id>[A-Za-z0-9_.-]+)$"
)


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
        raise SpecError("history index contains non-finite JSON") from exc


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(value, dict):
        flattened: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str) or not key:
                continue
            path = f"{prefix}.{key}" if prefix else key
            flattened.update(_flatten(item, path))
        return flattened
    return {prefix: value} if prefix else {}


def _item_path(item: Any) -> str:
    nested = list(item.nested_key)
    if nested:
        return ".".join(nested)
    return item.key


def _decode_json(text: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise SpecError("local W&B record contains invalid JSON") from exc


def _scan_wandb_file(
    path: Path,
    config_path_map: dict[str, str],
    context_path_map: dict[str, str],
    metric_key_map: dict[str, str],
    progress_key: str,
    max_points: int,
) -> dict[str, Any]:
    try:
        from wandb.proto import wandb_internal_pb2
        from wandb.sdk.internal.datastore import DataStore
    except ImportError as exc:
        raise SpecError(
            "the approved isaacsim-5.1 environment cannot read local W&B records"
        ) from exc

    store = DataStore()
    try:
        store.open_for_scan(str(path))
    except Exception as exc:
        raise SpecError(f"cannot open local W&B record: {exc}") from exc
    config: dict[str, Any] = {}
    metric_samples = {
        metric: deque(maxlen=max_points) for metric in metric_key_map
    }
    run_identity: dict[str, Any] | None = None
    exit_code: int | None = None
    record_count = 0

    def update_config(items: Any) -> None:
        for item in items:
            path_key = _item_path(item)
            if path_key:
                config[path_key] = _decode_json(item.value_json)

    try:
        while True:
            data = store.scan_data()
            if data is None:
                break
            record_count += 1
            record = wandb_internal_pb2.Record()
            record.ParseFromString(data)
            kind = record.WhichOneof("record_type")
            if kind == "run":
                run_identity = {
                    "run_id": record.run.run_id,
                    "display_name": record.run.display_name,
                    "project": record.run.project,
                    "source_git_commit": (
                        record.run.git.commit or None
                    ),
                }
                update_config(record.run.config.update)
            elif kind == "config":
                update_config(record.config.update)
            elif kind == "history":
                by_key: dict[str, Any] = {}
                for item in record.history.item:
                    path_key = (
                        "/".join(item.nested_key)
                        if item.nested_key
                        else item.key
                    )
                    if path_key:
                        by_key[path_key] = _decode_json(item.value_json)
                progress = by_key.get(progress_key)
                if (
                    isinstance(progress, bool)
                    or not isinstance(progress, (int, float))
                    or not math.isfinite(float(progress))
                ):
                    continue
                for metric, wandb_key in metric_key_map.items():
                    value = by_key.get(wandb_key)
                    if (
                        not isinstance(value, bool)
                        and isinstance(value, (int, float))
                        and math.isfinite(float(value))
                    ):
                        metric_samples[metric].append(
                            (float(progress), float(value))
                        )
            elif kind == "exit":
                exit_code = int(record.exit.exit_code)
    except Exception as exc:
        if isinstance(exc, SpecError):
            raise
        raise SpecError(f"cannot scan local W&B record: {exc}") from exc
    finally:
        store.close()

    flattened = _flatten(config)
    overrides: dict[str, Any] = {}
    for parameter_path, wandb_path in config_path_map.items():
        if wandb_path not in flattened:
            raise SpecError(
                f"W&B config is missing approved path {wandb_path}"
            )
        overrides[parameter_path] = flattened[wandb_path]
    observed_context: dict[str, Any] = {}
    for context_key, wandb_path in context_path_map.items():
        if wandb_path not in flattened:
            raise SpecError(
                f"W&B config is missing compatibility path {wandb_path}"
            )
        observed_context[context_key] = flattened[wandb_path]
    metric_statistics: dict[str, dict[str, float | int]] = {}
    for metric, samples in metric_samples.items():
        if not samples:
            raise SpecError(f"W&B history is missing finite metric {metric}")
        x_values = [item[0] for item in samples]
        y_values = [item[1] for item in samples]
        mean = sum(y_values) / len(y_values)
        variance = sum((value - mean) ** 2 for value in y_values) / len(
            y_values
        )
        x_mean = sum(x_values) / len(x_values)
        denominator = sum((value - x_mean) ** 2 for value in x_values)
        slope = (
            sum(
                (x_value - x_mean) * (y_value - mean)
                for x_value, y_value in samples
            )
            / denominator
            if denominator > 0
            else 0.0
        )
        metric_statistics[metric] = {
            "count": len(samples),
            "mean": mean,
            "standard_deviation": math.sqrt(variance),
            "slope": slope,
        }
    final_progress = min(
        max(sample[0] for sample in samples)
        for samples in metric_samples.values()
    )
    return {
        "identity": run_identity,
        "exit_code": exit_code,
        "overrides": overrides,
        "observed_context": observed_context,
        "metrics": {
            metric: float(statistics["mean"])
            for metric, statistics in metric_statistics.items()
        },
        "metric_statistics": metric_statistics,
        "final_progress": final_progress,
        "retained_points": {
            metric: len(samples)
            for metric, samples in metric_samples.items()
        },
        "record_count": record_count,
    }


def _allowed_values(spec: dict[str, Any]) -> dict[str, list[Any]]:
    return {
        parameter["path"]: _expand_parameter(parameter)
        for parameter in spec["tuning"]["allowed_parameters"]
    }


def _same_scalar(left: Any, right: Any) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return type(left) is type(right) and left == right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return math.isclose(
            float(left),
            float(right),
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
    return type(left) is type(right) and left == right


def _overrides_are_authorized(
    overrides: dict[str, Any],
    allowed: dict[str, list[Any]],
) -> bool:
    return set(overrides) == set(allowed) and all(
        any(_same_scalar(value, candidate) for candidate in allowed[path])
        for path, value in overrides.items()
    )


def _candidate_directories(
    root: Path,
    history: dict[str, Any],
    now: datetime,
) -> list[tuple[datetime, str, Path]]:
    if not root.is_dir() or root.is_symlink():
        raise SpecError(f"local W&B root is not a regular directory: {root}")
    explicit = set(history["explicit_run_ids"])
    cutoff = now - timedelta(days=history["lookback_days"])
    candidates: list[tuple[datetime, str, Path]] = []
    for child in root.iterdir():
        if not child.is_dir() or child.is_symlink():
            continue
        match = RUN_DIR_RE.fullmatch(child.name)
        if match is None:
            continue
        run_id = match.group("run_id")
        observed = datetime.strptime(
            match.group("date") + match.group("time"),
            "%Y%m%d%H%M%S",
        ).replace(tzinfo=now.tzinfo)
        if explicit:
            if run_id not in explicit:
                continue
        elif observed < cutoff:
            continue
        candidates.append((observed, run_id, child))
    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    if explicit:
        missing = sorted(explicit - {item[1] for item in candidates})
        if missing:
            raise SpecError(f"explicit local W&B run IDs were not found: {missing}")
        return candidates
    return candidates[: history["max_selected_runs"] * 2]


def build_history_index(
    spec: dict[str, Any],
    worker_id: str,
    *,
    now: datetime | None = None,
) -> dict[str, Any]:
    history = spec.get("history_prior")
    if not isinstance(history, dict) or not history.get("enabled"):
        raise SpecError("session does not enable a local W&B history prior")
    if worker_id not in history["worker_roots"]:
        raise SpecError("worker is not authorized for a local W&B history root")
    observed_now = now or datetime.now().astimezone()
    root = Path(history["worker_roots"][worker_id]).resolve()
    allowed = _allowed_values(spec)
    compatibility = history["compatibility"]
    quality_gates = history["quality_gates"]
    selected: list[dict[str, Any]] = []
    excluded: list[dict[str, str]] = []
    seen_overrides: set[str] = set()
    candidates = _candidate_directories(root, history, observed_now)
    for observed, directory_run_id, directory in candidates:
        if len(selected) >= history["max_selected_runs"]:
            break
        wandb_files = sorted(directory.glob("run-*.wandb"))
        if len(wandb_files) != 1 or wandb_files[0].is_symlink():
            excluded.append(
                {
                    "run_id": directory_run_id,
                    "reason": "run directory must contain exactly one regular .wandb file",
                }
            )
            continue
        try:
            scanned = _scan_wandb_file(
                wandb_files[0],
                history["config_path_map"],
                compatibility["context_path_map"],
                history["metric_key_map"],
                quality_gates["progress_key"],
                history["max_points_per_run"],
            )
            identity = scanned["identity"]
            if (
                not isinstance(identity, dict)
                or identity.get("run_id") != directory_run_id
                or identity.get("project") != history["wandb_project"]
            ):
                raise SpecError("W&B run identity or project is incompatible")
            completed = scanned["exit_code"] == 0
            if not completed and not history["include_failed_runs"]:
                raise SpecError("W&B run has no successful terminal exit record")
            if not _overrides_are_authorized(scanned["overrides"], allowed):
                raise SpecError(
                    "W&B parameters are incomplete or outside the approved grid"
                )
            overrides_sha = _sha256(scanned["overrides"])
            if overrides_sha in seen_overrides:
                raise SpecError("duplicate approved parameter combination")
            approved_source_commit = spec["training"].get(
                "source_git_commit"
            )
            source_git_match = (
                isinstance(approved_source_commit, str)
                and identity["source_git_commit"]
                == approved_source_commit
            )
            expected_context = compatibility["expected_context"]
            observed_context = scanned["observed_context"]
            context_match = (
                set(observed_context) == set(expected_context)
                and all(
                    _same_scalar(
                        observed_context[key],
                        expected_context[key],
                    )
                    for key in expected_context
                )
            )
            source_policy = compatibility["source_policy"]
            guidance_eligible = source_git_match or context_match
            if source_policy == "exact" and not source_git_match:
                raise SpecError(
                    "W&B source commit does not match exact history policy"
                )
            if source_policy == "compatible" and not guidance_eligible:
                raise SpecError(
                    "W&B run matches neither source commit nor compatibility context"
                )
            retained = scanned["retained_points"]
            if any(
                count < quality_gates["minimum_points_per_metric"]
                for count in retained.values()
            ):
                raise SpecError(
                    "W&B run has insufficient retained metric points"
                )
            if (
                scanned["final_progress"]
                < quality_gates["minimum_final_progress"]
            ):
                raise SpecError(
                    "W&B run did not reach the approved minimum progress"
                )
            stability = quality_gates["stability"]
            stability_statistics = scanned["metric_statistics"][
                stability["metric"]
            ]
            if (
                stability_statistics["standard_deviation"]
                > stability["max_standard_deviation"]
            ):
                raise SpecError(
                    "W&B stability metric exceeds the approved standard deviation"
                )
            if (
                abs(float(stability_statistics["slope"]))
                > stability["max_abs_slope"]
            ):
                raise SpecError(
                    "W&B stability metric exceeds the approved absolute slope"
                )
            stat = wandb_files[0].stat()
            seen_overrides.add(overrides_sha)
            selected.append(
                {
                    "run_id": directory_run_id,
                    "display_name": identity["display_name"],
                    "project": identity["project"],
                    "observed_at": observed.astimezone(UTC).isoformat().replace(
                        "+00:00",
                        "Z",
                    ),
                    "status": "completed" if completed else "failed",
                    "source_git_commit": identity["source_git_commit"],
                    "source_git_match": source_git_match,
                    "source_policy": source_policy,
                    "observed_context": observed_context,
                    "context_match": context_match,
                    "guidance_eligible": guidance_eligible,
                    "overrides": scanned["overrides"],
                    "overrides_sha256": overrides_sha,
                    "metrics": scanned["metrics"],
                    "retained_points": scanned["retained_points"],
                    "quality": {
                        "passed": True,
                        "final_progress": scanned["final_progress"],
                        "metric_statistics": scanned[
                            "metric_statistics"
                        ],
                    },
                    "evidence": {
                        "wandb_path": str(wandb_files[0].resolve()),
                        "size_bytes": stat.st_size,
                        "mtime_ns": stat.st_mtime_ns,
                        "record_count": scanned["record_count"],
                    },
                }
            )
        except (OSError, SpecError) as exc:
            excluded.append(
                {"run_id": directory_run_id, "reason": str(exc)}
            )
    base = {
        "schema_version": 2,
        "event": "local_wandb_history_indexed",
        "worker_id": worker_id,
        "session_sha256": _sha256(spec),
        "wandb_project": history["wandb_project"],
        "root": str(root),
        "lookback_days": history["lookback_days"],
        "max_selected_runs": history["max_selected_runs"],
        "max_points_per_run": history["max_points_per_run"],
        "candidate_read_limit": history["max_selected_runs"] * 2,
        "selected_runs": selected,
        "excluded_runs": excluded,
    }
    return {**base, "index_sha256": _sha256(base)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session")
    parser.add_argument("--worker-id", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    try:
        spec = load_and_validate(args.session)
        index = build_history_index(spec, args.worker_id)
        output = Path(args.output)
        if (
            not output.is_absolute()
            or output.exists()
            or not output.parent.is_dir()
            or output.parent.is_symlink()
        ):
            raise SpecError(
                "history index output must be a new absolute file under an "
                "existing regular parent"
            )
        output.write_bytes(
            json.dumps(
                index,
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
