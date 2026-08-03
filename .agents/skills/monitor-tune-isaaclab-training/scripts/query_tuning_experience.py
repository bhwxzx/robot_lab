#!/usr/bin/env python3
"""Query immutable tuning history without changing files or external state."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
from datetime import datetime
from pathlib import Path
from typing import Any

from record_tuning_experience import ExperienceError, SLUG_RE, validate_event


REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_ROOT = REPO_ROOT / "learnings" / "policy_tuning"
DEFAULT_MAX_EVENTS = 10_000
DEFAULT_MAX_EVENT_BYTES = 4 * 1024 * 1024
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
FINGERPRINT_FIELDS = (
    "observation_fingerprint",
    "reward_fingerprint",
    "deployment_fingerprint",
)


class ExperienceQueryError(ValueError):
    """Raised when a history query is unsafe, unbounded, or inconsistent."""


def _reject_symlink_components(path: Path, *, label: str) -> None:
    if not path.is_absolute():
        raise ExperienceQueryError(f"{label} must be absolute")
    current = Path(path.anchor)
    for component in path.parts[1:]:
        current /= component
        if current.is_symlink():
            raise ExperienceQueryError(
                f"{label} contains a symlinked path component: {current}"
            )


def _validate_nonempty(name: str, value: Any) -> None:
    if not isinstance(value, str) or not value:
        raise ExperienceQueryError(f"{name} must be a non-empty string")


def _validate_query_inputs(
    *,
    task: str,
    algorithm: str,
    host_id: str,
    observation_fingerprint: str,
    reward_fingerprint: str,
    deployment_fingerprint: str,
    max_events: int,
    max_event_bytes: int,
) -> None:
    for name, value in (("task", task), ("algorithm", algorithm), ("host-id", host_id)):
        if not isinstance(value, str) or not SLUG_RE.fullmatch(value):
            raise ExperienceQueryError(f"{name} must be a safe ASCII identifier")
    for name, value in (
        ("observation-fingerprint", observation_fingerprint),
        ("reward-fingerprint", reward_fingerprint),
        ("deployment-fingerprint", deployment_fingerprint),
    ):
        _validate_nonempty(name, value)
    if isinstance(max_events, bool) or not isinstance(max_events, int) or max_events <= 0:
        raise ExperienceQueryError("max-events must be a positive integer")
    if (
        isinstance(max_event_bytes, bool)
        or not isinstance(max_event_bytes, int)
        or max_event_bytes <= 0
    ):
        raise ExperienceQueryError("max-event-bytes must be a positive integer")


def _is_unknown(value: Any) -> bool:
    return value is None or (
        isinstance(value, str) and value.strip().casefold() == "unknown"
    )


def _event_paths(root: Path, task: str, max_events: int) -> list[Path]:
    task_dir = root / task
    _reject_symlink_components(task_dir, label="task history directory")
    if not task_dir.exists():
        return []
    if not task_dir.is_dir():
        raise ExperienceQueryError("task history path must be a directory")

    event_paths: list[Path] = []
    with os.scandir(task_dir) as run_entries:
        for run_entry in sorted(run_entries, key=lambda entry: entry.name):
            run_path = Path(run_entry.path)
            if run_entry.is_symlink():
                raise ExperienceQueryError(
                    f"task history contains a symlinked run directory: {run_path}"
                )
            if not run_entry.is_dir(follow_symlinks=False):
                continue
            if not SLUG_RE.fullmatch(run_entry.name):
                raise ExperienceQueryError(
                    f"task history contains an unsafe run directory: {run_path}"
                )
            with os.scandir(run_path) as candidate_entries:
                for candidate in sorted(candidate_entries, key=lambda entry: entry.name):
                    if candidate.name.startswith(".") or not candidate.name.endswith(".json"):
                        continue
                    candidate_path = Path(candidate.path)
                    if candidate.is_symlink():
                        raise ExperienceQueryError(
                            f"task history contains a symlinked event file: {candidate_path}"
                        )
                    if not candidate.is_file(follow_symlinks=False):
                        continue
                    event_paths.append(candidate_path)
                    if len(event_paths) > max_events:
                        raise ExperienceQueryError(
                            f"history exceeds max-events={max_events}; narrow the query"
                        )
    return event_paths


def _read_stable_event(path: Path, max_event_bytes: int) -> tuple[bytes, os.stat_result]:
    before = path.lstat()
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise ExperienceQueryError(f"event is not a regular non-symlinked file: {path}")
    if before.st_size > max_event_bytes:
        raise ExperienceQueryError(
            f"event exceeds max-event-bytes={max_event_bytes}: {path}"
        )
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        opened = os.fstat(descriptor)
        if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
            raise ExperienceQueryError(f"event changed before read: {path}")
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            encoded = stream.read(max_event_bytes + 1)
    finally:
        os.close(descriptor)
    after = path.lstat()
    stable_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns")
    if any(getattr(before, field) != getattr(after, field) for field in stable_fields):
        raise ExperienceQueryError(f"event changed during read: {path}")
    if len(encoded) > max_event_bytes:
        raise ExperienceQueryError(
            f"event exceeds max-event-bytes={max_event_bytes}: {path}"
        )
    return encoded, after


def _timestamp_slug(timestamp: str) -> str:
    return re.sub(r"[^0-9A-Za-z]+", "-", timestamp).strip("-")


def _validate_storage_binding(event: dict[str, Any], path: Path, task: str) -> None:
    run_id = path.parent.name
    if event["task"] != task or event["run_id"] != run_id:
        raise ExperienceQueryError(
            "event scope does not match its task/run history directory"
        )
    expected_name = (
        f"{_timestamp_slug(event['recorded_at'])}__{event['event_id']}.json"
    )
    if path.name != expected_name:
        raise ExperienceQueryError("event filename does not match recorded_at/event_id")


def _extract_evidence_refs(value: Any, json_path: str = "evidence") -> list[dict[str, Any]]:
    references: list[dict[str, Any]] = []
    if isinstance(value, dict):
        direct_path = value.get("path")
        if isinstance(direct_path, str) and Path(direct_path).is_absolute():
            direct_hash = value.get("sha256")
            references.append(
                {
                    "json_path": f"{json_path}.path",
                    "path": direct_path,
                    "sha256": (
                        direct_hash
                        if isinstance(direct_hash, str)
                        and SHA256_RE.fullmatch(direct_hash)
                        else None
                    ),
                }
            )
        for key in sorted(value):
            child = value[key]
            child_json_path = f"{json_path}.{key}"
            if (
                key.endswith("_path")
                and isinstance(child, str)
                and Path(child).is_absolute()
            ):
                companion = value.get(f"{key[:-5]}_sha256")
                references.append(
                    {
                        "json_path": child_json_path,
                        "path": child,
                        "sha256": (
                            companion
                            if isinstance(companion, str)
                            and SHA256_RE.fullmatch(companion)
                            else None
                        ),
                    }
                )
            references.extend(_extract_evidence_refs(child, child_json_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            references.extend(_extract_evidence_refs(child, f"{json_path}[{index}]"))
    unique = {
        (item["json_path"], item["path"], item["sha256"]): item
        for item in references
    }
    return [unique[key] for key in sorted(unique)]


def _classify_event(
    event: dict[str, Any],
    *,
    algorithm: str,
    host_id: str,
    observation_fingerprint: str,
    reward_fingerprint: str,
    deployment_fingerprint: str,
) -> tuple[str, list[str]]:
    event_host = (
        event["run_identity"]["host_id"] if event["version"] == 2 else None
    )
    query_values = {
        "algorithm": algorithm,
        "host_id": host_id,
        "observation_fingerprint": observation_fingerprint,
        "reward_fingerprint": reward_fingerprint,
        "deployment_fingerprint": deployment_fingerprint,
    }
    event_values = {
        "algorithm": event["algorithm"],
        "host_id": event_host,
        **{field: event["context"].get(field) for field in FINGERPRINT_FIELDS},
    }
    conflicts: list[str] = []
    unknowns: list[str] = []
    for field in query_values:
        query_value = query_values[field]
        event_value = event_values[field]
        if _is_unknown(query_value):
            unknowns.append(f"query_{field}_unknown")
        elif _is_unknown(event_value):
            unknowns.append(f"event_{field}_unknown")
        elif query_value != event_value:
            conflicts.append(f"{field}_mismatch")
    if conflicts:
        return "conflicting", conflicts + unknowns
    if unknowns:
        return "unknown", unknowns
    return "compatible", []


def _event_sort_key(item: dict[str, Any]) -> tuple[float, str, str]:
    timestamp = item["recorded_at"].replace("Z", "+00:00")
    return (
        datetime.fromisoformat(timestamp).timestamp(),
        item["event_id"],
        item["event_path"],
    )


def query_tuning_experience(
    root: Path,
    *,
    task: str,
    algorithm: str,
    host_id: str,
    observation_fingerprint: str,
    reward_fingerprint: str,
    deployment_fingerprint: str,
    max_events: int = DEFAULT_MAX_EVENTS,
    max_event_bytes: int = DEFAULT_MAX_EVENT_BYTES,
) -> dict[str, Any]:
    """Return deterministic history classifications without changing state."""
    _validate_query_inputs(
        task=task,
        algorithm=algorithm,
        host_id=host_id,
        observation_fingerprint=observation_fingerprint,
        reward_fingerprint=reward_fingerprint,
        deployment_fingerprint=deployment_fingerprint,
        max_events=max_events,
        max_event_bytes=max_event_bytes,
    )
    _reject_symlink_components(root, label="history root")
    if not root.is_dir():
        raise ExperienceQueryError("history root must be an existing directory")

    event_paths = _event_paths(root, task, max_events)
    buckets: dict[str, list[dict[str, Any]]] = {
        "compatible": [],
        "conflicting": [],
        "unknown": [],
    }
    invalid_events: list[dict[str, str | None]] = []
    for path in event_paths:
        encoded: bytes | None = None
        event_hash: str | None = None
        try:
            encoded, _ = _read_stable_event(path, max_event_bytes)
            event_hash = hashlib.sha256(encoded).hexdigest()
            event = json.loads(encoded.decode("utf-8"))
            if not isinstance(event, dict):
                raise ExperienceQueryError("event must be a JSON object")
            try:
                validate_event(event)
            except ExperienceError as exc:
                raise ExperienceQueryError(str(exc)) from exc
            _validate_storage_binding(event, path, task)
            classification, reasons = _classify_event(
                event,
                algorithm=algorithm,
                host_id=host_id,
                observation_fingerprint=observation_fingerprint,
                reward_fingerprint=reward_fingerprint,
                deployment_fingerprint=deployment_fingerprint,
            )
            event_host = (
                event["run_identity"]["host_id"]
                if event["version"] == 2
                else None
            )
            buckets[classification].append(
                {
                    "event_path": str(path),
                    "event_sha256": event_hash,
                    "version": event["version"],
                    "event_id": event["event_id"],
                    "event_type": event["event_type"],
                    "recorded_at": event["recorded_at"],
                    "task": event["task"],
                    "run_id": event["run_id"],
                    "algorithm": event["algorithm"],
                    "host_id": event_host,
                    "context": event["context"],
                    "parameters": event["parameters"],
                    "evidence_refs": _extract_evidence_refs(event["evidence"]),
                    "analysis": {
                        "summary": event["analysis"].get("summary"),
                        "confidence": event["analysis"]["confidence"],
                    },
                    "next_suggestion": event["next_suggestion"],
                    "classification_reasons": reasons,
                }
            )
        except (ExperienceQueryError, OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            invalid_events.append(
                {
                    "event_path": str(path),
                    "event_sha256": event_hash,
                    "error": str(exc),
                }
            )

    for events in buckets.values():
        events.sort(key=_event_sort_key)
    invalid_events.sort(key=lambda item: item["event_path"])
    query_context_complete = not any(
        _is_unknown(value)
        for value in (
            algorithm,
            host_id,
            observation_fingerprint,
            reward_fingerprint,
            deployment_fingerprint,
        )
    )
    if not query_context_complete:
        support_status = "query_context_incomplete"
    elif invalid_events:
        support_status = "history_invalid"
    elif buckets["compatible"]:
        support_status = "compatible_history_available"
    else:
        support_status = "no_compatible_history"
    confidence_counts = {"low": 0, "medium": 0, "high": 0}
    for item in buckets["compatible"]:
        confidence_counts[item["analysis"]["confidence"]] += 1

    return {
        "version": 1,
        "read_only": True,
        "query": {
            "root": str(root),
            "task": task,
            "algorithm": algorithm,
            "host_id": host_id,
            "observation_fingerprint": observation_fingerprint,
            "reward_fingerprint": reward_fingerprint,
            "deployment_fingerprint": deployment_fingerprint,
            "max_events": max_events,
            "max_event_bytes": max_event_bytes,
        },
        "scan": {
            "event_files": len(event_paths),
            "valid_events": sum(len(events) for events in buckets.values()),
            "invalid_events": len(invalid_events),
            "complete": not invalid_events,
        },
        "summary": {
            "compatible": len(buckets["compatible"]),
            "conflicting": len(buckets["conflicting"]),
            "unknown": len(buckets["unknown"]),
            "invalid": len(invalid_events),
            "compatible_confidence_counts": confidence_counts,
        },
        "historical_support": {
            "status": support_status,
            "query_context_complete": query_context_complete,
            "compatible_history_is_candidate_evidence_only": True,
            "direct_parameter_change_supported": False,
        },
        "compatible_events": buckets["compatible"],
        "conflicting_events": buckets["conflicting"],
        "unknown_events": buckets["unknown"],
        "invalid_events": invalid_events,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(DEFAULT_ROOT))
    parser.add_argument("--task", required=True)
    parser.add_argument("--algorithm", required=True)
    parser.add_argument("--host-id", required=True)
    parser.add_argument("--observation-fingerprint", required=True)
    parser.add_argument("--reward-fingerprint", required=True)
    parser.add_argument("--deployment-fingerprint", required=True)
    parser.add_argument("--max-events", type=int, default=DEFAULT_MAX_EVENTS)
    parser.add_argument(
        "--max-event-bytes",
        type=int,
        default=DEFAULT_MAX_EVENT_BYTES,
    )
    args = parser.parse_args()
    try:
        result = query_tuning_experience(
            Path(args.root),
            task=args.task,
            algorithm=args.algorithm,
            host_id=args.host_id,
            observation_fingerprint=args.observation_fingerprint,
            reward_fingerprint=args.reward_fingerprint,
            deployment_fingerprint=args.deployment_fingerprint,
            max_events=args.max_events,
            max_event_bytes=args.max_event_bytes,
        )
    except (ExperienceQueryError, OSError) as exc:
        parser.error(str(exc))
    print(
        json.dumps(
            result,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
