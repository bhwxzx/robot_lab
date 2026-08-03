#!/usr/bin/env python3
"""Write one immutable, structured policy-tuning experience event."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Any

from capture_effective_training_config import (
    EffectiveConfigError,
    SHA256_RE,
    load_and_validate_effective_config,
)
from capture_run_identity import RunIdentityError, validate_run_identity


EVENT_TYPES = {
    "run_snapshot",
    "assessment",
    "decision",
    "checkpoint_evaluation",
    "checkpoint_selection",
    "export",
    "archive",
    "feedback",
    "recommendation",
}
SLUG_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
TIMESTAMP_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})$")


class ExperienceError(ValueError):
    """Raised when an experience event is unsafe or malformed."""


def _reject_symlink_components(path: Path, *, label: str) -> None:
    if not path.is_absolute():
        raise ExperienceError(f"{label} must be absolute")
    current = Path(path.anchor)
    for component in path.parts[1:]:
        current /= component
        if current.is_symlink():
            raise ExperienceError(
                f"{label} contains a symlinked path component: {current}"
            )


def _write_new_output(path: Path, encoded: str) -> None:
    if not path.is_absolute():
        raise ExperienceError("--output must be a new absolute path")
    _reject_symlink_components(path.parent, label="--output")
    if not path.parent.is_dir():
        raise ExperienceError("--output parent directory does not exist")
    if path.exists() or path.is_symlink():
        raise ExperienceError("--output must be a new absolute path")
    try:
        with path.open("x", encoding="utf-8") as stream:
            stream.write(encoded)
    except FileExistsError as exc:
        raise ExperienceError("--output must be a new absolute path") from exc


def _load(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ExperienceError(f"event file does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ExperienceError(f"invalid event JSON at line {exc.lineno}: {exc.msg}") from exc
    if not isinstance(value, dict):
        raise ExperienceError("event must be a JSON object")
    return value


def validate_event(event: dict[str, Any]) -> None:
    version = event.get("version")
    if version not in {1, 2, 3}:
        raise ExperienceError("event.version must be 1, 2, or 3")
    for field in ("event_id", "task", "run_id", "algorithm"):
        value = event.get(field)
        if not isinstance(value, str) or not SLUG_RE.fullmatch(value):
            raise ExperienceError(f"{field} must be a safe ASCII identifier")
    if event.get("event_type") not in EVENT_TYPES:
        raise ExperienceError("event_type is unsupported")
    timestamp = event.get("recorded_at")
    if not isinstance(timestamp, str) or not TIMESTAMP_RE.fullmatch(timestamp):
        raise ExperienceError("recorded_at must be an ISO-8601 timestamp with timezone")
    for field in ("context", "parameters", "evidence", "analysis"):
        if not isinstance(event.get(field), dict):
            raise ExperienceError(f"{field} must be an object")
    context = event["context"]
    for field in (
        "observation_fingerprint",
        "reward_fingerprint",
        "deployment_fingerprint",
    ):
        value = context.get(field)
        if not isinstance(value, str) or not value:
            raise ExperienceError(f"context.{field} must be a non-empty string")
    confidence = event["analysis"].get("confidence")
    if confidence not in {"low", "medium", "high"}:
        raise ExperienceError("analysis.confidence must be low, medium, or high")
    if not isinstance(event.get("next_suggestion"), str):
        raise ExperienceError("next_suggestion must be a string")
    if event["event_type"] == "feedback" and event["evidence"].get("source") not in {"sim2sim", "sim2real"}:
        raise ExperienceError("feedback evidence.source must be sim2sim or sim2real")
    if version in {2, 3}:
        try:
            validate_run_identity(event.get("run_identity"))
        except RunIdentityError as exc:
            raise ExperienceError(str(exc)) from exc
        run_identity = event["run_identity"]
        for field in ("task", "run_id", "algorithm"):
            if event[field] != run_identity[field]:
                raise ExperienceError(
                    f"event.{field} must match run_identity.{field}"
                )
    if version == 3:
        reference = event["evidence"].get("effective_config")
        expected_keys = {
            "effective_config_fingerprint",
            "path",
            "reward_fingerprint",
            "sha256",
        }
        if not isinstance(reference, dict) or set(reference) != expected_keys:
            raise ExperienceError(
                "version-3 evidence.effective_config must contain path, sha256, "
                "effective_config_fingerprint, and reward_fingerprint"
            )
        path = reference["path"]
        if (
            not isinstance(path, str)
            or not Path(path).is_absolute()
            or ".." in Path(path).parts
        ):
            raise ExperienceError("effective config evidence path must be absolute")
        for field in (
            "sha256",
            "effective_config_fingerprint",
            "reward_fingerprint",
        ):
            value = reference[field]
            if not isinstance(value, str) or not SHA256_RE.fullmatch(value):
                raise ExperienceError(f"effective config {field} must be SHA-256")
        if event["context"]["reward_fingerprint"] != reference["reward_fingerprint"]:
            raise ExperienceError(
                "context.reward_fingerprint must match effective config reference"
            )


def validate_effective_config_binding(
    root: Path,
    event: dict[str, Any],
) -> dict[str, Any]:
    """Validate a version-3 event's host-local immutable config reference."""
    if event.get("version") != 3:
        raise ExperienceError("new experience events must use version 3")
    reference = event["evidence"]["effective_config"]
    path = Path(reference["path"])
    expected_parent = root / event["task"] / event["run_id"] / "evidence" / "source"
    if path.parent != expected_parent or not re.fullmatch(
        r"effective-config-[A-Za-z0-9][A-Za-z0-9._-]{0,127}\.json",
        path.name,
    ):
        raise ExperienceError(
            "effective config evidence must be a direct source artifact for this run"
        )
    try:
        config, _ = load_and_validate_effective_config(
            path,
            expected_sha256=reference["sha256"],
            run_identity=event["run_identity"],
        )
    except EffectiveConfigError as exc:
        raise ExperienceError(str(exc)) from exc
    if (
        config["fingerprints"]["effective_config"]
        != reference["effective_config_fingerprint"]
    ):
        raise ExperienceError("effective config fingerprint reference mismatch")
    if config["fingerprints"]["reward"] != reference["reward_fingerprint"]:
        raise ExperienceError("effective config reward fingerprint reference mismatch")
    return config


def write_event(root: Path, event: dict[str, Any]) -> dict[str, Any]:
    validate_event(event)
    if not root.is_absolute():
        raise ExperienceError("experience root must be absolute")
    _reject_symlink_components(root, label="experience root")
    if not root.is_dir():
        raise ExperienceError("experience root must be an existing directory")
    run_dir = root / event["task"] / event["run_id"]
    _reject_symlink_components(run_dir, label="experience run directory")
    validate_effective_config_binding(root, event)
    run_dir.mkdir(parents=True, exist_ok=True)
    _reject_symlink_components(run_dir, label="experience run directory")
    timestamp_slug = re.sub(r"[^0-9A-Za-z]+", "-", event["recorded_at"]).strip("-")
    destination = run_dir / f"{timestamp_slug}__{event['event_id']}.json"
    if destination.exists() or destination.is_symlink():
        raise ExperienceError(f"event already exists: {destination}")
    encoded = json.dumps(event, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
    temporary = run_dir / f".{destination.name}.tmp-{os.getpid()}-{time.time_ns()}"
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, destination, follow_symlinks=False)
        except FileExistsError as exc:
            raise ExperienceError(f"event already exists: {destination}") from exc
    finally:
        if temporary.exists():
            temporary.unlink()
    return {
        "version": 1,
        "event_path": str(destination),
        "sha256": hashlib.sha256(encoded.encode("utf-8")).hexdigest(),
        "immutable": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("event")
    parser.add_argument(
        "--root",
        default=str(Path(__file__).resolve().parents[4] / "learnings" / "policy_tuning"),
    )
    parser.add_argument("--output")
    args = parser.parse_args()
    try:
        receipt = write_event(Path(args.root).resolve(), _load(Path(args.event)))
    except (OSError, ExperienceError) as exc:
        parser.error(str(exc))
    encoded = json.dumps(receipt, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    if args.output:
        try:
            _write_new_output(Path(args.output), encoded)
        except ExperienceError as exc:
            parser.error(str(exc))
    else:
        print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
