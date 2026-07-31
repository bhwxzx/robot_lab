#!/usr/bin/env python3
"""Write one immutable, structured policy-tuning experience event."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any


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
    if event.get("version") != 1:
        raise ExperienceError("event.version must be 1")
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


def write_event(root: Path, event: dict[str, Any]) -> dict[str, Any]:
    validate_event(event)
    if not root.is_absolute():
        raise ExperienceError("experience root must be absolute")
    run_dir = root / event["task"] / event["run_id"]
    run_dir.mkdir(parents=True, exist_ok=True)
    timestamp_slug = re.sub(r"[^0-9A-Za-z]+", "-", event["recorded_at"]).strip("-")
    destination = run_dir / f"{timestamp_slug}__{event['event_id']}.json"
    if destination.exists():
        raise ExperienceError(f"event already exists: {destination}")
    encoded = json.dumps(event, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
    temporary = run_dir / f".{destination.name}.tmp-{os.getpid()}"
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(destination)
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
    except ExperienceError as exc:
        parser.error(str(exc))
    encoded = json.dumps(receipt, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    if args.output:
        output = Path(args.output)
        if not output.is_absolute() or output.exists():
            parser.error("--output must be a new absolute path")
        output.write_text(encoded, encoding="utf-8")
    else:
        print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
