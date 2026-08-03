#!/usr/bin/env python3
"""Prepare deterministic, non-overwriting policy-tuning evidence paths."""

from __future__ import annotations

import argparse
import json
import re
import shlex
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_TUNING_ROOT = REPO_ROOT / "learnings" / "policy_tuning"
SAFE_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


class EvidenceLayoutError(ValueError):
    """Raised when an evidence layout would be unsafe or overwrite evidence."""


def _validate_identifier(name: str, value: str) -> None:
    if not SAFE_IDENTIFIER_RE.fullmatch(value):
        raise EvidenceLayoutError(
            f"{name} must be a safe ASCII identifier (letters, digits, '.', '_', '-')"
        )


def _reject_symlink_components(path: Path) -> None:
    if not path.is_absolute():
        raise EvidenceLayoutError("tuning root must be absolute")
    current = Path(path.anchor)
    for component in path.parts[1:]:
        current /= component
        if current.is_symlink():
            raise EvidenceLayoutError(f"symlinked path component is not allowed: {current}")


def _require_new_targets(paths: list[Path]) -> None:
    for path in paths:
        _reject_symlink_components(path)
        if path.exists() or path.is_symlink():
            raise EvidenceLayoutError(f"evidence target already exists: {path}")


def prepare_evidence_layout(
    tuning_root: Path,
    *,
    task: str,
    run_id: str,
    snapshot_id: str,
    evaluation_id: str | None = None,
) -> dict[str, Any]:
    """Create safe evidence directories and return new absolute artifact paths."""
    for name, value in (
        ("task", task),
        ("run-id", run_id),
        ("snapshot-id", snapshot_id),
    ):
        _validate_identifier(name, value)
    if evaluation_id is not None:
        _validate_identifier("evaluation-id", evaluation_id)

    if not tuning_root.is_absolute():
        raise EvidenceLayoutError("tuning root must be absolute")
    _reject_symlink_components(tuning_root)
    if tuning_root.exists() and not tuning_root.is_dir():
        raise EvidenceLayoutError(f"tuning root is not a directory: {tuning_root}")

    run_root = tuning_root / task / run_id
    evidence_root = run_root / "evidence"
    criteria_dir = evidence_root / "criteria"
    health_dir = evidence_root / "health"
    source_dir = evidence_root / "source"
    training_dir = evidence_root / "training"
    play_dir = evidence_root / "play"
    evaluation_dir = play_dir / evaluation_id if evaluation_id is not None else None

    paths: dict[str, Path | None] = {
        "criteria": criteria_dir / f"criteria-{snapshot_id}.json",
        "health": health_dir / f"health-{snapshot_id}.json",
        "source_identity": source_dir / f"identity-{snapshot_id}.json",
        "effective_config": source_dir / f"effective-config-{snapshot_id}.json",
        "source_patch": source_dir / f"source-{snapshot_id}.patch",
        "summary": training_dir / f"summary-{snapshot_id}.json",
        "assessment": training_dir / f"assessment-{snapshot_id}.json",
        "play_result": evaluation_dir / "result.json" if evaluation_dir else None,
        "telemetry": evaluation_dir / "telemetry.json" if evaluation_dir else None,
        "video": evaluation_dir / "video.mp4" if evaluation_dir else None,
    }
    _require_new_targets([path for path in paths.values() if path is not None])

    directories = [criteria_dir, health_dir, source_dir, training_dir, play_dir]
    if evaluation_dir is not None:
        directories.append(evaluation_dir)
    try:
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            _reject_symlink_components(directory)
            if not directory.is_dir():
                raise EvidenceLayoutError(f"evidence path is not a directory: {directory}")
    except OSError as exc:
        raise EvidenceLayoutError(f"cannot prepare evidence directories: {exc}") from exc

    _require_new_targets([path for path in paths.values() if path is not None])
    return {
        "version": 1,
        "task": task,
        "run_id": run_id,
        "snapshot_id": snapshot_id,
        "evaluation_id": evaluation_id,
        "run_root": str(run_root),
        "evidence_root": str(evidence_root),
        "directories": {
            "criteria": str(criteria_dir),
            "health": str(health_dir),
            "source": str(source_dir),
            "training": str(training_dir),
            "play": str(play_dir),
            "evaluation": str(evaluation_dir) if evaluation_dir else None,
        },
        "paths": {
            name: str(path) if path is not None else None
            for name, path in paths.items()
        },
    }


def _shell_assignments(layout: dict[str, Any]) -> str:
    values = {
        "RUN_ROOT": layout["run_root"],
        "EVIDENCE_ROOT": layout["evidence_root"],
        "CRITERIA_PATH": layout["paths"]["criteria"],
        "HEALTH_PATH": layout["paths"]["health"],
        "SOURCE_IDENTITY_PATH": layout["paths"]["source_identity"],
        "EFFECTIVE_CONFIG_PATH": layout["paths"]["effective_config"],
        "SOURCE_PATCH_PATH": layout["paths"]["source_patch"],
        "SUMMARY_PATH": layout["paths"]["summary"],
        "ASSESSMENT_PATH": layout["paths"]["assessment"],
        "PLAY_RESULT_PATH": layout["paths"]["play_result"],
        "TELEMETRY_PATH": layout["paths"]["telemetry"],
        "VIDEO_PATH": layout["paths"]["video"],
    }
    assignments = [
        f"{name}={shlex.quote(value)}"
        for name, value in values.items()
        if value is not None
    ]
    missing = [name for name, value in values.items() if value is None]
    if missing:
        assignments.append(f"unset {' '.join(missing)}")
    return "\n".join(assignments)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--snapshot-id", required=True)
    parser.add_argument("--evaluation-id")
    parser.add_argument("--format", choices=("json", "shell"), default="json")
    args = parser.parse_args()
    try:
        layout = prepare_evidence_layout(
            DEFAULT_TUNING_ROOT,
            task=args.task,
            run_id=args.run_id,
            snapshot_id=args.snapshot_id,
            evaluation_id=args.evaluation_id,
        )
    except EvidenceLayoutError as exc:
        parser.error(str(exc))
    if args.format == "shell":
        print(_shell_assignments(layout))
    else:
        print(json.dumps(layout, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
