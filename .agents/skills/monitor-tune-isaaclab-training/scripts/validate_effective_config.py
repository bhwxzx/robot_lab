#!/usr/bin/env python3
"""Verify that a trial effective config differs only by authorized overrides."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from validate_session_spec import SpecError, load_and_validate


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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_leaf(value: Any, path: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise SpecError(f"{path} contains a non-finite value")
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str) or not key:
                raise SpecError(f"{path} contains an invalid object key")
            _validate_leaf(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _validate_leaf(item, f"{path}[{index}]")


def flatten_config(
    value: dict[str, Any],
    prefix: str = "",
) -> dict[str, Any]:
    """Flatten nested objects to the same dotted paths used by tuning overrides."""
    flattened: dict[str, Any] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not key or "." in key:
            raise SpecError("effective config keys must be non-empty and contain no dots")
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(item, dict):
            if not item:
                flattened[path] = {}
            else:
                flattened.update(flatten_config(item, path))
        else:
            _validate_leaf(item, path)
            flattened[path] = item
    return flattened


def _same_scalar(actual: Any, expected: Any) -> bool:
    if type(actual) is not type(expected):
        return False
    return actual == expected


def validate_effective_config(
    spec: dict[str, Any],
    baseline_path: Path,
    candidate_path: Path,
    overrides: dict[str, Any],
    runtime_values: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a hash-bound diff report or reject any unauthorized difference."""
    if spec.get("version", 0) < 6 or spec.get("mode") != "tune":
        raise SpecError(
            "effective config validation requires a version-6-or-newer tune session"
        )
    if (
        not baseline_path.is_absolute()
        or not candidate_path.is_absolute()
        or not baseline_path.is_file()
        or not candidate_path.is_file()
        or baseline_path.is_symlink()
        or candidate_path.is_symlink()
    ):
        raise SpecError(
            "effective config paths must be existing absolute regular files"
        )
    authorized = {
        item["path"] for item in spec["tuning"]["allowed_parameters"]
    }
    unknown_override = sorted(set(overrides) - authorized)
    if unknown_override:
        raise SpecError(
            f"overrides contain unauthorized paths: {unknown_override}"
        )
    baseline = flatten_config(_load_object(baseline_path, "baseline config"))
    candidate = flatten_config(_load_object(candidate_path, "candidate config"))
    normalized_runtime: dict[str, dict[str, Any]] = {}
    for path, expected in (runtime_values or {}).items():
        if path not in baseline or path not in candidate:
            raise SpecError(f"runtime config path is missing: {path}")
        if not _same_scalar(candidate[path], expected):
            raise SpecError(
                f"runtime config path {path} does not match this run identity"
            )
        normalized_runtime[path] = {
            "baseline": baseline[path],
            "candidate": candidate[path],
            "expected": expected,
        }
        candidate[path] = baseline[path]
    changed: dict[str, dict[str, Any]] = {}
    for path in sorted(set(baseline) | set(candidate)):
        before = baseline.get(path, {"__missing__": True})
        after = candidate.get(path, {"__missing__": True})
        if _same_scalar(before, after):
            continue
        changed[path] = {"baseline": before, "candidate": after}
    unauthorized_changed = sorted(set(changed) - set(overrides))
    if unauthorized_changed:
        raise SpecError(
            "effective config contains unauthorized differences: "
            + ", ".join(unauthorized_changed)
        )
    missing_or_mismatched: list[str] = []
    for path, expected in overrides.items():
        if path not in candidate or not _same_scalar(candidate[path], expected):
            missing_or_mismatched.append(path)
            continue
        if path not in changed and not _same_scalar(baseline.get(path), expected):
            missing_or_mismatched.append(path)
    if missing_or_mismatched:
        raise SpecError(
            "effective config did not apply exact overrides: "
            + ", ".join(sorted(missing_or_mismatched))
        )
    return {
        "version": 1,
        "status": "valid",
        "baseline_path": str(baseline_path),
        "baseline_sha256": _sha256(baseline_path),
        "candidate_path": str(candidate_path),
        "candidate_sha256": _sha256(candidate_path),
        "authorized_parameter_paths": sorted(authorized),
        "verified_overrides": overrides,
        "verified_runtime_values": normalized_runtime,
        "changed_paths": changed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session", help="Validated version-6-or-newer tune session")
    parser.add_argument("baseline", help="Baseline effective config JSON")
    parser.add_argument("candidate", help="Candidate effective config JSON")
    parser.add_argument(
        "--overrides-json",
        required=True,
        help="Exact JSON object of approved overrides",
    )
    parser.add_argument("--output", help="Optional report JSON")
    args = parser.parse_args()
    try:
        overrides = json.loads(args.overrides_json)
        if not isinstance(overrides, dict):
            raise SpecError("--overrides-json must decode to an object")
        report = validate_effective_config(
            load_and_validate(args.session),
            Path(args.baseline),
            Path(args.candidate),
            overrides,
        )
    except (SpecError, json.JSONDecodeError) as exc:
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
