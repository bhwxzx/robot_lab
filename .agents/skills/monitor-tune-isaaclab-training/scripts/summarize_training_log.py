#!/usr/bin/env python3
"""Parse profile-driven training console logs into bounded JSON summaries."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import deque
from pathlib import Path
from statistics import fmean
from typing import Any

from algorithm_profiles import (
    DEFAULT_REGISTRY_PATH,
    ProfileError,
    load_registry,
    normalize_metric_name,
    resolve_profile,
)


ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
COMPUTATION_RE = re.compile(
    r"Computation:\s+([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s+steps/s "
    r"\(collection:\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+))s,\s*"
    r"learning\s+([+-]?(?:\d+(?:\.\d*)?|\.\d+))s\)"
)
VALUE_RE = re.compile(
    r"^\s*([^:]+):\s*([+-]?(?:(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?|nan|inf|-inf))s?\s*$",
    re.IGNORECASE,
)


def _progress_match(
    line: str,
    patterns: list[dict[str, Any]],
) -> tuple[str, int, int | None] | None:
    for pattern in patterns:
        match = re.search(pattern["regex"], line)
        if not match:
            continue
        current = int(match.group("current"))
        target_text = match.groupdict().get("target")
        target = int(target_text) if target_text is not None else None
        return pattern["name"], current, target
    return None


def _finish_progress(
    current: dict[str, Any] | None,
    rows: deque[dict[str, Any]],
) -> None:
    if current is not None:
        rows.append(current)


def parse_log(
    path: Path,
    last: int,
    profile: dict[str, Any],
) -> dict[str, Any]:
    """Parse a log while retaining only the requested number of progress records."""
    rows: deque[dict[str, Any]] = deque(maxlen=last)
    current: dict[str, Any] | None = None
    non_finite: deque[dict[str, Any]] = deque(maxlen=1000)
    aliases = profile["metric_aliases"]

    with path.open("r", encoding="utf-8", errors="replace") as stream:
        for raw_line in stream:
            line = ANSI_RE.sub("", raw_line.rstrip("\n"))
            progress = _progress_match(line, profile["progress_patterns"])
            if progress:
                _finish_progress(current, rows)
                progress_name, progress_value, target = progress
                current = {
                    "progress_name": progress_name,
                    "progress": progress_value,
                    "target_progress": target,
                }
                if progress_name == "learning_iteration":
                    current["iteration"] = progress_value
                    current["target_iteration"] = target
                continue
            if current is None:
                continue
            computation_match = COMPUTATION_RE.search(line)
            if computation_match:
                current["steps_per_second"] = float(computation_match.group(1))
                current["collection_time_seconds"] = float(computation_match.group(2))
                current["learning_time_seconds"] = float(computation_match.group(3))
                continue
            value_match = VALUE_RE.match(line)
            if not value_match:
                continue
            label = value_match.group(1).strip()
            metric = aliases.get(label, normalize_metric_name(label))
            if not metric:
                continue
            value = float(value_match.group(2))
            if math.isfinite(value):
                current[metric] = value
            else:
                current[metric] = None
                non_finite.append(
                    {
                        "progress": current["progress"],
                        "metric": metric,
                    }
                )
    _finish_progress(current, rows)

    retained = list(rows)
    aggregate: dict[str, float] = {}
    excluded = {
        "progress",
        "target_progress",
        "iteration",
        "target_iteration",
    }
    metric_names = sorted(
        {
            key
            for row in retained
            for key, value in row.items()
            if key not in excluded
            and isinstance(value, (int, float))
            and not isinstance(value, bool)
        }
    )
    for metric in metric_names:
        values = [
            float(row[metric])
            for row in retained
            if isinstance(row.get(metric), (int, float))
        ]
        if values:
            aggregate[metric] = fmean(values)

    latest = retained[-1] if retained else None
    return {
        "log_path": str(path),
        "profile_id": profile["id"],
        "window_size": len(retained),
        "first_progress": retained[0]["progress"] if retained else None,
        "last_progress": latest["progress"] if latest else None,
        "first_iteration": retained[0].get("iteration") if retained else None,
        "last_iteration": latest.get("iteration") if latest else None,
        "latest": latest,
        "mean": aggregate,
        "non_finite_metrics": list(non_finite),
        "records": retained,
        "iterations": retained,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", help="Training stdout/stderr log")
    parser.add_argument("--last", type=int, default=100)
    parser.add_argument("--profile-id", default="generic")
    parser.add_argument("--registry", default=str(DEFAULT_REGISTRY_PATH))
    parser.add_argument("--output")
    args = parser.parse_args()
    if not 1 <= args.last <= 10000:
        parser.error("--last must be between 1 and 10000")
    try:
        profile = resolve_profile(load_registry(args.registry), args.profile_id)
    except ProfileError as exc:
        parser.error(str(exc))
    result = parse_log(Path(args.log), args.last, profile)
    encoded = json.dumps(
        result,
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
