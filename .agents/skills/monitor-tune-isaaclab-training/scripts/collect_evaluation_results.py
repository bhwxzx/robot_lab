#!/usr/bin/env python3
"""Collect per-run evaluation JSON files into one validation document."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from validate_policy_evaluation import load_evaluation_plan
from validate_session_spec import SpecError


def _load_reviews(path: Path | None) -> list[dict[str, Any]]:
    if path is None:
        return []
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise SpecError(f"visual review file does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise SpecError(
            f"invalid visual review JSON at line {exc.lineno}: {exc.msg}"
        ) from exc
    if isinstance(value, dict):
        value = value.get("visual_reviews")
    if not isinstance(value, list):
        raise SpecError(
            "visual review file must be an array or contain visual_reviews"
        )
    return value


def collect(
    plan: dict[str, Any],
    visual_reviews: list[dict[str, Any]],
) -> dict[str, Any]:
    """Read every planned result path without treating missing runs as success."""
    runs: list[dict[str, Any]] = []
    for expected in plan["runs"]:
        result_path = Path(expected["result_path"])
        try:
            value = json.loads(result_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            value = {
                "version": 1,
                "run_id": expected["run_id"],
                "candidate_id": expected["candidate_id"],
                "artifact": expected["artifact"],
                "scenario_id": expected["scenario_id"],
                "seed": expected["seed"],
                "status": "missing",
                "video_path": "",
                "metrics": {},
            }
        except json.JSONDecodeError as exc:
            raise SpecError(
                f"invalid run result {result_path} at line {exc.lineno}: {exc.msg}"
            ) from exc
        if not isinstance(value, dict):
            raise SpecError(f"run result must be an object: {result_path}")
        runs.append(value)
    return {
        "version": 1,
        "runs": runs,
        "visual_reviews": visual_reviews,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", help="Evaluation plan JSON")
    parser.add_argument(
        "--visual-reviews",
        help="Optional JSON containing manual or agent visual reviews",
    )
    parser.add_argument("--output", help="Optional consolidated JSON output path")
    args = parser.parse_args()
    try:
        result = collect(
            load_evaluation_plan(Path(args.plan)),
            _load_reviews(
                Path(args.visual_reviews) if args.visual_reviews else None
            ),
        )
    except SpecError as exc:
        parser.error(str(exc))
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
