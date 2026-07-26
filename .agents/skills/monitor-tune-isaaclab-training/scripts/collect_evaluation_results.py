#!/usr/bin/env python3
"""Collect per-run evaluation JSON files into one validation document."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from validate_policy_evaluation import load_evaluation_plan
from validate_session_spec import SpecError


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_execution_state(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise SpecError(f"evaluation execution state does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise SpecError(
            f"invalid evaluation execution state at line {exc.lineno}: {exc.msg}"
        ) from exc
    if (
        not isinstance(value, dict)
        or value.get("version") != 1
        or not isinstance(value.get("runs"), dict)
    ):
        raise SpecError("evaluation execution state is invalid")
    return value


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
    execution_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Read every planned result path without treating missing runs as success."""
    runs: list[dict[str, Any]] = []
    for expected in plan["runs"]:
        result_path = Path(expected["result_path"])
        state_run = (
            execution_state["runs"].get(expected["run_id"])
            if execution_state is not None
            else None
        )
        if execution_state is not None:
            if not isinstance(state_run, dict):
                raise SpecError(
                    f"execution state is missing run {expected['run_id']}"
                )
            if (
                state_run.get("status") != "completed"
                or state_run.get("canonical_result_path")
                != expected["result_path"]
                or not isinstance(state_run.get("result_sha256"), str)
            ):
                raise SpecError(
                    f"execution state run is not completed: {expected['run_id']}"
                )
            if not result_path.is_file() or result_path.is_symlink():
                raise SpecError(
                    f"completed evaluation result is not a regular file: "
                    f"{result_path}"
                )
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
        if execution_state is not None:
            if _sha256(result_path) != state_run["result_sha256"]:
                raise SpecError(
                    f"evaluation result changed after execution: {result_path}"
                )
            video_hash = state_run.get("video_sha256")
            if video_hash is not None:
                video_path = Path(expected["video_path"])
                if (
                    not video_path.is_file()
                    or _sha256(video_path) != video_hash
                ):
                    raise SpecError(
                        f"evaluation video changed after execution: {video_path}"
                    )
        runs.append(value)
    result = {
        "version": 1,
        "runs": runs,
        "visual_reviews": visual_reviews,
    }
    if execution_state is not None:
        result["execution_state_sha256"] = hashlib.sha256(
            json.dumps(
                execution_state,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", help="Evaluation plan JSON")
    parser.add_argument(
        "--visual-reviews",
        help="Optional JSON containing manual or agent visual reviews",
    )
    parser.add_argument(
        "--execution-state",
        help="Optional executor state that hash-binds completed results and videos",
    )
    parser.add_argument("--output", help="Optional consolidated JSON output path")
    args = parser.parse_args()
    try:
        result = collect(
            load_evaluation_plan(Path(args.plan)),
            _load_reviews(
                Path(args.visual_reviews) if args.visual_reviews else None
            ),
            _load_execution_state(
                Path(args.execution_state) if args.execution_state else None
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
