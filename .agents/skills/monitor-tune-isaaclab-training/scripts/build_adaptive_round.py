#!/usr/bin/env python3
"""Append one deterministic adaptive-search round from completed results."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from build_trial_plan import extend_adaptive_plan, validate_trial_plan
from validate_session_spec import SpecError, load_and_validate


def _load_json(path: Path, label: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise SpecError(f"{label} does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise SpecError(
            f"invalid {label} JSON at line {exc.lineno}: {exc.msg}"
        ) from exc


def _results(
    value: Any,
    plan: dict[str, Any],
) -> list[dict[str, Any]]:
    if isinstance(value, list):
        records = value
    elif isinstance(value, dict) and isinstance(
        value.get("accepted_results"),
        list,
    ):
        if value.get("invalid_results"):
            raise SpecError(
                "invalid mailbox results block adaptive round construction"
            )
        expected_job_ids = {run["run_id"] for run in plan["runs"]}
        records = [
            envelope.get("result")
            for envelope in value["accepted_results"]
            if isinstance(envelope, dict)
            and envelope.get("job_id") in expected_job_ids
        ]
    else:
        raise SpecError(
            "adaptive results must be a result array or mailbox collection report"
        )
    if not all(isinstance(record, dict) for record in records):
        raise SpecError("adaptive results contain a non-object record")
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session")
    parser.add_argument("current_plan")
    parser.add_argument("completed_results")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    try:
        spec = load_and_validate(args.session)
        plan = _load_json(Path(args.current_plan), "current plan")
        if not isinstance(plan, dict):
            raise SpecError("current plan must be a JSON object")
        validate_trial_plan(spec, plan)
        expanded = extend_adaptive_plan(
            spec,
            plan,
            _results(
                _load_json(Path(args.completed_results), "completed results"),
                plan,
            ),
        )
        output = Path(args.output)
        if (
            not output.is_absolute()
            or output.exists()
            or not output.parent.is_dir()
            or output.parent.is_symlink()
        ):
            raise SpecError(
                "adaptive plan output must be a new absolute file under an "
                "existing regular parent"
            )
        output.write_bytes(
            json.dumps(
                expanded,
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
