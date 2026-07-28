#!/usr/bin/env python3
"""Advance one synchronized multi-fidelity rung from completed results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from build_trial_plan import advance_multifidelity_plan, validate_trial_plan
from validate_session_spec import SpecError, load_and_validate


def _load_json(path: Path, label: str) -> Any:
    if not path.is_absolute() or not path.is_file() or path.is_symlink():
        raise SpecError(f"{label} must be an existing absolute regular file")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SpecError(
            f"invalid {label} JSON at line {exc.lineno}: {exc.msg}"
        ) from exc


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session")
    parser.add_argument("plan")
    parser.add_argument("results")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    try:
        spec = load_and_validate(args.session)
        plan = _load_json(Path(args.plan), "multi-fidelity plan")
        results = _load_json(Path(args.results), "multi-fidelity results")
        if not isinstance(plan, dict):
            raise SpecError("multi-fidelity plan must be an object")
        if not isinstance(results, dict) or set(results) != {"runs"}:
            raise SpecError(
                "multi-fidelity results must contain exactly a runs array"
            )
        if not isinstance(results["runs"], list):
            raise SpecError("multi-fidelity results.runs must be an array")
        validate_trial_plan(spec, plan)
        expanded = advance_multifidelity_plan(
            spec,
            plan,
            results["runs"],
        )
        output = Path(args.output)
        if (
            not output.is_absolute()
            or output.exists()
            or not output.parent.is_dir()
            or output.parent.is_symlink()
        ):
            raise SpecError(
                "output must be a new absolute file under an existing regular "
                "directory"
            )
        output.write_text(
            json.dumps(
                expanded,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
    except (OSError, SpecError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
