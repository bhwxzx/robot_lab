#!/usr/bin/env python3
"""Extract final-rung results for normal ranking from a terminal plan."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from build_trial_plan import validate_trial_plan
from validate_session_spec import SpecError, load_and_validate


def _load_object(path: Path, label: str) -> dict[str, Any]:
    if not path.is_absolute() or not path.is_file() or path.is_symlink():
        raise SpecError(f"{label} must be an existing absolute regular file")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SpecError(
            f"invalid {label} JSON at line {exc.lineno}: {exc.msg}"
        ) from exc
    if not isinstance(value, dict):
        raise SpecError(f"{label} must be a JSON object")
    return value


def final_rung_results(
    spec: dict[str, Any],
    plan: dict[str, Any],
) -> dict[str, Any]:
    validate_trial_plan(spec, plan)
    fidelity = plan.get("multi_fidelity")
    decisions = (
        fidelity.get("decisions") if isinstance(fidelity, dict) else None
    )
    if (
        not isinstance(fidelity, dict)
        or fidelity.get("status") != "completed"
        or not isinstance(decisions, list)
        or not decisions
        or decisions[-1].get("action") != "complete"
    ):
        raise SpecError(
            "final ranking requires a completed multi-fidelity plan"
        )
    final_runs = [
        {
            "trial_id": result["trial_id"],
            "seed": result["seed"],
            "status": result["status"],
            "metrics": result["metrics"],
        }
        for result in decisions[-1]["input_results"]
    ]
    if not any(run["trial_id"] == "baseline" for run in final_runs):
        raise SpecError("final multi-fidelity rung is missing the baseline")
    return {
        "source": "completed_multi_fidelity_final_rung",
        "evaluated_rung": decisions[-1]["evaluated_rung"],
        "selected_trial_ids": decisions[-1]["selected_trial_ids"],
        "runs": final_runs,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session")
    parser.add_argument("plan")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    try:
        spec = load_and_validate(args.session)
        plan = _load_object(Path(args.plan), "multi-fidelity plan")
        results = final_rung_results(spec, plan)
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
                results,
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
