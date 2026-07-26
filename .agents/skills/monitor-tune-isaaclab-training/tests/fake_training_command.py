#!/usr/bin/env python3
"""Small deterministic child command used only by execution-state tests."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path


def _set_path(root, path, value):
    target = root
    parts = path.split(".")
    for part in parts[:-1]:
        target = target[part]
    target[parts[-1]] = value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--effective-config", required=True)
    parser.add_argument("--result", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--trial-id", required=True)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--gpu-index", required=True, type=int)
    parser.add_argument("--overrides-json", required=True)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    baseline = json.loads(Path(args.baseline).read_text(encoding="utf-8"))
    effective = copy.deepcopy(baseline)
    overrides = json.loads(args.overrides_json)
    for path, value in overrides.items():
        _set_path(effective, path, value)
    Path(args.effective_config).write_text(
        json.dumps(effective),
        encoding="utf-8",
    )
    learning_rate = float(effective["agent"]["learning_rate"])
    score = 10.0 + learning_rate * 10.0 + (args.seed - 42) * 0.001
    result = {
        "trial_id": args.trial_id,
        "seed": args.seed,
        "status": "completed",
        "metrics": {
            "score": score,
            "unsafe": 0.0,
        },
    }
    Path(args.result).write_text(json.dumps(result), encoding="utf-8")
    summary = {
        "records": [
            {"progress": 1, "score": score, "steps_per_second": 1000.0},
            {"progress": 2, "score": score, "steps_per_second": 1000.0},
            {"progress": 3, "score": score, "steps_per_second": 1000.0},
        ],
        "non_finite_metrics": [],
    }
    Path(args.summary).write_text(json.dumps(summary), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
