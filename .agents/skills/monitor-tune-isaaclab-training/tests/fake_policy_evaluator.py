#!/usr/bin/env python3
"""Small evaluator double for transactional execution fault tests."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-kind", required=True)
    parser.add_argument("--artifact-path", required=True)
    parser.add_argument("--artifact-sha256", required=True)
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--command-schedule-json", required=True)
    parser.add_argument("--duration-steps", type=int, required=True)
    parser.add_argument("--executor-run-id", required=True)
    parser.add_argument("--gpu-index", required=True)
    parser.add_argument("--result-path", required=True)
    parser.add_argument("--require_idle_gpu", action="store_true")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--scenario-id", required=True)
    parser.add_argument("--scenario-overrides-json", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--video-path", required=True)
    parser.add_argument(
        "--fake-mode",
        choices={"healthy", "crash", "missing-video", "sleep"},
        default="healthy",
    )
    args = parser.parse_args()
    if args.executor_run_id != args.run_id:
        return 9
    if args.fake_mode == "crash":
        return 7
    if args.fake_mode == "sleep":
        time.sleep(300)
        return 0

    result_path = Path(args.result_path)
    video_path = Path(args.video_path)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    if args.fake_mode != "missing-video":
        video_path.write_bytes(b"synthetic-mp4" * 32)
    tracking = 0.2 if args.artifact_kind == "native" else 0.22
    result = {
        "version": 1,
        "run_id": args.run_id,
        "candidate_id": args.candidate_id,
        "artifact": args.artifact_kind,
        "scenario_id": args.scenario_id,
        "seed": args.seed,
        "duration_steps": args.duration_steps,
        "status": "completed",
        "video_path": (
            str(video_path) if args.fake_mode != "missing-video" else ""
        ),
        "metrics": {
            "termination_rate": 0.0,
            "tracking_xy_rmse": tracking,
            "max_abs_action_error": (
                0.0 if args.artifact_kind == "native" else 1.0e-6
            ),
        },
        "motion_evidence": {
            "step_dt_seconds": 0.02,
            "peak_steps": {"max_tilt": 5},
            "termination_first_steps": {},
            "review_windows": [
                {
                    "start_step": 0,
                    "end_step": 9,
                    "start_seconds": 0.0,
                    "end_seconds": 0.2,
                    "evidence": ["max_tilt"],
                }
            ],
        },
    }
    result_path.write_text(
        json.dumps(result, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
