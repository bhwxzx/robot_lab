#!/usr/bin/env python3
"""Tests for hash-bound, human-readable policy visual-review bundles."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from algorithm_profiles import (  # noqa: E402
    load_registry,
    profile_fingerprint,
    resolve_profile,
)
from build_evaluation_plan import build_plan  # noqa: E402
from build_visual_review_bundle import build_review_bundle  # noqa: E402
from validate_session_spec import SpecError, validate_spec  # noqa: E402


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class VisualReviewBundleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.output = self.root / "evaluation"
        self.state_dir = self.output / ".executor"
        self.checkpoint = self.root / "model.pt"
        self.onnx = self.root / "policy.onnx"
        self.checkpoint.write_bytes(b"native-policy")
        self.onnx.write_bytes(b"onnx-policy")
        profile = resolve_profile(load_registry(), "rsl-rl-ppo")
        self.spec = {
            "version": 4,
            "mode": "monitor",
            "algorithm": {
                "backend": "rsl_rl",
                "name": "PPO",
                "runner_class": "OnPolicyRunner",
                "profile_id": profile["id"],
                "profile_version": profile["profile_version"],
                "profile_fingerprint": profile_fingerprint(profile),
                "unknown_algorithm_policy": "reject",
            },
            "training": {
                "command": ["python", "train.py", "--headless"],
                "cwd": str(self.root),
                "log_path": str(self.root / "training.log"),
                "run_id": "review-source",
            },
            "monitoring": {
                "check_interval_seconds": 60,
                "stale_after_seconds": 120,
                "pid": None,
                "gpu_index": 0,
                "tensorboard_path": None,
                "expected_process_pattern": "train.py",
                "low_gpu_utilization_percent": 5,
            },
            "recovery": {
                "enabled": False,
                "max_restarts": 0,
                "cooldown_seconds": 0,
            },
            "tuning": None,
            "evaluation": {
                "enabled": True,
                "require_for_final_selection": True,
                "artifacts": [
                    {
                        "kind": kind,
                        "required": True,
                        "command": self._command(),
                    }
                    for kind in ("native", "onnx")
                ],
                "scenarios": [
                    self._scenario(
                        "nominal-command-sweep",
                        "nominal",
                        [0.3, 0.0, 0.0],
                    ),
                    self._scenario(
                        "rough-push-turn-reversal",
                        "disturbance",
                        [-0.2, 0.0, 0.3],
                    ),
                ],
                "gates": [
                    {
                        "metric": "termination_rate",
                        "op": "<=",
                        "value": 0.05,
                        "aggregation": "max",
                        "artifacts": ["*"],
                        "scenarios": ["*"],
                    }
                ],
                "parity": {
                    "required": True,
                    "reference_artifact": "native",
                    "max_abs_action_error": 1.0e-5,
                    "closed_loop_metrics": [
                        {
                            "metric": "tracking_xy_rmse",
                            "max_abs_delta": 0.05,
                            "aggregation": "max",
                        }
                    ],
                },
                "visual_review": {
                    "required": True,
                    "minimum_reviewed_videos": 4,
                    "require_notes": True,
                },
                "output_dir": str(self.output),
                "gpu_index": 0,
                "require_idle_gpu": True,
                "max_concurrent_runs": 1,
                "run_timeout_minutes": 1,
                "allow_reject_candidate": True,
                "allow_retune_on_failure": False,
                "execution": {
                    "state_dir": str(self.state_dir),
                    "max_retries_per_run": 1,
                    "stop_grace_seconds": 1,
                    "min_free_disk_gb": 0.001,
                    "max_gpu_temperature_c": 90,
                    "minimum_video_bytes": 32,
                },
            },
            "archive": None,
            "hardware_feedback": None,
            "execution": None,
            "cleanup": {"remove_created_temp_files": True},
        }
        self.spec = validate_spec(self.spec)
        candidates = [self._candidate("candidate-1")]
        self.plan = build_plan(self.spec, candidates)
        self.session_path = self.root / "session.json"
        self.plan_path = self.root / "evaluation-plan.json"
        self.state_path = self.state_dir / "evaluation_state.json"
        self.session_path.write_text(
            json.dumps(self.spec, sort_keys=True),
            encoding="utf-8",
        )
        self.plan_path.write_text(
            json.dumps(self.plan, sort_keys=True),
            encoding="utf-8",
        )
        self.state = self._completed_state()
        self.state_dir.mkdir(parents=True)
        self.state_path.write_text(
            json.dumps(self.state, sort_keys=True),
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _command(self) -> list[str]:
        return [
            "fake-evaluator",
            "{run_id}",
            "{artifact_kind}",
            "{artifact_path}",
            "{artifact_sha256}",
            "{candidate_id}",
            "{checkpoint_path}",
            "{checkpoint_sha256}",
            "{command_schedule_json}",
            "{duration_steps}",
            "{executor_run_id}",
            "{gpu_index}",
            "{result_path}",
            "{scenario_id}",
            "{scenario_overrides_json}",
            "{seed}",
            "{video_path}",
            "{require_idle_gpu_flag}",
        ]

    def _scenario(
        self,
        scenario_id: str,
        category: str,
        command: list[float],
    ) -> dict[str, object]:
        return {
            "id": scenario_id,
            "category": category,
            "required": True,
            "seeds": [42],
            "duration_steps": 10,
            "overrides": {},
            "command_schedule": [
                {
                    "start_step": 0,
                    "end_step": 9,
                    "command": command,
                }
            ],
            "video": True,
        }

    def _candidate(self, candidate_id: str) -> dict[str, object]:
        return {
            "candidate_id": candidate_id,
            "checkpoint_path": str(self.checkpoint),
            "checkpoint_sha256": _sha256(self.checkpoint),
            "artifacts": {
                "native": str(self.checkpoint),
                "onnx": str(self.onnx),
            },
            "artifact_sha256": {
                "native": _sha256(self.checkpoint),
                "onnx": _sha256(self.onnx),
            },
        }

    def _completed_state(self) -> dict[str, object]:
        runs = {}
        for index, run in enumerate(self.plan["runs"]):
            video_path = Path(run["video_path"])
            result_path = Path(run["result_path"])
            video_path.parent.mkdir(parents=True, exist_ok=True)
            video_path.write_bytes(
                (f"video-{run['run_id']}".encode("utf-8") + b"x" * 128)
            )
            video_hash = _sha256(video_path)
            result = {
                "version": 1,
                "run_id": run["run_id"],
                "candidate_id": run["candidate_id"],
                "artifact": run["artifact"],
                "scenario_id": run["scenario_id"],
                "seed": run["seed"],
                "status": "completed",
                "video_path": run["video_path"],
                "metrics": {
                    "termination_rate": 0.0,
                    "tracking_xy_rmse": 0.1 + index * 0.01,
                },
                "motion_evidence": {
                    "step_dt_seconds": 0.02,
                    "review_windows": [
                        {
                            "start_step": 2,
                            "end_step": 4,
                            "start_seconds": 0.04,
                            "end_seconds": 0.1,
                            "evidence": ["max_tilt"],
                        }
                    ],
                },
                "execution_evidence": {
                    "video_sha256": video_hash,
                },
            }
            result_path.write_text(
                json.dumps(result, sort_keys=True),
                encoding="utf-8",
            )
            runs[run["run_id"]] = {
                "run_id": run["run_id"],
                "status": "completed",
                "canonical_result_path": run["result_path"],
                "canonical_video_path": run["video_path"],
                "result_sha256": _sha256(result_path),
                "video_sha256": video_hash,
            }
        return {
            "version": 1,
            "session_sha256": _sha256(self.session_path),
            "plan_sha256": _sha256(self.plan_path),
            "algorithm": self.spec["algorithm"],
            "training_run_id": self.spec["training"]["run_id"],
            "stage": "awaiting_visual_review",
            "runs": runs,
        }

    def _build(self) -> dict[str, object]:
        return build_review_bundle(
            self.spec,
            self.session_path,
            self.plan,
            self.plan_path,
            self.state,
            self.state_path,
        )

    def test_builds_readable_hash_bound_bundle_without_copying_videos(self) -> None:
        summary = self._build()
        review_dir = self.output / "review"
        self.assertEqual(summary["state"], "review_bundle_ready")
        self.assertEqual(summary["video_count"], 4)
        aliases = sorted((review_dir / "required_videos").iterdir())
        self.assertEqual(len(aliases), 4)
        self.assertTrue(all(path.is_symlink() for path in aliases))
        self.assertIn(
            "candidate-1__native__nominal-command-sweep__seed-42",
            aliases[0].name,
        )
        for alias in aliases:
            self.assertTrue(alias.resolve().is_file())
        index = (review_dir / "REVIEW_INDEX.md").read_text(encoding="utf-8")
        self.assertIn("nominal-command-sweep", index)
        self.assertIn("rough-push-turn-reversal", index)
        self.assertIn("0.040s–0.100s", index)
        manifest = json.loads(
            (review_dir / "review_manifest.json").read_text(encoding="utf-8")
        )
        self.assertEqual(manifest["execution_state_sha256"], _sha256(self.state_path))
        self.assertEqual(manifest["video_count"], 4)
        draft = json.loads(
            (review_dir / "visual_reviews.draft.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(draft["visual_reviews"][0]["status"], "pending")
        self.assertEqual(len(draft["visual_reviews"][0]["reviewed_video_paths"]), 4)

    def test_cli_builds_bundle_from_fully_validated_session(self) -> None:
        completed = subprocess.run(
            [
                sys.executable,
                str(SCRIPTS / "build_visual_review_bundle.py"),
                str(self.session_path),
                str(self.plan_path),
                "--execution-state",
                str(self.state_path),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        summary = json.loads(completed.stdout)
        self.assertEqual(summary["state"], "review_bundle_ready")
        self.assertEqual(summary["video_count"], 4)

    def test_multi_candidate_aliases_preserve_candidate_identity(self) -> None:
        self.plan = build_plan(
            self.spec,
            [
                self._candidate("candidate-1"),
                self._candidate("candidate-2"),
            ],
        )
        self.plan_path.write_text(
            json.dumps(self.plan, sort_keys=True),
            encoding="utf-8",
        )
        self.state = self._completed_state()
        self.state_path.write_text(
            json.dumps(self.state, sort_keys=True),
            encoding="utf-8",
        )

        summary = self._build()
        aliases = sorted((self.output / "review" / "required_videos").iterdir())
        self.assertEqual(summary["video_count"], 8)
        self.assertTrue(any("__candidate-1__" in path.name for path in aliases))
        self.assertTrue(any("__candidate-2__" in path.name for path in aliases))
        draft = json.loads(
            (self.output / "review" / "visual_reviews.draft.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(
            [review["candidate_id"] for review in draft["visual_reviews"]],
            ["candidate-1", "candidate-2"],
        )

    def test_identical_rerun_is_idempotent_and_preserves_edited_draft(self) -> None:
        first = self._build()
        draft_path = Path(first["draft_path"])
        draft = json.loads(draft_path.read_text(encoding="utf-8"))
        draft["visual_reviews"][0]["notes"] = "review in progress"
        draft_path.write_text(
            json.dumps(draft, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        second = self._build()
        self.assertEqual(
            second["created"],
            {"manifest": False, "index": False, "draft": False},
        )
        preserved = json.loads(draft_path.read_text(encoding="utf-8"))
        self.assertEqual(
            preserved["visual_reviews"][0]["notes"],
            "review in progress",
        )

    def test_rejects_non_terminal_execution_state(self) -> None:
        self.state["stage"] = "executing"
        with self.assertRaisesRegex(SpecError, "awaiting_visual_review"):
            self._build()

    def test_rejects_tampered_canonical_video(self) -> None:
        video_path = Path(self.plan["required_videos"][0])
        video_path.write_bytes(b"tampered" * 20)
        with self.assertRaisesRegex(SpecError, "video hash is invalid"):
            self._build()

    def test_rejects_malformed_motion_review_window(self) -> None:
        run = self.plan["runs"][0]
        result_path = Path(run["result_path"])
        result = json.loads(result_path.read_text(encoding="utf-8"))
        result["motion_evidence"]["review_windows"][0]["start_seconds"] = "bad"
        result_path.write_text(
            json.dumps(result, sort_keys=True),
            encoding="utf-8",
        )
        self.state["runs"][run["run_id"]]["result_sha256"] = _sha256(
            result_path
        )
        self.state_path.write_text(
            json.dumps(self.state, sort_keys=True),
            encoding="utf-8",
        )
        with self.assertRaisesRegex(SpecError, "finite number"):
            self._build()

    def test_rejects_alias_collision(self) -> None:
        aliases = self.output / "review" / "required_videos"
        aliases.mkdir(parents=True)
        aliases.joinpath(
            "001__candidate-1__native__nominal-command-sweep__seed-42.mp4"
        ).write_bytes(b"collision")
        with self.assertRaisesRegex(SpecError, "alias collision"):
            self._build()

    def test_rejects_video_path_outside_evaluation_root(self) -> None:
        run = self.plan["runs"][0]
        run["video_path"] = str(self.root / "escaped.mp4")
        run["run_dir"] = str(self.root)
        self.plan["required_videos"][0] = run["video_path"]
        self.plan_path.write_text(
            json.dumps(self.plan, sort_keys=True),
            encoding="utf-8",
        )
        self.state["plan_sha256"] = _sha256(self.plan_path)
        with self.assertRaisesRegex(SpecError, "evaluation.output_dir"):
            self._build()


if __name__ == "__main__":
    unittest.main()
