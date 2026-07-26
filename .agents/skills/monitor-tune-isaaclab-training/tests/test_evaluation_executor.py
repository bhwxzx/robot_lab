#!/usr/bin/env python3
"""Fault-injection tests for transactional policy-evaluation execution."""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
FAKE_EVALUATOR = Path(__file__).resolve().parent / "fake_policy_evaluator.py"
sys.path.insert(0, str(SCRIPTS))

import execute_evaluation_plan as executor  # noqa: E402
from algorithm_profiles import (  # noqa: E402
    load_registry,
    profile_fingerprint,
    resolve_profile,
)
from build_evaluation_plan import (  # noqa: E402
    _load_candidates,
    build_plan,
)
from collect_evaluation_results import collect  # noqa: E402
from validate_policy_evaluation import evaluate_results  # noqa: E402
from validate_session_spec import SpecError, validate_spec  # noqa: E402


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class EvaluationExecutorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.output = self.root / "evaluation"
        self.checkpoint = self.root / "model.pt"
        self.onnx = self.root / "policy.onnx"
        self.checkpoint.write_bytes(b"native-policy")
        self.onnx.write_bytes(b"onnx-policy")

    def tearDown(self) -> None:
        for process in list(executor._ACTIVE_CHILDREN.values()):
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=3)
                except Exception:
                    process.kill()
                    process.wait(timeout=3)
        executor._ACTIVE_CHILDREN.clear()
        self.temp.cleanup()

    def _command(self, mode: str) -> list[str]:
        return [
            sys.executable,
            str(FAKE_EVALUATOR),
            "--artifact-kind",
            "{artifact_kind}",
            "--artifact-path",
            "{artifact_path}",
            "--artifact-sha256",
            "{artifact_sha256}",
            "--candidate-id",
            "{candidate_id}",
            "--checkpoint-path",
            "{checkpoint_path}",
            "--checkpoint-sha256",
            "{checkpoint_sha256}",
            "--command-schedule-json",
            "{command_schedule_json}",
            "--duration-steps",
            "{duration_steps}",
            "--executor-run-id",
            "{executor_run_id}",
            "--gpu-index",
            "{gpu_index}",
            "{require_idle_gpu_flag}",
            "--result-path",
            "{result_path}",
            "--run-id",
            "{run_id}",
            "--scenario-id",
            "{scenario_id}",
            "--scenario-overrides-json",
            "{scenario_overrides_json}",
            "--seed",
            "{seed}",
            "--video-path",
            "{video_path}",
            "--fake-mode",
            mode,
        ]

    def _session(
        self,
        mode: str = "healthy",
        retries: int = 1,
    ) -> dict[str, object]:
        profile = resolve_profile(load_registry(), "rsl-rl-ppo")
        artifacts = [
            {
                "kind": kind,
                "required": True,
                "command": self._command(mode),
            }
            for kind in ("native", "onnx")
        ]
        scenarios = [
            {
                "id": "nominal",
                "category": "nominal",
                "required": True,
                "seeds": [42],
                "duration_steps": 10,
                "overrides": {},
                "command_schedule": [
                    {
                        "start_step": 0,
                        "end_step": 9,
                        "command": [0.0, 0.0, 0.0],
                    }
                ],
                "video": True,
            },
            {
                "id": "stress",
                "category": "dynamics",
                "required": True,
                "seeds": [42],
                "duration_steps": 10,
                "overrides": {},
                "command_schedule": [
                    {
                        "start_step": 0,
                        "end_step": 9,
                        "command": [0.2, 0.0, 0.0],
                    }
                ],
                "video": True,
            },
        ]
        return {
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
                "run_id": "evaluation-source",
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
                "artifacts": artifacts,
                "scenarios": scenarios,
                "gates": [
                    {
                        "metric": "termination_rate",
                        "op": "<=",
                        "value": 0.01,
                        "aggregation": "max",
                        "artifacts": ["*"],
                        "scenarios": ["*"],
                    },
                    {
                        "metric": "tracking_xy_rmse",
                        "op": "<=",
                        "value": 1.0,
                        "aggregation": "mean",
                        "artifacts": ["*"],
                        "scenarios": ["*"],
                    },
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
                    "minimum_reviewed_videos": 2,
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
                    "state_dir": str(self.output / ".executor"),
                    "max_retries_per_run": retries,
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

    def _case(
        self,
        mode: str = "healthy",
        retries: int = 1,
    ) -> tuple[dict[str, object], dict[str, object]]:
        spec = validate_spec(self._session(mode, retries))
        manifest_path = self.root / "candidates.json"
        manifest_path.write_text(
            json.dumps(
                {
                    "candidates": [
                        {
                            "candidate_id": "candidate-1",
                            "checkpoint_path": str(self.checkpoint),
                            "checkpoint_sha256": _sha256(self.checkpoint),
                            "artifacts": {"onnx": str(self.onnx)},
                            "artifact_sha256": {"onnx": _sha256(self.onnx)},
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        candidates = _load_candidates(
            manifest_path,
            {"native", "onnx"},
        )
        return spec, build_plan(spec, candidates)

    def _state(
        self,
        spec: dict[str, object],
        plan: dict[str, object],
    ) -> tuple[dict[str, object], Path, Path]:
        session_path = self.root / "session.json"
        plan_path = self.root / "plan.json"
        session_path.write_text(
            json.dumps(spec, sort_keys=True),
            encoding="utf-8",
        )
        plan_path.write_text(
            json.dumps(plan, sort_keys=True),
            encoding="utf-8",
        )
        state = executor.initialize_state(
            spec,
            session_path,
            plan,
            plan_path,
        )
        return state, session_path, plan_path

    def _launch_and_finish(
        self,
        spec: dict[str, object],
        plan: dict[str, object],
        state: dict[str, object],
    ) -> dict[str, object]:
        with (
            mock.patch.object(
                executor,
                "_resource_preflight",
                return_value={"synthetic": True},
            ),
            mock.patch.object(executor, "_gpu_idle", return_value=True),
        ):
            executor.launch_next(spec, plan, state)
        run = next(
            run
            for run in state["runs"].values()
            if run["status"] == "running"
        )
        executor._ACTIVE_CHILDREN[run["pid"]].wait(timeout=5)
        return executor.reconcile(spec, plan, state)

    def test_healthy_attempt_promotes_hash_bound_result_and_video(self) -> None:
        spec, plan = self._case()
        state, _, _ = self._state(spec, plan)
        self._launch_and_finish(spec, plan, state)
        run = state["runs"][plan["runs"][0]["run_id"]]
        self.assertEqual(run["status"], "completed")
        result = json.loads(
            Path(run["canonical_result_path"]).read_text(encoding="utf-8")
        )
        self.assertEqual(result["execution_evidence"]["attempt"], 1)
        self.assertEqual(
            result["execution_evidence"]["video_sha256"],
            run["video_sha256"],
        )
        self.assertTrue(result["motion_evidence"]["review_windows"])

    def test_crash_consumes_attempt_and_uses_isolated_retry(self) -> None:
        spec, plan = self._case("crash", retries=1)
        state, _, _ = self._state(spec, plan)
        self._launch_and_finish(spec, plan, state)
        run = state["runs"][plan["runs"][0]["run_id"]]
        self.assertEqual(run["status"], "pending")
        self.assertEqual(run["attempts"], 1)
        first_dir = run["attempt_dir"]
        self._launch_and_finish(spec, plan, state)
        self.assertEqual(run["status"], "failed")
        self.assertEqual(run["attempts"], 2)
        self.assertNotEqual(first_dir, run["attempt_dir"])

    def test_missing_required_video_fails_closed(self) -> None:
        spec, plan = self._case("missing-video", retries=0)
        state, _, _ = self._state(spec, plan)
        self._launch_and_finish(spec, plan, state)
        run = state["runs"][plan["runs"][0]["run_id"]]
        self.assertEqual(run["status"], "failed")
        self.assertIn("video", run["failure_reason"])

    def test_artifact_tamper_is_rejected_before_launch(self) -> None:
        spec, plan = self._case()
        state, _, _ = self._state(spec, plan)
        self.onnx.write_bytes(b"changed")
        first_onnx = next(
            run for run in plan["runs"] if run["artifact"] == "onnx"
        )
        state["runs"] = {
            first_onnx["run_id"]: state["runs"][first_onnx["run_id"]]
        }
        with (
            mock.patch.object(
                executor,
                "_resource_preflight",
                return_value={"synthetic": True},
            ),
            mock.patch.object(executor, "_gpu_idle", return_value=True),
            self.assertRaisesRegex(SpecError, "artifact hash changed"),
        ):
            executor.launch_next(spec, plan, state)
        self.assertEqual(
            state["runs"][first_onnx["run_id"]]["attempts"],
            0,
        )

    def test_plan_command_tamper_is_rejected_before_state_creation(self) -> None:
        spec, plan = self._case()
        plan["runs"][0]["command"][0] = "/bin/false"
        session_path = self.root / "session.json"
        plan_path = self.root / "plan.json"
        session_path.write_text(json.dumps(spec), encoding="utf-8")
        plan_path.write_text(json.dumps(plan), encoding="utf-8")
        with self.assertRaisesRegex(SpecError, "approved template"):
            executor.initialize_state(
                spec,
                session_path,
                plan,
                plan_path,
            )

    def test_busy_gpu_is_rejected_without_consuming_attempt(self) -> None:
        spec, plan = self._case()
        state, _, _ = self._state(spec, plan)
        with (
            mock.patch.object(
                executor,
                "_resource_preflight",
                return_value={"synthetic": True},
            ),
            mock.patch.object(executor, "_gpu_idle", return_value=False),
            self.assertRaisesRegex(SpecError, "GPU is not idle"),
        ):
            executor.launch_next(spec, plan, state)
        run = state["runs"][plan["runs"][0]["run_id"]]
        self.assertEqual(run["attempts"], 0)

    def test_launch_receipt_restores_exact_process_after_scheduler_loss(
        self,
    ) -> None:
        spec, plan = self._case("sleep", retries=0)
        state, _, _ = self._state(spec, plan)
        with (
            mock.patch.object(
                executor,
                "_resource_preflight",
                return_value={"synthetic": True},
            ),
            mock.patch.object(executor, "_gpu_idle", return_value=True),
        ):
            executor.launch_next(spec, plan, state)
        run = state["runs"][plan["runs"][0]["run_id"]]
        process = executor._ACTIVE_CHILDREN[run["pid"]]
        run["status"] = "launching"
        run["pid"] = None
        run["process_group"] = None
        run["process_start_ticks"] = None
        run["launched_at"] = None
        executor.reconcile(spec, plan, state)
        self.assertEqual(run["status"], "running")
        self.assertEqual(run["pid"], process.pid)
        executor._stop_exact_process(run)
        process.wait(timeout=5)

    def test_timeout_stops_only_recorded_process_group(self) -> None:
        spec, plan = self._case("sleep", retries=0)
        state, _, _ = self._state(spec, plan)
        with (
            mock.patch.object(
                executor,
                "_resource_preflight",
                return_value={"synthetic": True},
            ),
            mock.patch.object(executor, "_gpu_idle", return_value=True),
        ):
            executor.launch_next(spec, plan, state)
        run = state["runs"][plan["runs"][0]["run_id"]]
        process = executor._ACTIVE_CHILDREN[run["pid"]]
        run["launched_at"] = time.time() - 61
        executor.reconcile(spec, plan, state)
        self.assertEqual(run["status"], "stopping_timeout")
        process.wait(timeout=5)
        executor.reconcile(spec, plan, state)
        self.assertEqual(run["status"], "failed")

    def test_truncated_journal_tail_recovers_verified_prefix(self) -> None:
        spec, plan = self._case()
        state, session_path, plan_path = self._state(spec, plan)
        executor._persist_state(spec, state, "initialize-evaluation")
        journal = Path(spec["evaluation"]["execution"]["state_dir"]) / (
            executor.JOURNAL_FILENAME
        )
        with journal.open("a", encoding="utf-8") as stream:
            stream.write('{"version":1')
        Path(executor._state_path(spec)).write_text("{}", encoding="utf-8")
        recovered = executor.recover_state_from_journal(
            spec,
            session_path,
            plan_path,
        )
        self.assertEqual(recovered["plan_sha256"], _sha256(plan_path))
        self.assertTrue(journal.read_text(encoding="utf-8").endswith("\n"))

    def test_closed_loop_artifact_delta_rejects_behavioral_drift(self) -> None:
        spec, plan = self._case()
        state, _, _ = self._state(spec, plan)
        for _ in plan["runs"]:
            self._launch_and_finish(spec, plan, state)
        onnx_result = next(
            Path(run["result_path"])
            for run in plan["runs"]
            if run["artifact"] == "onnx"
        )
        changed = json.loads(onnx_result.read_text(encoding="utf-8"))
        changed["metrics"]["tracking_xy_rmse"] = 0.8
        onnx_result.write_text(
            json.dumps(changed, sort_keys=True),
            encoding="utf-8",
        )
        reviews = [
            {
                "candidate_id": "candidate-1",
                "status": "pass",
                "reviewer": "test",
                "reviewed_video_paths": plan["required_videos"][:2],
                "notes": "synthetic review",
            }
        ]
        report = evaluate_results(spec, plan, collect(plan, reviews))
        candidate = report["candidate_results"][0]
        self.assertFalse(candidate["passed"])
        self.assertTrue(
            any(
                failure.get("metric") == "tracking_xy_rmse"
                and "absolute_delta" in failure
                for failure in candidate["parity_failures"]
            )
        )

    def test_collection_rejects_result_changed_after_execution(self) -> None:
        spec, plan = self._case()
        state, _, _ = self._state(spec, plan)
        for _ in plan["runs"]:
            self._launch_and_finish(spec, plan, state)
        result_path = Path(plan["runs"][0]["result_path"])
        changed = json.loads(result_path.read_text(encoding="utf-8"))
        changed["metrics"]["termination_rate"] = 0.5
        result_path.write_text(
            json.dumps(changed, sort_keys=True),
            encoding="utf-8",
        )
        with self.assertRaisesRegex(SpecError, "changed after execution"):
            collect(plan, [], state)


if __name__ == "__main__":
    unittest.main()
