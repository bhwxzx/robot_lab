#!/usr/bin/env python3
"""Synthetic tests for version-6 staged execution and robust ranking."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


SKILL = Path(__file__).resolve().parents[1]
SCRIPTS = SKILL / "scripts"
FAKE_TRAINING = Path(__file__).resolve().parent / "fake_training_command.py"
sys.path.insert(0, str(SCRIPTS))

from algorithm_profiles import (  # noqa: E402
    load_registry,
    profile_fingerprint,
    resolve_profile,
)
from build_trial_plan import build_plan  # noqa: E402
from detect_training_anomalies import detect_anomalies  # noqa: E402
from execute_trial_plan import (  # noqa: E402
    _ACTIVE_CHILDREN,
    _append_confirmation_runs,
    _collect_completed_results,
    initialize_state,
    launch_next,
    reconcile,
    state_summary,
)
from rank_trials import rank  # noqa: E402
from validate_effective_config import validate_effective_config  # noqa: E402
from validate_session_spec import SpecError, validate_spec  # noqa: E402


class ExecutionRoundOneTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.baseline = self.root / "baseline.json"
        self.baseline.write_text(
            json.dumps(
                {
                    "agent": {"learning_rate": 0.1},
                    "env": {"penalty": -0.1},
                }
            ),
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _algorithm(self) -> dict[str, object]:
        profile = resolve_profile(load_registry(), "rsl-rl-ppo")
        return {
            "backend": "rsl_rl",
            "name": "PPO",
            "runner_class": "OnPolicyRunner",
            "profile_id": "rsl-rl-ppo",
            "profile_version": profile["profile_version"],
            "profile_fingerprint": profile_fingerprint(profile),
            "unknown_algorithm_policy": "reject",
        }

    def _session(self) -> dict[str, object]:
        return {
            "version": 6,
            "mode": "tune",
            "algorithm": self._algorithm(),
            "training": {
                "command": [sys.executable, "train.py"],
                "cwd": str(self.root),
                "log_path": str(self.root / "training.log"),
                "run_id": "round-one-test",
            },
            "monitoring": {
                "check_interval_seconds": 60,
                "stale_after_seconds": 120,
                "pid": None,
                "gpu_index": 0,
                "tensorboard_path": None,
                "expected_process_pattern": "fake_training_command.py",
                "low_gpu_utilization_percent": 5,
            },
            "recovery": {
                "enabled": False,
                "max_restarts": 0,
                "cooldown_seconds": 0,
            },
            "tuning": {
                "allowed_parameters": [
                    {
                        "path": "agent.learning_rate",
                        "values": [0.1, 0.2, 0.3],
                        "baseline": 0.1,
                    }
                ],
                "protected_parameters_unlocked": [],
                "max_trials": 3,
                "seeds": [42, 43, 44],
                "seed_strategy": {
                    "screening_seeds": [42],
                    "confirmation_seeds": [42, 43, 44],
                    "confirmation_top_k": 1,
                },
                "ranking": {
                    "require_paired_baseline": True,
                    "constraint_scope": "each_seed",
                    "minimum_final_training_seeds": 3,
                    "pareto_front_required": True,
                },
                "trial_timeout_minutes": 60,
                "max_concurrent_trials": 1,
                "mutation_scope": "overrides_only",
                "objectives": [
                    {
                        "metric": "score",
                        "goal": "maximize",
                        "weight": 1,
                        "minimum_improvement": 0.1,
                    }
                ],
                "constraints": [
                    {
                        "metric": "unsafe",
                        "op": "<=",
                        "value": 0,
                        "scope": "each_seed",
                    }
                ],
            },
            "evaluation": None,
            "archive": None,
            "hardware_feedback": None,
            "execution": {
                "enabled": True,
                "state_dir": str(self.root / "state"),
                "run_command": [
                    sys.executable,
                    str(FAKE_TRAINING),
                    "--baseline",
                    str(self.baseline),
                    "--effective-config",
                    "{effective_config_path}",
                    "--result",
                    "{result_path}",
                    "--summary",
                    "{summary_path}",
                    "--run-dir",
                    "{run_dir}",
                    "--run-id",
                    "{run_id}",
                    "--trial-id",
                    "{trial_id}",
                    "--stage",
                    "{stage}",
                    "--seed",
                    "{seed}",
                    "--gpu-index",
                    "{gpu_index}",
                    "--overrides-json",
                    "{overrides_json}",
                ],
                "gpu_index": 0,
                "require_idle_gpu": True,
                "max_retries_per_run": 0,
                "effective_config": {
                    "enabled": True,
                    "baseline_path": str(self.baseline),
                    "require_exact_override_match": True,
                },
                "quality_rules": [
                    {
                        "id": "reward-collapse",
                        "metric": "score",
                        "op": "<",
                        "value": 0,
                        "consecutive_windows": 3,
                        "action": "stop_trial",
                    }
                ],
                "nonfinite_action": "stop_trial",
            },
            "cleanup": {"remove_created_temp_files": True},
        }

    def _write_contract(
        self,
        session: dict[str, object],
        plan: dict[str, object],
    ) -> tuple[Path, Path]:
        session_path = self.root / "session.json"
        plan_path = self.root / "plan.json"
        session_path.write_text(json.dumps(session), encoding="utf-8")
        plan_path.write_text(json.dumps(plan), encoding="utf-8")
        return session_path, plan_path

    def test_staged_executor_is_single_run_and_resumable(self) -> None:
        spec = validate_spec(self._session())
        plan = build_plan(spec)
        self.assertEqual(len(plan["stages"]["screening"]["runs"]), 3)
        session_path, plan_path = self._write_contract(spec, plan)
        state_path, state = initialize_state(
            spec,
            session_path,
            plan,
            plan_path,
        )
        with patch("execute_trial_plan._gpu_idle", return_value=True):
            while state["stage"] not in {"completed", "blocked"}:
                pending = any(
                    run["status"] == "pending"
                    for run in state["runs"].values()
                )
                if pending:
                    state = launch_next(spec, state)
                    with self.assertRaisesRegex(SpecError, "already active"):
                        launch_next(spec, state)
                    active = next(
                        run
                        for run in state["runs"].values()
                        if run["status"] == "running"
                    )
                    _ACTIVE_CHILDREN[active["pid"]].wait()
                state = reconcile(spec, plan, state)
        self.assertEqual(state["stage"], "completed")
        self.assertEqual(state["screening_selection"], ["trial-002"])
        self.assertEqual(len(state["runs"]), 7)
        self.assertTrue(
            all(run["status"] == "completed" for run in state["runs"].values())
        )
        _, loaded = initialize_state(spec, session_path, plan, plan_path)
        self.assertEqual(loaded["session_sha256"], state["session_sha256"])
        self.assertTrue(state_path.is_file())
        self.assertEqual(state_summary(state)["counts"], {"completed": 7})
        ranked = rank(spec, _collect_completed_results(state))
        self.assertEqual(
            ranked["recommended_for_evaluation"],
            "trial-002",
        )
        self.assertEqual(ranked["expected_seeds"], [42, 43, 44])
        self.assertTrue(ranked["ranking"][0]["pareto_optimal"])

    def test_effective_config_rejects_unapproved_difference(self) -> None:
        spec = validate_spec(self._session())
        candidate = self.root / "candidate.json"
        candidate.write_text(
            json.dumps(
                {
                    "agent": {"learning_rate": 0.2},
                    "env": {"penalty": -0.2},
                }
            ),
            encoding="utf-8",
        )
        with self.assertRaisesRegex(SpecError, "unauthorized differences"):
            validate_effective_config(
                spec,
                self.baseline,
                candidate,
                {"agent.learning_rate": 0.2},
            )

    def test_effective_config_rejects_nested_nonfinite_value(self) -> None:
        spec = validate_spec(self._session())
        candidate = self.root / "candidate.json"
        candidate.write_text(
            '{"agent":{"learning_rate":0.2},'
            '"env":{"penalty":-0.1},'
            '"extra":[{"bad":NaN}]}',
            encoding="utf-8",
        )
        with self.assertRaisesRegex(SpecError, "non-finite"):
            validate_effective_config(
                spec,
                self.baseline,
                candidate,
                {"agent.learning_rate": 0.2},
            )

    def test_anomaly_requires_consecutive_windows(self) -> None:
        spec = validate_spec(self._session())
        healthy = {
            "records": [{"score": -1.0}, {"score": 1.0}, {"score": -1.0}],
            "non_finite_metrics": [],
        }
        stopped = {
            "records": [{"score": -1.0}, {"score": -2.0}, {"score": -3.0}],
            "non_finite_metrics": [],
        }
        self.assertEqual(detect_anomalies(spec, healthy)["status"], "healthy")
        self.assertEqual(
            detect_anomalies(spec, stopped)["status"],
            "stop_approved",
        )

    def test_each_seed_constraint_rejects_unsafe_candidate(self) -> None:
        spec = validate_spec(self._session())
        runs = []
        for trial_id, base_score in (("baseline", 10.0), ("trial-001", 12.0)):
            for seed in (42, 43, 44):
                runs.append(
                    {
                        "trial_id": trial_id,
                        "seed": seed,
                        "status": "completed",
                        "metrics": {
                            "score": base_score,
                            "unsafe": (
                                1.0
                                if trial_id == "trial-001" and seed == 44
                                else 0.0
                            ),
                        },
                    }
                )
        ranked = rank(spec, runs)
        self.assertEqual(ranked["ranking"][0]["trial_id"], "baseline")
        rejected = next(
            item for item in ranked["ineligible"]
            if item["trial_id"] == "trial-001"
        )
        self.assertEqual(
            rejected["constraint_failures"][0]["failures"][0]["seed"],
            44,
        )

    def test_version6_requires_execution_contract(self) -> None:
        session = self._session()
        session["execution"] = None
        with self.assertRaisesRegex(SpecError, "require execution"):
            validate_spec(session)

    def test_executor_rejects_plan_not_bound_to_session(self) -> None:
        spec = validate_spec(self._session())
        plan = build_plan(spec)
        plan["authorized_parameter_paths"] = ["env.penalty"]
        session_path, plan_path = self._write_contract(spec, plan)
        with self.assertRaisesRegex(SpecError, "does not exactly match"):
            initialize_state(spec, session_path, plan, plan_path)

    def test_execution_template_rejects_invalid_formatting(self) -> None:
        session = self._session()
        session["execution"]["run_command"][0] = "{seed!r}"
        with self.assertRaisesRegex(SpecError, "cannot format or convert"):
            validate_spec(session)

    def test_final_selection_requires_pareto_candidate(self) -> None:
        spec = validate_spec(self._session())
        runs = []
        for trial_id, score in (("baseline", 10.0), ("trial-001", 12.0)):
            for seed in (42, 43, 44):
                runs.append(
                    {
                        "trial_id": trial_id,
                        "seed": seed,
                        "status": "completed",
                        "metrics": {"score": score, "unsafe": 0.0},
                    }
                )
        spec["evaluation"] = {"enabled": True}
        report = rank(
            spec,
            runs,
            {
                "algorithm": spec["algorithm"],
                "simulation_qualified_candidates": ["baseline"],
            },
        )
        self.assertEqual(
            report["selection_status"],
            "no_simulation_qualified_candidate",
        )
        self.assertIsNone(report["final_selection"])

    def test_completed_quality_stop_is_not_retried(self) -> None:
        session = self._session()
        session["execution"]["max_retries_per_run"] = 1
        spec = validate_spec(session)
        plan = build_plan(spec)
        session_path, plan_path = self._write_contract(spec, plan)
        _, state = initialize_state(spec, session_path, plan, plan_path)
        run = next(iter(state["runs"].values()))
        attempt_dir = Path(run["run_dir"]) / "attempt-1"
        attempt_dir.mkdir(parents=True)
        run.update(
            {
                "status": "running",
                "attempts": 1,
                "pid": 2**30,
                "process_group": 2**30,
                "process_start_ticks": 1,
                "argv": ["not-live"],
                "summary_path": str(attempt_dir / "training_summary.json"),
                "result_path": str(attempt_dir / "result.json"),
                "effective_config_path": str(
                    attempt_dir / "effective_config.json"
                ),
            }
        )
        Path(run["summary_path"]).write_text(
            json.dumps(
                {
                    "records": [
                        {"score": -1.0},
                        {"score": -2.0},
                        {"score": -3.0},
                    ],
                    "non_finite_metrics": [],
                }
            ),
            encoding="utf-8",
        )
        state = reconcile(spec, plan, state)
        self.assertEqual(run["status"], "failed")
        self.assertEqual(state["stage"], "blocked")
        self.assertEqual(run["attempts"], 1)

    def test_versions3_through5_keep_static_plan_and_ranking(self) -> None:
        for version in (3, 4, 5):
            with self.subTest(version=version):
                session = self._session()
                session["version"] = version
                session.pop("execution")
                session["tuning"].pop("seed_strategy")
                session["tuning"].pop("ranking")
                session["tuning"]["objectives"][0].pop(
                    "minimum_improvement"
                )
                session["tuning"]["constraints"][0].pop("scope")
                spec = validate_spec(session)
                plan = build_plan(spec)
                self.assertEqual(plan["version"], 3)
                self.assertNotIn("runs", plan)
                runs = []
                for trial_id, score in (
                    ("baseline", 10.0),
                    ("trial-001", 11.0),
                    ("trial-002", 12.0),
                ):
                    for seed in (42, 43, 44):
                        runs.append(
                            {
                                "trial_id": trial_id,
                                "seed": seed,
                                "status": "completed",
                                "metrics": {
                                    "score": score,
                                    "unsafe": 0.0,
                                },
                            }
                        )
                report = rank(spec, runs)
                self.assertEqual(report["version"], 3)
                self.assertEqual(
                    report["recommended_for_evaluation"],
                    "trial-002",
                )

    def test_screening_must_leave_confirmation_seeds(self) -> None:
        session = self._session()
        session["tuning"]["seed_strategy"]["screening_seeds"] = [42, 43, 44]
        with self.assertRaisesRegex(SpecError, "proper subset"):
            validate_spec(session)

    def test_insufficient_screening_candidates_persists_blocked_state(
        self,
    ) -> None:
        spec = validate_spec(self._session())
        plan = build_plan(spec)
        session_path, plan_path = self._write_contract(spec, plan)
        _, state = initialize_state(spec, session_path, plan, plan_path)
        for run in state["runs"].values():
            result_path = Path(run["result_path"])
            result_path.parent.mkdir(parents=True, exist_ok=True)
            result_path.write_text(
                json.dumps(
                    {
                        "trial_id": run["trial_id"],
                        "seed": run["seed"],
                        "status": "completed",
                        "metrics": {
                            "score": 10.0,
                            "unsafe": (
                                0.0
                                if run["trial_id"] == "baseline"
                                else 1.0
                            ),
                        },
                    }
                ),
                encoding="utf-8",
            )
            run["status"] = "completed"
        _append_confirmation_runs(spec, plan, state)
        self.assertEqual(state["stage"], "blocked")
        self.assertIn(
            "fewer eligible candidates",
            state["selection_failure_reason"],
        )


if __name__ == "__main__":
    unittest.main()
