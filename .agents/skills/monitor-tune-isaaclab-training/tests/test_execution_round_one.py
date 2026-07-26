#!/usr/bin/env python3
"""Synthetic tests for version-6 staged execution and robust ranking."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch


SKILL = Path(__file__).resolve().parents[1]
SCRIPTS = SKILL / "scripts"
REPO = SKILL.parents[2]
FAKE_TRAINING = Path(__file__).resolve().parent / "fake_training_command.py"
FAKE_RSL = Path(__file__).resolve().parent / "fake_rsl_rl_train.py"
RSL_ADAPTER = SCRIPTS / "rsl_rl_trial_adapter.py"
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
    _persist_state,
    _process_start_ticks,
    execution_state_lock,
    initialize_state,
    launch_next,
    reconcile,
    recover_state_from_journal,
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

    def _adapter_session(
        self,
        mode: str = "healthy",
        delay_seconds: float = 0.0,
    ) -> dict[str, object]:
        session = self._session()
        self.baseline.unlink(missing_ok=True)
        training_command = [
            sys.executable,
            str(FAKE_RSL),
            "--fake-log-root",
            str(self.root / "rsl-logs"),
        ]
        if mode != "healthy":
            training_command.extend(["--fake-mode", mode])
        if delay_seconds:
            training_command.extend(
                ["--fake-delay-seconds", str(delay_seconds)]
            )
        session["training"]["command"] = training_command
        parameter = session["tuning"]["allowed_parameters"][0]
        parameter.update(
            {
                "path": "agent.algorithm.learning_rate",
                "values": [0.001, 0.002, 0.003],
                "baseline": 0.001,
            }
        )
        session["tuning"]["objectives"][0]["metric"] = "mean_reward"
        session["tuning"]["constraints"][0]["metric"] = "illegal_contact"
        execution = session["execution"]
        execution["run_command"] = [
            sys.executable,
            str(RSL_ADAPTER),
            "--contract",
            "{adapter_contract_path}",
            "--executor-run-id",
            "{run_id}",
            "--overrides-json",
            "{overrides_json}",
            "--effective-config",
            "{effective_config_path}",
            "--result",
            "{result_path}",
            "--summary",
            "{summary_path}",
            "--terminal",
            "{terminal_path}",
            "--log-path",
            "{log_path}",
        ]
        execution["effective_config"]["allow_baseline_bootstrap"] = True
        execution["adapter"] = {
            "id": "rsl-rl",
            "parameter_cli_map": {
                "agent.algorithm.learning_rate":
                    "agent.algorithm.learning_rate",
            },
            "runtime_config_paths": {
                "agent.run_name": "run_id",
                "agent.seed": "seed",
                "env.seed": "seed",
            },
            "summary_last": 5,
            "require_checkpoint": True,
        }
        execution["resource_limits"] = {
            "campaign_timeout_minutes": 120,
            "min_free_disk_gb": 0,
            "max_gpu_temperature_c": 100,
            "stop_grace_seconds": 1,
        }
        execution["quality_rules"][0]["metric"] = "mean_reward"
        return session

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

    def test_quality_rule_honors_minimum_progress(self) -> None:
        session = self._session()
        session["execution"]["quality_rules"][0]["minimum_progress"] = 10
        spec = validate_spec(session)
        warmup = {
            "records": [
                {"progress": 7, "score": -1.0},
                {"progress": 8, "score": -2.0},
                {"progress": 9, "score": -3.0},
            ],
            "non_finite_metrics": [],
        }
        eligible = {
            "records": [
                {"progress": 10, "score": -1.0},
                {"progress": 11, "score": -2.0},
                {"progress": 12, "score": -3.0},
            ],
            "non_finite_metrics": [],
        }
        warmup_report = detect_anomalies(spec, warmup)
        self.assertEqual(warmup_report["status"], "healthy")
        self.assertEqual(
            warmup_report["insufficient_data_rules"][0]["minimum_progress"],
            10,
        )
        self.assertEqual(
            detect_anomalies(spec, eligible)["status"],
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

    def test_state_lock_rejects_concurrent_transition(self) -> None:
        spec = validate_spec(self._session())
        with execution_state_lock(spec):
            with self.assertRaisesRegex(SpecError, "state lock"):
                with execution_state_lock(spec):
                    pass

    def test_hash_chained_journal_recovers_latest_state(self) -> None:
        spec = validate_spec(self._session())
        plan = build_plan(spec)
        session_path, plan_path = self._write_contract(spec, plan)
        state_path, state = initialize_state(
            spec,
            session_path,
            plan,
            plan_path,
        )
        _persist_state(spec, state_path, state, "initialize")
        state["selection_failure_reason"] = "recover-this-snapshot"
        _persist_state(spec, state_path, state, "test-transition")
        state_path.write_text('{"corrupt":true}', encoding="utf-8")
        recovered_path, recovered = recover_state_from_journal(
            spec,
            session_path,
            plan_path,
        )
        self.assertEqual(recovered_path, state_path)
        self.assertEqual(
            recovered["selection_failure_reason"],
            "recover-this-snapshot",
        )

    def test_journal_recovery_discards_only_truncated_tail(self) -> None:
        spec = validate_spec(self._session())
        plan = build_plan(spec)
        session_path, plan_path = self._write_contract(spec, plan)
        state_path, state = initialize_state(
            spec,
            session_path,
            plan,
            plan_path,
        )
        _persist_state(spec, state_path, state, "initialize")
        journal_path = Path(spec["execution"]["state_dir"]) / (
            "execution_events.jsonl"
        )
        with journal_path.open("ab") as stream:
            stream.write(b'{"version":1,"sequence":2')
        state_path.write_text('{"corrupt":true}', encoding="utf-8")
        _, recovered = recover_state_from_journal(
            spec,
            session_path,
            plan_path,
        )
        self.assertEqual(recovered["stage"], "screening")
        self.assertTrue(journal_path.read_bytes().endswith(b"\n"))
        self.assertNotIn(
            b'"sequence":2',
            journal_path.read_bytes(),
        )

    def test_launch_failure_consumes_attempt_and_retry_uses_new_directory(
        self,
    ) -> None:
        session = self._session()
        session["execution"]["max_retries_per_run"] = 1
        spec = validate_spec(session)
        plan = build_plan(spec)
        session_path, plan_path = self._write_contract(spec, plan)
        _, state = initialize_state(spec, session_path, plan, plan_path)
        approved_command = spec["execution"]["run_command"]
        spec["execution"]["run_command"] = [
            "/path/that/does/not/exist",
            *approved_command[1:],
        ]
        transitions: list[tuple[str, str]] = []

        def record_transition(
            current_state: dict[str, object],
            action: str,
        ) -> None:
            run = next(iter(current_state["runs"].values()))
            transitions.append((action, run["status"]))

        with patch("execute_trial_plan._gpu_idle", return_value=True):
            state = launch_next(spec, state, record_transition)
            run = next(iter(state["runs"].values()))
            self.assertEqual(run["status"], "pending")
            self.assertEqual(run["attempts"], 1)
            self.assertTrue(
                (Path(run["run_dir"]) / "attempt-1").is_dir()
            )
            spec["execution"]["run_command"] = approved_command
            state = launch_next(spec, state, record_transition)
            self.assertEqual(run["attempts"], 2)
            self.assertIn("attempt-2", run["log_path"])
            _ACTIVE_CHILDREN[run["pid"]].wait(timeout=10)
            state = reconcile(spec, plan, state)
        self.assertEqual(run["status"], "completed")
        self.assertIn(("reserve-attempt", "launching"), transitions)
        self.assertIn(("launch-failed", "pending"), transitions)
        self.assertIn(("launch-started", "running"), transitions)

    def test_reconcile_restores_process_identity_from_launch_receipt(
        self,
    ) -> None:
        spec = validate_spec(self._session())
        plan = build_plan(spec)
        session_path, plan_path = self._write_contract(spec, plan)
        _, state = initialize_state(spec, session_path, plan, plan_path)
        with patch("execute_trial_plan._gpu_idle", return_value=True):
            state = launch_next(spec, state)
            run = next(
                item for item in state["runs"].values()
                if item["status"] == "running"
            )
            _ACTIVE_CHILDREN[run["pid"]].wait(timeout=10)
            run["status"] = "launching"
            run["pid"] = None
            run["process_group"] = None
            run["process_start_ticks"] = None
            run["launched_at"] = None
            run["launch_receipt_sha256"] = None
            state = reconcile(spec, plan, state)
        self.assertEqual(run["status"], "completed")
        self.assertIsInstance(run["pid"], int)
        self.assertIsInstance(run["process_start_ticks"], int)

    def test_trial_timeout_escalates_exact_process_to_sigkill(self) -> None:
        spec = validate_spec(self._session())
        spec["execution"]["resource_limits"] = {
            "campaign_timeout_minutes": 120,
            "min_free_disk_gb": 0,
            "max_gpu_temperature_c": 100,
            "stop_grace_seconds": 1,
        }
        plan = build_plan(spec)
        session_path, plan_path = self._write_contract(spec, plan)
        _, state = initialize_state(spec, session_path, plan, plan_path)
        process = subprocess.Popen(
            [
                sys.executable,
                "-c",
                (
                    "import signal,time;"
                    "signal.signal(signal.SIGTERM,lambda *_:None);"
                    "time.sleep(60)"
                ),
                "timeout-test-token",
            ],
            start_new_session=True,
        )
        try:
            time.sleep(0.05)
            run = next(iter(state["runs"].values()))
            run.update(
                {
                    "status": "running",
                    "attempts": 1,
                    "pid": process.pid,
                    "process_group": process.pid,
                    "process_start_ticks": _process_start_ticks(process.pid),
                    "argv": [
                        sys.executable,
                        "-c",
                        (
                            "import signal,time;"
                            "signal.signal(signal.SIGTERM,lambda *_:None);"
                            "time.sleep(60)"
                        ),
                        "timeout-test-token",
                    ],
                    "launched_at": time.time() - 3601,
                }
            )
            state = reconcile(spec, plan, state)
            self.assertEqual(run["status"], "stopping_trial_timeout")
            run["stop_requested_at"] = time.time() - 2
            state = reconcile(spec, plan, state)
            self.assertEqual(run["status"], "stopping_forced")
            process.wait(timeout=5)
            state = reconcile(spec, plan, state)
            self.assertEqual(run["status"], "failed")
            self.assertEqual(run["stop_reason"], "approved_trial_timeout")
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=5)

    def test_executor_runs_adapter_and_bootstraps_exact_baseline(self) -> None:
        session = self._session()
        self.baseline.unlink()
        session["training"]["command"] = [
            sys.executable,
            str(FAKE_RSL),
            "--fake-log-root",
            str(self.root / "rsl-logs"),
        ]
        parameter = session["tuning"]["allowed_parameters"][0]
        parameter.update(
            {
                "path": "agent.algorithm.learning_rate",
                "values": [0.001, 0.002, 0.003],
                "baseline": 0.001,
            }
        )
        session["tuning"]["objectives"][0]["metric"] = "mean_reward"
        session["tuning"]["constraints"][0]["metric"] = "illegal_contact"
        execution = session["execution"]
        execution["run_command"] = [
            sys.executable,
            str(RSL_ADAPTER),
            "--contract",
            "{adapter_contract_path}",
            "--executor-run-id",
            "{run_id}",
            "--overrides-json",
            "{overrides_json}",
            "--effective-config",
            "{effective_config_path}",
            "--result",
            "{result_path}",
            "--summary",
            "{summary_path}",
            "--terminal",
            "{terminal_path}",
            "--log-path",
            "{log_path}",
        ]
        execution["effective_config"]["allow_baseline_bootstrap"] = True
        execution["adapter"] = {
            "id": "rsl-rl",
            "parameter_cli_map": {
                "agent.algorithm.learning_rate":
                    "agent.algorithm.learning_rate",
            },
            "runtime_config_paths": {
                "agent.run_name": "run_id",
                "agent.seed": "seed",
                "env.seed": "seed",
            },
            "summary_last": 3,
            "require_checkpoint": True,
        }
        execution["resource_limits"] = {
            "campaign_timeout_minutes": 120,
            "min_free_disk_gb": 0,
            "max_gpu_temperature_c": 100,
            "stop_grace_seconds": 1,
        }
        execution["quality_rules"][0]["metric"] = "mean_reward"
        spec = validate_spec(session)
        plan = build_plan(spec)
        session_path, plan_path = self._write_contract(spec, plan)
        _, state = initialize_state(spec, session_path, plan, plan_path)
        with (
            patch("execute_trial_plan._gpu_idle", return_value=True),
            patch(
                "execute_trial_plan._resource_preflight",
                return_value={
                    "checked_at": time.time(),
                    "free_disk_gb": 100.0,
                    "gpu_temperature_c": 40,
                },
            ),
        ):
            state = launch_next(spec, state)
            run = next(
                item for item in state["runs"].values()
                if item["status"] == "running"
            )
            _ACTIVE_CHILDREN[run["pid"]].wait(timeout=10)
            state = reconcile(spec, plan, state)
        self.assertEqual(
            run["status"],
            "completed",
            (
                f"{run['failure_reason']}\n"
                f"{Path(run['log_path']).read_text(encoding='utf-8')}"
            ),
        )
        self.assertTrue(self.baseline.is_file())
        self.assertEqual(run["terminal_receipt"]["status"], "completed")
        self.assertEqual(
            run["config_report"]["verified_runtime_values"]["agent.seed"][
                "expected"
            ],
            42,
        )

    def test_live_collapse_stops_exact_adapter_process_group(self) -> None:
        session = self._adapter_session("collapse", 0.3)
        rule = session["execution"]["quality_rules"][0]
        rule["consecutive_windows"] = 3
        rule["minimum_progress"] = 0
        spec = validate_spec(session)
        plan = build_plan(spec)
        session_path, plan_path = self._write_contract(spec, plan)
        _, state = initialize_state(spec, session_path, plan, plan_path)
        with (
            patch("execute_trial_plan._gpu_idle", return_value=True),
            patch(
                "execute_trial_plan._resource_preflight",
                return_value={
                    "checked_at": time.time(),
                    "free_disk_gb": 100.0,
                    "gpu_temperature_c": 40,
                },
            ),
        ):
            state = launch_next(spec, state)
            run = next(
                item for item in state["runs"].values()
                if item["status"] == "running"
            )
            deadline = time.time() + 10
            while time.time() < deadline:
                summary_path = Path(run["summary_path"])
                if summary_path.exists():
                    summary = json.loads(summary_path.read_text())
                    if len(summary["records"]) >= 3:
                        break
                time.sleep(0.02)
            self.assertTrue(_process_start_ticks(run["pid"]))
            state = reconcile(spec, plan, state)
            self.assertEqual(run["status"], "stopping_quality_rule")
            _ACTIVE_CHILDREN[run["pid"]].wait(timeout=10)
            state = reconcile(spec, plan, state)
        self.assertEqual(run["status"], "failed")
        self.assertEqual(run["stop_reason"], "approved_quality_stop_rule")

    def test_trial_reproducibility_manifest_is_hash_bound(self) -> None:
        session = self._session()
        session["training"]["cwd"] = str(REPO)
        session["execution"]["reproducibility"] = {
            "enabled": True,
            "capture_git_diff": False,
            "capture_gpu": False,
            "package_names": ["PyYAML"],
            "input_paths": [str(self.baseline)],
        }
        spec = validate_spec(session)
        plan = build_plan(spec)
        session_path, plan_path = self._write_contract(spec, plan)
        _, state = initialize_state(spec, session_path, plan, plan_path)
        with patch("execute_trial_plan._gpu_idle", return_value=True):
            state = launch_next(spec, state)
            run = next(
                item for item in state["runs"].values()
                if item["status"] == "running"
            )
            _ACTIVE_CHILDREN[run["pid"]].wait(timeout=10)
            state = reconcile(spec, plan, state)
        self.assertEqual(run["status"], "completed")
        manifest_path = Path(run["reproducibility_path"])
        manifest = json.loads(manifest_path.read_text())
        self.assertEqual(manifest["git"]["root"], str(REPO))
        self.assertEqual(
            manifest["inputs"][0]["sha256"],
            hashlib.sha256(self.baseline.read_bytes()).hexdigest(),
        )
        self.assertEqual(
            run["reproducibility_sha256"],
            hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        )

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
