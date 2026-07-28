#!/usr/bin/env python3
"""Synchronized multi-fidelity planning, resume, and executor tests."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import unittest
from pathlib import Path


SKILL = Path(__file__).resolve().parents[1]
SCRIPTS = SKILL / "scripts"
TESTS = Path(__file__).resolve().parent
FAKE_RSL = TESTS / "fake_rsl_rl_train.py"
RSL_ADAPTER = SCRIPTS / "rsl_rl_trial_adapter.py"
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(TESTS))

from build_trial_plan import (  # noqa: E402
    advance_multifidelity_plan,
    build_plan,
    validate_trial_plan,
)
from execute_trial_plan import (  # noqa: E402
    _adapter_contract,
    adopt_expanded_plan,
    initialize_state,
)
from git_mailbox import (  # noqa: E402
    _build_jobs,
    claim,
    publish,
    publish_multifidelity_rung,
    result as publish_result,
)
from finalize_multifidelity_results import final_rung_results  # noqa: E402
from rank_trials import rank  # noqa: E402
from rsl_rl_trial_adapter import build_child_argv  # noqa: E402
import test_fixed_single_seed as fixed_tests  # noqa: E402
import test_git_mailbox as mailbox_tests  # noqa: E402
from validate_session_spec import SpecError, validate_spec  # noqa: E402


class MultiFidelityTrainingTests(unittest.TestCase):
    def setUp(self) -> None:
        helper = fixed_tests.FixedSingleSeedTests(methodName="runTest")
        helper.setUp()
        self.helper = helper
        self.root = helper.root
        self.session = helper.session
        self.baseline = self.root / "baseline.json"
        self.baseline.write_text(
            json.dumps(
                {
                    "agent": {
                        "algorithm": {"learning_rate": 0.001},
                        "max_iterations": 10,
                        "run_name": "baseline-template",
                        "seed": 0,
                    },
                    "env": {
                        "seed": 0,
                        "terrain": {"difficulty": 1},
                    },
                }
            ),
            encoding="utf-8",
        )
        self.session["training"]["command"] = [
            sys.executable,
            str(FAKE_RSL),
            "--fake-log-root",
            str(self.root / "rsl-logs"),
        ]
        self.session["tuning"]["allowed_parameters"] = [
            {
                "path": "agent.algorithm.learning_rate",
                "values": [0.001, 0.002, 0.003],
                "baseline": 0.001,
            }
        ]
        self.session["tuning"]["objectives"][0]["metric"] = "mean_reward"
        self.session["tuning"]["constraints"][0][
            "metric"
        ] = "illegal_contact"
        execution = self.session["execution"]
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
        execution["effective_config"]["baseline_path"] = str(self.baseline)
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
            "multi_fidelity": {
                "budget_cli_path": "agent.max_iterations",
                "resume_cli_paths": {
                    "enabled": "agent.resume",
                    "load_run": "agent.load_run",
                    "load_checkpoint": "agent.load_checkpoint",
                },
                "load_run_reference": "basename",
            },
        }
        execution["resource_limits"] = {
            "campaign_timeout_minutes": 120,
            "min_free_disk_gb": 0,
            "max_gpu_temperature_c": 100,
            "stop_grace_seconds": 1,
        }
        execution["quality_rules"][0]["metric"] = "mean_reward"
        self.session["multi_fidelity"] = {
            "enabled": True,
            "metric": "mean_reward",
            "minimum_margin": 1.0,
            "minimum_rungs_before_performance_pruning": 2,
            "required_consecutive_underperformance": 2,
            "resume_same_worker": True,
            "rungs": [
                {"budget": 10, "target_promoted_candidates": 2},
                {"budget": 20, "target_promoted_candidates": 1},
                {"budget": 30, "target_promoted_candidates": 0},
            ],
        }

    def tearDown(self) -> None:
        self.helper.tearDown()

    def _results(
        self,
        plan: dict[str, object],
        values: dict[str, tuple[float, float]],
    ) -> list[dict[str, object]]:
        current_rung = len(plan["multi_fidelity"]["rungs"])
        results: list[dict[str, object]] = []
        for run in plan["runs"]:
            if run.get("rung") != current_rung:
                continue
            rsl_dir = self.root / "checkpoints" / run["run_id"]
            rsl_dir.mkdir(parents=True)
            checkpoint = rsl_dir / f"model_{run['target_budget']}.pt"
            checkpoint.write_bytes(run["run_id"].encode())
            score, unsafe = values[run["trial_id"]]
            results.append(
                {
                    "run_id": run["run_id"],
                    "trial_id": run["trial_id"],
                    "seed": run["seed"],
                    "status": "completed",
                    "metrics": {
                        "mean_reward": score,
                        "illegal_contact": unsafe,
                    },
                    "checkpoint": {
                        "path": str(checkpoint),
                        "sha256": hashlib.sha256(
                            checkpoint.read_bytes()
                        ).hexdigest(),
                        "step": run["target_budget"],
                        "rsl_rl_run_dir": str(rsl_dir),
                    },
                    "rung": run["rung"],
                    "target_budget": run["target_budget"],
                }
            )
        return results

    def test_conservative_rungs_prune_then_complete(self) -> None:
        spec = validate_spec(self.session)
        plan = build_plan(spec)
        self.assertEqual(plan["version"], 6)
        self.assertEqual(len(plan["runs"]), 3)
        first = advance_multifidelity_plan(
            spec,
            plan,
            self._results(
                plan,
                {
                    "baseline": (10.0, 0.0),
                    "trial-001": (5.0, 0.0),
                    "trial-002": (12.0, 0.0),
                },
            ),
        )
        self.assertEqual(
            first["multi_fidelity"]["decisions"][0][
                "promoted_trial_ids"
            ],
            ["baseline", "trial-002", "trial-001"],
        )
        self.assertEqual(len(first["runs"]), 6)
        second = advance_multifidelity_plan(
            spec,
            first,
            self._results(
                first,
                {
                    "baseline": (10.0, 0.0),
                    "trial-001": (5.0, 0.0),
                    "trial-002": (13.0, 0.0),
                },
            ),
        )
        second_decision = second["multi_fidelity"]["decisions"][-1]
        self.assertEqual(
            second_decision["promoted_trial_ids"],
            ["baseline", "trial-002"],
        )
        self.assertFalse(second_decision["target_exceeded_for_safety"])
        self.assertEqual(len(second["runs"]), 8)
        self.assertTrue(
            all(
                run["resume_from"] is not None
                for run in second["runs"]
                if run.get("rung") in {2, 3}
            )
        )
        final = advance_multifidelity_plan(
            spec,
            second,
            self._results(
                second,
                {
                    "baseline": (10.0, 0.0),
                    "trial-002": (14.0, 0.0),
                },
            ),
        )
        self.assertEqual(final["multi_fidelity"]["status"], "completed")
        self.assertEqual(
            final["multi_fidelity"]["decisions"][-1][
                "selected_trial_ids"
            ],
            ["trial-002"],
        )
        validate_trial_plan(spec, final)
        ranking_input = final_rung_results(spec, final)
        ranking = rank(spec, ranking_input["runs"])
        self.assertEqual(
            ranking["training_evidence"],
            "single_seed_selected",
        )

    def test_constraint_failure_eliminates_before_performance_pruning(
        self,
    ) -> None:
        spec = validate_spec(self.session)
        plan = build_plan(spec)
        expanded = advance_multifidelity_plan(
            spec,
            plan,
            self._results(
                plan,
                {
                    "baseline": (10.0, 0.0),
                    "trial-001": (20.0, 1.0),
                    "trial-002": (11.0, 0.0),
                },
            ),
        )
        decision = expanded["multi_fidelity"]["decisions"][0]
        self.assertNotIn("trial-001", decision["promoted_trial_ids"])
        eliminated = next(
            record
            for record in decision["trial_records"]
            if record["trial_id"] == "trial-001"
        )
        self.assertEqual(
            eliminated["disposition"],
            "hard_constraint_eliminated",
        )

    def test_resume_argv_and_executor_plan_adoption_are_hash_bound(
        self,
    ) -> None:
        spec = validate_spec(self.session)
        plan = build_plan(spec)
        results = self._results(
            plan,
            {
                "baseline": (10.0, 0.0),
                "trial-001": (9.0, 0.0),
                "trial-002": (12.0, 0.0),
            },
        )
        expanded = advance_multifidelity_plan(spec, plan, results)
        resumed_run = next(
            run
            for run in expanded["runs"]
            if run["rung"] == 2 and run["trial_id"] == "trial-002"
        )
        contract = _adapter_contract(spec, resumed_run)
        argv = build_child_argv(contract, resumed_run["overrides"])
        self.assertIn("agent.max_iterations=20", argv)
        self.assertIn("agent.resume=true", argv)
        self.assertIn(
            "agent.load_checkpoint=\"model_10.pt\"",
            argv,
        )
        outputs = self.root / "adapter-rung-2"
        outputs.mkdir()
        contract_path = outputs / "contract.json"
        contract_path.write_text(json.dumps(contract), encoding="utf-8")
        adapter_log = outputs / "adapter.log"
        with adapter_log.open("wb") as stream:
            completed = subprocess.run(
                [
                    sys.executable,
                    str(RSL_ADAPTER),
                    "--contract",
                    str(contract_path),
                    "--executor-run-id",
                    resumed_run["run_id"],
                    "--overrides-json",
                    json.dumps(resumed_run["overrides"]),
                    "--effective-config",
                    str(outputs / "effective.json"),
                    "--result",
                    str(outputs / "result.json"),
                    "--summary",
                    str(outputs / "summary.json"),
                    "--terminal",
                    str(outputs / "terminal.json"),
                    "--log-path",
                    str(outputs / "training.log"),
                ],
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=False,
            )
        self.assertEqual(completed.returncode, 0, adapter_log.read_text())
        adapter_result = json.loads(
            (outputs / "result.json").read_text(encoding="utf-8")
        )
        self.assertEqual(adapter_result["rung"], 2)
        self.assertEqual(adapter_result["checkpoint"]["step"], 20)
        effective = json.loads(
            (outputs / "effective.json").read_text(encoding="utf-8")
        )
        self.assertEqual(effective["agent"]["max_iterations"], 20)
        self.assertTrue(effective["agent"]["resume"])

        session_path = self.root / "mf-session.json"
        plan_path = self.root / "mf-plan-1.json"
        expanded_path = self.root / "mf-plan-2.json"
        session_path.write_text(json.dumps(spec), encoding="utf-8")
        plan_path.write_text(json.dumps(plan), encoding="utf-8")
        expanded_path.write_text(json.dumps(expanded), encoding="utf-8")
        state_path, state = initialize_state(
            spec,
            session_path,
            plan,
            plan_path,
        )
        for record in state["runs"].values():
            record["status"] = "completed"
        state_path.write_text(json.dumps(state), encoding="utf-8")
        _, adopted = adopt_expanded_plan(
            spec,
            session_path,
            expanded,
            expanded_path,
        )
        self.assertEqual(adopted["stage"], "fidelity-rung-002")
        self.assertEqual(
            sum(
                record["status"] == "pending"
                for record in adopted["runs"].values()
            ),
            3,
        )

        parent = Path(resumed_run["resume_from"]["checkpoint_path"])
        parent.write_bytes(b"tampered")
        with self.assertRaisesRegex(SpecError, "hash-mismatched"):
            build_child_argv(contract, resumed_run["overrides"])

    def test_session_rejects_unsafe_multi_fidelity_combinations(self) -> None:
        adaptive = json.loads(json.dumps(self.session))
        adaptive["adaptive_search"] = {
            "enabled": True,
            "max_rounds": 2,
            "trials_per_round": 1,
            "exploration_fraction": 0.5,
            "stop_policy": {
                "enabled": True,
                "metric": "mean_reward",
                "minimum_improvement": 1.0,
                "patience_rounds": 1,
                "minimum_feasible_trials": 1,
            },
        }
        adaptive["history_prior"] = {}
        with self.assertRaises(SpecError):
            validate_spec(adaptive)

        early_prune = json.loads(json.dumps(self.session))
        early_prune["multi_fidelity"]["rungs"][0][
            "target_promoted_candidates"
        ] = 1
        with self.assertRaisesRegex(SpecError, "protect every candidate"):
            validate_spec(early_prune)

        no_resume = json.loads(json.dumps(self.session))
        no_resume["execution"]["adapter"].pop("multi_fidelity")
        with self.assertRaisesRegex(SpecError, "adapter-specific"):
            validate_spec(no_resume)


class MultiFidelityGitMailboxTests(unittest.TestCase):
    def setUp(self) -> None:
        helper = mailbox_tests.GitMailboxTests(methodName="runTest")
        helper.setUp()
        self.helper = helper
        self.root = helper.root
        session = helper.session
        session["tuning"]["seeds"] = [42]
        session["tuning"]["seed_strategy"] = {
            "mode": "fixed_single_seed",
            "screening_seeds": [42],
            "confirmation_seeds": [42],
            "confirmation_top_k": 1,
            "final_authority": "supervised_hardware",
        }
        session["tuning"]["ranking"]["minimum_final_training_seeds"] = 1
        session["hardware_feedback"] = {
            "enabled": True,
            "output_mode": "prepare_authorized_draft",
            "output_dir": str(self.root / "hardware-feedback"),
            "require_policy_manifest": True,
            "verify_artifact_hashes": True,
            "stop_on_safety_event": True,
            "require_new_session_approval": True,
            "qualification": {
                "enabled": True,
                "final_authority": "supervised_hardware",
                "minimum_total_tests": 4,
                "required_scenarios": [
                    "standing",
                    "start_stop",
                    "low_speed",
                    "turn",
                ],
                "minimum_tests_per_scenario": 1,
                "require_high_evidence_confidence": True,
                "required_telemetry_channels": [
                    "action",
                    "control_timestamp",
                    "imu_roll",
                ],
                "require_all_assessments_pass": True,
                "require_zero_safety_events": True,
                "status_label": "hardware_validated_for_test_envelope",
            },
        }
        session["distributed"]["assignment_mode"] = "by_trial"
        session["distributed"]["calibration"] = {
            "enabled": False,
            "seed": 42,
            "worker_ids": [],
        }
        for worker in session["distributed"]["workers"]:
            worker["assigned_seeds"] = [42]
        session["execution"]["adapter"] = {
            "id": "rsl-rl",
            "parameter_cli_map": {
                "agent.learning_rate": "agent.learning_rate",
            },
            "runtime_config_paths": {
                "agent.run_name": "run_id",
                "agent.seed": "seed",
            },
            "summary_last": 5,
            "require_checkpoint": True,
            "multi_fidelity": {
                "budget_cli_path": "agent.max_iterations",
                "resume_cli_paths": {
                    "enabled": "agent.resume",
                    "load_run": "agent.load_run",
                    "load_checkpoint": "agent.load_checkpoint",
                },
                "load_run_reference": "basename",
            },
        }
        session["execution"]["resource_limits"] = {
            "campaign_timeout_minutes": 120,
            "min_free_disk_gb": 0,
            "max_gpu_temperature_c": 100,
            "stop_grace_seconds": 1,
        }
        session["execution"]["run_command"] = [
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
        session["multi_fidelity"] = {
            "enabled": True,
            "metric": "score",
            "minimum_margin": 1.0,
            "minimum_rungs_before_performance_pruning": 2,
            "required_consecutive_underperformance": 2,
            "resume_same_worker": True,
            "rungs": [
                {"budget": 10, "target_promoted_candidates": 2},
                {"budget": 20, "target_promoted_candidates": 0},
            ],
        }
        self.spec = validate_spec(session)
        self.plan = build_plan(self.spec)
        self.session_path = self.root / "mf-mailbox-session.json"
        self.plan_path = self.root / "mf-mailbox-plan-1.json"
        self.session_path.write_text(json.dumps(self.spec), encoding="utf-8")
        self.plan_path.write_text(json.dumps(self.plan), encoding="utf-8")
        self.helper.session = self.spec
        self.helper.session_path = self.session_path

    def tearDown(self) -> None:
        self.helper.tearDown()

    def _publish_rung_results(
        self,
        plan: dict[str, object],
        scores: dict[str, float],
    ) -> list[dict[str, object]]:
        current_rung = len(plan["multi_fidelity"]["rungs"])
        runs = [
            run for run in plan["runs"] if run.get("rung") == current_rung
        ]
        jobs = _build_jobs(
            self.spec,
            plan,
            runs=runs,
            include_calibration=False,
        )
        payloads: list[dict[str, object]] = []
        for job in jobs:
            run = job["run"]
            worker_id = job["worker_id"]
            claim(
                self.helper.worker_clone,
                self.session_path,
                worker_id,
                run["run_id"],
                1,
            )
            rsl_dir = self.root / "mailbox-checkpoints" / run["run_id"]
            rsl_dir.mkdir(parents=True)
            checkpoint = rsl_dir / f"model_{run['target_budget']}.pt"
            checkpoint.write_bytes(run["run_id"].encode())
            checkpoint_hash = hashlib.sha256(
                checkpoint.read_bytes()
            ).hexdigest()
            payload = {
                "run_id": run["run_id"],
                "trial_id": run["trial_id"],
                "seed": run["seed"],
                "status": "completed",
                "metrics": {
                    "score": scores[run["trial_id"]],
                    "unsafe": 0.0,
                },
                "checkpoint": {
                    "path": str(checkpoint),
                    "sha256": checkpoint_hash,
                    "step": run["target_budget"],
                    "rsl_rl_run_dir": str(rsl_dir),
                },
                "rung": run["rung"],
                "target_budget": run["target_budget"],
            }
            result_path = self.root / f"{run['run_id']}-result.json"
            artifacts_path = self.root / f"{run['run_id']}-artifacts.json"
            result_path.write_text(json.dumps(payload), encoding="utf-8")
            artifacts_path.write_text(
                json.dumps(
                    {
                        "artifacts": [
                            {
                                "kind": "checkpoint",
                                "path": str(checkpoint),
                                "sha256": checkpoint_hash,
                                "size_bytes": checkpoint.stat().st_size,
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            publish_result(
                self.helper.worker_clone,
                self.session_path,
                worker_id,
                run["run_id"],
                1,
                result_path,
                artifacts_path,
            )
            payloads.append(payload)
        return payloads

    def test_rung_publication_preserves_worker_affinity_and_terminal_zero_jobs(
        self,
    ) -> None:
        publish(
            self.helper.coordinator,
            self.session_path,
            self.plan_path,
        )
        first_results = self._publish_rung_results(
            self.plan,
            {
                "baseline": 10.0,
                "trial-001": 9.0,
                "trial-002": 12.0,
            },
        )
        expanded = advance_multifidelity_plan(
            self.spec,
            self.plan,
            first_results,
        )
        expanded_path = self.root / "mf-mailbox-plan-2.json"
        expanded_path.write_text(json.dumps(expanded), encoding="utf-8")
        published = publish_multifidelity_rung(
            self.helper.coordinator,
            self.session_path,
            self.plan_path,
            expanded_path,
        )
        self.assertEqual(
            published["state"],
            "multi_fidelity_rung_published",
        )
        self.assertEqual(published["published_jobs"], 3)
        old_workers = {
            job["run"]["trial_id"]: job["worker_id"]
            for job in _build_jobs(
                self.spec,
                self.plan,
                include_calibration=False,
            )
        }
        new_workers = {
            job["run"]["trial_id"]: job["worker_id"]
            for job in _build_jobs(
                self.spec,
                expanded,
                runs=expanded["runs"][len(self.plan["runs"]):],
                include_calibration=False,
            )
        }
        self.assertEqual(old_workers, new_workers)

        final_results = self._publish_rung_results(
            expanded,
            {
                "baseline": 10.0,
                "trial-001": 9.0,
                "trial-002": 13.0,
            },
        )
        final = advance_multifidelity_plan(
            self.spec,
            expanded,
            final_results,
        )
        final_path = self.root / "mf-mailbox-plan-final.json"
        final_path.write_text(json.dumps(final), encoding="utf-8")
        completed = publish_multifidelity_rung(
            self.helper.coordinator,
            self.session_path,
            expanded_path,
            final_path,
        )
        self.assertEqual(completed["state"], "multi_fidelity_completed")
        self.assertEqual(completed["published_jobs"], 0)


if __name__ == "__main__":
    unittest.main()
