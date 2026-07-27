#!/usr/bin/env python3
"""Contracts for fixed-single-seed selection and by-trial distribution."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


SKILL = Path(__file__).resolve().parents[1]
SCRIPTS = SKILL / "scripts"
TESTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(TESTS))

from build_trial_plan import build_plan  # noqa: E402
from git_mailbox import _build_jobs  # noqa: E402
from rank_trials import rank, select_confirmation_candidates  # noqa: E402
import test_evaluation_executor as evaluation_tests  # noqa: E402
import test_execution_round_one as execution_tests  # noqa: E402
from validate_session_spec import SpecError, validate_spec  # noqa: E402


class FixedSingleSeedTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        execution_helper = execution_tests.ExecutionRoundOneTests(methodName="runTest")
        execution_helper.temp = self.temp
        execution_helper.root = self.root
        execution_helper.baseline = self.root / "baseline.json"
        execution_helper.baseline.write_text(
            json.dumps(
                {
                    "agent": {"learning_rate": 0.1},
                    "env": {"penalty": -0.1},
                }
            ),
            encoding="utf-8",
        )
        session = execution_helper._session()

        evaluation_helper = evaluation_tests.EvaluationExecutorTests(methodName="runTest")
        evaluation_helper.temp = self.temp
        evaluation_helper.root = self.root
        evaluation_helper.output = self.root / "evaluation"
        evaluation = evaluation_helper._session()["evaluation"]
        evaluation["allow_retune_on_failure"] = True
        session["evaluation"] = evaluation
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
        session["tuning"]["seeds"] = [42]
        session["tuning"]["seed_strategy"] = {
            "mode": "fixed_single_seed",
            "screening_seeds": [42],
            "confirmation_seeds": [42],
            "confirmation_top_k": 1,
            "final_authority": "supervised_hardware",
        }
        session["tuning"]["ranking"]["minimum_final_training_seeds"] = 1
        self.session = session

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_fixed_seed_plan_has_selection_without_extra_seed_runs(self) -> None:
        spec = validate_spec(self.session)
        plan = build_plan(spec)
        self.assertEqual(plan["stages"]["confirmation"]["remaining_seeds"], [])
        self.assertEqual(
            plan["stages"]["confirmation"]["status"],
            "selection_only_after_screening",
        )
        self.assertEqual(plan["planned_run_count"], len(plan["runs"]))

        runs = [
            {
                "trial_id": trial_id,
                "seed": 42,
                "status": "completed",
                "metrics": {"score": score, "unsafe": 0.0},
            }
            for trial_id, score in (
                ("baseline", 10.0),
                ("trial-001", 11.0),
                ("trial-002", 12.0),
            )
        ]
        self.assertEqual(
            select_confirmation_candidates(spec, runs),
            ["trial-002"],
        )
        report = rank(spec, runs)
        self.assertEqual(report["training_evidence"], "single_seed_selected")
        self.assertEqual(report["final_authority"], "supervised_hardware")
        self.assertFalse(report["generalization_claim"])
        self.assertEqual(
            report["policy_acceptance_status"],
            "awaiting_supervised_hardware",
        )
        self.assertEqual(report["selection_status"], "awaiting_policy_evaluation")

    def test_fixed_seed_rejects_hidden_multi_seed_or_missing_hardware_gate(self) -> None:
        multi = json.loads(json.dumps(self.session))
        multi["tuning"]["seeds"] = [42, 43]
        with self.assertRaisesRegex(
            SpecError,
            "exactly equal confirmation_seeds|exactly one identical",
        ):
            validate_spec(multi)

        no_hardware = json.loads(json.dumps(self.session))
        no_hardware["hardware_feedback"] = None
        with self.assertRaisesRegex(SpecError, "hardware qualification"):
            validate_spec(no_hardware)

    def test_two_workers_receive_trials_not_distinct_seeds(self) -> None:
        session = json.loads(json.dumps(self.session))
        session["version"] = 7
        session["training"]["source_git_commit"] = "a" * 40
        session["training"]["source_git_dirty"] = False
        session["distributed"] = {
            "enabled": True,
            "transport": "git_mailbox",
            "campaign_id": "fixed-seed-test",
            "remote_url": "https://example.invalid/private/mailbox.git",
            "coordinator_id": "pc-a",
            "coordinator_branch": "tune/fixed-seed/coordinator",
            "poll_interval_seconds": 600,
            "remote_state_unknown_after_seconds": 1800,
            "artifact_policy": "metadata_only",
            "assignment_mode": "by_trial",
            "workers": [
                {
                    "id": "pc-a",
                    "branch": "tune/fixed-seed/worker-pc-a",
                    "assigned_seeds": [42],
                    "source_repo": str(self.root),
                    "state_dir": str(self.root / "pc-a-state"),
                    "effective_config_baseline_path": str(
                        self.root / "pc-a-effective.json"
                    ),
                    "gpu_index": 0,
                    "max_active_jobs": 1,
                },
                {
                    "id": "pc-b",
                    "branch": "tune/fixed-seed/worker-pc-b",
                    "assigned_seeds": [42],
                    "source_repo": str(self.root),
                    "state_dir": str(self.root / "pc-b-state"),
                    "effective_config_baseline_path": str(
                        self.root / "pc-b-effective.json"
                    ),
                    "gpu_index": 0,
                    "max_active_jobs": 1,
                },
            ],
            "calibration": {
                "enabled": False,
                "seed": 42,
                "worker_ids": [],
            },
        }
        spec = validate_spec(session)
        plan = build_plan(spec)
        jobs = _build_jobs(spec, plan)
        assignments = {
            job["run"]["trial_id"]: job["worker_id"] for job in jobs
        }
        self.assertEqual(
            assignments,
            {
                "baseline": "pc-a",
                "trial-001": "pc-a",
                "trial-002": "pc-b",
            },
        )
        self.assertEqual(len(jobs), len(plan["runs"]))
        exact_runs = {
            (
                job["run"]["seed"],
                json.dumps(job["run"]["overrides"], sort_keys=True),
            )
            for job in jobs
        }
        self.assertEqual(len(exact_runs), len(jobs))

        wrong_mode = json.loads(json.dumps(session))
        wrong_mode["distributed"]["assignment_mode"] = "by_seed"
        with self.assertRaisesRegex(SpecError, "assignment_mode=by_trial"):
            validate_spec(wrong_mode)

        calibrated = json.loads(json.dumps(session))
        calibrated["distributed"]["calibration"] = {
            "enabled": True,
            "seed": 42,
            "worker_ids": ["pc-a", "pc-b"],
        }
        calibrated_spec = validate_spec(calibrated)
        calibrated_jobs = _build_jobs(calibrated_spec, plan)
        self.assertEqual(len(calibrated_jobs), len(plan["runs"]) + 2)
        self.assertEqual(
            {job["worker_id"] for job in calibrated_jobs if job["kind"] == "calibration"},
            {"pc-a", "pc-b"},
        )


if __name__ == "__main__":
    unittest.main()
