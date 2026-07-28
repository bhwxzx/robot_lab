#!/usr/bin/env python3
"""Git-mailbox exchange for bounded history and adaptive rounds."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


SKILL = Path(__file__).resolve().parents[1]
SCRIPTS = SKILL / "scripts"
TESTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(TESTS))

from build_trial_plan import build_plan, extend_adaptive_plan  # noqa: E402
from build_adaptive_round import _results  # noqa: E402
from git_mailbox import (  # noqa: E402
    _build_jobs,
    _sha256,
    collect,
    history_collect,
    history_initialize,
    history_publish,
    publish,
    publish_adaptive_round,
)
import test_git_mailbox as mailbox_tests  # noqa: E402
from validate_session_spec import validate_spec  # noqa: E402


class AdaptiveGitMailboxTests(unittest.TestCase):
    def setUp(self) -> None:
        helper = mailbox_tests.GitMailboxTests(methodName="runTest")
        helper.setUp()
        self.helper = helper
        self.root = helper.root
        self.session = helper.session
        self.session["tuning"]["allowed_parameters"] = [
            {
                "path": "agent.learning_rate",
                "values": [0.1, 0.2, 0.3, 0.4],
                "baseline": 0.1,
            },
            {
                "path": "env.penalty",
                "values": [-0.1, -0.2],
                "baseline": -0.1,
            },
        ]
        self.session["tuning"]["max_trials"] = 5
        self.session["tuning"]["seeds"] = [42]
        self.session["tuning"]["seed_strategy"] = {
            "mode": "fixed_single_seed",
            "screening_seeds": [42],
            "confirmation_seeds": [42],
            "confirmation_top_k": 1,
            "final_authority": "supervised_hardware",
        }
        self.session["tuning"]["ranking"]["minimum_final_training_seeds"] = 1
        self.session["hardware_feedback"] = {
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
        self.session["distributed"]["assignment_mode"] = "by_trial"
        self.session["distributed"]["calibration"] = {
            "enabled": False,
            "seed": 42,
            "worker_ids": [],
        }
        for worker in self.session["distributed"]["workers"]:
            worker["assigned_seeds"] = [42]
        roots = {
            "pc-a": self.root / "pc-a-wandb",
            "pc-b": self.root / "pc-b-wandb",
        }
        for root in roots.values():
            root.mkdir()
        self.expected_context = {
            "task_id": "adaptive-mailbox-task",
            "profile_fingerprint": self.session["algorithm"][
                "profile_fingerprint"
            ],
            "observation_contract_sha256": "b" * 64,
            "reward_config_sha256": "c" * 64,
        }
        self.session["history_prior"] = {
            "enabled": True,
            "source": "local_wandb",
            "wandb_project": "mailbox-history",
            "lookback_days": 30,
            "max_selected_runs": 2,
            "max_points_per_run": 3,
            "include_failed_runs": False,
            "max_first_round_fraction": 0.5,
            "explicit_run_ids": [],
            "config_path_map": {
                "agent.learning_rate": "alg_cfg.learning_rate",
                "env.penalty": "env_cfg.penalty",
            },
            "metric_key_map": {
                "score": "Train/score",
                "unsafe": "Safety/unsafe",
            },
            "worker_roots": {
                worker_id: str(root) for worker_id, root in roots.items()
            },
            "compatibility": {
                "source_policy": "compatible",
                "expected_context": self.expected_context,
                "context_path_map": {
                    key: f"context.{key}"
                    for key in self.expected_context
                },
            },
            "quality_gates": {
                "progress_key": "_step",
                "minimum_final_progress": 3,
                "minimum_points_per_metric": 3,
                "stability": {
                    "metric": "score",
                    "max_standard_deviation": 1.0,
                    "max_abs_slope": 1.0,
                },
            },
        }
        self.session["adaptive_search"] = {
            "enabled": True,
            "max_rounds": 2,
            "trials_per_round": 2,
            "exploration_fraction": 0.5,
            "stop_policy": {
                "enabled": True,
                "metric": "score",
                "minimum_improvement": 0.5,
                "patience_rounds": 1,
                "minimum_feasible_trials": 1,
            },
        }
        self.session["distributed"]["calibration"] = {
            "enabled": True,
            "seed": 42,
            "worker_ids": ["pc-a", "pc-b"],
        }
        self.spec = validate_spec(self.session)
        self.session_path = self.root / "adaptive-mailbox-session.json"
        self.session_path.write_text(json.dumps(self.spec), encoding="utf-8")
        self.helper.session = self.spec
        self.helper.session_path = self.session_path

    def tearDown(self) -> None:
        self.helper.tearDown()

    def _index(
        self,
        worker_id: str,
        run_id: str,
        overrides: dict[str, float],
        score: float,
    ) -> dict[str, object]:
        run = {
            "run_id": run_id,
            "display_name": run_id,
            "project": "mailbox-history",
            "observed_at": "2026-07-27T12:00:00Z",
            "status": "completed",
            "source_git_commit": self.spec["training"]["source_git_commit"],
            "source_git_match": True,
            "source_policy": "compatible",
            "observed_context": self.expected_context,
            "context_match": True,
            "guidance_eligible": True,
            "overrides": overrides,
            "overrides_sha256": _sha256(overrides),
            "metrics": {"score": score, "unsafe": 0.0},
            "retained_points": {"score": 3, "unsafe": 3},
            "quality": {
                "passed": True,
                "final_progress": 3,
                "metric_statistics": {
                    "score": {
                        "count": 3,
                        "mean": score,
                        "standard_deviation": 0.1,
                        "slope": 0.0,
                    },
                    "unsafe": {
                        "count": 3,
                        "mean": 0.0,
                        "standard_deviation": 0.0,
                        "slope": 0.0,
                    },
                },
            },
            "evidence": {
                "wandb_path": f"/history/{worker_id}/{run_id}.wandb",
                "size_bytes": 100,
                "mtime_ns": 1,
                "record_count": 10,
            },
        }
        base = {
            "schema_version": 2,
            "event": "local_wandb_history_indexed",
            "worker_id": worker_id,
            "session_sha256": _sha256(self.spec),
            "wandb_project": "mailbox-history",
            "root": self.spec["history_prior"]["worker_roots"][worker_id],
            "lookback_days": 30,
            "max_selected_runs": 2,
            "max_points_per_run": 3,
            "candidate_read_limit": 4,
            "selected_runs": [run],
            "excluded_runs": [],
        }
        return {**base, "index_sha256": _sha256(base)}

    def test_history_metadata_and_next_round_are_distributed(self) -> None:
        history_initialize(
            self.helper.coordinator,
            self.session_path,
        )
        index_a = self._index(
            "pc-a",
            "history-a",
            {"agent.learning_rate": 0.2, "env.penalty": -0.1},
            10.0,
        )
        index_b = self._index(
            "pc-b",
            "history-b",
            {"agent.learning_rate": 0.3, "env.penalty": -0.2},
            20.0,
        )
        for worker_id, index in (("pc-a", index_a), ("pc-b", index_b)):
            path = self.root / f"{worker_id}-history-index.json"
            path.write_text(json.dumps(index), encoding="utf-8")
            history_publish(
                self.helper.worker_clone,
                self.session_path,
                worker_id,
                path,
            )
        prior_path = self.root / "merged-history-prior.json"
        collected = history_collect(
            self.helper.coordinator,
            self.session_path,
            prior_path,
        )
        self.assertEqual(collected["selected_run_count"], 2)
        prior = json.loads(prior_path.read_text(encoding="utf-8"))
        plan = build_plan(self.spec, prior)
        plan_path = self.root / "adaptive-plan-1.json"
        plan_path.write_text(json.dumps(plan), encoding="utf-8")
        publish(
            self.helper.coordinator,
            self.session_path,
            plan_path,
        )
        jobs = _build_jobs(self.spec, plan)
        for job in jobs:
            self.helper._publish_training_result(
                job["worker_id"],
                job["run"],
                10.0 + len(job["run"]["trial_id"]),
            )
        result_path = self.root / "adaptive-round-1-results.json"
        collect(
            self.helper.coordinator,
            self.session_path,
            result_path,
        )
        report = json.loads(result_path.read_text(encoding="utf-8"))
        results = sorted(
            _results(report, plan),
            key=lambda item: item["trial_id"],
        )
        self.assertEqual(
            len(report["accepted_results"]),
            len(plan["runs"]) + 2,
        )
        expanded = extend_adaptive_plan(self.spec, plan, results)
        expanded_path = self.root / "adaptive-plan-2.json"
        expanded_path.write_text(json.dumps(expanded), encoding="utf-8")
        published = publish_adaptive_round(
            self.helper.coordinator,
            self.session_path,
            plan_path,
            expanded_path,
        )
        self.assertEqual(published["state"], "adaptive_round_published")
        self.assertEqual(published["published_jobs"], 2)
        new_runs = expanded["runs"][len(plan["runs"]):]
        for job in _build_jobs(
            self.spec,
            expanded,
            runs=new_runs,
            include_calibration=False,
        ):
            self.helper._publish_training_result(
                job["worker_id"],
                job["run"],
                30.0,
            )
        final_result_path = self.root / "adaptive-round-2-results.json"
        collect(
            self.helper.coordinator,
            self.session_path,
            final_result_path,
        )
        final_report = json.loads(
            final_result_path.read_text(encoding="utf-8")
        )
        stopped = extend_adaptive_plan(
            self.spec,
            expanded,
            sorted(
                _results(final_report, expanded),
                key=lambda item: item["trial_id"],
            ),
        )
        stopped_path = self.root / "adaptive-plan-stopped.json"
        stopped_path.write_text(json.dumps(stopped), encoding="utf-8")
        stopped_publication = publish_adaptive_round(
            self.helper.coordinator,
            self.session_path,
            expanded_path,
            stopped_path,
        )
        self.assertEqual(
            stopped_publication["state"],
            "adaptive_search_stopped",
        )
        self.assertEqual(stopped_publication["published_jobs"], 0)
        self.assertEqual(
            stopped_publication["reason"],
            "max_rounds_reached",
        )


if __name__ == "__main__":
    unittest.main()
