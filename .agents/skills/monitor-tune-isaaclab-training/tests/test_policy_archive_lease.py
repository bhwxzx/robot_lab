#!/usr/bin/env python3
"""End-to-end contract for a lease-bound version-7 policy archive."""

from __future__ import annotations

import json
import os
import sys
import unittest
from pathlib import Path
from unittest import mock


SKILL = Path(__file__).resolve().parents[1]
SCRIPTS = SKILL / "scripts"
TESTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(TESTS))

import execute_evaluation_plan as evaluation_executor  # noqa: E402
import test_evaluation_executor as evaluation_tests  # noqa: E402
import test_fixed_single_seed as fixed_tests  # noqa: E402
import test_git_mailbox as mailbox_tests  # noqa: E402
from archive_policy_candidate import (  # noqa: E402
    _canonical_sha256,
    archive_candidate,
    build_distributed_archive_request,
)
from build_evaluation_plan import _load_candidates, build_plan  # noqa: E402
from collect_evaluation_results import collect  # noqa: E402
from validate_session_spec import SpecError, validate_spec  # noqa: E402


class PolicyArchiveLeaseTests(unittest.TestCase):
    def setUp(self) -> None:
        fixed = fixed_tests.FixedSingleSeedTests(methodName="runTest")
        fixed.setUp()
        self.fixed = fixed
        self.root = fixed.root
        self.session = fixed.session
        self.checkpoint = self.root / "candidate-native.pt"
        self.onnx = self.root / "candidate.onnx"
        self.checkpoint.write_bytes(b"jit-and-native-policy")
        self.onnx.write_bytes(b"onnx-policy")
        (
            self.policy_origin,
            self.policy_a,
            self.policy_b,
            self.policy_base_commit,
        ) = mailbox_tests._init_policy_storage(self.root)

        onnx_artifact = next(
            artifact
            for artifact in self.session["evaluation"]["artifacts"]
            if artifact["kind"] == "onnx"
        )
        self.session["evaluation"]["artifacts"].append(
            {**onnx_artifact, "kind": "jit"}
        )
        self.session["version"] = 7
        self.session["training"]["source_git_commit"] = "a" * 40
        self.session["training"]["source_git_dirty"] = False
        self.session["distributed"] = {
            "enabled": True,
            "transport": "git_mailbox",
            "campaign_id": "policy-archive-lease-test",
            "remote_url": "https://example.invalid/private/mailbox.git",
            "coordinator_id": "pc-a",
            "coordinator_branch": "tune/archive-test/coordinator",
            "poll_interval_seconds": 600,
            "remote_state_unknown_after_seconds": 1800,
            "artifact_policy": "metadata_only",
            "assignment_mode": "by_trial",
            "workers": [
                {
                    "id": "pc-a",
                    "branch": "tune/archive-test/worker-pc-a",
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
                    "branch": "tune/archive-test/worker-pc-b",
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
        self.session["archive"] = {
            "enabled": True,
            "copy_after_qualification": True,
            "storage_root": None,
            "collection": "LW/leg_loco",
            "directory_naming": "local_timestamp_seconds",
            "timezone": "Asia/Shanghai",
            "required_artifacts": ["jit", "onnx"],
            "require_clean_git_worktree": True,
            "write_manifest": True,
            "description_notes": "lease-bound archive test",
            "git_action": "none",
            "distributed_lease": {
                "enabled": True,
                "storage_remote_url": (
                    "https://example.invalid/private/policy-storage.git"
                ),
                "storage_branch": "master",
                "authorized_worker_ids": ["pc-a", "pc-b"],
                "worker_storage_roots": {
                    "pc-a": str(self.policy_a),
                    "pc-b": str(self.policy_b),
                },
                "takeover_policy": "explicit_revoke_only",
            },
        }
        self.spec = validate_spec(self.session)
        self.remote_override = mock.patch.dict(
            os.environ,
            {"POLICY_ARCHIVE_ALLOW_TEST_REMOTE": "1"},
        )
        self.remote_override.start()

    def tearDown(self) -> None:
        self.remote_override.stop()
        for process in list(evaluation_executor._ACTIVE_CHILDREN.values()):
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=3)
        evaluation_executor._ACTIVE_CHILDREN.clear()
        self.fixed.tearDown()

    def _evaluation(self) -> tuple[dict[str, object], dict[str, object]]:
        manifest_path = self.root / "candidates.json"
        manifest_path.write_text(
            json.dumps(
                {
                    "candidates": [
                        {
                            "candidate_id": "trial-002",
                            "checkpoint_path": str(self.checkpoint),
                            "checkpoint_sha256": mailbox_tests._file_sha256(
                                self.checkpoint
                            ),
                            "artifacts": {
                                "jit": str(self.checkpoint),
                                "onnx": str(self.onnx),
                            },
                            "artifact_sha256": {
                                "jit": mailbox_tests._file_sha256(
                                    self.checkpoint
                                ),
                                "onnx": mailbox_tests._file_sha256(self.onnx),
                            },
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        candidates = _load_candidates(
            manifest_path,
            {"native", "jit", "onnx"},
        )
        plan = build_plan(self.spec, candidates)
        helper = evaluation_tests.EvaluationExecutorTests(methodName="runTest")
        helper.temp = self.fixed.temp
        helper.root = self.root
        helper.output = Path(self.spec["evaluation"]["output_dir"])
        helper.checkpoint = self.checkpoint
        helper.onnx = self.onnx
        state, _, _ = helper._state(self.spec, plan)
        for _ in plan["runs"]:
            helper._launch_and_finish(self.spec, plan, state)
        reviews = [
            {
                "candidate_id": "trial-002",
                "status": "pass",
                "reviewer": "lease-test",
                "reviewed_video_paths": plan["required_videos"],
                "notes": "synthetic closed-loop motion review passed",
            }
        ]
        return plan, collect(plan, reviews)

    def test_version7_archive_requires_and_binds_active_grant(self) -> None:
        plan, results = self._evaluation()
        training_runs = [
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
        request = build_distributed_archive_request(
            self.spec,
            training_runs,
            plan,
            results,
            "pc-a",
        )
        grant = {
            "schema_version": 1,
            "event": "policy_archive_granted",
            "campaign_id": self.spec["distributed"]["campaign_id"],
            "session_sha256": _canonical_sha256(self.spec),
            "lease_id": request["request_id"],
            "worker_id": "pc-a",
            "coordinator_id": "pc-a",
            "request_sha256": _canonical_sha256(request),
            "request": request,
            "granted_at": "2026-07-27T10:00:00Z",
            "takeover_policy": "explicit_revoke_only",
        }
        with self.assertRaisesRegex(SpecError, "requires worker ID and lease"):
            archive_candidate(
                self.spec,
                training_runs,
                plan,
                results,
            )
        receipt = archive_candidate(
            self.spec,
            training_runs,
            plan,
            results,
            worker_id="pc-a",
            lease_grant=grant,
        )
        self.assertEqual(
            receipt["distributed_archive_lease"]["lease_id"],
            request["request_id"],
        )
        self.assertEqual(
            receipt["storage_base_commit"],
            self.policy_base_commit,
        )
        self.assertTrue(Path(receipt["files"]["manifest"]).is_file())


if __name__ == "__main__":
    unittest.main()
