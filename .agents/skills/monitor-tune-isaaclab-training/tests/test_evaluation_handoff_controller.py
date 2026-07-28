#!/usr/bin/env python3
"""Training-ranking to policy-evaluation handoff controller tests."""

from __future__ import annotations

import copy
import hashlib
import json
import sys
import time
import unittest
from pathlib import Path
from unittest import mock


SKILL = Path(__file__).resolve().parents[1]
SCRIPTS = SKILL / "scripts"
TESTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(TESTS))

import execute_evaluation_plan as evaluation_executor  # noqa: E402
from evaluation_handoff_controller import inspect_or_advance  # noqa: E402
from rank_trials import rank  # noqa: E402
import test_fixed_single_seed as fixed_tests  # noqa: E402
from validate_session_spec import SpecError, validate_spec  # noqa: E402


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class EvaluationHandoffControllerTests(unittest.TestCase):
    def setUp(self) -> None:
        helper = fixed_tests.FixedSingleSeedTests(methodName="runTest")
        helper.setUp()
        self.helper = helper
        self.root = helper.root
        self.checkpoint_dir = self.root / "selected-policy"
        self.checkpoint_dir.mkdir()
        self.checkpoint = self.checkpoint_dir / "model_100.pt"
        self.onnx = self.checkpoint_dir / "policy.onnx"
        self.checkpoint.write_bytes(b"native")
        self.onnx.write_bytes(b"onnx")

    def tearDown(self) -> None:
        for process in list(evaluation_executor._ACTIVE_CHILDREN.values()):
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=3)
                except Exception:
                    process.kill()
                    process.wait(timeout=3)
        evaluation_executor._ACTIVE_CHILDREN.clear()
        self.helper.tearDown()

    def _session(self, handoff_mode: str) -> dict[str, object]:
        session = copy.deepcopy(self.helper.session)
        session["campaign_controller"] = {
            "enabled": True,
            "mode": "execute",
            "role": "single_host",
            "auto_launch_trials": True,
            "auto_advance_plans": True,
            "stop_before_evaluation": True,
            "worker_mailbox_repos": {},
        }
        session["evaluation_handoff"] = {
            "enabled": True,
            "mode": handoff_mode,
            "top_k": 1,
            "require_pareto": True,
            "checkpoint_seed": 42,
            "evaluation_worker_id": None,
            "artifact_path_templates": {
                "onnx": "{checkpoint_dir}/policy.onnx",
            },
            "auto_build_plan": True,
            "auto_execute_evaluation": True,
            "stop_before_visual_review": True,
        }
        return session

    def _sources(
        self,
        handoff_mode: str,
        *,
        worker_id: str | None = None,
        session: dict[str, object] | None = None,
    ) -> tuple[Path, Path, Path]:
        spec = validate_spec(session or self._session(handoff_mode))
        session_path = self.root / f"handoff-{handoff_mode}-session.json"
        session_path.write_text(
            json.dumps(spec, sort_keys=True),
            encoding="utf-8",
        )
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
        ranking = rank(spec, runs)
        ranking_path = self.root / f"handoff-{handoff_mode}-ranking.json"
        ranking_path.write_text(
            json.dumps(ranking, sort_keys=True),
            encoding="utf-8",
        )
        inventory = {
            "version": 1,
            "session_sha256": _sha256(session_path),
            "training_ranking_sha256": _sha256(ranking_path),
            "training_run_id": spec["training"]["run_id"],
            "algorithm": spec["algorithm"],
            "entries": [
                {
                    "trial_id": "trial-002",
                    "seed": 42,
                    "run_id": "trial-002__screening__seed-42",
                    "worker_id": worker_id,
                    "checkpoint_path": str(self.checkpoint),
                    "checkpoint_sha256": _sha256(self.checkpoint),
                    "checkpoint_step": 100,
                    "rsl_rl_run_dir": str(self.checkpoint_dir),
                    "rung": None,
                    "target_budget": None,
                }
            ],
        }
        inventory_path = self.root / f"handoff-{handoff_mode}-inventory.json"
        inventory_path.write_text(
            json.dumps(inventory, sort_keys=True),
            encoding="utf-8",
        )
        return session_path, ranking_path, inventory_path

    def test_shadow_is_read_only_and_execute_requires_permission(self) -> None:
        session_path, ranking_path, inventory_path = self._sources("shadow")
        output = self.root / "evaluation" / ".handoff"
        report = inspect_or_advance(
            session_path,
            ranking_path,
            inventory_path,
            execute=False,
        )
        self.assertEqual(report["next_action"], "initialize_handoff")
        self.assertFalse(output.exists())
        with self.assertRaisesRegex(SpecError, "mode=execute"):
            inspect_or_advance(
                session_path,
                ranking_path,
                inventory_path,
                execute=True,
            )
        self.assertFalse(output.exists())

    def test_executes_complete_matrix_then_stops_for_visual_review(self) -> None:
        session_path, ranking_path, inventory_path = self._sources("execute")
        actions: list[str] = []
        with (
            mock.patch.object(
                evaluation_executor,
                "_resource_preflight",
                return_value={"synthetic": True},
            ),
            mock.patch.object(
                evaluation_executor,
                "_gpu_idle",
                return_value=True,
            ),
        ):
            report: dict[str, object] = {}
            for _ in range(120):
                report = inspect_or_advance(
                    session_path,
                    ranking_path,
                    inventory_path,
                    execute=True,
                )
                actions.append(str(report.get("action_taken")))
                if report.get("action_taken") == "launch_next":
                    for process in list(
                        evaluation_executor._ACTIVE_CHILDREN.values()
                    ):
                        process.wait(timeout=5)
                if report.get("next_action") == "awaiting_visual_review":
                    break
                time.sleep(0.02)
            for process in list(
                evaluation_executor._ACTIVE_CHILDREN.values()
            ):
                process.wait(timeout=5)
            evaluation_executor._ACTIVE_CHILDREN.clear()
        self.assertEqual(report["next_action"], "awaiting_visual_review")
        self.assertIn("prepare_evaluation", actions)
        self.assertIn("initialize_evaluation", actions)
        self.assertIn("launch_next", actions)
        state_path = Path(report["state_path"])
        state_before = state_path.read_bytes()
        repeated = inspect_or_advance(
            session_path,
            ranking_path,
            inventory_path,
            execute=True,
        )
        self.assertEqual(repeated["action_taken"], "none")
        self.assertEqual(state_path.read_bytes(), state_before)
        self.assertFalse((self.root / "policy-storage").exists())

    def test_missing_or_changed_artifact_blocks_plan_creation(self) -> None:
        session = self._session("execute")
        session["evaluation_handoff"]["artifact_path_templates"]["onnx"] = (
            "{checkpoint_dir}/missing-policy.onnx"
        )
        session_path, ranking_path, inventory_path = self._sources(
            "execute",
            session=session,
        )
        inspect_or_advance(
            session_path,
            ranking_path,
            inventory_path,
            execute=True,
        )
        with self.assertRaisesRegex(SpecError, "onnx artifact is unavailable"):
            inspect_or_advance(
                session_path,
                ranking_path,
                inventory_path,
                execute=True,
            )
        self.assertFalse(
            (
                self.root
                / "evaluation"
                / ".handoff"
                / "candidate_manifest.json"
            ).exists()
        )

    def test_changed_checkpoint_blocks_candidate_selection(self) -> None:
        session_path, ranking_path, inventory_path = self._sources("execute")
        self.checkpoint.write_bytes(b"changed-after-ranking")
        inspect_or_advance(
            session_path,
            ranking_path,
            inventory_path,
            execute=True,
        )
        with self.assertRaisesRegex(SpecError, "checkpoint hash changed"):
            inspect_or_advance(
                session_path,
                ranking_path,
                inventory_path,
                execute=True,
            )

    def test_version_seven_rejects_non_designated_worker(self) -> None:
        session = self._session("execute")
        session["version"] = 7
        session["training"]["source_git_commit"] = "a" * 40
        session["training"]["source_git_dirty"] = False
        session["distributed"] = {
            "enabled": True,
            "transport": "git_mailbox",
            "campaign_id": "evaluation-worker-test",
            "remote_url": "https://example.invalid/private/mailbox.git",
            "coordinator_id": "pc-a",
            "coordinator_branch": "tune/eval/coordinator",
            "poll_interval_seconds": 600,
            "remote_state_unknown_after_seconds": 1800,
            "artifact_policy": "metadata_only",
            "assignment_mode": "by_trial",
            "workers": [
                {
                    "id": worker_id,
                    "branch": f"tune/eval/worker-{worker_id}",
                    "assigned_seeds": [42],
                    "source_repo": str(self.root),
                    "state_dir": str(self.root / f"{worker_id}-state"),
                    "effective_config_baseline_path": str(
                        self.root / f"{worker_id}-effective.json"
                    ),
                    "gpu_index": 0,
                    "max_active_jobs": 1,
                }
                for worker_id in ("pc-a", "pc-b")
            ],
            "calibration": {
                "enabled": False,
                "seed": 42,
                "worker_ids": [],
            },
        }
        session["campaign_controller"]["role"] = "distributed"
        session["campaign_controller"]["worker_mailbox_repos"] = {
            "pc-a": str(self.root / "mailbox-a"),
            "pc-b": str(self.root / "mailbox-b"),
        }
        session["evaluation_handoff"]["evaluation_worker_id"] = "pc-a"
        session_path, ranking_path, inventory_path = self._sources(
            "execute",
            worker_id="pc-a",
            session=session,
        )
        with self.assertRaisesRegex(SpecError, "approved evaluation worker"):
            inspect_or_advance(
                session_path,
                ranking_path,
                inventory_path,
                execute=False,
                worker_id="pc-b",
            )


if __name__ == "__main__":
    unittest.main()
