#!/usr/bin/env python3
"""Idempotent campaign-controller and shadow-mode tests."""

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

from build_trial_plan import build_plan  # noqa: E402
from campaign_controller import inspect_or_advance  # noqa: E402
import execute_trial_plan as executor_module  # noqa: E402
from git_mailbox import _build_jobs, claim, prepare_job, publish  # noqa: E402
import test_git_mailbox as mailbox_tests  # noqa: E402
import test_multifidelity_training as multifidelity_tests  # noqa: E402
from validate_session_spec import SpecError, validate_spec  # noqa: E402


def _controller_contract(
    *,
    mode: str,
    role: str,
    worker_mailbox_repos: dict[str, str],
) -> dict[str, object]:
    return {
        "enabled": True,
        "mode": mode,
        "role": role,
        "auto_launch_trials": True,
        "auto_advance_plans": True,
        "stop_before_evaluation": True,
        "worker_mailbox_repos": worker_mailbox_repos,
    }


class SingleHostCampaignControllerTests(unittest.TestCase):
    def setUp(self) -> None:
        helper = multifidelity_tests.MultiFidelityTrainingTests(
            methodName="runTest"
        )
        helper.setUp()
        self.helper = helper
        self.root = helper.root
        helper.session["multi_fidelity"]["rungs"] = [
            {"budget": 10, "target_promoted_candidates": 2},
            {"budget": 20, "target_promoted_candidates": 0},
        ]

    def tearDown(self) -> None:
        self.helper.tearDown()

    def _write(
        self,
        *,
        mode: str,
    ) -> tuple[dict[str, object], Path, Path]:
        session = copy.deepcopy(self.helper.session)
        session["campaign_controller"] = _controller_contract(
            mode=mode,
            role="single_host",
            worker_mailbox_repos={},
        )
        spec = validate_spec(session)
        plan = build_plan(spec)
        session_path = self.root / f"controller-{mode}-session.json"
        plan_path = self.root / f"controller-{mode}-plan.json"
        session_path.write_text(json.dumps(spec), encoding="utf-8")
        plan_path.write_text(json.dumps(plan), encoding="utf-8")
        return spec, session_path, plan_path

    def test_shadow_is_read_only_and_execute_requires_authorization(self) -> None:
        spec, session_path, plan_path = self._write(mode="shadow")
        controller_root = (
            Path(spec["execution"]["state_dir"]) / "controller"
        )
        report = inspect_or_advance(
            session_path,
            plan_path,
            execute=False,
        )
        self.assertEqual(report["next_action"], "initialize_controller")
        self.assertFalse(controller_root.exists())
        with self.assertRaisesRegex(SpecError, "mode=execute"):
            inspect_or_advance(
                session_path,
                plan_path,
                execute=True,
            )
        self.assertFalse(controller_root.exists())

    def test_controller_completes_multifidelity_and_is_idempotent(self) -> None:
        spec, session_path, plan_path = self._write(mode="execute")
        with (
            mock.patch("execute_trial_plan._gpu_idle", return_value=True),
            mock.patch(
                "execute_trial_plan._resource_preflight",
                return_value={"test": "approved"},
            ),
        ):
            ranking_path = None
            actions: list[str] = []
            for _ in range(240):
                report = inspect_or_advance(
                    session_path,
                    plan_path,
                    execute=True,
                )
                actions.append(report["action_taken"])
                ranking_path = report.get("training_ranking_path")
                if ranking_path:
                    break
                time.sleep(0.02)
            for process in list(executor_module._ACTIVE_CHILDREN.values()):
                process.wait(timeout=5)
            executor_module._ACTIVE_CHILDREN.clear()
        self.assertIsNotNone(ranking_path)
        self.assertTrue(Path(ranking_path).is_file())
        inventory_path = report.get("checkpoint_inventory_path")
        self.assertIsNotNone(inventory_path)
        inventory = json.loads(Path(inventory_path).read_text(encoding="utf-8"))
        self.assertTrue(inventory["entries"])
        self.assertEqual(inventory["training_ranking_sha256"], hashlib.sha256(
            Path(ranking_path).read_bytes()
        ).hexdigest())
        self.assertIn("advance_plan", actions)
        self.assertIn("finalize_training", actions)
        status = inspect_or_advance(
            session_path,
            plan_path,
            execute=False,
        )
        self.assertEqual(status["next_action"], "evaluation_required")
        state_path = Path(status["state_path"])
        state_before = state_path.read_bytes()
        repeated = inspect_or_advance(
            session_path,
            plan_path,
            execute=True,
        )
        self.assertEqual(repeated["action_taken"], "none")
        self.assertEqual(state_path.read_bytes(), state_before)

    def test_contract_rejects_machine_specific_distributed_shape(self) -> None:
        invalid = copy.deepcopy(self.helper.session)
        invalid["campaign_controller"] = _controller_contract(
            mode="execute",
            role="distributed",
            worker_mailbox_repos={},
        )
        with self.assertRaisesRegex(
            SpecError,
            "version-6.*single_host",
        ):
            validate_spec(invalid)

    def test_controller_rejects_tampered_state(self) -> None:
        _, session_path, plan_path = self._write(mode="execute")
        initialized = inspect_or_advance(
            session_path,
            plan_path,
            execute=True,
        )
        state_path = Path(initialized["state_path"])
        state = json.loads(state_path.read_text(encoding="utf-8"))
        state["last_action"] = "tampered"
        state_path.write_text(json.dumps(state), encoding="utf-8")
        with self.assertRaisesRegex(SpecError, "state binding"):
            inspect_or_advance(
                session_path,
                plan_path,
                execute=False,
            )


class DistributedCampaignControllerTests(unittest.TestCase):
    def setUp(self) -> None:
        helper = mailbox_tests.GitMailboxTests(methodName="runTest")
        helper.setUp()
        self.helper = helper
        self.root = helper.root
        session = copy.deepcopy(helper.session)
        session["campaign_controller"] = _controller_contract(
            mode="execute",
            role="distributed",
            worker_mailbox_repos={
                session["distributed"]["workers"][0]["id"]:
                    str(helper.coordinator),
                session["distributed"]["workers"][1]["id"]:
                    str(helper.worker_clone),
            },
        )
        self.spec = validate_spec(session)
        self.plan = build_plan(self.spec)
        self.session_path = self.root / "controller-mailbox-session.json"
        self.plan_path = self.root / "controller-mailbox-plan.json"
        self.session_path.write_text(json.dumps(self.spec), encoding="utf-8")
        self.plan_path.write_text(json.dumps(self.plan), encoding="utf-8")

    def tearDown(self) -> None:
        self.helper.tearDown()

    def test_mailbox_prepares_exact_plan_snapshot(self) -> None:
        publish(self.helper.coordinator, self.session_path, self.plan_path)
        published_job = _build_jobs(
            self.spec,
            self.plan,
            include_calibration=False,
        )[0]
        worker = next(
            item
            for item in self.spec["distributed"]["workers"]
            if item["id"] == published_job["worker_id"]
        )
        job = published_job["run"]
        claim(
            self.helper.worker_clone,
            self.session_path,
            worker["id"],
            job["run_id"],
            1,
        )
        prepared_path = self.root / "prepared-with-plan.json"
        prepare_job(
            self.helper.worker_clone,
            self.session_path,
            worker["id"],
            job["run_id"],
            1,
            prepared_path,
        )
        prepared = json.loads(prepared_path.read_text(encoding="utf-8"))
        self.assertEqual(prepared["plan"], self.plan)
        self.assertEqual(
            prepared["job"]["plan_sha256"],
            __import__("hashlib").sha256(
                json.dumps(
                    self.plan,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                    allow_nan=False,
                ).encode()
            ).hexdigest(),
        )

    def test_coordinator_publishes_then_worker_claims_idempotently(self) -> None:
        coordinator_id = self.spec["distributed"]["coordinator_id"]
        initialized = inspect_or_advance(
            self.session_path,
            self.plan_path,
            execute=True,
            worker_id=coordinator_id,
        )
        self.assertEqual(initialized["action_taken"], "initialize_controller")
        published = inspect_or_advance(
            self.session_path,
            self.plan_path,
            execute=True,
            worker_id=coordinator_id,
        )
        self.assertEqual(published["action_taken"], "publish_initial_plan")

        worker_id = next(
            worker["id"]
            for worker in self.spec["distributed"]["workers"]
            if worker["id"] != coordinator_id
        )
        worker_init = inspect_or_advance(
            self.session_path,
            self.plan_path,
            execute=True,
            worker_id=worker_id,
        )
        self.assertEqual(worker_init["action_taken"], "initialize_controller")
        claimed = inspect_or_advance(
            self.session_path,
            self.plan_path,
            execute=True,
            worker_id=worker_id,
        )
        self.assertEqual(claimed["action_taken"], "claim_job")
        self.assertIsNotNone(claimed["active_job"])
        shadow = inspect_or_advance(
            self.session_path,
            self.plan_path,
            execute=False,
            worker_id=worker_id,
        )
        self.assertIn(
            shadow["next_action"],
            {"initialize_worker_executor", "launch_worker"},
        )


if __name__ == "__main__":
    unittest.main()
