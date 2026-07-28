#!/usr/bin/env python3
"""Transactional policy-export executor and parity-gate tests."""

from __future__ import annotations

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

from build_policy_export_plan import (  # noqa: E402
    build_plan,
    validate_sources,
)
import execute_policy_export_plan as executor  # noqa: E402
import test_evaluation_handoff_controller as handoff_tests  # noqa: E402
from validate_session_spec import validate_spec  # noqa: E402
from validate_session_spec import SpecError  # noqa: E402


class PolicyExportExecutorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.helper = handoff_tests.EvaluationHandoffControllerTests(
            methodName="runTest"
        )
        self.helper.setUp()

    def tearDown(self) -> None:
        self.helper.tearDown()

    def _fixture(
        self,
        fake_mode: str = "healthy",
    ) -> tuple[dict, Path, Path, dict]:
        session = self.helper._session_with_export(fake_mode=fake_mode)
        session_path, ranking_path, inventory_path = self.helper._sources(
            "execute",
            session=session,
        )
        spec = validate_spec(session)
        ranking, inventory = validate_sources(
            spec,
            session_path,
            ranking_path,
            inventory_path,
        )
        plan = build_plan(
            spec,
            ranking,
            inventory,
            session_path=session_path,
            ranking_path=ranking_path,
            inventory_path=inventory_path,
            worker_id=None,
        )
        plan_path = self.helper.root / f"export-plan-{fake_mode}.json"
        plan_path.write_text(
            json.dumps(plan, sort_keys=True),
            encoding="utf-8",
        )
        return spec, session_path, plan_path, plan

    def _execute(
        self,
        spec: dict,
        session_path: Path,
        plan_path: Path,
        plan: dict,
    ) -> dict:
        state = executor.initialize_state(
            spec,
            session_path,
            plan,
            plan_path,
        )
        executor._persist_state(spec, state, "test-initialize")
        with (
            mock.patch.object(
                executor,
                "_resource_preflight",
                return_value={"synthetic": True},
            ),
            mock.patch.object(executor, "_gpu_idle", return_value=True),
        ):
            executor.launch_next(spec, plan, state)
            for process in list(executor._ACTIVE_CHILDREN.values()):
                process.wait(timeout=5)
            for _ in range(20):
                state = executor.reconcile(spec, plan, state)
                if not any(
                    run["status"] in executor.ACTIVE_STATUSES
                    for run in state["runs"].values()
                ):
                    break
                time.sleep(0.02)
        return state

    def test_healthy_export_emits_hash_bound_manifest(self) -> None:
        spec, session_path, plan_path, plan = self._fixture()
        state = self._execute(spec, session_path, plan_path, plan)
        self.assertEqual(state["stage"], "completed")
        manifest = json.loads(
            Path(state["manifest_path"]).read_text(encoding="utf-8")
        )
        self.assertEqual(manifest["plan_sha256"], state["plan_sha256"])
        self.assertEqual(
            set(manifest["candidates"][0]["artifact_sha256"]),
            {"jit", "onnx"},
        )

    def test_bad_parity_blocks_without_publishing_manifest(self) -> None:
        spec, session_path, plan_path, plan = self._fixture("bad-parity")
        state = self._execute(spec, session_path, plan_path, plan)
        self.assertEqual(state["stage"], "blocked")
        run = next(iter(state["runs"].values()))
        self.assertEqual(run["status"], "failed")
        self.assertIn("parity gate failed", run["failure_reason"])
        self.assertIsNone(state["manifest_path"])

    def test_truncated_last_journal_record_is_repaired_before_append(self) -> None:
        spec, session_path, plan_path, plan = self._fixture()
        state = executor.initialize_state(
            spec,
            session_path,
            plan,
            plan_path,
        )
        executor._persist_state(spec, state, "test-initialize")
        journal = executor._journal_path(spec)
        with journal.open("ab") as stream:
            stream.write(b'{"truncated":')
        state["updated_at"] = time.time()
        executor._persist_state(spec, state, "test-after-truncation")
        events = executor._read_journal(journal)
        self.assertEqual(len(events), 2)
        self.assertEqual(events[-1]["action"], "test-after-truncation")

    def test_plan_with_different_session_hash_is_rejected(self) -> None:
        spec, session_path, plan_path, plan = self._fixture()
        plan["session_sha256"] = "f" * 64
        plan_path.write_text(
            json.dumps(plan, sort_keys=True),
            encoding="utf-8",
        )
        with self.assertRaisesRegex(SpecError, "exact session"):
            executor.initialize_state(
                spec,
                session_path,
                plan,
                plan_path,
            )


if __name__ == "__main__":
    unittest.main()
