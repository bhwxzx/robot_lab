#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))


def load_script(name: str):
    path = SCRIPT_DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


LAYOUT = load_script("prepare_evidence_layout")
EXPERIENCE = load_script("record_tuning_experience")


def experience_event() -> dict:
    return {
        "version": 1,
        "event_id": "snapshot-001",
        "event_type": "run_snapshot",
        "recorded_at": "2026-08-01T18:00:00+08:00",
        "task": "task-a",
        "run_id": "run-001",
        "algorithm": "AMP-ROA",
        "context": {
            "observation_fingerprint": "unknown",
            "reward_fingerprint": "unknown",
            "deployment_fingerprint": "unknown",
        },
        "parameters": {},
        "evidence": {},
        "analysis": {"summary": "", "confidence": "low"},
        "next_suggestion": "",
    }


class EvidenceLayoutTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.root = Path(self.temporary_directory.name).resolve() / "policy_tuning"

    def prepare(self, snapshot: str = "snapshot-001", evaluation: str | None = "eval-001") -> dict:
        return LAYOUT.prepare_evidence_layout(
            self.root,
            task="task-a",
            run_id="run-001",
            snapshot_id=snapshot,
            evaluation_id=evaluation,
        )

    def test_returns_deterministic_absolute_paths_and_creates_only_directories(self) -> None:
        layout = self.prepare()
        self.assertEqual(layout, self.prepare())
        for path in layout["paths"].values():
            self.assertTrue(Path(path).is_absolute())
            self.assertFalse(Path(path).exists())
        self.assertTrue(Path(layout["evidence_root"]).is_dir())
        self.assertEqual(
            Path(layout["paths"]["health"]),
            self.root / "task-a" / "run-001" / "evidence" / "health" / "health-snapshot-001.json",
        )

    def test_snapshots_and_evaluations_do_not_conflict(self) -> None:
        first = self.prepare("snapshot-001", "eval-001")
        second = self.prepare("snapshot-002", "eval-002")
        for name in (
            "criteria",
            "health",
            "source_identity",
            "source_patch",
            "summary",
            "assessment",
            "play_result",
            "telemetry",
            "video",
        ):
            self.assertNotEqual(first["paths"][name], second["paths"][name])

    def test_optional_evaluation_omits_play_targets(self) -> None:
        layout = self.prepare(evaluation=None)
        for name in ("play_result", "telemetry", "video"):
            self.assertIsNone(layout["paths"][name])
        self.assertIsNone(layout["directories"]["evaluation"])
        self.assertIn(
            "unset PLAY_RESULT_PATH TELEMETRY_PATH VIDEO_PATH",
            LAYOUT._shell_assignments(layout),
        )

    def test_rejects_unsafe_identifiers(self) -> None:
        invalid_values = ("../escape", "nested/name", ".", "..", " hidden", "非ascii")
        for value in invalid_values:
            with self.subTest(value=value), self.assertRaisesRegex(
                LAYOUT.EvidenceLayoutError,
                "safe ASCII identifier",
            ):
                LAYOUT.prepare_evidence_layout(
                    self.root,
                    task=value,
                    run_id="run-001",
                    snapshot_id="snapshot-001",
                )

    def test_rejects_symlinked_path_component(self) -> None:
        self.root.mkdir(parents=True)
        real_task = self.root / "real-task"
        real_task.mkdir()
        (self.root / "task-a").symlink_to(real_task, target_is_directory=True)
        with self.assertRaisesRegex(LAYOUT.EvidenceLayoutError, "symlinked path component"):
            self.prepare()

    def test_rejects_existing_evidence_target(self) -> None:
        layout = self.prepare()
        Path(layout["paths"]["health"]).write_text("{}\n", encoding="utf-8")
        with self.assertRaisesRegex(LAYOUT.EvidenceLayoutError, "already exists"):
            self.prepare()

    def test_raw_evidence_and_immutable_events_have_separate_roots(self) -> None:
        layout = self.prepare()
        receipt = EXPERIENCE.write_event(self.root, experience_event())
        event_path = Path(receipt["event_path"])
        run_root = Path(layout["run_root"])
        evidence_root = Path(layout["evidence_root"])
        self.assertEqual(event_path.parent, run_root)
        self.assertEqual(evidence_root.parent, run_root)
        self.assertNotIn(evidence_root, event_path.parents)


if __name__ == "__main__":
    unittest.main()
