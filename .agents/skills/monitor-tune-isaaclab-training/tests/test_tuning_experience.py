#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))
SCRIPT = SCRIPT_DIR / "record_tuning_experience.py"
SPEC = importlib.util.spec_from_file_location("record_tuning_experience", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

import capture_run_identity as IDENTITY  # noqa: E402


def event(event_type: str = "assessment") -> dict:
    value = {
        "version": 1,
        "event_id": "assessment-001",
        "event_type": event_type,
        "recorded_at": "2026-07-31T18:00:00+08:00",
        "task": "lw-leg-rough",
        "run_id": "run-001",
        "algorithm": "amp-roa",
        "context": {
            "observation_fingerprint": "obs-hash",
            "reward_fingerprint": "reward-hash",
            "deployment_fingerprint": "deploy-hash",
        },
        "parameters": {"action_rate_l2": -0.15},
        "evidence": {},
        "analysis": {"summary": "healthy", "confidence": "medium"},
        "next_suggestion": "recheck at iteration 40000",
    }
    if event_type == "feedback":
        value["evidence"] = {"source": "sim2real"}
    return value


def version_two_event() -> dict:
    value = event()
    value["version"] = 2
    scenario = {
        "scenario_id": "quick-native",
        "scenario_overrides": {},
        "command_schedule": [],
        "duration_steps": 500,
        "num_envs": 1,
        "seed": 42,
    }
    identity = {
        "version": 1,
        "task": value["task"],
        "run_id": value["run_id"],
        "host_id": "younghit",
        "backend": "isaaclab",
        "algorithm": value["algorithm"],
        "runner": "OnPolicyRunnerAmpROA",
        "seed": 42,
        "source": {
            "repository_root": "/absolute/robot_lab",
            "branch": "main",
            "head": "1" * 40,
            "dirty": False,
            "dirty_paths": [],
            "diff_sha256": None,
            "patch_evidence": None,
        },
        "training": {
            "command": ["python", "train.py", "env.scene.num_envs=4096"],
            "hydra_overrides": ["env.scene.num_envs=4096"],
        },
        "config_files": [{"path": "config.yaml", "sha256": "2" * 64}],
        "evaluation_scenario": {
            "contract": scenario,
            "sha256": IDENTITY._sha256_bytes(
                IDENTITY._canonical_json(scenario).encode("utf-8")
            ),
        },
    }
    identity["identity_sha256"] = IDENTITY._sha256_bytes(
        IDENTITY._canonical_json(identity).encode("utf-8")
    )
    value["run_identity"] = identity
    return value


class TuningExperienceTests(unittest.TestCase):
    def test_writes_immutable_event(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            receipt = MODULE.write_event(Path(directory).resolve(), event())
            self.assertTrue(Path(receipt["event_path"]).is_file())
            self.assertTrue(receipt["immutable"])
            with self.assertRaises(MODULE.ExperienceError):
                MODULE.write_event(Path(directory).resolve(), event())

    def test_concurrent_writers_publish_exactly_one_event(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            barrier = Barrier(2)

            def write_once() -> str:
                barrier.wait()
                try:
                    MODULE.write_event(root, event())
                except MODULE.ExperienceError:
                    return "rejected"
                return "published"

            with ThreadPoolExecutor(max_workers=2) as executor:
                outcomes = sorted(executor.map(lambda _: write_once(), range(2)))
            self.assertEqual(outcomes, ["published", "rejected"])
            self.assertEqual(len(list(root.rglob("*.json"))), 1)
            self.assertEqual(len(list(root.rglob("*.tmp-*"))), 0)

    def test_feedback_requires_source(self) -> None:
        value = event("feedback")
        value["evidence"] = {}
        with self.assertRaises(MODULE.ExperienceError):
            MODULE.validate_event(value)

    def test_feedback_accepts_sim2sim_and_sim2real(self) -> None:
        MODULE.validate_event(event("feedback"))
        value = event("feedback")
        value["evidence"]["source"] = "sim2sim"
        MODULE.validate_event(value)

    def test_version_two_requires_valid_run_identity(self) -> None:
        value = version_two_event()
        MODULE.validate_event(value)
        del value["run_identity"]
        with self.assertRaisesRegex(MODULE.ExperienceError, "run_identity.version"):
            MODULE.validate_event(value)

    def test_version_two_scope_must_match_run_identity(self) -> None:
        value = version_two_event()
        value["run_id"] = "different-run"
        with self.assertRaisesRegex(MODULE.ExperienceError, "must match"):
            MODULE.validate_event(value)

    def test_version_one_remains_compatible(self) -> None:
        value = event()
        MODULE.validate_event(value)

    def test_rejects_symlinked_experience_root(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory).resolve()
            real_root = parent / "real-root"
            real_root.mkdir()
            linked_root = parent / "linked-root"
            linked_root.symlink_to(real_root, target_is_directory=True)
            with self.assertRaisesRegex(MODULE.ExperienceError, "symlinked"):
                MODULE.write_event(linked_root, event())

    def test_rejects_symlinked_run_component_before_creation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve() / "events"
            root.mkdir()
            external = Path(directory).resolve() / "external"
            external.mkdir()
            (root / "lw-leg-rough").symlink_to(external, target_is_directory=True)
            with self.assertRaisesRegex(MODULE.ExperienceError, "symlinked"):
                MODULE.write_event(root, event())
            self.assertFalse((external / "run-001").exists())

    def test_receipt_output_is_exclusive(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory).resolve() / "receipt.json"
            MODULE._write_new_output(output, "{}\n")
            with self.assertRaisesRegex(MODULE.ExperienceError, "new absolute"):
                MODULE._write_new_output(output, "changed\n")
            self.assertEqual(output.read_text(encoding="utf-8"), "{}\n")


if __name__ == "__main__":
    unittest.main()
