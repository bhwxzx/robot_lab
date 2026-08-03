#!/usr/bin/env python3

from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))
SCRIPT = SCRIPT_DIR / "query_tuning_experience.py"
SPEC = importlib.util.spec_from_file_location("query_tuning_experience", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

import capture_run_identity as IDENTITY  # noqa: E402
import record_tuning_experience as RECORD  # noqa: E402


def make_event(
    *,
    event_id: str = "assessment-001",
    recorded_at: str = "2026-08-01T18:00:00+08:00",
    run_id: str = "run-001",
    algorithm: str = "amp-roa",
    host_id: str = "younghit",
    observation: str = "obs-hash",
    reward: str = "reward-hash",
    deployment: str = "deploy-hash",
    version: int = 2,
    evidence: dict | None = None,
) -> dict:
    event = {
        "version": version,
        "event_id": event_id,
        "event_type": "assessment",
        "recorded_at": recorded_at,
        "task": "lw-leg-rough",
        "run_id": run_id,
        "algorithm": algorithm,
        "context": {
            "observation_fingerprint": observation,
            "reward_fingerprint": reward,
            "deployment_fingerprint": deployment,
        },
        "parameters": {"action_rate_l2": -0.15},
        "evidence": evidence or {},
        "analysis": {"summary": "healthy", "confidence": "medium"},
        "next_suggestion": "recheck before changing parameters",
    }
    if version == 1:
        return event
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
        "task": event["task"],
        "run_id": event["run_id"],
        "host_id": host_id,
        "backend": "isaaclab",
        "algorithm": algorithm,
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
    event["run_identity"] = identity
    return event


class ExperienceQueryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.root = Path(self.temporary_directory.name).resolve() / "history"
        self.root.mkdir()

    def query(self, **overrides) -> dict:
        arguments = {
            "task": "lw-leg-rough",
            "algorithm": "amp-roa",
            "host_id": "younghit",
            "observation_fingerprint": "obs-hash",
            "reward_fingerprint": "reward-hash",
            "deployment_fingerprint": "deploy-hash",
        }
        arguments.update(overrides)
        return MODULE.query_tuning_experience(self.root, **arguments)

    def write(self, event: dict) -> Path:
        receipt = RECORD.write_event(self.root, event)
        return Path(receipt["event_path"])

    def test_exact_version_two_match_is_compatible_with_evidence_refs(self) -> None:
        health_path = "/archive/run-001/evidence/health/health.json"
        event_path = self.write(
            make_event(
                evidence={
                    "health": {"path": health_path, "sha256": "a" * 64},
                    "video_path": "/archive/run-001/evidence/play/video.mp4",
                    "video_sha256": "b" * 64,
                }
            )
        )
        result = self.query()
        self.assertEqual(result["summary"]["compatible"], 1)
        self.assertEqual(result["historical_support"]["status"], "compatible_history_available")
        self.assertFalse(result["historical_support"]["direct_parameter_change_supported"])
        item = result["compatible_events"][0]
        self.assertEqual(item["event_path"], str(event_path))
        self.assertEqual(
            item["event_sha256"], hashlib.sha256(event_path.read_bytes()).hexdigest()
        )
        self.assertEqual(
            {reference["path"] for reference in item["evidence_refs"]},
            {health_path, "/archive/run-001/evidence/play/video.mp4"},
        )

    def test_known_mismatches_are_conflicting_with_explicit_reasons(self) -> None:
        cases = [
            ("algorithm", {"algorithm": "ppo"}, "algorithm_mismatch"),
            ("host", {"host_id": "server5090"}, "host_id_mismatch"),
            (
                "observation",
                {"observation": "different-observation"},
                "observation_fingerprint_mismatch",
            ),
            (
                "reward",
                {"reward": "different-reward"},
                "reward_fingerprint_mismatch",
            ),
            (
                "deployment",
                {"deployment": "different-deployment"},
                "deployment_fingerprint_mismatch",
            ),
        ]
        expected_reasons = {}
        for index, (name, overrides, reason) in enumerate(cases, start=1):
            event_id = f"assessment-{name}"
            self.write(
                make_event(
                    event_id=event_id,
                    run_id=f"run-{index:03d}",
                    **overrides,
                )
            )
            expected_reasons[event_id] = [reason]
        result = self.query()
        observed_reasons = {
            item["event_id"]: item["classification_reasons"]
            for item in result["conflicting_events"]
        }
        self.assertEqual(
            observed_reasons,
            expected_reasons,
        )
        self.assertEqual(result["historical_support"]["status"], "no_compatible_history")

    def test_legacy_and_explicit_unknown_events_never_become_compatible(self) -> None:
        self.write(make_event(version=1))
        self.write(make_event(event_id="assessment-002", observation="unknown"))
        result = self.query()
        self.assertEqual(result["summary"]["unknown"], 2)
        reasons = [item["classification_reasons"] for item in result["unknown_events"]]
        self.assertIn(["event_host_id_unknown"], reasons)
        self.assertIn(["event_observation_fingerprint_unknown"], reasons)

    def test_unknown_query_context_suppresses_historical_support(self) -> None:
        self.write(make_event())
        result = self.query(observation_fingerprint="unknown")
        self.assertEqual(result["summary"]["unknown"], 1)
        self.assertEqual(
            result["historical_support"]["status"],
            "query_context_incomplete",
        )

    def test_invalid_json_is_reported_and_suppresses_support(self) -> None:
        self.write(make_event())
        run_dir = self.root / "lw-leg-rough" / "run-002"
        run_dir.mkdir(parents=True)
        invalid = run_dir / "invalid.json"
        invalid.write_text("{invalid", encoding="utf-8")
        result = self.query()
        self.assertEqual(result["summary"]["compatible"], 1)
        self.assertEqual(result["summary"]["invalid"], 1)
        self.assertEqual(result["historical_support"]["status"], "history_invalid")
        self.assertFalse(result["scan"]["complete"])

    def test_storage_scope_and_filename_mismatch_are_invalid(self) -> None:
        event = make_event()
        run_dir = self.root / "lw-leg-rough" / "wrong-run"
        run_dir.mkdir(parents=True)
        (run_dir / "wrong.json").write_text(
            json.dumps(event),
            encoding="utf-8",
        )
        filename_event = make_event(event_id="assessment-filename", run_id="run-002")
        filename_run_dir = self.root / "lw-leg-rough" / "run-002"
        filename_run_dir.mkdir()
        (filename_run_dir / "wrong.json").write_text(
            json.dumps(filename_event),
            encoding="utf-8",
        )
        result = self.query()
        self.assertEqual(result["summary"]["invalid"], 2)
        errors = [item["error"] for item in result["invalid_events"]]
        self.assertTrue(any("scope" in error for error in errors))
        self.assertTrue(any("filename" in error for error in errors))

    def test_results_are_deterministic_and_chronologically_sorted(self) -> None:
        self.write(
            make_event(
                event_id="assessment-later",
                recorded_at="2026-08-01T18:00:00+08:00",
                run_id="run-z",
            )
        )
        self.write(
            make_event(
                event_id="assessment-earlier",
                recorded_at="2026-08-01T09:30:00Z",
                run_id="run-a",
            )
        )
        first = self.query()
        second = self.query()
        self.assertEqual(first, second)
        self.assertEqual(
            [item["event_id"] for item in first["compatible_events"]],
            ["assessment-earlier", "assessment-later"],
        )

    def test_nested_evidence_json_is_not_scanned(self) -> None:
        self.write(make_event())
        nested = (
            self.root
            / "lw-leg-rough"
            / "run-001"
            / "evidence"
            / "health"
            / "health.json"
        )
        nested.parent.mkdir(parents=True)
        nested.write_text("{invalid", encoding="utf-8")
        result = self.query()
        self.assertEqual(result["scan"]["event_files"], 1)
        self.assertEqual(result["summary"]["invalid"], 0)

    def test_rejects_symlinked_root_run_and_event_paths(self) -> None:
        real_root = Path(self.temporary_directory.name).resolve() / "real-root"
        real_root.mkdir()
        linked_root = Path(self.temporary_directory.name).resolve() / "linked-root"
        linked_root.symlink_to(real_root, target_is_directory=True)
        with self.assertRaisesRegex(MODULE.ExperienceQueryError, "symlinked"):
            MODULE.query_tuning_experience(
                linked_root,
                task="lw-leg-rough",
                algorithm="amp-roa",
                host_id="younghit",
                observation_fingerprint="obs-hash",
                reward_fingerprint="reward-hash",
                deployment_fingerprint="deploy-hash",
            )

        task_dir = self.root / "lw-leg-rough"
        task_dir.mkdir()
        external_run = Path(self.temporary_directory.name).resolve() / "external-run"
        external_run.mkdir()
        (task_dir / "run-linked").symlink_to(external_run, target_is_directory=True)
        with self.assertRaisesRegex(MODULE.ExperienceQueryError, "symlinked run"):
            self.query()

        (task_dir / "run-linked").unlink()
        run_dir = task_dir / "run-001"
        run_dir.mkdir()
        external_event = Path(self.temporary_directory.name).resolve() / "external.json"
        external_event.write_text("{}", encoding="utf-8")
        (run_dir / "event.json").symlink_to(external_event)
        with self.assertRaisesRegex(MODULE.ExperienceQueryError, "symlinked event"):
            self.query()

    def test_scan_and_file_size_bounds_are_enforced(self) -> None:
        self.write(make_event())
        self.write(make_event(event_id="assessment-002", run_id="run-002"))
        with self.assertRaisesRegex(MODULE.ExperienceQueryError, "max-events=1"):
            self.query(max_events=1)
        result = self.query(max_event_bytes=10)
        self.assertEqual(result["summary"]["invalid"], 2)
        self.assertEqual(result["historical_support"]["status"], "history_invalid")

    def test_cli_prints_json_without_changing_history(self) -> None:
        self.write(make_event())

        def snapshot() -> dict[str, bytes | None]:
            return {
                str(path.relative_to(self.root)): path.read_bytes() if path.is_file() else None
                for path in sorted(self.root.rglob("*"))
            }

        before = snapshot()
        stdout = io.StringIO()
        argv = [
            str(SCRIPT),
            "--root",
            str(self.root),
            "--task",
            "lw-leg-rough",
            "--algorithm",
            "amp-roa",
            "--host-id",
            "younghit",
            "--observation-fingerprint",
            "obs-hash",
            "--reward-fingerprint",
            "reward-hash",
            "--deployment-fingerprint",
            "deploy-hash",
        ]
        with patch.object(sys, "argv", argv), contextlib.redirect_stdout(stdout):
            self.assertEqual(MODULE.main(), 0)
        result = json.loads(stdout.getvalue())
        self.assertTrue(result["read_only"])
        self.assertEqual(snapshot(), before)


if __name__ == "__main__":
    unittest.main()
