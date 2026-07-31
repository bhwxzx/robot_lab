#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "archive_advised_policy.py"
SPEC = importlib.util.spec_from_file_location("archive_advised_policy", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class AdvisorArchiveTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name).resolve()
        self.storage = root / "storage"
        self.artifacts = root / "artifacts"
        self.storage.mkdir()
        self.artifacts.mkdir()
        (self.storage / "LW" / "leg_loco").mkdir(parents=True)
        subprocess.run(["git", "init", "-q", str(self.storage)], check=True)
        subprocess.run(["git", "-C", str(self.storage), "config", "user.email", "test@example.invalid"], check=True)
        subprocess.run(["git", "-C", str(self.storage), "config", "user.name", "Test"], check=True)
        (self.storage / ".gitkeep").write_text("\n", encoding="utf-8")
        (self.storage / "LW" / "leg_loco" / ".gitkeep").write_text("\n", encoding="utf-8")
        subprocess.run(["git", "-C", str(self.storage), "add", "."], check=True)
        subprocess.run(["git", "-C", str(self.storage), "commit", "-qm", "init"], check=True)
        self.checkpoint = self.artifacts / "model_10.pt"
        self.jit = self.artifacts / "policy.pt"
        self.onnx = self.artifacts / "policy.onnx"
        self.checkpoint.write_bytes(b"checkpoint")
        self.jit.write_bytes(b"jit")
        self.onnx.write_bytes(b"onnx")

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def manifest(self) -> dict:
        return {
            "version": 1,
            "archive_authorized": True,
            "storage_root": str(self.storage),
            "collection": "LW/leg_loco",
            "task": "lw-leg-rough",
            "algorithm": "AMP-ROA",
            "runner": "OnPolicyRunnerAmpROA",
            "selected_checkpoint": {
                "path": str(self.checkpoint),
                "sha256": sha(self.checkpoint),
                "iteration": 10,
            },
            "artifacts": {
                "jit": {"path": str(self.jit), "sha256": sha(self.jit)},
                "onnx": {"path": str(self.onnx), "sha256": sha(self.onnx)},
            },
            "source": {"commit": "abc123", "dirty": True},
            "parameters": {"action_rate_l2": -0.15},
            "evaluation": {"result_paths": ["/tmp/result.json"]},
            "description_notes": "supervised test candidate",
        }

    def test_archives_four_files_without_git_action(self) -> None:
        receipt = MODULE.archive_policy(
            self.manifest(), timestamp="2026-07-31-18-00-00"
        )
        destination = Path(receipt["archive_path"])
        self.assertEqual(
            {path.name for path in destination.iterdir()},
            {"policy.pt", "policy.onnx", "策略说明.txt", "archive_manifest.json"},
        )
        self.assertEqual(receipt["git_action"], "none")
        self.assertFalse(receipt["hardware_ready"])
        archived_manifest = json.loads((destination / "archive_manifest.json").read_text(encoding="utf-8"))
        self.assertFalse(archived_manifest["hardware_ready"])

    def test_refuses_dirty_storage(self) -> None:
        (self.storage / "dirty.txt").write_text("dirty", encoding="utf-8")
        with self.assertRaisesRegex(MODULE.ArchiveError, "must be clean"):
            MODULE.archive_policy(self.manifest(), timestamp="2026-07-31-18-00-00")

    def test_refuses_duplicate_pair(self) -> None:
        MODULE.archive_policy(self.manifest(), timestamp="2026-07-31-18-00-00")
        subprocess.run(["git", "-C", str(self.storage), "add", "."], check=True)
        subprocess.run(["git", "-C", str(self.storage), "commit", "-qm", "archive"], check=True)
        with self.assertRaisesRegex(MODULE.ArchiveError, "already exists"):
            MODULE.archive_policy(self.manifest(), timestamp="2026-07-31-18-00-01")


if __name__ == "__main__":
    unittest.main()
