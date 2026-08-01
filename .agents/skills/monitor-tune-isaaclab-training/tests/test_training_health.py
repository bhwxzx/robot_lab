#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))
SCRIPT = SCRIPT_DIR / "collect_training_health.py"
SPEC = importlib.util.spec_from_file_location("collect_training_health", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


PROFILE = {
    "id": "test-profile",
    "progress_patterns": [
        {
            "name": "learning_iteration",
            "regex": r"Learning iteration\s+(?P<current>\d+)(?:/(?P<target>\d+))?",
            "completion_offset": 1,
        }
    ],
}


def tensorboard(step: int | None, wall_time: float | None = None) -> dict:
    return {
        "path": "/tmp/events",
        "available": step is not None,
        "step": step,
        "wall_time_unix": wall_time,
        "tag": "Loss/value_function" if step is not None else None,
        "error": None if step is not None else "no scalar events",
    }


def process(pid: int = 123, *, alive: bool = True, matches: bool = True) -> dict:
    return {
        "pid": pid,
        "alive": alive,
        "matches_expected": matches if alive else None,
        "cmdline": "python train.py" if alive else None,
    }


def gpu(utilization: float = 50.0) -> dict:
    return {
        "index": 0,
        "available": True,
        "utilization_percent": utilization,
        "memory_used_mb": 1000.0,
        "error": None,
    }


def previous_health(
    log_path: Path,
    *,
    timestamp: float = 100.0,
    log_step: int | None = 10,
    tensorboard_step: int | None = None,
    pid: int = 123,
) -> dict:
    log_progress = (
        {
            "name": "learning_iteration",
            "current": log_step,
            "target": 100,
            "completion_offset": 1,
        }
        if log_step is not None
        else None
    )
    return {
        "version": 1,
        "timestamp_unix": timestamp,
        "profile_id": PROFILE["id"],
        "progress": {
            "log": log_progress,
            "tensorboard": {"step": tensorboard_step},
        },
        "log": {"path": str(log_path)},
        "process": {"pid": pid},
    }


class TrainingHealthTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.log_path = Path(self.temporary_directory.name) / "train.log"

    def collect(
        self,
        log_step: int,
        *,
        now: float = 200.0,
        target: int = 100,
        previous: dict | None = None,
        tensorboard_value: dict | None = None,
        process_value: dict | None = None,
        gpu_value: dict | None = None,
    ) -> dict:
        self.log_path.write_text(
            f"Learning iteration {log_step}/{target}\n",
            encoding="utf-8",
        )
        with (
            patch.object(
                MODULE,
                "tensorboard_progress",
                return_value=tensorboard_value or tensorboard(None),
            ),
            patch.object(
                MODULE,
                "_process_info",
                return_value=process_value or process(),
            ),
            patch.object(
                MODULE,
                "_gpu_info",
                return_value=gpu_value or gpu(),
            ),
        ):
            return MODULE.collect_health(
                log_path=self.log_path,
                profile=PROFILE,
                stale_after_seconds=60,
                pid=123,
                expected_process_pattern="train.py",
                gpu_index=0,
                low_gpu_utilization_percent=5.0,
                tensorboard_path=Path("/tmp/events"),
                previous_health=previous,
                now=now,
            )

    def test_first_recent_progress_is_observing_not_healthy(self) -> None:
        report = self.collect(
            10,
            tensorboard_value=tensorboard(50, wall_time=199.0),
        )
        self.assertEqual(report["state"], "observing")
        self.assertFalse(report["comparison"]["baseline_available"])
        self.assertIn("tensorboard_recency_is_auxiliary_only", report["evidence"])

    def test_advanced_log_step_is_healthy(self) -> None:
        previous = previous_health(self.log_path, log_step=10)
        report = self.collect(11, previous=previous)
        self.assertEqual(report["state"], "healthy")
        self.assertTrue(report["comparison"]["advanced"])
        self.assertEqual(report["baseline_for_next_check"]["log_progress"]["current"], 11)

    def test_advanced_tensorboard_step_is_healthy(self) -> None:
        previous = previous_health(
            self.log_path,
            log_step=None,
            tensorboard_step=50,
        )
        report = self.collect(
            10,
            previous=previous,
            tensorboard_value=tensorboard(51, wall_time=199.0),
        )
        self.assertEqual(report["state"], "healthy")
        self.assertTrue(report["comparison"]["tensorboard"]["advanced"])

    def test_unchanged_progress_before_threshold_is_suspect(self) -> None:
        previous = previous_health(self.log_path, timestamp=170.0, log_step=10)
        report = self.collect(10, now=200.0, previous=previous)
        self.assertEqual(report["state"], "suspect")
        self.assertTrue(report["comparison"]["unchanged"])
        self.assertFalse(report["progress"]["stale"])

    def test_unchanged_progress_with_live_process_and_low_gpu_is_stalled(self) -> None:
        previous = previous_health(self.log_path, timestamp=100.0, log_step=10)
        report = self.collect(
            10,
            now=200.0,
            previous=previous,
            gpu_value=gpu(2.0),
        )
        self.assertEqual(report["state"], "stalled")
        self.assertTrue(report["progress"]["stale"])

    def test_regressed_progress_is_unknown(self) -> None:
        previous = previous_health(self.log_path, log_step=11)
        report = self.collect(10, previous=previous)
        self.assertEqual(report["state"], "unknown")
        self.assertTrue(report["comparison"]["regressed"])

    def test_previous_pid_mismatch_is_unknown(self) -> None:
        previous = previous_health(self.log_path, log_step=10, pid=999)
        report = self.collect(11, previous=previous)
        self.assertEqual(report["state"], "unknown")
        self.assertIn("previous_pid_mismatch", report["comparison"]["errors"])

    def test_stopped_process_without_completion_is_stopped(self) -> None:
        previous = previous_health(self.log_path, log_step=10)
        report = self.collect(
            10,
            previous=previous,
            process_value=process(alive=False),
            gpu_value=gpu(0.0),
        )
        self.assertEqual(report["state"], "stopped")

    def test_target_progress_is_completed(self) -> None:
        report = self.collect(99, target=100)
        self.assertEqual(report["state"], "completed")


if __name__ == "__main__":
    unittest.main()
