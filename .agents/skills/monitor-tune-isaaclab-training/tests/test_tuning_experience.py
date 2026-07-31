#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "record_tuning_experience.py"
SPEC = importlib.util.spec_from_file_location("record_tuning_experience", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


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


class TuningExperienceTests(unittest.TestCase):
    def test_writes_immutable_event(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            receipt = MODULE.write_event(Path(directory).resolve(), event())
            self.assertTrue(Path(receipt["event_path"]).is_file())
            self.assertTrue(receipt["immutable"])
            with self.assertRaises(MODULE.ExperienceError):
                MODULE.write_event(Path(directory).resolve(), event())

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


if __name__ == "__main__":
    unittest.main()
