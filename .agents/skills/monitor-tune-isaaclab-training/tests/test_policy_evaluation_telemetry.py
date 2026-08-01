#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[4]
    / "scripts"
    / "reinforcement_learning"
    / "rsl_rl"
    / "policy_evaluation_telemetry.py"
)
SPEC = importlib.util.spec_from_file_location("policy_evaluation_telemetry", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class SignalLedgerTests(unittest.TestCase):
    def test_one_signal_failure_does_not_hide_another_signal(self) -> None:
        ledger = MODULE.SignalLedger(
            {"joint_velocity": 1, "applied_torque": 1},
            required_signals={"joint_velocity", "applied_torque"},
        )
        missing = ledger.capture(
            "joint_velocity",
            lambda: (_ for _ in ()).throw(AttributeError("joint_vel missing")),
        )
        torque = ledger.capture("applied_torque", lambda: [1.0, 2.0])
        report = ledger.report()
        self.assertIsNone(missing)
        self.assertEqual(torque, [1.0, 2.0])
        self.assertEqual(report["status"], "partial")
        self.assertFalse(report["signals"]["joint_velocity"]["available"])
        self.assertTrue(report["signals"]["applied_torque"]["complete"])
        self.assertIn("AttributeError", report["signals"]["joint_velocity"]["error"])

    def test_intermittent_signal_is_available_but_incomplete(self) -> None:
        ledger = MODULE.SignalLedger(
            {"action": 2},
            required_signals={"action"},
        )
        ledger.capture("action", lambda: [0.0])
        ledger.capture(
            "action",
            lambda: (_ for _ in ()).throw(RuntimeError("sample failed")),
        )
        status = ledger.report()["signals"]["action"]
        self.assertTrue(status["available"])
        self.assertFalse(status["complete"])
        self.assertEqual(status["sample_count"], 1)
        self.assertEqual(status["error_count"], 1)

    def test_all_required_failures_are_unavailable(self) -> None:
        ledger = MODULE.SignalLedger(
            {"command": 1},
            required_signals={"command"},
        )
        ledger.capture(
            "command",
            lambda: (_ for _ in ()).throw(KeyError("base_velocity")),
        )
        self.assertEqual(ledger.report()["status"], "unavailable")

    def test_not_requested_lists_amp_roa_required_signals(self) -> None:
        report = MODULE.telemetry_report(
            requested=False,
            runner=MODULE.AMP_ROA_RUNNER,
            ledger=None,
        )
        self.assertEqual(report["telemetry_status"], "not_requested")
        self.assertTrue(report["telemetry_required_for_complete_assessment"])
        self.assertIn("joint_velocity", report["missing_required_signals"])
        self.assertIn("applied_torque", report["missing_required_signals"])

    def test_non_amp_runner_does_not_get_amp_specific_requirements(self) -> None:
        required = MODULE.required_signals_for_runner("OnPolicyRunner")
        self.assertEqual(required, MODULE.BASE_REQUIRED_SIGNALS)
        self.assertNotIn("joint_velocity", required)
        self.assertFalse(
            MODULE.runner_requires_complete_telemetry("OnPolicyRunner")
        )

    def test_incomplete_source_does_not_emit_fake_zero_metric(self) -> None:
        ledger = MODULE.SignalLedger(
            {"joint_velocity": 1},
            required_signals={"joint_velocity"},
        )
        ledger.capture(
            "joint_velocity",
            lambda: (_ for _ in ()).throw(AttributeError("missing")),
        )
        availability = MODULE.metric_availability_report(
            ledger.report()["signals"],
            {"max_abs_joint_velocity": ("joint_velocity",)},
        )
        metrics = {}
        MODULE.record_complete_metric(
            metrics,
            availability,
            "max_abs_joint_velocity",
            lambda: 0.0,
        )
        self.assertNotIn("max_abs_joint_velocity", metrics)
        self.assertFalse(availability["max_abs_joint_velocity"]["available"])


if __name__ == "__main__":
    unittest.main()
