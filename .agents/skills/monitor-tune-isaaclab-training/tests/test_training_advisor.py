#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "assess_training_run.py"
SPEC = importlib.util.spec_from_file_location("assess_training_run", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def criteria() -> dict:
    return {
        "version": 1,
        "window_size": 2,
        "minimum_records": 4,
        "plateau_required_metrics": 2,
        "metrics": {
            "mean_reward": {
                "direction": "maximize",
                "plateau_relative_tolerance": 0.01,
                "required": True,
            },
            "error_vel_xy": {
                "direction": "minimize",
                "plateau_relative_tolerance": 0.01,
                "hard_max": 1.0,
                "required": True,
            },
        },
        "play_gates": {"termination_rate": {"op": "<=", "value": 0.1}},
    }


def summary(rows: list[tuple[float, float]], non_finite: list | None = None) -> dict:
    return {
        "profile_id": "rsl-rl-amp-roa",
        "records": [
            {"progress": index, "mean_reward": reward, "error_vel_xy": error}
            for index, (reward, error) in enumerate(rows)
        ],
        "non_finite_metrics": non_finite or [],
    }


class TrainingAdvisorTests(unittest.TestCase):
    def test_continue_when_required_metrics_improve(self) -> None:
        report = MODULE.assess_training(
            summary([(10.0, 0.8), (11.0, 0.7), (14.0, 0.5), (15.0, 0.4)]),
            criteria(),
            health={"state": "healthy"},
        )
        self.assertEqual(report["recommendation"], "continue")
        self.assertTrue(report["advisory_only"])

    def test_suspect_health_requires_recheck_even_when_metrics_improve(self) -> None:
        report = MODULE.assess_training(
            summary([(10.0, 0.8), (11.0, 0.7), (14.0, 0.5), (15.0, 0.4)]),
            criteria(),
            health={"state": "suspect"},
        )
        self.assertEqual(report["recommendation"], "continue_and_recheck")

    def test_unknown_health_is_insufficient_even_when_metrics_improve(self) -> None:
        report = MODULE.assess_training(
            summary([(10.0, 0.8), (11.0, 0.7), (14.0, 0.5), (15.0, 0.4)]),
            criteria(),
            health={"state": "unknown"},
        )
        self.assertEqual(report["recommendation"], "insufficient_evidence")

    def test_observing_health_requires_recheck_even_when_metrics_plateau(self) -> None:
        report = MODULE.assess_training(
            summary([(10.0, 0.5), (10.0, 0.5), (10.0, 0.5), (10.0, 0.5)]),
            criteria(),
            health={"state": "observing"},
        )
        self.assertEqual(report["recommendation"], "continue_and_recheck")

    def test_plateau_is_advisory(self) -> None:
        report = MODULE.assess_training(
            summary([(10.0, 0.5), (10.0, 0.5), (10.0, 0.5), (10.0, 0.5)]),
            criteria(),
            health={"state": "healthy"},
        )
        self.assertEqual(report["recommendation"], "consider_stop_plateau")
        self.assertTrue(report["pending_user_decision"])

    def test_nonfinite_recommends_stop_without_action(self) -> None:
        report = MODULE.assess_training(
            summary(
                [(10.0, 0.5), (10.0, 0.5), (10.0, 0.5), (10.0, 0.5)],
                [{"progress": 3, "metric": "mean_reward"}],
            ),
            criteria(),
            health={"state": "healthy"},
        )
        self.assertEqual(report["recommendation"], "recommend_stop_invalid")
        self.assertNotIn("signal", report)

    def test_stalled_health_recommends_stop_without_action(self) -> None:
        report = MODULE.assess_training(
            summary([(10.0, 0.8), (11.0, 0.7), (14.0, 0.5), (15.0, 0.4)]),
            criteria(),
            health={"state": "stalled"},
        )
        self.assertEqual(report["recommendation"], "recommend_stop_invalid")
        self.assertNotIn("signal", report)

    def test_completed_plateau_with_passing_play_is_converged(self) -> None:
        report = MODULE.assess_training(
            summary([(10.0, 0.5), (10.0, 0.5), (10.0, 0.5), (10.0, 0.5)]),
            criteria(),
            health={"state": "completed"},
            play_results=[{"status": "completed", "metrics": {"termination_rate": 0.0}}],
            training_finished=True,
        )
        self.assertEqual(report["convergence"], "converged")

    def test_completed_without_play_is_indeterminate(self) -> None:
        report = MODULE.assess_training(
            summary([(10.0, 0.5), (10.0, 0.5), (10.0, 0.5), (10.0, 0.5)]),
            criteria(),
            health={"state": "completed"},
            training_finished=True,
        )
        self.assertEqual(report["convergence"], "indeterminate")


if __name__ == "__main__":
    unittest.main()
