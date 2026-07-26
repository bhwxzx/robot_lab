#!/usr/bin/env python3
"""End-to-end synthetic tests for the concrete RSL-RL trial adapter."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path


SKILL = Path(__file__).resolve().parents[1]
SCRIPTS = SKILL / "scripts"
ADAPTER = SCRIPTS / "rsl_rl_trial_adapter.py"
FAKE_RSL = Path(__file__).resolve().parent / "fake_rsl_rl_train.py"
sys.path.insert(0, str(SCRIPTS))

from algorithm_profiles import load_registry, resolve_profile  # noqa: E402
from summarize_training_log import (  # noqa: E402
    StreamingLogSummary,
    parse_log,
)


class RslRlTrialAdapterTests(unittest.TestCase):
    def test_streaming_and_batch_parser_are_equivalent_at_eof(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "training.log"
            lines = [
                "Learning iteration 0/2\n",
                "Mean reward: 10.0\n",
                "Computation: 1000 steps/s "
                "(collection: 0.010s, learning 0.020s)\n",
                "Learning iteration 1/2\n",
                "Mean reward: 12.0\n",
            ]
            path.write_text("".join(lines), encoding="utf-8")
            profile = resolve_profile(load_registry(), "rsl-rl-ppo")
            streaming = StreamingLogSummary(path, 10, profile)
            for line in lines:
                streaming.feed_line(line)
            streaming.finish()
            self.assertEqual(
                streaming.snapshot(),
                parse_log(path, 10, profile),
            )

    def test_adapter_builds_exact_argv_and_emits_verified_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            outputs = root / "outputs"
            outputs.mkdir()
            log_path = outputs / "training.log"
            contract = {
                "version": 1,
                "adapter_id": "rsl-rl",
                "profile_id": "rsl-rl-ppo",
                "training_argv": [
                    sys.executable,
                    str(FAKE_RSL),
                    "--fake-log-root",
                    str(root / "rsl-logs"),
                ],
                "training_cwd": str(root),
                "parameter_cli_map": {
                    "agent.algorithm.learning_rate":
                        "agent.algorithm.learning_rate",
                },
                "summary_last": 3,
                "required_metrics": ["illegal_contact", "mean_reward"],
                "require_checkpoint": True,
                "run_id": "screening-trial-001-seed-42",
                "trial_id": "trial-001",
                "stage": "screening",
                "seed": 42,
            }
            contract_path = outputs / "adapter_contract.json"
            contract_path.write_text(json.dumps(contract), encoding="utf-8")
            command = [
                sys.executable,
                str(ADAPTER),
                "--contract",
                str(contract_path),
                "--executor-run-id",
                contract["run_id"],
                "--overrides-json",
                '{"agent.algorithm.learning_rate":0.002}',
                "--effective-config",
                str(outputs / "effective_config.json"),
                "--result",
                str(outputs / "result.json"),
                "--summary",
                str(outputs / "summary.json"),
                "--terminal",
                str(outputs / "terminal.json"),
                "--log-path",
                str(log_path),
            ]
            with log_path.open("wb") as stream:
                completed = subprocess.run(
                    command,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            self.assertEqual(completed.returncode, 0, log_path.read_text())

            effective = json.loads(
                (outputs / "effective_config.json").read_text()
            )
            self.assertEqual(
                effective["agent"]["algorithm"]["learning_rate"],
                0.002,
            )
            result = json.loads((outputs / "result.json").read_text())
            self.assertEqual(
                result["metrics"],
                {"illegal_contact": 0.0, "mean_reward": 12.0},
            )
            terminal = json.loads((outputs / "terminal.json").read_text())
            self.assertEqual(terminal["status"], "completed")
            checkpoint = Path(terminal["checkpoint"]["path"])
            self.assertEqual(
                terminal["checkpoint"]["sha256"],
                hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            )
            received = json.loads(
                (Path(terminal["rsl_rl_run_dir"]) / "received.json").read_text()
            )
            self.assertEqual(received["seed"], 42)
            self.assertEqual(
                received["overrides"],
                {"agent.algorithm.learning_rate": 0.002},
            )

    def test_adapter_refreshes_summary_before_child_exits(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            outputs = root / "outputs"
            outputs.mkdir()
            log_path = outputs / "training.log"
            run_id = "screening-baseline-seed-42"
            contract = {
                "version": 1,
                "adapter_id": "rsl-rl",
                "profile_id": "rsl-rl-ppo",
                "training_argv": [
                    sys.executable,
                    str(FAKE_RSL),
                    "--fake-log-root",
                    str(root / "rsl-logs"),
                    "--fake-delay-seconds",
                    "0.2",
                ],
                "training_cwd": str(root),
                "parameter_cli_map": {
                    "agent.algorithm.learning_rate":
                        "agent.algorithm.learning_rate",
                },
                "summary_last": 4,
                "required_metrics": ["illegal_contact", "mean_reward"],
                "require_checkpoint": True,
                "run_id": run_id,
                "trial_id": "baseline",
                "stage": "screening",
                "seed": 42,
            }
            contract_path = outputs / "adapter_contract.json"
            contract_path.write_text(json.dumps(contract), encoding="utf-8")
            command = [
                sys.executable,
                str(ADAPTER),
                "--contract",
                str(contract_path),
                "--executor-run-id",
                run_id,
                "--overrides-json",
                "{}",
                "--effective-config",
                str(outputs / "effective_config.json"),
                "--result",
                str(outputs / "result.json"),
                "--summary",
                str(outputs / "summary.json"),
                "--terminal",
                str(outputs / "terminal.json"),
                "--log-path",
                str(log_path),
            ]
            observed_live = False
            with log_path.open("wb") as stream:
                process = subprocess.Popen(
                    command,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                )
                deadline = time.time() + 5
                while time.time() < deadline and process.poll() is None:
                    summary_path = outputs / "summary.json"
                    if summary_path.exists():
                        summary = json.loads(summary_path.read_text())
                        if summary["window_size"] >= 1:
                            observed_live = True
                            break
                    time.sleep(0.02)
                process.wait(timeout=10)
            self.assertTrue(observed_live)
            terminal = json.loads((outputs / "terminal.json").read_text())
            self.assertGreaterEqual(terminal["summary_updates"], 2)

    def test_nonfinite_metric_fails_without_completed_result(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            outputs = root / "outputs"
            outputs.mkdir()
            log_path = outputs / "training.log"
            run_id = "screening-nonfinite-seed-42"
            contract = {
                "version": 1,
                "adapter_id": "rsl-rl",
                "profile_id": "rsl-rl-ppo",
                "training_argv": [
                    sys.executable,
                    str(FAKE_RSL),
                    "--fake-log-root",
                    str(root / "rsl-logs"),
                    "--fake-mode",
                    "nonfinite",
                ],
                "training_cwd": str(root),
                "parameter_cli_map": {},
                "summary_last": 3,
                "required_metrics": ["mean_reward"],
                "require_checkpoint": True,
                "run_id": run_id,
                "trial_id": "baseline",
                "stage": "screening",
                "seed": 42,
            }
            contract_path = outputs / "adapter_contract.json"
            contract_path.write_text(json.dumps(contract), encoding="utf-8")
            command = [
                sys.executable,
                str(ADAPTER),
                "--contract",
                str(contract_path),
                "--executor-run-id",
                run_id,
                "--overrides-json",
                "{}",
                "--effective-config",
                str(outputs / "effective_config.json"),
                "--result",
                str(outputs / "result.json"),
                "--summary",
                str(outputs / "summary.json"),
                "--terminal",
                str(outputs / "terminal.json"),
                "--log-path",
                str(log_path),
            ]
            with log_path.open("wb") as stream:
                completed = subprocess.run(
                    command,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            self.assertEqual(completed.returncode, 2)
            self.assertFalse((outputs / "result.json").exists())
            summary = json.loads((outputs / "summary.json").read_text())
            self.assertEqual(
                summary["non_finite_metrics"],
                [{"metric": "mean_reward", "progress": 0}],
            )


if __name__ == "__main__":
    unittest.main()
