#!/usr/bin/env python3
"""Tests for approval-gated first-run skill configuration."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from configure_skill import (  # noqa: E402
    SetupError,
    apply_plan,
    build_plan,
    locate_configuration,
    normalize_answers,
    verify_configuration,
)


def _run(*args: str, cwd: Path | None = None) -> str:
    completed = subprocess.run(
        list(args),
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        raise AssertionError(
            f"command failed: {' '.join(args)}\n{completed.stderr}"
        )
    return completed.stdout.strip()


def _init_git(path: Path) -> None:
    _run("git", "init", "-b", "main", str(path))
    (path / ".gitignore").write_text("# initialized\n", encoding="utf-8")
    _run("git", "add", ".gitignore", cwd=path)
    _run(
        "git",
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.invalid",
        "commit",
        "-m",
        "initialize",
        cwd=path,
    )


class FirstRunConfigurationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.source_a = self.root / "source-a"
        _init_git(self.source_a)
        _run(
            "git",
            "remote",
            "add",
            "origin",
            "https://example.invalid/private/robot_lab.git",
            cwd=self.source_a,
        )

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _machine(
        self,
        machine_id: str,
        source: Path,
        *,
        distributed: bool,
    ) -> dict[str, object]:
        prefix = self.root / machine_id
        return {
            "id": machine_id,
            "source_repo": str(source),
            "mailbox_repo": (
                str(prefix / "mailbox") if distributed else None
            ),
            "state_dir": str(prefix / "state"),
            "effective_config_baseline_path": str(
                prefix / "state" / "effective-config.json"
            ),
            "evaluation_output_dir": str(prefix / "evaluation"),
            "hardware_feedback_output_dir": str(prefix / "hardware-feedback"),
            "policy_storage_root": None,
            "gpu_index": 0,
            "worker_branch": (
                f"tune/setup/worker-{machine_id}" if distributed else None
            ),
        }

    def _single_answers(self) -> dict[str, object]:
        return {
            "version": 1,
            "setup_id": "single-test",
            "setup_mode": "single_host",
            "configuration_dir": str(self.root / "single-config"),
            "conda_env": "isaacsim-5.1",
            "default_seed": 42,
            "source_remote_url": "https://example.invalid/private/robot_lab.git",
            "local_machine_id": "pc-a",
            "machines": [
                self._machine("pc-a", self.source_a, distributed=False)
            ],
            "distributed": None,
        }

    def _distributed_answers(self) -> dict[str, object]:
        source_b = self.root / "other-host" / "source-b"
        answers = {
            "version": 1,
            "setup_id": "distributed-test",
            "setup_mode": "git_mailbox",
            "configuration_dir": str(self.root / "distributed-config"),
            "conda_env": "isaacsim-5.1",
            "default_seed": 42,
            "source_remote_url": "https://example.invalid/private/robot_lab.git",
            "local_machine_id": "pc-a",
            "machines": [
                self._machine("pc-a", self.source_a, distributed=True),
                self._machine("pc-b", source_b, distributed=True),
            ],
            "distributed": {
                "transport": "git_mailbox",
                "remote_url": "https://example.invalid/private/mailbox.git",
                "coordinator_id": "pc-a",
                "coordinator_branch": "tune/setup/coordinator",
                "poll_interval_seconds": 600,
                "remote_state_unknown_after_seconds": 1800,
                "artifact_policy": "metadata_only",
                "assignment_mode_default": "by_trial",
                "host_effect_calibration_default_enabled": False,
            },
        }
        mailbox = Path(answers["machines"][0]["mailbox_repo"])
        if not mailbox.exists():
            _init_git(mailbox)
            _run(
                "git",
                "remote",
                "add",
                "origin",
                answers["distributed"]["remote_url"],
                cwd=mailbox,
            )
        return answers

    def test_single_host_plan_apply_verify_and_idempotence(self) -> None:
        answers = self._single_answers()
        with patch.dict(
            os.environ,
            {
                "ROBOT_LAB_TUNER_CONFIG": str(
                    Path(answers["configuration_dir"]) / "configuration.json"
                )
            },
        ):
            plan = build_plan(answers)
            self.assertTrue(plan["discovery"]["matches_current_discovery"])
        self.assertFalse(plan["remote_push_permitted"])
        self.assertFalse(plan["credential_storage_permitted"])
        applied = apply_plan(plan, plan["plan_sha256"])
        self.assertEqual(applied["state"], "configured")
        self.assertFalse(applied["configuration_discoverable"])
        self.assertEqual(
            applied["activation_required"]["environment_variable"],
            "ROBOT_LAB_TUNER_CONFIG",
        )
        self.assertFalse(applied["remote_push_performed"])
        configuration = json.loads(
            Path(applied["configuration_path"]).read_text(encoding="utf-8")
        )
        report = verify_configuration(
            configuration,
            check_runtime=False,
            check_remote=False,
            loaded_from=applied["configuration_path"],
        )
        self.assertTrue(report["ready_for_training"])
        self.assertFalse(report["remote_write_tested"])
        repeated = apply_plan(plan, plan["plan_sha256"])
        self.assertEqual(repeated["state"], "already_configured")

    def test_locator_uses_explicit_absolute_configuration(self) -> None:
        expected = self.root / "custom" / "configuration.json"
        with patch.dict(
            os.environ,
            {"ROBOT_LAB_TUNER_CONFIG": str(expected)},
        ):
            located = locate_configuration()
        self.assertEqual(located["configuration_path"], str(expected))
        self.assertEqual(located["source"], "environment")
        with patch.dict(
            os.environ,
            {"ROBOT_LAB_TUNER_CONFIG": "relative/configuration.json"},
        ):
            with self.assertRaisesRegex(SetupError, "absolute path"):
                locate_configuration()

    def test_apply_requires_exact_approval_and_refuses_overwrite(self) -> None:
        plan = build_plan(self._single_answers())
        with self.assertRaisesRegex(SetupError, "approval SHA-256"):
            apply_plan(plan, "0" * 64)
        applied = apply_plan(plan, plan["plan_sha256"])
        path = Path(applied["configuration_path"])
        changed = json.loads(path.read_text(encoding="utf-8"))
        changed["default_seed"] = 43
        path.write_text(json.dumps(changed), encoding="utf-8")
        with self.assertRaisesRegex(SetupError, "differs"):
            apply_plan(plan, plan["plan_sha256"])

    def test_verify_blocks_changed_setup_receipt(self) -> None:
        plan = build_plan(self._single_answers())
        applied = apply_plan(plan, plan["plan_sha256"])
        configuration_path = Path(applied["configuration_path"])
        receipt_path = Path(applied["receipt_path"])
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        receipt["remote_push_performed"] = True
        receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
        configuration = json.loads(
            configuration_path.read_text(encoding="utf-8")
        )
        report = verify_configuration(
            configuration,
            check_runtime=False,
            check_remote=False,
            loaded_from=configuration_path,
        )
        self.assertFalse(report["ready_for_training"])
        self.assertIn(
            "setup receipt identity or safety binding is invalid",
            report["blockers"],
        )

    def test_distributed_setup_uses_existing_clone_and_detects_origin_drift(
        self,
    ) -> None:
        answers = self._distributed_answers()
        plan = build_plan(answers)
        applied = apply_plan(plan, plan["plan_sha256"])
        self.assertFalse(applied["git_clone_performed"])
        configuration = json.loads(
            Path(applied["configuration_path"]).read_text(encoding="utf-8")
        )
        self.assertFalse(
            configuration["distributed"][
                "host_effect_calibration_default_enabled"
            ]
        )
        report = verify_configuration(
            configuration,
            check_runtime=False,
            check_remote=False,
        )
        self.assertTrue(report["ready_for_training"])
        self.assertIn(
            "coordination remote connectivity was not checked",
            report["warnings"],
        )
        self.assertIn(
            "source remote connectivity was not checked",
            report["warnings"],
        )
        mailbox = Path(configuration["machines"][0]["mailbox_repo"])
        _run(
            "git",
            "remote",
            "set-url",
            "origin",
            "https://example.invalid/other/mailbox.git",
            cwd=mailbox,
        )
        drift = verify_configuration(
            configuration,
            check_runtime=False,
            check_remote=False,
        )
        self.assertFalse(drift["ready_for_training"])
        self.assertIn(
            "mailbox_repo origin differs from configuration",
            drift["blockers"],
        )

    def test_credentials_source_tree_outputs_and_invalid_branches_are_rejected(
        self,
    ) -> None:
        credentials = self._distributed_answers()
        credentials["distributed"]["remote_url"] = (
            "https://user:secret@example.invalid/mailbox.git"
        )
        with self.assertRaisesRegex(SetupError, "without credentials"):
            normalize_answers(credentials)

        calibration_default = self._distributed_answers()
        calibration_default["distributed"][
            "host_effect_calibration_default_enabled"
        ] = "false"
        with self.assertRaisesRegex(SetupError, "must be a boolean"):
            normalize_answers(calibration_default)

        inside = self._single_answers()
        inside["configuration_dir"] = str(self.source_a / ".setup")
        with self.assertRaisesRegex(SetupError, "outside every source"):
            normalize_answers(inside)

        branch = self._distributed_answers()
        branch["machines"][1]["worker_branch"] = (
            branch["machines"][0]["worker_branch"]
        )
        with self.assertRaisesRegex(SetupError, "branches must all be unique"):
            normalize_answers(branch)

        overlap = self._distributed_answers()
        overlap["machines"][0]["state_dir"] = str(
            Path(overlap["machines"][0]["mailbox_repo"]) / "state"
        )
        with self.assertRaisesRegex(SetupError, "outside mailbox_repo"):
            normalize_answers(overlap)

    def test_plan_tampering_is_rejected(self) -> None:
        plan = build_plan(self._single_answers())
        tampered = json.loads(json.dumps(plan))
        tampered["configuration"]["default_seed"] = 43
        with self.assertRaisesRegex(SetupError, "SHA-256"):
            apply_plan(tampered, plan["plan_sha256"])


if __name__ == "__main__":
    unittest.main()
