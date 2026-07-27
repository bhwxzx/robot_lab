#!/usr/bin/env python3
"""Local bare-Git integration tests for version-7 distributed tuning."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


SKILL = Path(__file__).resolve().parents[1]
SCRIPTS = SKILL / "scripts"
TESTS = Path(__file__).resolve().parent
import sys

sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(TESTS))

from build_trial_plan import build_plan  # noqa: E402
from git_mailbox import (  # noqa: E402
    MailboxError,
    _build_jobs,
    cancel,
    claim,
    collect,
    prepare_job,
    publish,
    publish_confirmation,
    result,
    status,
)
from execute_trial_plan import _prepare_distributed_job, initialize_state  # noqa: E402
from validate_session_spec import SpecError, validate_spec  # noqa: E402


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


def _init_source(root: Path) -> tuple[Path, Path, str]:
    origin = root / "source-origin.git"
    seed = root / "source-seed"
    pc_a = root / "source-pc-a"
    pc_b = root / "source-pc-b"
    _run("git", "init", "--bare", str(origin))
    _run("git", "init", "-b", "main", str(seed))
    (seed / "source.txt").write_text("approved source\n", encoding="utf-8")
    _run("git", "add", "source.txt", cwd=seed)
    _run(
        "git",
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.invalid",
        "commit",
        "-m",
        "source",
        cwd=seed,
    )
    _run("git", "remote", "add", "origin", str(origin), cwd=seed)
    _run("git", "push", "-u", "origin", "main", cwd=seed)
    _run("git", "symbolic-ref", "HEAD", "refs/heads/main", cwd=origin)
    _run("git", "clone", str(origin), str(pc_a))
    _run("git", "clone", str(origin), str(pc_b))
    commit = _run("git", "rev-parse", "HEAD", cwd=pc_a)
    return pc_a, pc_b, commit


def _init_mailbox(root: Path) -> tuple[Path, Path, Path]:
    origin = root / "mailbox-origin.git"
    seed = root / "mailbox-seed"
    coordinator = root / "mailbox-coordinator"
    worker = root / "mailbox-worker"
    _run("git", "init", "--bare", str(origin))
    _run("git", "init", "-b", "main", str(seed))
    (seed / ".gitignore").write_text("# coordination repository\n", encoding="utf-8")
    _run("git", "add", ".gitignore", cwd=seed)
    _run(
        "git",
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.invalid",
        "commit",
        "-m",
        "initialize",
        cwd=seed,
    )
    _run("git", "remote", "add", "origin", str(origin), cwd=seed)
    _run("git", "push", "-u", "origin", "main", cwd=seed)
    _run("git", "symbolic-ref", "HEAD", "refs/heads/main", cwd=origin)
    _run("git", "clone", str(origin), str(coordinator))
    _run("git", "clone", str(origin), str(worker))
    return origin, coordinator, worker


class GitMailboxTests(unittest.TestCase):
    def setUp(self) -> None:
        from test_execution_round_one import ExecutionRoundOneTests

        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        helper = ExecutionRoundOneTests(methodName="runTest")
        helper.temp = self.temp
        helper.root = self.root
        helper.baseline = self.root / "baseline.json"
        helper.baseline.write_text(
            json.dumps({"agent": {"learning_rate": 0.1}, "env": {"penalty": -0.1}}),
            encoding="utf-8",
        )
        self.session = helper._session()
        self.source_a, self.source_b, source_commit = _init_source(self.root)
        self.origin, self.coordinator, self.worker_clone = _init_mailbox(self.root)
        self.session["version"] = 7
        self.session["training"]["source_git_commit"] = source_commit
        self.session["training"]["source_git_dirty"] = False
        self.session["distributed"] = {
            "enabled": True,
            "transport": "git_mailbox",
            "campaign_id": "mailbox-test",
            "remote_url": "https://example.invalid/private/mailbox.git",
            "coordinator_id": "pc-a",
            "coordinator_branch": "tune/mailbox-test/coordinator",
            "poll_interval_seconds": 600,
            "remote_state_unknown_after_seconds": 1800,
            "artifact_policy": "metadata_only",
            "workers": [
                {
                    "id": "pc-a",
                    "branch": "tune/mailbox-test/worker-pc-a",
                    "assigned_seeds": [42, 44],
                    "source_repo": str(self.source_a),
                    "state_dir": str(self.root / "pc-a-state"),
                    "effective_config_baseline_path": str(
                        self.root / "pc-a-effective-config.json"
                    ),
                    "gpu_index": 0,
                    "max_active_jobs": 1,
                },
                {
                    "id": "pc-b",
                    "branch": "tune/mailbox-test/worker-pc-b",
                    "assigned_seeds": [43],
                    "source_repo": str(self.source_b),
                    "state_dir": str(self.root / "pc-b-state"),
                    "effective_config_baseline_path": str(
                        self.root / "pc-b-effective-config.json"
                    ),
                    "gpu_index": 0,
                    "max_active_jobs": 1,
                },
            ],
            "calibration": {
                "enabled": True,
                "seed": 42,
                "worker_ids": ["pc-a", "pc-b"],
            },
        }
        validate_spec(self.session)
        self.plan = build_plan(self.session)
        self.session_path = self.root / "session.json"
        self.plan_path = self.root / "plan.json"
        self.session_path.write_text(json.dumps(self.session), encoding="utf-8")
        self.plan_path.write_text(json.dumps(self.plan), encoding="utf-8")
        self.remote_override = patch.dict(
            os.environ, {"GIT_MAILBOX_ALLOW_TEST_REMOTE": "1"}
        )
        self.remote_override.start()

    def tearDown(self) -> None:
        self.remote_override.stop()
        self.temp.cleanup()

    def _trial_job_id(self) -> str:
        return self.plan["runs"][0]["run_id"]

    def _publish_training_result(
        self,
        worker_id: str,
        job: dict[str, object],
        score: float,
    ) -> None:
        job_id = str(job["run_id"])
        claim(self.worker_clone, self.session_path, worker_id, job_id, 1)
        result_path = self.root / f"{job_id}-result.json"
        artifacts_path = self.root / f"{job_id}-artifacts.json"
        result_path.write_text(
            json.dumps(
                {
                    "trial_id": job["trial_id"],
                    "seed": job["seed"],
                    "status": "completed",
                    "metrics": {"score": score, "unsafe": 0.0},
                }
            ),
            encoding="utf-8",
        )
        artifacts_path.write_text(json.dumps({"artifacts": []}), encoding="utf-8")
        result(
            self.worker_clone,
            self.session_path,
            worker_id,
            job_id,
            1,
            result_path,
            artifacts_path,
        )

    def test_round_trip_publish_claim_result_collect_and_cancel(self) -> None:
        published = publish(self.coordinator, self.session_path, self.plan_path)
        self.assertEqual(published["published_jobs"], len(self.plan["runs"]) + 2)

        worker_status = status(
            self.worker_clone, self.session_path, "pc-a"
        )
        self.assertTrue(any(job["state"] == "pending" for job in worker_status["jobs"]))

        job_id = self._trial_job_id()
        claimed = claim(
            self.worker_clone, self.session_path, "pc-a", job_id, 1
        )
        self.assertEqual(claimed["state"], "claim_published")

        result_path = self.root / "result.json"
        artifacts_path = self.root / "artifacts.json"
        result_path.write_text(
            json.dumps({"status": "completed", "metrics": {"score": 1.25}}),
            encoding="utf-8",
        )
        artifacts_path.write_text(
            json.dumps(
                {
                    "artifacts": [
                        {
                            "kind": "checkpoint",
                            "path": str(self.root / "model.pt"),
                            "sha256": "a" * 64,
                            "size_bytes": 123,
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        completed = result(
            self.worker_clone,
            self.session_path,
            "pc-a",
            job_id,
            1,
            result_path,
            artifacts_path,
        )
        self.assertEqual(completed["state"], "result_published")

        report_path = self.root / "collected.json"
        report = collect(self.coordinator, self.session_path, report_path)
        self.assertEqual(report["accepted_result_count"], 1)
        self.assertEqual(report["invalid_result_count"], 0)

        other_job = self.plan["runs"][1]["run_id"]
        cancelled = cancel(
            self.coordinator,
            self.session_path,
            "pc-a",
            other_job,
            "bounded test cancellation",
        )
        self.assertEqual(cancelled["state"], "cancel_published")
        refreshed = status(self.worker_clone, self.session_path, "pc-a")
        states = {job["job_id"]: job["state"] for job in refreshed["jobs"]}
        self.assertEqual(states[other_job], "cancel_requested")

    def test_failed_claim_push_is_idempotently_recoverable(self) -> None:
        publish(self.coordinator, self.session_path, self.plan_path)
        invalid_push = self.root / "missing-remote.git"
        _run(
            "git",
            "remote",
            "set-url",
            "--push",
            "origin",
            str(invalid_push),
            cwd=self.worker_clone,
        )
        job_id = self._trial_job_id()
        with self.assertRaises(MailboxError):
            claim(self.worker_clone, self.session_path, "pc-a", job_id, 1)
        _run(
            "git",
            "remote",
            "set-url",
            "--push",
            "origin",
            str(self.origin),
            cwd=self.worker_clone,
        )
        recovered = claim(
            self.worker_clone, self.session_path, "pc-a", job_id, 1
        )
        self.assertEqual(recovered["state"], "claim_published")

    def test_prepared_job_limits_existing_executor_to_one_run(self) -> None:
        publish(self.coordinator, self.session_path, self.plan_path)
        job_id = self._trial_job_id()
        claim(self.worker_clone, self.session_path, "pc-a", job_id, 1)
        job_path = self.root / "prepared-job.json"
        prepared = prepare_job(
            self.worker_clone,
            self.session_path,
            "pc-a",
            job_id,
            1,
            job_path,
        )
        self.assertEqual(prepared["state"], "job_prepared")
        prepared_job = json.loads(job_path.read_text(encoding="utf-8"))
        runtime_spec, runtime_plan = _prepare_distributed_job(
            self.session,
            self.plan,
            prepared_job,
            "pc-a",
        )
        state_path, state = initialize_state(
            runtime_spec,
            self.session_path,
            runtime_plan,
            self.plan_path,
            plan_already_validated=True,
        )
        self.assertEqual(set(state["runs"]), {job_id})
        self.assertEqual(state["distributed_job"]["job_id"], job_id)
        self.assertEqual(
            state_path.parent,
            Path(self.session["distributed"]["workers"][0]["state_dir"])
            / "jobs"
            / job_id,
        )

        tampered = json.loads(json.dumps(prepared_job))
        tampered["job"]["run"]["seed"] = 43
        with self.assertRaisesRegex(SpecError, "hash is invalid|exact approved"):
            _prepare_distributed_job(
                self.session,
                self.plan,
                tampered,
                "pc-a",
            )

    def test_complete_screening_publishes_only_selected_confirmation(self) -> None:
        publish(self.coordinator, self.session_path, self.plan_path)
        for index, run in enumerate(self.plan["runs"]):
            self._publish_training_result("pc-a", run, 10.0 + index)
        confirmation = publish_confirmation(
            self.coordinator,
            self.session_path,
            self.plan_path,
        )
        self.assertEqual(confirmation["state"], "confirmation_published")
        self.assertEqual(len(confirmation["selected_trial_ids"]), 1)
        self.assertEqual(confirmation["published_jobs"], 4)

        pc_b = status(self.worker_clone, self.session_path, "pc-b")
        confirmation_jobs = [
            job
            for job in pc_b["jobs"]
            if "--confirmation--" in job["job_id"]
        ]
        self.assertEqual(len(confirmation_jobs), 2)
        selected_job_id = confirmation_jobs[0]["job_id"]
        claim(
            self.worker_clone,
            self.session_path,
            "pc-b",
            selected_job_id,
            1,
        )
        prepared_path = self.root / "confirmation-job.json"
        prepare_job(
            self.worker_clone,
            self.session_path,
            "pc-b",
            selected_job_id,
            1,
            prepared_path,
        )
        prepared = json.loads(prepared_path.read_text(encoding="utf-8"))
        runtime_spec, runtime_plan = _prepare_distributed_job(
            self.session,
            self.plan,
            prepared,
            "pc-b",
        )
        self.assertEqual(runtime_plan["runs"][0]["run_id"], selected_job_id)
        self.assertEqual(
            runtime_spec["execution"]["gpu_index"],
            self.session["distributed"]["workers"][1]["gpu_index"],
        )

    def test_source_mismatch_blocks_worker(self) -> None:
        publish(self.coordinator, self.session_path, self.plan_path)
        (self.source_a / "untracked.txt").write_text("dirty\n", encoding="utf-8")
        with self.assertRaisesRegex(MailboxError, "must be clean"):
            status(self.worker_clone, self.session_path, "pc-a")

    def test_changed_publication_collides(self) -> None:
        publish(self.coordinator, self.session_path, self.plan_path)
        changed = json.loads(json.dumps(self.plan))
        changed["runs"][0]["overrides"] = {"agent.learning_rate": 999}
        changed_path = self.root / "changed-plan.json"
        changed_path.write_text(json.dumps(changed), encoding="utf-8")
        with self.assertRaisesRegex(MailboxError, "immutable mailbox collision"):
            publish(self.coordinator, self.session_path, changed_path)

    def test_validator_rejects_overlap_credentials_and_invalid_calibration(self) -> None:
        overlap = json.loads(json.dumps(self.session))
        overlap["distributed"]["workers"][1]["assigned_seeds"] = [42, 43]
        with self.assertRaisesRegex(SpecError, "overlaps"):
            validate_spec(overlap)

        credentials = json.loads(json.dumps(self.session))
        credentials["distributed"]["remote_url"] = (
            "https://user:secret@example.invalid/mailbox.git"
        )
        with self.assertRaisesRegex(SpecError, "without embedded credentials"):
            validate_spec(credentials)

        no_calibration = json.loads(json.dumps(self.session))
        no_calibration["distributed"]["calibration"] = {
            "enabled": False,
            "seed": 42,
            "worker_ids": [],
        }
        validate_spec(no_calibration)
        no_calibration_jobs = _build_jobs(
            validate_spec(no_calibration),
            build_plan(no_calibration),
        )
        self.assertEqual(len(no_calibration_jobs), len(self.plan["runs"]))

        disabled_with_workers = json.loads(json.dumps(no_calibration))
        disabled_with_workers["distributed"]["calibration"]["worker_ids"] = [
            "pc-a"
        ]
        with self.assertRaisesRegex(SpecError, "must be empty"):
            validate_spec(disabled_with_workers)

        partial_calibration = json.loads(json.dumps(self.session))
        partial_calibration["distributed"]["calibration"]["worker_ids"] = [
            "pc-a"
        ]
        with self.assertRaisesRegex(SpecError, "every worker exactly once"):
            validate_spec(partial_calibration)


if __name__ == "__main__":
    unittest.main()
