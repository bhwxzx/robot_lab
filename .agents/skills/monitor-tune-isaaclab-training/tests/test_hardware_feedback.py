#!/usr/bin/env python3
"""Synthetic tests for the hardware-feedback retuning boundary."""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from algorithm_profiles import (  # noqa: E402
    load_registry,
    profile_fingerprint,
    resolve_profile,
)
from build_feedback_retune_proposal import build_proposal  # noqa: E402
from validate_hardware_feedback import (  # noqa: E402
    authorized_output_path,
    load_and_validate_feedback,
)
from validate_hardware_qualification import qualify  # noqa: E402
from validate_session_spec import SpecError, validate_spec  # noqa: E402


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class HardwareFeedbackTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _algorithm(
        self,
        profile_id: str = "rsl-rl-ppo",
        name: str = "PPO",
        runner: str = "OnPolicyRunner",
        unknown_policy: str = "reject",
    ) -> dict[str, object]:
        profile = resolve_profile(load_registry(), profile_id)
        return {
            "backend": "rsl_rl",
            "name": name,
            "runner_class": runner,
            "profile_id": profile_id,
            "profile_version": profile["profile_version"],
            "profile_fingerprint": profile_fingerprint(profile),
            "unknown_algorithm_policy": unknown_policy,
        }

    def _session(
        self,
        algorithm: dict[str, object],
        mode: str = "tune",
        output_mode: str = "prepare_authorized_draft",
    ) -> dict[str, object]:
        tuning = None
        if mode == "tune":
            tuning = {
                "allowed_parameters": [
                    {
                        "path": "env.rewards.action_rate_l2.weight",
                        "values": [-0.1, -0.2],
                        "baseline": -0.1,
                    },
                    {
                        "path": "env.rewards.track_lin_vel_xy_exp.weight",
                        "range": {"min": 1.0, "max": 2.0, "step": 0.5},
                        "baseline": 1.0,
                    },
                ],
                "protected_parameters_unlocked": [],
                "max_trials": 3,
                "seeds": [42, 43],
                "trial_timeout_minutes": 60,
                "max_concurrent_trials": 1,
                "mutation_scope": "overrides_only",
                "objectives": [
                    {"metric": "tilt_rms", "goal": "minimize", "weight": 1}
                ],
                "constraints": [
                    {"metric": "termination_rate", "op": "<=", "value": 0.01}
                ],
            }
        return {
            "version": 5,
            "mode": mode,
            "algorithm": algorithm,
            "training": {
                "command": ["python", "train.py", "--headless"],
                "cwd": str(self.root),
                "log_path": str(self.root / "training.log"),
                "run_id": "feedback-test",
            },
            "monitoring": {
                "check_interval_seconds": 60,
                "stale_after_seconds": 120,
                "pid": None,
                "gpu_index": 0,
                "tensorboard_path": None,
                "expected_process_pattern": "train.py",
                "low_gpu_utilization_percent": 5,
            },
            "recovery": {
                "enabled": False,
                "max_restarts": 0,
                "cooldown_seconds": 0,
            },
            "tuning": tuning,
            "evaluation": None,
            "archive": None,
            "hardware_feedback": {
                "enabled": True,
                "output_mode": output_mode,
                "output_dir": str(self.root / "output"),
                "require_policy_manifest": True,
                "verify_artifact_hashes": True,
                "stop_on_safety_event": True,
                "require_new_session_approval": True,
            },
            "cleanup": {"remove_created_temp_files": True},
        }

    def _write_case(
        self,
        session: dict[str, object],
    ) -> tuple[Path, Path, dict[str, object]]:
        archive = self.root / "archive"
        archive.mkdir()
        jit = archive / "policy.pt"
        onnx = archive / "policy.onnx"
        config = self.root / "deployment.yaml"
        video = self.root / "motion.mp4"
        telemetry = self.root / "telemetry.csv"
        jit.write_bytes(b"jit-policy")
        onnx.write_bytes(b"onnx-policy")
        config.write_text("control_hz: 50\n", encoding="utf-8")
        video.write_bytes(b"synthetic-video")
        telemetry.write_text("t,roll,action\n0,0,0\n", encoding="utf-8")
        manifest = {
            "version": 1,
            "candidate_id": "trial-001",
            "hardware_ready": False,
            "algorithm": session["algorithm"],
            "artifacts": {
                "jit": {"source_path": str(jit), "sha256": _sha256(jit)},
                "onnx": {"source_path": str(onnx), "sha256": _sha256(onnx)},
            },
        }
        manifest_path = archive / "archive_manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, sort_keys=True),
            encoding="utf-8",
        )
        feedback: dict[str, object] = {
            "version": 1,
            "feedback_id": "feedback-001",
            "policy": {
                "archive_manifest_path": str(manifest_path),
                "archive_manifest_sha256": _sha256(manifest_path),
                "candidate_id": "trial-001",
                "artifacts": {
                    "jit": _sha256(jit),
                    "onnx": _sha256(onnx),
                },
            },
            "deployment": {
                "runtime": "synthetic-runtime-v1",
                "artifact_kind": "jit",
                "robot_id": "synthetic-robot",
                "firmware": "synthetic-firmware",
                "control_frequency_hz": 50,
                "config_files": [
                    {"path": str(config), "sha256": _sha256(config)}
                ],
                "observation_contract_verified": True,
                "history_initialized": True,
                "emergency_stop_verified": True,
                "notes": "",
            },
            "test": {
                "started_at": "2026-07-26T14:00:00+08:00",
                "operator": "user",
                "supervision": "supervised",
                "scenario": "standing",
                "surface": "level floor",
                "payload_kg": 0,
                "duration_seconds": 30,
                "command_envelope": {
                    "max_linear_speed_mps": 0,
                    "max_yaw_rate_rps": 0,
                },
            },
            "observations": [
                {
                    "symptom": "standing_roll_oscillation",
                    "severity": "moderate",
                    "start_seconds": 5,
                    "end_seconds": 18,
                    "notes": "repeatable roll motion",
                }
            ],
            "safety": {
                "emergency_stop": False,
                "fall": False,
                "joint_limit_violation": False,
                "torque_limit_violation": False,
                "communication_timeout": False,
                "mechanical_damage": False,
                "operator_intervention": False,
                "notes": "",
            },
            "evidence": {
                "video_files": [
                    {"path": str(video), "sha256": _sha256(video)}
                ],
                "telemetry_files": [
                    {"path": str(telemetry), "sha256": _sha256(telemetry)}
                ],
                "telemetry_channels": ["imu_roll", "action", "control_timestamp"],
                "sample_rate_hz": 100,
                "clock_synchronized": True,
            },
            "user_assessment": {"overall": "fail", "notes": "not acceptable"},
        }
        session_path = self.root / "session.json"
        feedback_path = self.root / "feedback.json"
        session_path.write_text(json.dumps(session), encoding="utf-8")
        feedback_path.write_text(json.dumps(feedback), encoding="utf-8")
        return session_path, feedback_path, feedback

    def _proposal(
        self,
        session: dict[str, object],
        mutate_feedback=None,
    ) -> dict[str, object]:
        session_path, feedback_path, feedback = self._write_case(session)
        if mutate_feedback is not None:
            mutate_feedback(feedback)
            feedback_path.write_text(json.dumps(feedback), encoding="utf-8")
        loaded_session, loaded_feedback, _ = load_and_validate_feedback(
            session_path,
            feedback_path,
        )
        return build_proposal(loaded_session, loaded_feedback)

    def test_evidenced_oscillation_prepares_only_pending_draft(self) -> None:
        proposal = self._proposal(self._session(self._algorithm()))
        self.assertEqual(
            proposal["classification"]["root_cause"],
            "training_or_sim2real_candidate",
        )
        self.assertTrue(
            proposal["retune"]["may_prepare_authorization_draft"]
        )
        self.assertFalse(proposal["executable"])
        self.assertEqual(
            proposal["authorization_draft"]["selected_parameter_paths"],
            [],
        )
        self.assertTrue(proposal["requires_user_approval"])

    def test_subjective_only_feedback_does_not_enable_retune(self) -> None:
        def subjective(feedback):
            feedback["evidence"] = {
                "video_files": [],
                "telemetry_files": [],
                "telemetry_channels": [],
                "sample_rate_hz": None,
                "clock_synchronized": False,
            }

        proposal = self._proposal(
            self._session(self._algorithm()),
            subjective,
        )
        self.assertEqual(
            proposal["classification"]["root_cause"],
            "insufficient_evidence_for_retune",
        )
        self.assertFalse(
            proposal["retune"]["may_prepare_authorization_draft"]
        )

    def test_safety_event_stops_physical_testing(self) -> None:
        def unsafe(feedback):
            feedback["safety"]["emergency_stop"] = True
            feedback["user_assessment"]["overall"] = "unsafe"

        proposal = self._proposal(self._session(self._algorithm()), unsafe)
        self.assertEqual(
            proposal["classification"]["root_cause"],
            "hardware_safety_incident",
        )
        self.assertTrue(proposal["safety"]["stop_physical_testing"])
        self.assertFalse(
            proposal["retune"]["may_prepare_authorization_draft"]
        )

    def test_unverified_deployment_is_diagnosed_before_training(self) -> None:
        def unverified(feedback):
            feedback["deployment"]["observation_contract_verified"] = False

        proposal = self._proposal(
            self._session(self._algorithm()),
            unverified,
        )
        self.assertEqual(
            proposal["classification"]["root_cause"],
            "deployment_contract_incomplete",
        )
        self.assertFalse(
            proposal["retune"]["may_prepare_authorization_draft"]
        )

    def test_artifact_hash_mismatch_is_rejected(self) -> None:
        def changed_hash(feedback):
            feedback["policy"]["artifacts"]["jit"] = "0" * 64

        with self.assertRaisesRegex(SpecError, "does not match"):
            self._proposal(self._session(self._algorithm()), changed_hash)

    def test_future_algorithm_remains_monitor_only_and_profile_neutral(self) -> None:
        algorithm = self._algorithm(
            profile_id="rsl-rl-generic",
            name="FutureRL",
            runner="FutureRunner",
            unknown_policy="runtime_generic",
        )
        session = self._session(
            algorithm,
            mode="monitor",
            output_mode="proposal_only",
        )
        proposal = self._proposal(session)
        self.assertEqual(proposal["eligible_existing_parameters"], [])
        self.assertIn(
            "session_has_no_tuning_authority",
            proposal["retune"]["blockers"],
        )
        self.assertNotIn("authorization_draft", proposal)
        self.assertFalse(proposal["executable"])

    def test_version4_without_feedback_remains_valid(self) -> None:
        session = self._session(self._algorithm())
        session["version"] = 4
        session.pop("hardware_feedback")
        self.assertEqual(validate_spec(session)["version"], 4)

    def test_version4_cannot_enable_feedback(self) -> None:
        session = self._session(self._algorithm())
        session["version"] = 4
        with self.assertRaisesRegex(SpecError, "requires session version 5"):
            validate_spec(session)

    def test_version6_monitor_feedback_remains_valid(self) -> None:
        session = self._session(
            self._algorithm(),
            mode="monitor",
            output_mode="proposal_only",
        )
        session["version"] = 6
        self.assertEqual(validate_spec(session)["version"], 6)

    def test_output_must_remain_beneath_authorized_directory(self) -> None:
        session = self._session(self._algorithm())
        with self.assertRaisesRegex(SpecError, "escapes"):
            authorized_output_path(
                session,
                str(self.root / "outside.json"),
            )
        accepted = authorized_output_path(
            session,
            str(self.root / "output" / "proposal.json"),
        )
        self.assertEqual(accepted.name, "proposal.json")

    def test_repeated_physical_matrix_gets_only_bounded_qualification(self) -> None:
        session = self._session(self._algorithm())
        session["hardware_feedback"]["qualification"] = {
            "enabled": True,
            "final_authority": "supervised_hardware",
            "minimum_total_tests": 4,
            "required_scenarios": [
                "standing",
                "start_stop",
                "low_speed",
                "turn",
            ],
            "minimum_tests_per_scenario": 1,
            "require_high_evidence_confidence": True,
            "required_telemetry_channels": [
                "action",
                "control_timestamp",
                "imu_roll",
            ],
            "require_all_assessments_pass": True,
            "require_zero_safety_events": True,
            "status_label": "hardware_validated_for_test_envelope",
        }
        session_path, base_feedback_path, base_feedback = self._write_case(session)
        feedback_paths = []
        for index, scenario in enumerate(
            ("standing", "start_stop", "low_speed", "turn"),
            start=1,
        ):
            feedback = json.loads(json.dumps(base_feedback))
            feedback["feedback_id"] = f"feedback-{index:03d}"
            feedback["test"]["scenario"] = scenario
            feedback["test"]["started_at"] = (
                f"2026-07-26T14:0{index}:00+08:00"
            )
            feedback["observations"] = []
            feedback["user_assessment"] = {
                "overall": "pass",
                "notes": "supervised test passed",
            }
            video = self.root / f"motion-{index}.mp4"
            telemetry = self.root / f"telemetry-{index}.csv"
            video.write_bytes(f"video-{index}".encode())
            telemetry.write_text(
                f"t,roll,action\n0,0,{index}\n",
                encoding="utf-8",
            )
            feedback["evidence"]["video_files"] = [
                {"path": str(video), "sha256": _sha256(video)}
            ]
            feedback["evidence"]["telemetry_files"] = [
                {"path": str(telemetry), "sha256": _sha256(telemetry)}
            ]
            path = (
                base_feedback_path
                if index == 1
                else self.root / f"feedback-{index}.json"
            )
            path.write_text(json.dumps(feedback), encoding="utf-8")
            feedback_paths.append(str(path))
        bundle_path = self.root / "qualification.json"
        bundle_path.write_text(
            json.dumps(
                {
                    "version": 1,
                    "qualification_id": "qualification-001",
                    "feedback_paths": feedback_paths,
                }
            ),
            encoding="utf-8",
        )
        report = qualify(session_path, bundle_path)
        self.assertTrue(report["qualified"])
        self.assertEqual(
            report["status"],
            "hardware_validated_for_test_envelope",
        )
        self.assertFalse(report["hardware_ready"])
        self.assertFalse(report["generalization_claim"])
        self.assertEqual(report["tested_envelope"]["scenario_test_counts"]["turn"], 1)

        duplicate = json.loads(bundle_path.read_text(encoding="utf-8"))
        copied_feedback = json.loads(
            Path(duplicate["feedback_paths"][0]).read_text(encoding="utf-8")
        )
        copied_feedback["feedback_id"] = "feedback-duplicate"
        copied_feedback["test"]["started_at"] = "2026-07-26T14:05:00+08:00"
        duplicate_path = self.root / "feedback-duplicate.json"
        duplicate_path.write_text(json.dumps(copied_feedback), encoding="utf-8")
        duplicate["feedback_paths"][-1] = str(duplicate_path)
        duplicate_bundle = self.root / "qualification-duplicate.json"
        duplicate_bundle.write_text(json.dumps(duplicate), encoding="utf-8")
        blocked = qualify(session_path, duplicate_bundle)
        self.assertFalse(blocked["qualified"])
        self.assertIn(
            "video and telemetry evidence files cannot be reused",
            blocked["failures"],
        )


if __name__ == "__main__":
    unittest.main()
