#!/usr/bin/env python3
"""Validate physical deployment feedback against an exact archived policy."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from validate_session_spec import SpecError, load_and_validate


SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
SYMPTOMS = {
    "standing_roll_oscillation",
    "standing_pitch_oscillation",
    "action_chatter",
    "tracking_lag",
    "tracking_overshoot",
    "turn_instability",
    "foot_slip",
    "unexpected_contact",
    "recovery_failure",
    "joint_or_torque_margin",
    "deployment_output_mismatch",
    "startup_state_error",
    "control_timing_jitter",
    "communication_dropout",
    "calibration_bias",
    "mechanical_issue",
    "actuator_overheat",
    "fall",
    "other",
}
SEVERITIES = {"minor", "moderate", "major", "critical"}
SCENARIOS = {
    "standing",
    "start_stop",
    "low_speed",
    "turn",
    "disturbance",
    "terrain",
    "other",
}
SAFETY_FIELDS = {
    "emergency_stop",
    "fall",
    "joint_limit_violation",
    "torque_limit_violation",
    "communication_timeout",
    "mechanical_damage",
    "operator_intervention",
}


def _load_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise SpecError(f"{label} file does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise SpecError(
            f"invalid {label} JSON at line {exc.lineno}: {exc.msg}"
        ) from exc
    if not isinstance(value, dict):
        raise SpecError(f"{label} must be a JSON object")
    return value


def _check_keys(value: dict[str, Any], expected: set[str], path: str) -> None:
    missing = sorted(expected - set(value))
    unknown = sorted(set(value) - expected)
    if missing:
        raise SpecError(f"{path} is missing field(s): {', '.join(missing)}")
    if unknown:
        raise SpecError(f"{path} contains unknown field(s): {', '.join(unknown)}")


def _string(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SpecError(f"{path} must be a non-empty string")
    return value


def _number(value: Any, path: str, minimum: float = 0.0) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < minimum
    ):
        raise SpecError(f"{path} must be a finite number >= {minimum}")
    return float(value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _expected_hash(value: Any, path: str) -> str:
    digest = _string(value, path)
    if not SHA256_RE.fullmatch(digest):
        raise SpecError(f"{path} must be a lowercase SHA-256")
    return digest


def _verified_file(value: Any, path: str) -> dict[str, str]:
    if not isinstance(value, dict):
        raise SpecError(f"{path} must be an object")
    _check_keys(value, {"path", "sha256"}, path)
    file_path = Path(_string(value["path"], f"{path}.path"))
    if (
        not file_path.is_absolute()
        or not file_path.is_file()
        or file_path.is_symlink()
    ):
        raise SpecError(f"{path}.path must be an existing absolute regular file")
    expected = _expected_hash(value["sha256"], f"{path}.sha256")
    if _sha256(file_path) != expected:
        raise SpecError(f"{path}.sha256 does not match the current file")
    return {"path": str(file_path), "sha256": expected}


def _verified_file_list(value: Any, path: str) -> list[dict[str, str]]:
    if not isinstance(value, list):
        raise SpecError(f"{path} must be an array")
    if len(value) > 64:
        raise SpecError(f"{path} may contain at most 64 files")
    files = [_verified_file(item, f"{path}[{index}]") for index, item in enumerate(value)]
    paths = [item["path"] for item in files]
    if len(paths) != len(set(paths)):
        raise SpecError(f"{path} contains duplicate paths")
    return files


def _validate_policy(
    value: Any,
    session: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not isinstance(value, dict):
        raise SpecError("policy must be an object")
    _check_keys(
        value,
        {
            "archive_manifest_path",
            "archive_manifest_sha256",
            "candidate_id",
            "artifacts",
        },
        "policy",
    )
    manifest_path = Path(
        _string(value["archive_manifest_path"], "policy.archive_manifest_path")
    )
    if (
        not manifest_path.is_absolute()
        or not manifest_path.is_file()
        or manifest_path.is_symlink()
    ):
        raise SpecError(
            "policy.archive_manifest_path must be an existing absolute regular file"
        )
    manifest_hash = _expected_hash(
        value["archive_manifest_sha256"],
        "policy.archive_manifest_sha256",
    )
    if _sha256(manifest_path) != manifest_hash:
        raise SpecError("policy archive manifest SHA-256 changed")
    manifest = _load_object(manifest_path, "policy archive manifest")
    if manifest.get("version") != 1:
        raise SpecError("policy archive manifest must be version 1")
    if manifest.get("hardware_ready") is not False:
        raise SpecError("policy archive manifest must not claim hardware readiness")
    candidate_id = _string(value["candidate_id"], "policy.candidate_id")
    if candidate_id != manifest.get("candidate_id"):
        raise SpecError("feedback candidate_id does not match the archive manifest")

    feedback_artifacts = value["artifacts"]
    if not isinstance(feedback_artifacts, dict):
        raise SpecError("policy.artifacts must be an object")
    _check_keys(feedback_artifacts, {"jit", "onnx"}, "policy.artifacts")
    manifest_artifacts = manifest.get("artifacts")
    if not isinstance(manifest_artifacts, dict):
        raise SpecError("policy archive manifest has no artifacts object")
    expected_names = {"jit": "policy.pt", "onnx": "policy.onnx"}
    normalized_artifacts: dict[str, str] = {}
    for kind, filename in expected_names.items():
        digest = _expected_hash(
            feedback_artifacts[kind],
            f"policy.artifacts.{kind}",
        )
        manifest_entry = manifest_artifacts.get(kind)
        if not isinstance(manifest_entry, dict) or manifest_entry.get("sha256") != digest:
            raise SpecError(
                f"policy.artifacts.{kind} does not match the archive manifest"
            )
        archived_artifact = manifest_path.parent / filename
        if (
            not archived_artifact.is_file()
            or archived_artifact.is_symlink()
            or _sha256(archived_artifact) != digest
        ):
            raise SpecError(f"archived {filename} is missing, linked, or changed")
        normalized_artifacts[kind] = digest

    manifest_algorithm = manifest.get("algorithm")
    if not isinstance(manifest_algorithm, dict):
        raise SpecError("policy archive manifest has no algorithm object")
    session_algorithm = session["algorithm"]
    for field in (
        "backend",
        "name",
        "runner_class",
        "profile_id",
        "profile_version",
        "profile_fingerprint",
    ):
        if manifest_algorithm.get(field) != session_algorithm.get(field):
            raise SpecError(
                f"policy archive algorithm {field} does not match the session"
            )
    normalized = {
        "archive_manifest_path": str(manifest_path),
        "archive_manifest_sha256": manifest_hash,
        "candidate_id": candidate_id,
        "artifacts": normalized_artifacts,
    }
    return normalized, manifest


def _validate_deployment(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise SpecError("deployment must be an object")
    _check_keys(
        value,
        {
            "runtime",
            "artifact_kind",
            "robot_id",
            "firmware",
            "control_frequency_hz",
            "config_files",
            "observation_contract_verified",
            "history_initialized",
            "emergency_stop_verified",
            "notes",
        },
        "deployment",
    )
    normalized = {
        "runtime": _string(value["runtime"], "deployment.runtime"),
        "artifact_kind": value["artifact_kind"],
        "robot_id": _string(value["robot_id"], "deployment.robot_id"),
        "firmware": _string(value["firmware"], "deployment.firmware"),
        "control_frequency_hz": _number(
            value["control_frequency_hz"],
            "deployment.control_frequency_hz",
            0.000001,
        ),
        "config_files": _verified_file_list(
            value["config_files"],
            "deployment.config_files",
        ),
        "notes": value["notes"],
    }
    if value["artifact_kind"] not in {"jit", "onnx"}:
        raise SpecError("deployment.artifact_kind must be jit or onnx")
    for field in (
        "observation_contract_verified",
        "history_initialized",
        "emergency_stop_verified",
    ):
        if not isinstance(value[field], bool):
            raise SpecError(f"deployment.{field} must be a boolean")
        normalized[field] = value[field]
    if not isinstance(value["notes"], str):
        raise SpecError("deployment.notes must be a string")
    return normalized


def _validate_test(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise SpecError("test must be an object")
    _check_keys(
        value,
        {
            "started_at",
            "operator",
            "supervision",
            "scenario",
            "surface",
            "payload_kg",
            "duration_seconds",
            "command_envelope",
        },
        "test",
    )
    started_at = _string(value["started_at"], "test.started_at")
    try:
        parsed = datetime.fromisoformat(started_at)
    except ValueError as exc:
        raise SpecError("test.started_at must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None:
        raise SpecError("test.started_at must include a timezone")
    if value["supervision"] != "supervised":
        raise SpecError("hardware feedback accepts only supervised tests")
    if value["scenario"] not in SCENARIOS:
        raise SpecError(f"test.scenario must be one of {sorted(SCENARIOS)}")
    envelope = value["command_envelope"]
    if not isinstance(envelope, dict):
        raise SpecError("test.command_envelope must be an object")
    _check_keys(
        envelope,
        {"max_linear_speed_mps", "max_yaw_rate_rps"},
        "test.command_envelope",
    )
    return {
        "started_at": started_at,
        "operator": _string(value["operator"], "test.operator"),
        "supervision": "supervised",
        "scenario": value["scenario"],
        "surface": _string(value["surface"], "test.surface"),
        "payload_kg": _number(value["payload_kg"], "test.payload_kg"),
        "duration_seconds": _number(
            value["duration_seconds"],
            "test.duration_seconds",
            0.000001,
        ),
        "command_envelope": {
            "max_linear_speed_mps": _number(
                envelope["max_linear_speed_mps"],
                "test.command_envelope.max_linear_speed_mps",
            ),
            "max_yaw_rate_rps": _number(
                envelope["max_yaw_rate_rps"],
                "test.command_envelope.max_yaw_rate_rps",
            ),
        },
    }


def _validate_observations(value: Any, duration_seconds: float) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise SpecError("observations must be a non-empty array")
    if len(value) > 256:
        raise SpecError("observations may contain at most 256 entries")
    normalized: list[dict[str, Any]] = []
    for index, item in enumerate(value):
        path = f"observations[{index}]"
        if not isinstance(item, dict):
            raise SpecError(f"{path} must be an object")
        _check_keys(
            item,
            {"symptom", "severity", "start_seconds", "end_seconds", "notes"},
            path,
        )
        if item["symptom"] not in SYMPTOMS:
            raise SpecError(f"{path}.symptom must be one of {sorted(SYMPTOMS)}")
        if item["severity"] not in SEVERITIES:
            raise SpecError(f"{path}.severity must be one of {sorted(SEVERITIES)}")
        start = _number(item["start_seconds"], f"{path}.start_seconds")
        end = _number(item["end_seconds"], f"{path}.end_seconds")
        if end < start or end > duration_seconds:
            raise SpecError(
                f"{path} time range must be ordered and within test duration"
            )
        notes = item["notes"]
        if not isinstance(notes, str):
            raise SpecError(f"{path}.notes must be a string")
        normalized.append(
            {
                "symptom": item["symptom"],
                "severity": item["severity"],
                "start_seconds": start,
                "end_seconds": end,
                "notes": notes,
            }
        )
    return normalized


def _validate_safety(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise SpecError("safety must be an object")
    _check_keys(value, SAFETY_FIELDS | {"notes"}, "safety")
    normalized: dict[str, Any] = {}
    for field in sorted(SAFETY_FIELDS):
        if not isinstance(value[field], bool):
            raise SpecError(f"safety.{field} must be a boolean")
        normalized[field] = value[field]
    if not isinstance(value["notes"], str):
        raise SpecError("safety.notes must be a string")
    normalized["notes"] = value["notes"]
    return normalized


def _validate_evidence(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise SpecError("evidence must be an object")
    _check_keys(
        value,
        {
            "video_files",
            "telemetry_files",
            "telemetry_channels",
            "sample_rate_hz",
            "clock_synchronized",
        },
        "evidence",
    )
    video_files = _verified_file_list(value["video_files"], "evidence.video_files")
    telemetry_files = _verified_file_list(
        value["telemetry_files"],
        "evidence.telemetry_files",
    )
    channels = value["telemetry_channels"]
    if not isinstance(channels, list) or any(
        not isinstance(channel, str) or not channel.strip()
        for channel in channels
    ):
        raise SpecError("evidence.telemetry_channels must be an array of strings")
    if len(channels) != len(set(channels)):
        raise SpecError("evidence.telemetry_channels must be unique")
    sample_rate = value["sample_rate_hz"]
    if telemetry_files:
        if not channels:
            raise SpecError("telemetry evidence requires channel names")
        sample_rate = _number(
            sample_rate,
            "evidence.sample_rate_hz",
            0.000001,
        )
    elif sample_rate is not None:
        raise SpecError("sample_rate_hz must be null without telemetry files")
    if not isinstance(value["clock_synchronized"], bool):
        raise SpecError("evidence.clock_synchronized must be a boolean")
    return {
        "video_files": video_files,
        "telemetry_files": telemetry_files,
        "telemetry_channels": channels,
        "sample_rate_hz": sample_rate,
        "clock_synchronized": value["clock_synchronized"],
    }


def load_and_validate_feedback(
    session_path: str | Path,
    feedback_path: str | Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Load a version-5-or-6 session and feedback bound to one archived policy."""
    session = load_and_validate(session_path)
    contract = session.get("hardware_feedback")
    if not isinstance(contract, dict) or not contract.get("enabled"):
        raise SpecError("session hardware feedback processing is not enabled")
    feedback = _load_object(Path(feedback_path), "hardware feedback")
    _check_keys(
        feedback,
        {
            "version",
            "feedback_id",
            "policy",
            "deployment",
            "test",
            "observations",
            "safety",
            "evidence",
            "user_assessment",
        },
        "hardware_feedback",
    )
    if feedback["version"] != 1:
        raise SpecError("hardware feedback version must be 1")
    feedback_id = _string(feedback["feedback_id"], "feedback_id")
    policy, manifest = _validate_policy(feedback["policy"], session)
    deployment = _validate_deployment(feedback["deployment"])
    test = _validate_test(feedback["test"])
    observations = _validate_observations(
        feedback["observations"],
        test["duration_seconds"],
    )
    safety = _validate_safety(feedback["safety"])
    if (
        any(item["symptom"] == "fall" for item in observations)
        and not safety["fall"]
    ):
        raise SpecError("a fall observation requires safety.fall=true")
    evidence = _validate_evidence(feedback["evidence"])
    assessment = feedback["user_assessment"]
    if not isinstance(assessment, dict):
        raise SpecError("user_assessment must be an object")
    _check_keys(assessment, {"overall", "notes"}, "user_assessment")
    if assessment["overall"] not in {"pass", "mixed", "fail", "unsafe"}:
        raise SpecError(
            "user_assessment.overall must be pass, mixed, fail, or unsafe"
        )
    if not isinstance(assessment["notes"], str):
        raise SpecError("user_assessment.notes must be a string")
    normalized = {
        "version": 1,
        "feedback_id": feedback_id,
        "policy": policy,
        "deployment": deployment,
        "test": test,
        "observations": observations,
        "safety": safety,
        "evidence": evidence,
        "user_assessment": assessment,
    }
    return session, normalized, manifest


def validation_report(feedback: dict[str, Any]) -> dict[str, Any]:
    """Summarize evidence strength without interpreting root cause."""
    evidence = feedback["evidence"]
    has_video = bool(evidence["video_files"])
    has_telemetry = bool(evidence["telemetry_files"])
    channels = set(evidence["telemetry_channels"])
    has_core_telemetry = (
        {"action", "control_timestamp"} <= channels
        and bool(
            channels
            & {
                "imu_roll",
                "imu_pitch",
                "base_angular_velocity",
                "joint_position",
                "joint_velocity",
            }
        )
    )
    if (
        has_video
        and has_telemetry
        and evidence["clock_synchronized"]
        and has_core_telemetry
    ):
        confidence = "high"
    elif has_video or has_telemetry:
        confidence = "medium"
    else:
        confidence = "low"
    safety_events = sorted(
        field for field in SAFETY_FIELDS if feedback["safety"][field]
    )
    critical_observation = any(
        item["severity"] == "critical" for item in feedback["observations"]
    )
    return {
        "version": 1,
        "status": "valid",
        "feedback_id": feedback["feedback_id"],
        "candidate_id": feedback["policy"]["candidate_id"],
        "artifact_hashes_verified": True,
        "evidence_confidence": confidence,
        "has_video": has_video,
        "has_telemetry": has_telemetry,
        "has_core_telemetry": has_core_telemetry,
        "safety_events": safety_events,
        "critical_observation": critical_observation,
        "hardware_ready": False,
    }


def authorized_output_path(
    session: dict[str, Any],
    value: str,
) -> Path:
    """Resolve one new JSON output beneath the approved feedback directory."""
    output = Path(value)
    root = Path(session["hardware_feedback"]["output_dir"])
    if not output.is_absolute():
        raise SpecError("feedback output path must be absolute")
    if output.suffix.lower() != ".json":
        raise SpecError("feedback output path must use a .json suffix")
    if root.exists() and (not root.is_dir() or root.is_symlink()):
        raise SpecError("hardware_feedback.output_dir must be a regular directory")
    resolved_root = root.resolve(strict=False)
    resolved_output = output.resolve(strict=False)
    try:
        relative = resolved_output.relative_to(resolved_root)
    except ValueError as exc:
        raise SpecError(
            "feedback output path escapes hardware_feedback.output_dir"
        ) from exc
    if not relative.parts:
        raise SpecError("feedback output path must name a JSON file")
    if output.exists():
        raise SpecError("feedback output path already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session", help="Validated version-5-or-6 session JSON")
    parser.add_argument("feedback", help="Physical deployment feedback JSON")
    parser.add_argument("--output", help="Optional validation report path")
    args = parser.parse_args()
    try:
        session, feedback, _ = load_and_validate_feedback(
            args.session,
            args.feedback,
        )
        report = validation_report(feedback)
        output_path = (
            authorized_output_path(session, args.output)
            if args.output
            else None
        )
    except SpecError as exc:
        parser.error(str(exc))
    encoded = json.dumps(
        report,
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
    )
    if output_path is not None:
        output_path.write_text(encoded + "\n", encoding="utf-8")
    else:
        print(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
