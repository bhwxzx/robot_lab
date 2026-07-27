#!/usr/bin/env python3
"""Qualify one exact policy only for its supervised physical-test envelope."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from validate_hardware_feedback import (
    SAFETY_FIELDS,
    authorized_output_path,
    load_and_validate_feedback,
    validation_report,
)
from validate_session_spec import SpecError


def _load_bundle(path: str | Path) -> dict[str, Any]:
    bundle_path = Path(path)
    try:
        value = json.loads(bundle_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise SpecError(f"qualification bundle does not exist: {bundle_path}") from exc
    except json.JSONDecodeError as exc:
        raise SpecError(
            f"invalid qualification bundle JSON at line {exc.lineno}: {exc.msg}"
        ) from exc
    if not isinstance(value, dict) or set(value) != {
        "version",
        "qualification_id",
        "feedback_paths",
    }:
        raise SpecError(
            "qualification bundle must contain version, qualification_id, "
            "and feedback_paths"
        )
    if value["version"] != 1:
        raise SpecError("qualification bundle version must be 1")
    if (
        not isinstance(value["qualification_id"], str)
        or not value["qualification_id"].strip()
    ):
        raise SpecError("qualification_id must be a non-empty string")
    paths = value["feedback_paths"]
    if (
        not isinstance(paths, list)
        or not paths
        or len(paths) > 1000
        or any(not isinstance(item, str) or not Path(item).is_absolute() for item in paths)
        or len(paths) != len(set(paths))
    ):
        raise SpecError(
            "feedback_paths must be a non-empty unique array of absolute paths"
        )
    return value


def _deployment_identity(feedback: dict[str, Any]) -> dict[str, Any]:
    deployment = feedback["deployment"]
    return {
        "candidate_id": feedback["policy"]["candidate_id"],
        "archive_manifest_sha256": feedback["policy"]["archive_manifest_sha256"],
        "artifacts": feedback["policy"]["artifacts"],
        "runtime": deployment["runtime"],
        "artifact_kind": deployment["artifact_kind"],
        "robot_id": deployment["robot_id"],
        "firmware": deployment["firmware"],
        "control_frequency_hz": deployment["control_frequency_hz"],
        "config_files": deployment["config_files"],
    }


def qualify(
    session_path: str | Path,
    bundle_path: str | Path,
) -> dict[str, Any]:
    """Validate repeated evidence and emit a deliberately bounded status."""
    bundle = _load_bundle(bundle_path)
    session: dict[str, Any] | None = None
    feedback_items: list[dict[str, Any]] = []
    reports: list[dict[str, Any]] = []
    feedback_ids: set[str] = set()
    identity: dict[str, Any] | None = None

    for feedback_path in bundle["feedback_paths"]:
        current_session, feedback, _ = load_and_validate_feedback(
            session_path,
            feedback_path,
        )
        if session is None:
            session = current_session
        if feedback["feedback_id"] in feedback_ids:
            raise SpecError(
                f"duplicate feedback_id: {feedback['feedback_id']}"
            )
        feedback_ids.add(feedback["feedback_id"])
        current_identity = _deployment_identity(feedback)
        if identity is None:
            identity = current_identity
        elif current_identity != identity:
            raise SpecError(
                "all qualification tests must use the exact same policy, "
                "artifacts, robot, runtime, firmware, control frequency, and configs"
            )
        feedback_items.append(feedback)
        reports.append(validation_report(feedback))

    assert session is not None
    assert identity is not None
    contract = session["hardware_feedback"].get("qualification")
    if not isinstance(contract, dict) or not contract.get("enabled"):
        raise SpecError("session does not authorize hardware qualification")
    failures: list[str] = []
    started_at_values = [item["test"]["started_at"] for item in feedback_items]
    if len(started_at_values) != len(set(started_at_values)):
        failures.append("test.started_at values must be unique")
    evidence_paths = [
        evidence_file["path"]
        for item in feedback_items
        for field in ("video_files", "telemetry_files")
        for evidence_file in item["evidence"][field]
    ]
    if len(evidence_paths) != len(set(evidence_paths)):
        failures.append("video and telemetry evidence files cannot be reused")
    if len(feedback_items) < contract["minimum_total_tests"]:
        failures.append(
            f"test count {len(feedback_items)} is below "
            f"{contract['minimum_total_tests']}"
        )
    scenario_counts = Counter(
        feedback["test"]["scenario"] for feedback in feedback_items
    )
    for scenario in contract["required_scenarios"]:
        actual = scenario_counts.get(scenario, 0)
        if actual < contract["minimum_tests_per_scenario"]:
            failures.append(
                f"scenario {scenario} has {actual} test(s), requires "
                f"{contract['minimum_tests_per_scenario']}"
            )
    required_channels = set(contract["required_telemetry_channels"])
    for feedback, report in zip(feedback_items, reports, strict=True):
        feedback_id = feedback["feedback_id"]
        for field in (
            "observation_contract_verified",
            "history_initialized",
            "emergency_stop_verified",
        ):
            if not feedback["deployment"][field]:
                failures.append(
                    f"{feedback_id} deployment.{field} is not verified"
                )
        if feedback["user_assessment"]["overall"] != "pass":
            failures.append(f"{feedback_id} assessment is not pass")
        safety_events = [
            field for field in SAFETY_FIELDS if feedback["safety"][field]
        ]
        if safety_events:
            failures.append(
                f"{feedback_id} has safety events: {', '.join(sorted(safety_events))}"
            )
        severe = [
            item["severity"]
            for item in feedback["observations"]
            if item["severity"] in {"major", "critical"}
        ]
        if severe:
            failures.append(f"{feedback_id} has major or critical observations")
        if report["evidence_confidence"] != "high":
            failures.append(f"{feedback_id} evidence confidence is not high")
        missing_channels = sorted(
            required_channels - set(feedback["evidence"]["telemetry_channels"])
        )
        if missing_channels:
            failures.append(
                f"{feedback_id} missing telemetry channels: "
                f"{', '.join(missing_channels)}"
            )

    tests = [feedback["test"] for feedback in feedback_items]
    envelope = {
        "scenarios": sorted(scenario_counts),
        "scenario_test_counts": dict(sorted(scenario_counts.items())),
        "surfaces": sorted({item["surface"] for item in tests}),
        "payload_kg": {
            "min": min(item["payload_kg"] for item in tests),
            "max": max(item["payload_kg"] for item in tests),
        },
        "duration_seconds": {
            "total": sum(item["duration_seconds"] for item in tests),
            "min": min(item["duration_seconds"] for item in tests),
            "max": max(item["duration_seconds"] for item in tests),
        },
        "max_linear_speed_mps": max(
            item["command_envelope"]["max_linear_speed_mps"] for item in tests
        ),
        "max_yaw_rate_rps": max(
            item["command_envelope"]["max_yaw_rate_rps"] for item in tests
        ),
    }
    qualified = not failures
    return {
        "version": 1,
        "qualification_id": bundle["qualification_id"],
        "status": (
            "hardware_validated_for_test_envelope"
            if qualified
            else "hardware_qualification_failed"
        ),
        "qualified": qualified,
        "candidate_id": identity["candidate_id"],
        "exact_deployment_identity": identity,
        "test_count": len(feedback_items),
        "feedback_ids": sorted(feedback_ids),
        "tested_envelope": envelope,
        "failures": failures,
        "hardware_ready": False,
        "generalization_claim": False,
        "scope_notice": (
            "Qualification applies only to the exact artifact and tested "
            "physical envelope recorded in this report."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session", help="Validated session JSON")
    parser.add_argument("bundle", help="Hardware qualification bundle JSON")
    parser.add_argument("--output", help="Optional qualification report path")
    args = parser.parse_args()
    try:
        report = qualify(args.session, args.bundle)
        if args.output:
            session, _, _ = load_and_validate_feedback(
                args.session,
                _load_bundle(args.bundle)["feedback_paths"][0],
            )
            output = authorized_output_path(session, args.output)
            output.write_text(
                json.dumps(
                    report,
                    indent=2,
                    sort_keys=True,
                    ensure_ascii=False,
                    allow_nan=False,
                )
                + "\n",
                encoding="utf-8",
            )
        else:
            print(
                json.dumps(
                    report,
                    indent=2,
                    sort_keys=True,
                    ensure_ascii=False,
                    allow_nan=False,
                )
            )
    except SpecError as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
