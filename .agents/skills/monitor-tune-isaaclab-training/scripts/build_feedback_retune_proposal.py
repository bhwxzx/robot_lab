#!/usr/bin/env python3
"""Build a non-executable tuning proposal from validated hardware feedback."""

from __future__ import annotations

import argparse
import json
from typing import Any

from validate_hardware_feedback import (
    authorized_output_path,
    load_and_validate_feedback,
    validation_report,
)
from validate_session_spec import SpecError


DEPLOYMENT_SYMPTOMS = {
    "deployment_output_mismatch",
    "startup_state_error",
    "control_timing_jitter",
    "communication_dropout",
}
HARDWARE_SYMPTOMS = {
    "calibration_bias",
    "mechanical_issue",
    "actuator_overheat",
}
SYMPTOM_CATEGORIES = {
    "standing_roll_oscillation": {
        "stability",
        "action_smoothness",
        "sim2real_dynamics",
    },
    "standing_pitch_oscillation": {
        "stability",
        "action_smoothness",
        "sim2real_dynamics",
    },
    "action_chatter": {"action_smoothness", "sim2real_dynamics"},
    "tracking_lag": {"command_tracking", "sim2real_dynamics"},
    "tracking_overshoot": {
        "command_tracking",
        "action_smoothness",
        "sim2real_dynamics",
    },
    "turn_instability": {"stability", "command_tracking"},
    "foot_slip": {"contact_robustness", "sim2real_dynamics"},
    "unexpected_contact": {"contact_robustness", "safety_margin"},
    "recovery_failure": {"recovery", "sim2real_dynamics"},
    "joint_or_torque_margin": {"safety_margin", "action_smoothness"},
    "fall": {"stability", "safety_margin", "recovery"},
    "other": {"task_specific_investigation"},
}
CATEGORY_TOKENS = {
    "stability": ("orientation", "tilt", "roll", "pitch", "base", "ang_vel"),
    "action_smoothness": (
        "action_rate",
        "smooth",
        "jerk",
        "acceleration",
        "torque",
    ),
    "command_tracking": (
        "tracking",
        "command",
        "velocity",
        "lin_vel",
        "ang_vel",
    ),
    "sim2real_dynamics": (
        "friction",
        "mass",
        "motor",
        "actuator",
        "gain",
        "delay",
        "latency",
        "random",
    ),
    "contact_robustness": ("contact", "slip", "feet", "foot"),
    "recovery": ("termination", "disturbance", "push", "recovery"),
    "safety_margin": (
        "joint",
        "torque",
        "action_scale",
        "velocity_limit",
    ),
}
CATEGORY_SCENARIOS = {
    "stability": ["standing", "start_stop", "turn"],
    "action_smoothness": ["standing", "start_stop", "low_speed"],
    "command_tracking": ["start_stop", "low_speed", "turn"],
    "sim2real_dynamics": ["latency", "dynamics", "friction"],
    "contact_robustness": ["terrain", "friction"],
    "recovery": ["disturbance", "recovery"],
    "safety_margin": ["joint_and_torque_margin"],
    "task_specific_investigation": ["reproduce_reported_segment"],
}
CATEGORY_METRICS = {
    "stability": ["tilt_rms", "max_tilt", "base_angular_velocity"],
    "action_smoothness": ["action_rate_rms", "action_jerk", "torque_rate"],
    "command_tracking": ["tracking_xy_rmse", "tracking_yaw_rmse", "response_time"],
    "sim2real_dynamics": ["cross_domain_gate_pass_rate", "timing_jitter"],
    "contact_robustness": ["foot_slip", "unexpected_contact_rate"],
    "recovery": ["recovery_success_rate", "recovery_time", "termination_rate"],
    "safety_margin": ["joint_limit_margin", "torque_margin", "max_abs_action"],
    "task_specific_investigation": ["user_defined_reproduction_metric"],
}


def _root_cause(
    feedback: dict[str, Any],
    report: dict[str, Any],
) -> tuple[str, list[str]]:
    symptoms = {item["symptom"] for item in feedback["observations"]}
    deployment = feedback["deployment"]
    assessment = feedback["user_assessment"]["overall"]
    if (
        report["safety_events"]
        or report["critical_observation"]
        or assessment == "unsafe"
    ):
        return (
            "hardware_safety_incident",
            [
                "stop further physical tests",
                "preserve logs, telemetry, video, and robot state",
                "diagnose hardware, calibration, limits, and deployment runtime",
            ],
        )
    unverified = [
        field
        for field in (
            "observation_contract_verified",
            "history_initialized",
            "emergency_stop_verified",
        )
        if not deployment[field]
    ]
    if unverified or not deployment["config_files"]:
        return (
            "deployment_contract_incomplete",
            [
                "verify observation order, units, normalization, and history reset",
                "bind and hash the deployed configuration",
                "verify control frequency, emergency stop, and communication timeout",
            ],
        )
    if symptoms & DEPLOYMENT_SYMPTOMS:
        return (
            "deployment_runtime_or_tensor_path",
            [
                "compare Native, JIT, and ONNX actions on recorded observations",
                "inspect timing, state reset, normalization, and history",
                "reproduce the reported segment without changing training",
            ],
        )
    if symptoms & HARDWARE_SYMPTOMS:
        return (
            "hardware_or_calibration",
            [
                "inspect actuator, sensor, mechanism, and calibration evidence",
                "repeat only after the physical cause is resolved",
            ],
        )
    if assessment == "pass":
        return (
            "no_retune_requested",
            ["retain the feedback as qualification evidence"],
        )
    if report["evidence_confidence"] == "low":
        return (
            "insufficient_evidence_for_retune",
            [
                "collect time-aligned video or telemetry for the reported segment",
                "reproduce the symptom under the same supervised command envelope",
            ],
        )
    return (
        "training_or_sim2real_candidate",
        [
            "reproduce the reported segment in closed-loop simulation",
            "add a measurable signature and hard safety constraints",
            "compare unchanged baseline and bounded candidates across fixed seeds",
        ],
    )


def _parameter_categories(feedback: dict[str, Any]) -> list[str]:
    categories: set[str] = set()
    for observation in feedback["observations"]:
        categories.update(SYMPTOM_CATEGORIES.get(observation["symptom"], set()))
    return sorted(categories)


def _eligible_parameters(
    session: dict[str, Any],
    categories: list[str],
) -> list[dict[str, Any]]:
    if session["mode"] != "tune":
        return []
    tuning = session["tuning"]
    eligible: list[dict[str, Any]] = []
    for parameter in tuning["allowed_parameters"]:
        lowered = parameter["path"].lower()
        matched = sorted(
            category
            for category in categories
            if any(token in lowered for token in CATEGORY_TOKENS.get(category, ()))
        )
        if matched:
            eligible.append(
                {
                    "path": parameter["path"],
                    "authorized_domain": {
                        key: parameter[key]
                        for key in ("values", "range", "baseline")
                        if key in parameter
                    },
                    "matched_categories": matched,
                }
            )
    return eligible


def build_proposal(
    session: dict[str, Any],
    feedback: dict[str, Any],
) -> dict[str, Any]:
    """Create analysis and an optional pending-approval parameter-choice draft."""
    report = validation_report(feedback)
    root_cause, investigations = _root_cause(feedback, report)
    categories = _parameter_categories(feedback)
    eligible = _eligible_parameters(session, categories)
    scenarios = sorted(
        {
            scenario
            for category in categories
            for scenario in CATEGORY_SCENARIOS.get(category, [])
        }
    )
    metrics = sorted(
        {
            metric
            for category in categories
            for metric in CATEGORY_METRICS.get(category, [])
        }
    )
    retune_candidate = root_cause == "training_or_sim2real_candidate"
    contract = session["hardware_feedback"]
    draft_allowed = (
        retune_candidate
        and contract["output_mode"] == "prepare_authorized_draft"
        and session["mode"] == "tune"
        and bool(eligible)
    )
    blockers: list[str] = []
    if not retune_candidate:
        blockers.append(root_cause)
    if retune_candidate and not eligible:
        blockers.append("no_relevant_previously_authorized_parameter")
    if session["mode"] != "tune":
        blockers.append("session_has_no_tuning_authority")

    proposal: dict[str, Any] = {
        "version": 1,
        "feedback_id": feedback["feedback_id"],
        "candidate_id": feedback["policy"]["candidate_id"],
        "algorithm": session["algorithm"],
        "classification": {
            "root_cause": root_cause,
            "evidence_confidence": report["evidence_confidence"],
            "symptoms": sorted(
                {item["symptom"] for item in feedback["observations"]}
            ),
            "parameter_categories": categories,
        },
        "recommended_investigations": investigations,
        "proposed_simulation_scenarios": scenarios,
        "proposed_metrics": metrics,
        "eligible_existing_parameters": eligible,
        "retune": {
            "may_prepare_authorization_draft": draft_allowed,
            "blockers": sorted(set(blockers)),
            "new_parameter_authority_granted": False,
            "new_trial_authority_granted": False,
        },
        "safety": {
            "stop_physical_testing": root_cause == "hardware_safety_incident",
            "events": report["safety_events"],
        },
        "executable": False,
        "requires_user_approval": True,
        "hardware_ready": False,
    }
    if contract["output_mode"] == "prepare_authorized_draft":
        proposal["authorization_draft"] = {
            "authorization_state": "pending_user_selection_and_approval",
            "executable": False,
            "source_feedback_id": feedback["feedback_id"],
            "retained_profile_id": session["algorithm"]["profile_id"],
            "retained_profile_fingerprint": session["algorithm"][
                "profile_fingerprint"
            ],
            "parameter_options": eligible,
            "selected_parameter_paths": [],
            "proposed_scenarios": scenarios,
            "proposed_metrics": metrics,
            "required_user_decisions": [
                "selected_parameter_paths",
                "exact_parameter_domains",
                "objectives_and_weights",
                "hard_constraints",
                "seeds_and_trial_budget",
                "evaluation_scenarios_and_gates",
            ],
            "require_new_session_approval": True,
        }
    return proposal


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session", help="Validated version-5-or-6 session JSON")
    parser.add_argument("feedback", help="Validated physical feedback JSON")
    parser.add_argument("--output", help="Optional proposal JSON path")
    args = parser.parse_args()
    try:
        session, feedback, _ = load_and_validate_feedback(
            args.session,
            args.feedback,
        )
        proposal = build_proposal(session, feedback)
        output_path = (
            authorized_output_path(session, args.output)
            if args.output
            else None
        )
    except SpecError as exc:
        parser.error(str(exc))
    encoded = json.dumps(
        proposal,
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
