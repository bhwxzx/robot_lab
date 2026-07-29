#!/usr/bin/env python3
"""Build a hash-bound, human-readable visual-review bundle without copying videos."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from pathlib import Path
from typing import Any

from execute_evaluation_plan import _validate_plan_against_spec
from validate_policy_evaluation import load_evaluation_plan
from validate_session_spec import SpecError, load_and_validate


SAFE_COMPONENT_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise SpecError(f"review evidence is not finite JSON: {exc}") from exc


def _load_object(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise SpecError(f"{label} must be a regular file: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SpecError(
            f"invalid {label} JSON at line {exc.lineno}: {exc.msg}"
        ) from exc
    if not isinstance(value, dict):
        raise SpecError(f"{label} must be a JSON object")
    _canonical_bytes(value)
    return value


def _inside(path: Path, root: Path, label: str) -> None:
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise SpecError(f"{label} must be inside evaluation.output_dir") from exc


def _safe_component(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or SAFE_COMPONENT_RE.fullmatch(value) is None
    ):
        raise SpecError(f"{label} is not safe for a review filename")
    if len(value) <= 48:
        return value
    suffix = hashlib.sha256(value.encode("utf-8")).hexdigest()[:8]
    return f"{value[:39]}-{suffix}"


def _alias_name(index: int, run: dict[str, Any]) -> str:
    candidate = _safe_component(run.get("candidate_id"), "candidate_id")
    artifact = _safe_component(run.get("artifact"), "artifact")
    scenario = _safe_component(run.get("scenario_id"), "scenario_id")
    seed = run.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise SpecError("seed is not safe for a review filename")
    name = (
        f"{index:03d}__{candidate}__{artifact}__"
        f"{scenario}__seed-{seed}.mp4"
    )
    if len(name.encode("utf-8")) > 240:
        raise SpecError("review alias filename exceeds the safe length limit")
    return name


def _validated_digest(value: Any, label: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise SpecError(f"{label} must be a lowercase SHA-256")
    return value


def _finite_number(value: Any, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise SpecError(f"{label} must be a finite number")
    return float(value)


def _validate_review_data(
    run_id: str,
    metrics: dict[str, Any],
    motion_evidence: dict[str, Any],
) -> None:
    for name, value in metrics.items():
        if not isinstance(name, str) or not name:
            raise SpecError(f"{run_id} contains an invalid metric name")
        _finite_number(value, f"{run_id} metric {name}")
    windows = motion_evidence.get("review_windows")
    if not isinstance(windows, list):
        raise SpecError(f"{run_id} review_windows must be an array")
    for index, window in enumerate(windows):
        label = f"{run_id} review_windows[{index}]"
        if not isinstance(window, dict):
            raise SpecError(f"{label} must be an object")
        start_step = window.get("start_step")
        end_step = window.get("end_step")
        if (
            isinstance(start_step, bool)
            or not isinstance(start_step, int)
            or isinstance(end_step, bool)
            or not isinstance(end_step, int)
            or start_step < 0
            or end_step < start_step
        ):
            raise SpecError(f"{label} contains an invalid step range")
        start_seconds = _finite_number(
            window.get("start_seconds"),
            f"{label}.start_seconds",
        )
        end_seconds = _finite_number(
            window.get("end_seconds"),
            f"{label}.end_seconds",
        )
        if start_seconds < 0 or end_seconds < start_seconds:
            raise SpecError(f"{label} contains an invalid time range")
        evidence = window.get("evidence")
        if (
            not isinstance(evidence, list)
            or any(not isinstance(item, str) or not item for item in evidence)
        ):
            raise SpecError(f"{label}.evidence must contain non-empty strings")


def _validate_result(
    run: dict[str, Any],
    state_run: dict[str, Any],
    video_sha256: str,
) -> tuple[dict[str, Any], str]:
    result_path = Path(run["result_path"])
    if result_path != Path(state_run.get("canonical_result_path", "")):
        raise SpecError(f"{run['run_id']} canonical result path changed")
    result_sha256 = _validated_digest(
        state_run.get("result_sha256"),
        f"{run['run_id']} result_sha256",
    )
    if (
        not result_path.is_file()
        or result_path.is_symlink()
        or _sha256(result_path) != result_sha256
    ):
        raise SpecError(f"{run['run_id']} canonical result hash is invalid")
    result = _load_object(result_path, f"{run['run_id']} result")
    expected_identity = {
        "run_id": run["run_id"],
        "candidate_id": run["candidate_id"],
        "artifact": run["artifact"],
        "scenario_id": run["scenario_id"],
        "seed": run["seed"],
        "status": "completed",
        "video_path": run["video_path"],
    }
    for field, expected in expected_identity.items():
        if result.get(field) != expected:
            raise SpecError(f"{run['run_id']} result {field} changed")
    evidence = result.get("execution_evidence")
    if (
        not isinstance(evidence, dict)
        or evidence.get("video_sha256") != video_sha256
    ):
        raise SpecError(f"{run['run_id']} result video evidence changed")
    metrics = result.get("metrics")
    motion_evidence = result.get("motion_evidence")
    if not isinstance(metrics, dict) or not metrics:
        raise SpecError(f"{run['run_id']} result metrics are missing")
    if not isinstance(motion_evidence, dict):
        raise SpecError(f"{run['run_id']} motion evidence is missing")
    _canonical_bytes(metrics)
    _canonical_bytes(motion_evidence)
    _validate_review_data(run["run_id"], metrics, motion_evidence)
    return result, result_sha256


def _validate_review_inputs(
    spec: dict[str, Any],
    session_path: Path,
    plan: dict[str, Any],
    plan_path: Path,
    state: dict[str, Any],
    state_path: Path,
) -> list[dict[str, Any]]:
    _validate_plan_against_spec(spec, plan)
    if state.get("version") != 1:
        raise SpecError("evaluation state must be version 1")
    if state.get("stage") != "awaiting_visual_review":
        raise SpecError(
            "visual-review bundle requires stage awaiting_visual_review"
        )
    if (
        state.get("session_sha256") != _sha256(session_path)
        or state.get("plan_sha256") != _sha256(plan_path)
        or state.get("algorithm") != spec["algorithm"]
        or state.get("training_run_id") != spec["training"]["run_id"]
    ):
        raise SpecError("evaluation state is not bound to this session and plan")
    expected_state_path = Path(spec["evaluation"]["execution"]["state_dir"]) / (
        "evaluation_state.json"
    )
    if state_path != expected_state_path:
        raise SpecError("execution state path differs from the approved session")
    state_runs = state.get("runs")
    if not isinstance(state_runs, dict):
        raise SpecError("evaluation state runs must be an object")
    plan_run_ids = {run["run_id"] for run in plan["runs"]}
    if set(state_runs) != plan_run_ids:
        raise SpecError("evaluation state run set differs from the plan")
    if any(
        not isinstance(state_runs[run_id], dict)
        or state_runs[run_id].get("status") != "completed"
        for run_id in plan_run_ids
    ):
        raise SpecError("every evaluation run must be completed before review")

    output_root = Path(spec["evaluation"]["output_dir"])
    required_paths = set(plan["required_videos"])
    required_runs = [
        run for run in plan["runs"] if run.get("video_required")
    ]
    if {run["video_path"] for run in required_runs} != required_paths:
        raise SpecError("required video list differs from required matrix runs")
    minimum_bytes = spec["evaluation"]["execution"]["minimum_video_bytes"]
    entries: list[dict[str, Any]] = []
    for index, run in enumerate(required_runs, start=1):
        state_run = state_runs[run["run_id"]]
        video_path = Path(run["video_path"])
        _inside(video_path, output_root, f"{run['run_id']} video")
        if video_path != Path(state_run.get("canonical_video_path", "")):
            raise SpecError(f"{run['run_id']} canonical video path changed")
        video_sha256 = _validated_digest(
            state_run.get("video_sha256"),
            f"{run['run_id']} video_sha256",
        )
        if (
            not video_path.is_file()
            or video_path.is_symlink()
            or video_path.stat().st_size < minimum_bytes
            or _sha256(video_path) != video_sha256
        ):
            raise SpecError(f"{run['run_id']} canonical video hash is invalid")
        result, result_sha256 = _validate_result(
            run,
            state_run,
            video_sha256,
        )
        entries.append(
            {
                "index": index,
                "run_id": run["run_id"],
                "candidate_id": run["candidate_id"],
                "artifact": run["artifact"],
                "scenario_id": run["scenario_id"],
                "scenario_category": run["scenario_category"],
                "seed": run["seed"],
                "duration_steps": run["duration_steps"],
                "command_schedule": run["command_schedule"],
                "overrides": run["overrides"],
                "canonical_video_path": str(video_path),
                "video_sha256": video_sha256,
                "video_size_bytes": video_path.stat().st_size,
                "canonical_result_path": run["result_path"],
                "result_sha256": result_sha256,
                "metrics": result["metrics"],
                "motion_evidence": result["motion_evidence"],
                "alias_name": _alias_name(index, run),
            }
        )
    return entries


def _format_number(value: Any) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _index_markdown(manifest: dict[str, Any]) -> str:
    lines = [
        "# Policy visual review index",
        "",
        "This bundle contains links to hash-verified canonical videos. "
        "It does not copy or replace the originals.",
        "",
        "## Required videos",
        "",
        "| # | Video | Candidate | Artifact | Scenario | Category | Seed | Steps |",
        "|---:|---|---|---|---|---|---:|---:|",
    ]
    for entry in manifest["videos"]:
        alias = entry["alias_name"]
        lines.append(
            f"| {entry['index']} | [{alias}](required_videos/{alias}) | "
            f"{entry['candidate_id']} | {entry['artifact']} | "
            f"{entry['scenario_id']} | {entry['scenario_category']} | "
            f"{entry['seed']} | {entry['duration_steps']} |"
        )
    lines.extend(
        [
            "",
            "## Review details",
            "",
        ]
    )
    for entry in manifest["videos"]:
        lines.extend(
            [
                f"### {entry['index']:03d} — {entry['candidate_id']} / "
                f"{entry['artifact']} / {entry['scenario_id']} / "
                f"seed-{entry['seed']}",
                "",
                f"- Canonical video: `{entry['canonical_video_path']}`",
                f"- Video SHA-256: `{entry['video_sha256']}`",
                f"- Canonical result: `{entry['canonical_result_path']}`",
                "- Command schedule:",
            ]
        )
        schedule = entry["command_schedule"]
        if schedule:
            for segment in schedule:
                command = ", ".join(
                    _format_number(component)
                    for component in segment["command"]
                )
                lines.append(
                    f"  - steps {segment['start_step']}–"
                    f"{segment['end_step']}: [{command}]"
                )
        else:
            lines.append("  - backend-defined or unscheduled")
        lines.append(
            "- Scenario overrides: "
            f"`{json.dumps(entry['overrides'], sort_keys=True, ensure_ascii=False)}`"
        )
        lines.append("- Metrics:")
        for name, value in sorted(entry["metrics"].items()):
            lines.append(f"  - `{name}`: {_format_number(value)}")
        lines.append("- Peak review windows:")
        windows = entry["motion_evidence"].get("review_windows")
        if isinstance(windows, list) and windows:
            for window in windows:
                evidence = ", ".join(str(item) for item in window.get("evidence", []))
                lines.append(
                    f"  - {window.get('start_seconds', 0):.3f}s–"
                    f"{window.get('end_seconds', 0):.3f}s "
                    f"(steps {window.get('start_step')}–"
                    f"{window.get('end_step')}): {evidence}"
                )
        else:
            lines.append("  - none reported; watch the complete video")
        lines.append("")
    lines.extend(
        [
            "## Human review checklist",
            "",
            "- Watch every required video, not only the peak windows.",
            "- Check torso oscillation, jitter, foot drag, asymmetric placement, "
            "abrupt transitions, unstable turns, and unrealistic recovery.",
            "- Edit `visual_reviews.draft.json`: replace `pending` with `pass` "
            "or `fail`, set the reviewer, and add concrete notes.",
            "- Keep `reviewed_video_paths` as canonical paths. The readable "
            "aliases are navigation aids only.",
            "",
        ]
    )
    return "\n".join(lines)


def _draft_reviews(
    plan: dict[str, Any],
    entries: list[dict[str, Any]],
) -> dict[str, Any]:
    reviews = []
    for candidate_id in plan["candidate_ids"]:
        reviewed_paths = [
            entry["canonical_video_path"]
            for entry in entries
            if entry["candidate_id"] == candidate_id
        ]
        reviews.append(
            {
                "candidate_id": candidate_id,
                "status": "pending",
                "reviewer": "user",
                "reviewed_video_paths": reviewed_paths,
                "notes": "",
            }
        )
    return {"visual_reviews": reviews}


def _write_immutable(path: Path, content: bytes, label: str) -> bool:
    if path.exists() or path.is_symlink():
        if (
            not path.is_file()
            or path.is_symlink()
            or path.read_bytes() != content
        ):
            raise SpecError(f"existing {label} differs from approved evidence")
        return False
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_bytes(content)
    os.replace(temporary, path)
    return True


def build_review_bundle(
    spec: dict[str, Any],
    session_path: Path,
    plan: dict[str, Any],
    plan_path: Path,
    state: dict[str, Any],
    state_path: Path,
) -> dict[str, Any]:
    """Create or verify one deterministic review bundle."""
    entries = _validate_review_inputs(
        spec,
        session_path,
        plan,
        plan_path,
        state,
        state_path,
    )
    output_root = Path(spec["evaluation"]["output_dir"])
    review_dir = output_root / "review"
    aliases_dir = review_dir / "required_videos"
    for directory, label in (
        (output_root, "evaluation output directory"),
        (review_dir, "review directory"),
        (aliases_dir, "review alias directory"),
    ):
        if directory.exists() and (
            not directory.is_dir() or directory.is_symlink()
        ):
            raise SpecError(f"{label} must be a regular directory")

    manifest = {
        "version": 1,
        "stage": "awaiting_visual_review",
        "session_path": str(session_path),
        "session_sha256": _sha256(session_path),
        "plan_path": str(plan_path),
        "plan_sha256": _sha256(plan_path),
        "execution_state_path": str(state_path),
        "execution_state_sha256": _sha256(state_path),
        "evaluation_output_dir": str(output_root),
        "review_dir": str(review_dir),
        "video_count": len(entries),
        "videos": [],
    }
    for entry in entries:
        alias_path = aliases_dir / entry["alias_name"]
        enriched = dict(entry)
        enriched["alias_path"] = str(alias_path)
        enriched["alias_relative_path"] = (
            f"required_videos/{entry['alias_name']}"
        )
        manifest["videos"].append(enriched)

    manifest_bytes = (
        json.dumps(
            manifest,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    index_bytes = _index_markdown(manifest).encode("utf-8")
    draft_bytes = (
        json.dumps(
            _draft_reviews(plan, entries),
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")

    immutable_outputs = (
        (review_dir / "review_manifest.json", manifest_bytes, "review manifest"),
        (review_dir / "REVIEW_INDEX.md", index_bytes, "review index"),
    )
    for path, content, label in immutable_outputs:
        if path.exists() or path.is_symlink():
            if (
                not path.is_file()
                or path.is_symlink()
                or path.read_bytes() != content
            ):
                raise SpecError(f"existing {label} differs from approved evidence")
    for entry in manifest["videos"]:
        alias_path = Path(entry["alias_path"])
        target = os.path.relpath(
            entry["canonical_video_path"],
            start=alias_path.parent,
        )
        if alias_path.exists() or alias_path.is_symlink():
            if not alias_path.is_symlink() or os.readlink(alias_path) != target:
                raise SpecError(
                    f"existing review alias collision: {alias_path}"
                )

    review_dir.mkdir(parents=True, exist_ok=True)
    aliases_dir.mkdir(exist_ok=True)
    for entry in manifest["videos"]:
        alias_path = Path(entry["alias_path"])
        target = os.path.relpath(
            entry["canonical_video_path"],
            start=alias_path.parent,
        )
        if not alias_path.is_symlink():
            alias_path.symlink_to(target)
    created_manifest = _write_immutable(
        review_dir / "review_manifest.json",
        manifest_bytes,
        "review manifest",
    )
    created_index = _write_immutable(
        review_dir / "REVIEW_INDEX.md",
        index_bytes,
        "review index",
    )
    draft_path = review_dir / "visual_reviews.draft.json"
    draft_created = False
    if not draft_path.exists() and not draft_path.is_symlink():
        draft_created = _write_immutable(
            draft_path,
            draft_bytes,
            "visual review draft",
        )
    elif not draft_path.is_file() or draft_path.is_symlink():
        raise SpecError("visual review draft must be a regular file")
    return {
        "state": "review_bundle_ready",
        "review_dir": str(review_dir),
        "index_path": str(review_dir / "REVIEW_INDEX.md"),
        "manifest_path": str(review_dir / "review_manifest.json"),
        "draft_path": str(draft_path),
        "video_count": len(entries),
        "created": {
            "manifest": created_manifest,
            "index": created_index,
            "draft": draft_created,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session", help="Approved evaluation session JSON")
    parser.add_argument("plan", help="Version-1 evaluation plan JSON")
    parser.add_argument(
        "--execution-state",
        required=True,
        help="Hash-bound evaluation_state.json at awaiting_visual_review",
    )
    args = parser.parse_args()
    session_path = Path(args.session).resolve()
    plan_path = Path(args.plan).resolve()
    state_path = Path(args.execution_state).resolve()
    try:
        spec = load_and_validate(session_path)
        plan = load_evaluation_plan(plan_path)
        state = _load_object(state_path, "evaluation state")
        result = build_review_bundle(
            spec,
            session_path,
            plan,
            plan_path,
            state,
            state_path,
        )
    except (OSError, SpecError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
