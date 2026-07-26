#!/usr/bin/env python3
"""Build a deterministic Native/JIT/ONNX policy evaluation matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from algorithm_profiles import load_registry, resolve_profile
from validate_session_spec import SpecError, load_and_validate


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validated_sha256(value: Any, path: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise SpecError(f"{path} must be a lowercase SHA-256 hex digest")
    return value


def _load_candidates(path: Path, selected_artifacts: set[str]) -> list[dict[str, Any]]:
    try:
        root = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise SpecError(f"candidate manifest does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise SpecError(
            f"invalid candidate manifest JSON at line {exc.lineno}: {exc.msg}"
        ) from exc
    if not isinstance(root, dict) or set(root) != {"candidates"}:
        raise SpecError("candidate manifest must contain only a candidates array")
    candidates = root["candidates"]
    if not isinstance(candidates, list) or not candidates:
        raise SpecError("candidates must be a non-empty array")
    if len(candidates) > 64:
        raise SpecError("candidates may contain at most 64 entries")

    seen: set[str] = set()
    normalized: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates):
        path_prefix = f"candidates[{index}]"
        if not isinstance(candidate, dict):
            raise SpecError(f"{path_prefix} must be an object")
        unknown = sorted(
            set(candidate)
            - {
                "candidate_id",
                "checkpoint_path",
                "checkpoint_sha256",
                "artifacts",
                "artifact_sha256",
            }
        )
        if unknown:
            raise SpecError(
                f"{path_prefix} contains unknown field(s): {', '.join(unknown)}"
            )
        candidate_id = candidate.get("candidate_id")
        if (
            not isinstance(candidate_id, str)
            or not candidate_id
            or any(character not in "abcdefghijklmnopqrstuvwxyz"
                   "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-" for character in candidate_id)
        ):
            raise SpecError(
                f"{path_prefix}.candidate_id contains unsupported characters"
            )
        if candidate_id in seen:
            raise SpecError(f"duplicate candidate_id: {candidate_id}")
        seen.add(candidate_id)

        checkpoint = candidate.get("checkpoint_path")
        if not isinstance(checkpoint, str) or not checkpoint:
            raise SpecError(f"{path_prefix}.checkpoint_path must be a non-empty string")
        if not Path(checkpoint).is_absolute():
            raise SpecError(f"{path_prefix}.checkpoint_path must be absolute")
        checkpoint_file = Path(checkpoint)
        if not checkpoint_file.is_file():
            raise SpecError(f"{path_prefix}.checkpoint_path does not exist")
        checkpoint_sha256 = _validated_sha256(
            candidate.get("checkpoint_sha256"),
            f"{path_prefix}.checkpoint_sha256",
        )
        if _sha256(checkpoint_file) != checkpoint_sha256:
            raise SpecError(f"{path_prefix}.checkpoint_sha256 does not match file")
        artifacts = candidate.get("artifacts")
        if not isinstance(artifacts, dict):
            raise SpecError(f"{path_prefix}.artifacts must be an object")
        artifact_sha256 = candidate.get("artifact_sha256")
        if not isinstance(artifact_sha256, dict):
            raise SpecError(f"{path_prefix}.artifact_sha256 must be an object")
        unknown_artifacts = sorted(set(artifacts) - selected_artifacts)
        if unknown_artifacts:
            raise SpecError(
                f"{path_prefix}.artifacts contains unselected artifact(s): "
                f"{', '.join(unknown_artifacts)}"
            )
        if set(artifact_sha256) != set(artifacts):
            raise SpecError(
                f"{path_prefix}.artifact_sha256 keys must match artifacts"
            )
        artifact_paths: dict[str, str] = {"native": checkpoint}
        artifact_hashes: dict[str, str] = {"native": checkpoint_sha256}
        for kind, artifact_path in artifacts.items():
            if not isinstance(artifact_path, str) or not artifact_path:
                raise SpecError(
                    f"{path_prefix}.artifacts.{kind} must be a non-empty string"
                )
            if not Path(artifact_path).is_absolute():
                raise SpecError(
                    f"{path_prefix}.artifacts.{kind} must be absolute"
                )
            if kind == "native" and artifact_path != checkpoint:
                raise SpecError(
                    f"{path_prefix}.artifacts.native must equal checkpoint_path"
                )
            artifact_file = Path(artifact_path)
            if not artifact_file.is_file():
                raise SpecError(
                    f"{path_prefix}.artifacts.{kind} does not exist"
                )
            expected_hash = _validated_sha256(
                artifact_sha256[kind],
                f"{path_prefix}.artifact_sha256.{kind}",
            )
            if _sha256(artifact_file) != expected_hash:
                raise SpecError(
                    f"{path_prefix}.artifact_sha256.{kind} does not match file"
                )
            artifact_paths[kind] = artifact_path
            artifact_hashes[kind] = expected_hash
        missing = sorted(selected_artifacts - set(artifact_paths))
        if missing:
            raise SpecError(
                f"{path_prefix} is missing selected artifact(s): {', '.join(missing)}"
            )
        normalized.append(
            {
                "candidate_id": candidate_id,
                "checkpoint_path": checkpoint,
                "checkpoint_sha256": checkpoint_sha256,
                "artifacts": artifact_paths,
                "artifact_sha256": artifact_hashes,
            }
        )
    return normalized


def _render_command(command: list[str], values: dict[str, Any]) -> list[str]:
    try:
        return [token.format_map(values) for token in command]
    except (KeyError, ValueError) as exc:
        raise SpecError(f"cannot render evaluation command: {exc}") from exc


def build_plan(
    spec: dict[str, Any],
    candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    """Return the exact candidate/artifact/scenario/seed evaluation matrix."""
    evaluation = spec.get("evaluation")
    if not isinstance(evaluation, dict) or not evaluation.get("enabled"):
        raise SpecError("session evaluation must be enabled")
    artifact_specs = {
        artifact["kind"]: artifact for artifact in evaluation["artifacts"]
    }
    output_dir = Path(evaluation["output_dir"])
    runs: list[dict[str, Any]] = []
    required_videos: list[str] = []

    for candidate in candidates:
        candidate_id = candidate["candidate_id"]
        for artifact_kind, artifact_spec in artifact_specs.items():
            artifact_path = candidate["artifacts"][artifact_kind]
            for scenario in evaluation["scenarios"]:
                scenario_id = scenario["id"]
                for seed in scenario["seeds"]:
                    run_id = (
                        f"{candidate_id}__{artifact_kind}__"
                        f"{scenario_id}__seed-{seed}"
                    )
                    run_dir = (
                        output_dir
                        / candidate_id
                        / artifact_kind
                        / scenario_id
                        / f"seed-{seed}"
                    )
                    result_path = run_dir / "result.json"
                    video_path = run_dir / "motion.mp4"
                    values = {
                        "artifact_kind": artifact_kind,
                        "artifact_path": artifact_path,
                        "artifact_sha256": candidate["artifact_sha256"][
                            artifact_kind
                        ],
                        "candidate_id": candidate_id,
                        "checkpoint_path": candidate["checkpoint_path"],
                        "checkpoint_sha256": candidate["checkpoint_sha256"],
                        "command_schedule_json": json.dumps(
                            scenario["command_schedule"],
                            sort_keys=True,
                            separators=(",", ":"),
                            ensure_ascii=False,
                        ),
                        "duration_steps": scenario["duration_steps"],
                        "executor_run_id": run_id,
                        "gpu_index": evaluation["gpu_index"],
                        "result_path": str(result_path),
                        "require_idle_gpu_flag": "--require_idle_gpu",
                        "run_id": run_id,
                        "scenario_id": scenario_id,
                        "scenario_overrides_json": json.dumps(
                            scenario["overrides"],
                            sort_keys=True,
                            separators=(",", ":"),
                            ensure_ascii=False,
                        ),
                        "seed": seed,
                        "video_path": str(video_path),
                    }
                    command = _render_command(artifact_spec["command"], values)
                    required_run = bool(
                        artifact_spec["required"] and scenario["required"]
                    )
                    video_required = bool(required_run and scenario["video"])
                    if video_required:
                        required_videos.append(str(video_path))
                    runs.append(
                        {
                            "run_id": run_id,
                            "candidate_id": candidate_id,
                            "artifact": artifact_kind,
                            "artifact_path": artifact_path,
                            "artifact_sha256": candidate["artifact_sha256"][
                                artifact_kind
                            ],
                            "checkpoint_path": candidate["checkpoint_path"],
                            "checkpoint_sha256": candidate["checkpoint_sha256"],
                            "artifact_required": artifact_spec["required"],
                            "scenario_id": scenario_id,
                            "scenario_category": scenario["category"],
                            "scenario_required": scenario["required"],
                            "seed": seed,
                            "duration_steps": scenario["duration_steps"],
                            "overrides": scenario["overrides"],
                            "command_schedule": scenario["command_schedule"],
                            "command": command,
                            "run_dir": str(run_dir),
                            "result_path": str(result_path),
                            "video_path": str(video_path),
                            "video_required": video_required,
                        }
                    )

    reference = evaluation["parity"]["reference_artifact"]
    profile = resolve_profile(
        load_registry(),
        spec["algorithm"]["profile_id"],
    )
    required_artifacts = [
        artifact["kind"]
        for artifact in evaluation["artifacts"]
        if artifact["required"] and artifact["kind"] != reference
    ]
    parity_expectations = [
        {
            "candidate_id": candidate["candidate_id"],
            "reference_artifact": reference,
            "artifact": artifact,
            "max_abs_action_error": evaluation["parity"][
                "max_abs_action_error"
            ],
            "closed_loop_metrics": evaluation["parity"].get(
                "closed_loop_metrics",
                [],
            ),
        }
        for candidate in candidates
        for artifact in required_artifacts
    ]
    return {
        "version": 1,
        "session_version": spec["version"],
        "training_run_id": spec["training"]["run_id"],
        "algorithm": spec["algorithm"],
        "profile_id": spec["algorithm"]["profile_id"],
        "history_contract": profile["evaluation_capabilities"][
            "history_contract"
        ],
        "candidate_ids": [candidate["candidate_id"] for candidate in candidates],
        "required_videos": required_videos,
        "minimum_reviewed_videos": evaluation["visual_review"][
            "minimum_reviewed_videos"
        ],
        "parity_expectations": parity_expectations,
        "gates": evaluation["gates"],
        "run_count": len(runs),
        "runs": runs,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session", help="Validated version-3-through-6 session JSON")
    parser.add_argument("candidates", help="Candidate checkpoint/artifact manifest")
    parser.add_argument("--output", help="Optional JSON output path")
    args = parser.parse_args()
    try:
        spec = load_and_validate(args.session)
        evaluation = spec.get("evaluation")
        if not isinstance(evaluation, dict) or not evaluation.get("enabled"):
            raise SpecError("session evaluation must be enabled")
        selected_artifacts = {
            artifact["kind"] for artifact in evaluation["artifacts"]
        }
        candidates = _load_candidates(Path(args.candidates), selected_artifacts)
        plan = build_plan(spec, candidates)
    except SpecError as exc:
        parser.error(str(exc))
    encoded = json.dumps(
        plan,
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
    )
    if args.output:
        Path(args.output).write_text(encoded + "\n", encoding="utf-8")
    else:
        print(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
