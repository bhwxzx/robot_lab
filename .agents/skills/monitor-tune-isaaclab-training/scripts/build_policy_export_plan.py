#!/usr/bin/env python3
"""Build deterministic checkpoint-to-JIT/ONNX export jobs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from validate_session_spec import SpecError, load_and_validate


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise SpecError(f"{label} does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise SpecError(
            f"{label} is invalid JSON at line {exc.lineno}: {exc.msg}"
        ) from exc
    if not isinstance(value, dict):
        raise SpecError(f"{label} must be a JSON object")
    return value


def validate_sources(
    spec: dict[str, Any],
    session_path: Path,
    ranking_path: Path,
    inventory_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    ranking = _load_json(ranking_path, "training ranking")
    inventory = _load_json(inventory_path, "checkpoint inventory")
    if (
        ranking.get("algorithm") != spec["algorithm"]
        or ranking.get("final_selection") is not None
        or ranking.get("selection_status") != "awaiting_policy_evaluation"
        or not isinstance(ranking.get("ranking"), list)
    ):
        raise SpecError(
            "training ranking is not awaiting the approved policy evaluation"
        )
    if (
        inventory.get("version") != 1
        or inventory.get("session_sha256") != _sha256(session_path)
        or inventory.get("training_ranking_sha256") != _sha256(ranking_path)
        or inventory.get("training_run_id") != spec["training"]["run_id"]
        or inventory.get("algorithm") != spec["algorithm"]
        or not isinstance(inventory.get("entries"), list)
    ):
        raise SpecError("checkpoint inventory binding is invalid")
    return ranking, inventory


def select_checkpoint_sources(
    spec: dict[str, Any],
    ranking: dict[str, Any],
    inventory: dict[str, Any],
    worker_id: str | None,
) -> list[dict[str, Any]]:
    handoff = spec["evaluation_handoff"]
    if worker_id != handoff["evaluation_worker_id"]:
        raise SpecError("this is not the approved policy export worker")
    selected = [
        item
        for item in ranking["ranking"]
        if isinstance(item, dict) and item.get("pareto_optimal") is True
    ][: handoff["top_k"]]
    if len(selected) != handoff["top_k"]:
        raise SpecError("training ranking has too few Pareto candidates")
    seed = handoff["checkpoint_seed"]
    sources: list[dict[str, Any]] = []
    for ranked_item in selected:
        trial_id = ranked_item.get("trial_id")
        matches = [
            entry
            for entry in inventory["entries"]
            if isinstance(entry, dict)
            and entry.get("trial_id") == trial_id
            and entry.get("seed") == seed
            and entry.get("worker_id") == worker_id
        ]
        if not matches:
            raise SpecError(
                f"candidate {trial_id} has no checkpoint on the approved "
                "policy export worker"
            )
        matches.sort(
            key=lambda item: (
                item.get("checkpoint_step", -1),
                item.get("run_id", ""),
            ),
            reverse=True,
        )
        source = matches[0]
        if (
            len(matches) > 1
            and matches[1].get("checkpoint_step")
            == source.get("checkpoint_step")
            and (
                matches[1].get("checkpoint_path")
                != source.get("checkpoint_path")
                or matches[1].get("checkpoint_sha256")
                != source.get("checkpoint_sha256")
            )
        ):
            raise SpecError(f"candidate {trial_id} has ambiguous newest checkpoints")
        checkpoint = Path(source.get("checkpoint_path", ""))
        expected_hash = source.get("checkpoint_sha256")
        if (
            not checkpoint.is_absolute()
            or not checkpoint.is_file()
            or checkpoint.is_symlink()
            or not isinstance(expected_hash, str)
            or _sha256(checkpoint) != expected_hash
        ):
            raise SpecError(f"candidate {trial_id} checkpoint hash changed")
        sources.append(
            {
                "candidate_id": trial_id,
                "trial_id": trial_id,
                "seed": seed,
                "source_run_id": source["run_id"],
                "worker_id": worker_id,
                "checkpoint_path": str(checkpoint),
                "checkpoint_sha256": expected_hash,
                "checkpoint_step": source["checkpoint_step"],
                "rsl_rl_run_dir": source.get("rsl_rl_run_dir"),
            }
        )
    return sources


def build_plan(
    spec: dict[str, Any],
    ranking: dict[str, Any],
    inventory: dict[str, Any],
    *,
    session_path: Path,
    ranking_path: Path,
    inventory_path: Path,
    worker_id: str | None,
) -> dict[str, Any]:
    contract = spec.get("policy_export")
    if not isinstance(contract, dict) or not contract.get("enabled"):
        raise SpecError("session does not enable policy_export")
    sources = select_checkpoint_sources(
        spec,
        ranking,
        inventory,
        worker_id,
    )
    runs = [
        {
            **source,
            "run_id": (
                f"export__{source['candidate_id']}__seed-{source['seed']}"
            ),
            "command_template": contract["command"],
            "artifact_filenames": contract["artifact_filenames"],
        }
        for source in sources
    ]
    return {
        "version": 1,
        "session_sha256": _sha256(session_path),
        "training_ranking_sha256": _sha256(ranking_path),
        "checkpoint_inventory_sha256": _sha256(inventory_path),
        "training_run_id": spec["training"]["run_id"],
        "algorithm": spec["algorithm"],
        "worker_id": worker_id,
        "adapter_id": contract["adapter_id"],
        "output_dir": contract["output_dir"],
        "gpu_index": contract["gpu_index"],
        "parity": contract["parity"],
        "run_count": len(runs),
        "runs": runs,
    }


def validate_plan(spec: dict[str, Any], plan: dict[str, Any]) -> None:
    contract = spec["policy_export"]
    handoff = spec["evaluation_handoff"]
    if (
        plan.get("version") != 1
        or plan.get("training_run_id") != spec["training"]["run_id"]
        or plan.get("algorithm") != spec["algorithm"]
        or plan.get("worker_id") != contract["worker_id"]
        or plan.get("adapter_id") != contract["adapter_id"]
        or plan.get("output_dir") != contract["output_dir"]
        or plan.get("gpu_index") != contract["gpu_index"]
        or plan.get("parity") != contract["parity"]
        or not isinstance(plan.get("runs"), list)
        or plan.get("run_count") != len(plan["runs"])
        or len(plan["runs"]) != handoff["top_k"]
        or any(
            not isinstance(plan.get(key), str)
            or len(plan[key]) != 64
            or any(
                character not in "0123456789abcdef"
                for character in plan[key]
            )
            for key in (
                "session_sha256",
                "training_ranking_sha256",
                "checkpoint_inventory_sha256",
            )
        )
    ):
        raise SpecError("policy export plan differs from the approved session")
    seen: set[str] = set()
    for run in plan["runs"]:
        run_id = run.get("run_id")
        if (
            not isinstance(run_id, str)
            or run_id in seen
            or run_id
            != f"export__{run.get('candidate_id')}__seed-{run.get('seed')}"
            or run.get("trial_id") != run.get("candidate_id")
            or run.get("seed") != handoff["checkpoint_seed"]
            or run.get("worker_id") != contract["worker_id"]
            or run.get("command_template") != contract["command"]
            or run.get("artifact_filenames") != contract["artifact_filenames"]
        ):
            raise SpecError("policy export plan run identity is invalid")
        seen.add(run_id)
        checkpoint = Path(run.get("checkpoint_path", ""))
        if (
            not checkpoint.is_absolute()
            or not checkpoint.is_file()
            or checkpoint.is_symlink()
            or _sha256(checkpoint) != run.get("checkpoint_sha256")
        ):
            raise SpecError("policy export plan checkpoint changed")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session")
    parser.add_argument("training_ranking")
    parser.add_argument("checkpoint_inventory")
    parser.add_argument("--worker-id")
    parser.add_argument("--output")
    args = parser.parse_args()
    try:
        session_path = Path(args.session).resolve()
        ranking_path = Path(args.training_ranking).resolve()
        inventory_path = Path(args.checkpoint_inventory).resolve()
        spec = load_and_validate(session_path)
        ranking, inventory = validate_sources(
            spec,
            session_path,
            ranking_path,
            inventory_path,
        )
        plan = build_plan(
            spec,
            ranking,
            inventory,
            session_path=session_path,
            ranking_path=ranking_path,
            inventory_path=inventory_path,
            worker_id=args.worker_id,
        )
        validate_plan(spec, plan)
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
