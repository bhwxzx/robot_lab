#!/usr/bin/env python3
"""Deterministic fake JIT/ONNX exporter for executor contract tests."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--candidate-id", required=True)
parser.add_argument("--trial-id", required=True)
parser.add_argument("--checkpoint-path", required=True)
parser.add_argument("--checkpoint-sha256", required=True)
parser.add_argument("--export-run-id", required=True)
parser.add_argument("--jit-path", required=True)
parser.add_argument("--onnx-path", required=True)
parser.add_argument("--result-path", required=True)
parser.add_argument("--gpu-index", required=True)
parser.add_argument("--seed", required=True)
parser.add_argument("--history-contract", required=True)
parser.add_argument("--normalization-contract", required=True)
parser.add_argument("--minimum-parity-samples", type=int, required=True)
parser.add_argument("--max-abs-action-error", type=float, required=True)
parser.add_argument(
    "--require-idle-gpu",
    "--require_idle_gpu",
    action="store_true",
)
parser.add_argument(
    "--fake-mode",
    choices={"healthy", "bad-parity", "missing-onnx", "crash"},
    default="healthy",
)
args = parser.parse_args()

if args.fake_mode == "crash":
    raise SystemExit(9)

checkpoint = Path(args.checkpoint_path)
if _sha256(checkpoint) != args.checkpoint_sha256:
    raise SystemExit("checkpoint hash mismatch")

jit_path = Path(args.jit_path)
onnx_path = Path(args.onnx_path)
result_path = Path(args.result_path)
result_path.parent.mkdir(parents=True, exist_ok=True)
jit_path.write_bytes(b"fake-jit-policy")
if args.fake_mode != "missing-onnx":
    onnx_path.write_bytes(b"fake-onnx-policy")

error = (
    args.max_abs_action_error + 1.0
    if args.fake_mode == "bad-parity"
    else min(args.max_abs_action_error, 1.0e-7)
)
artifacts = {}
for kind, path in (("jit", jit_path), ("onnx", onnx_path)):
    if not path.exists():
        continue
    artifacts[kind] = {
        "path": str(path),
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
        "input_shape": [args.minimum_parity_samples, 10],
        "output_shape": [args.minimum_parity_samples, 4],
        "finite": True,
        "max_abs_action_error": error,
    }

result = {
    "version": 1,
    "export_run_id": args.export_run_id,
    "candidate_id": args.candidate_id,
    "checkpoint_path": str(checkpoint),
    "checkpoint_sha256": args.checkpoint_sha256,
    "status": "completed",
    "artifacts": artifacts,
    "parity": {
        "sample_count": args.minimum_parity_samples,
        "observation_batch_sha256": hashlib.sha256(
            b"observations"
        ).hexdigest(),
        "native_output_sha256": hashlib.sha256(b"native-actions").hexdigest(),
        "history_contract": args.history_contract,
        "normalization_contract": args.normalization_contract,
    },
}
result_path.write_text(
    json.dumps(result, sort_keys=True),
    encoding="utf-8",
)
