#!/usr/bin/env python3
"""Deterministic stand-in for the real RSL-RL training entry point."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import yaml


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fake-log-root", required=True)
    parser.add_argument(
        "--fake-mode",
        choices={"healthy", "collapse", "nonfinite", "crash"},
        default="healthy",
    )
    parser.add_argument("--fake-delay-seconds", type=float, default=0.0)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--run_name", required=True)
    args, overrides = parser.parse_known_args()

    learning_rate = 0.001
    parsed_overrides: dict[str, object] = {}
    for token in overrides:
        path, separator, encoded = token.partition("=")
        if not separator:
            parser.error(f"invalid Hydra override: {token}")
        value = json.loads(encoded)
        parsed_overrides[path] = value
        if path == "agent.algorithm.learning_rate":
            learning_rate = float(value)

    log_root = Path(args.fake_log_root).resolve()
    stamp = "2099-01-01_00-00-00"
    run_dir = log_root / f"{stamp}_{args.run_name}"
    params_dir = run_dir / "params"
    params_dir.mkdir(parents=True)
    (params_dir / "env.yaml").write_text(
        yaml.safe_dump({"seed": args.seed, "terrain": {"difficulty": 1}}),
        encoding="utf-8",
    )
    (params_dir / "agent.yaml").write_text(
        yaml.safe_dump(
            {
                "algorithm": {"learning_rate": learning_rate},
                "run_name": args.run_name,
                "seed": args.seed,
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "model_2.pt").write_bytes(b"fake-checkpoint")
    (run_dir / "received.json").write_text(
        json.dumps(
            {
                "seed": args.seed,
                "run_name": args.run_name,
                "overrides": parsed_overrides,
            }
        ),
        encoding="utf-8",
    )

    print(f"[INFO] Logging experiment in directory: {log_root}", flush=True)
    print(f"Exact experiment name requested from command line: {stamp}", flush=True)
    rewards = {
        "healthy": (10.0, 11.0, 12.0, 13.0),
        "collapse": (-1.0, -2.0, -3.0, -4.0, -5.0),
        "nonfinite": ("nan",),
        "crash": (10.0,),
    }[args.fake_mode]
    for iteration, reward in enumerate(rewards):
        print(f"Learning iteration {iteration}/3", flush=True)
        print(f"Mean reward: {reward}", flush=True)
        print("Episode_Termination/illegal_contact: 0.0", flush=True)
        print(
            "Computation: 1000 steps/s (collection: 0.010s, learning 0.020s)",
            flush=True,
        )
        if args.fake_delay_seconds:
            time.sleep(args.fake_delay_seconds)
    return 7 if args.fake_mode == "crash" else 0


if __name__ == "__main__":
    raise SystemExit(main())
