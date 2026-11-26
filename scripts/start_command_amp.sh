#!/usr/bin/env bash
set -euo pipefail

terrain=Flat
train_args=()
while (( $# )); do
    case "$1" in
        --terrain)
            terrain=${2:?请为 --terrain 指定 Flat}
            shift
            ;;
        --terrain=*) terrain=${1#*=} ;;
        *) train_args+=("$1") ;;
    esac
    shift
done

if [[ "$terrain" != Flat ]]; then
    printf 'LW Leg CommandAMP 当前仅支持 Flat 地形，收到 %s。\n' "$terrain" >&2
    exit 2
fi

cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
exec python scripts/reinforcement_learning/rsl_rl/train.py \
    --task=RobotLab-Isaac-Velocity-Flat-LW-leg-CommandAmp-v0 \
    --log_project_name=LW_leg_locomotion_command_amp \
    --logger=wandb \
    --num_envs=4096 \
    --headless \
    "${train_args[@]}"
