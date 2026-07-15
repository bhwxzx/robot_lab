#!/bin/bash

# 设置默认值
TERRAIN="Flat"

# 解析传入的命名参数
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --terrain) TERRAIN="$2"; shift ;;
        *) echo "未知参数: $1"; exit 1 ;;
    esac
    shift
done

python scripts/reinforcement_learning/rsl_rl/train.py \
    --task="RobotLab-Isaac-Velocity-${TERRAIN}-LW-leg-Amp-Roa-v0" \
    --log_project_name="LW_leg_locomotion_amp_roa" \
    --logger=wandb \
    --num_envs=4096 \
    --max_iterations=100000 \
    --headless