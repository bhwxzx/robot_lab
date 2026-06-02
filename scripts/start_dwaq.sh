#!/bin/bash

# 设置默认值
TERRAIN="Rough"
type="leg"

# 解析传入的命名参数
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --terrain) TERRAIN="$2"; shift ;;
        --type) TYPE="$2"; shift ;;
        *) echo "未知参数: $1"; exit 1 ;;
    esac
    shift
done

python scripts/reinforcement_learning/rsl_rl/train.py \
    --task="RobotLab-Isaac-Velocity-${TERRAIN}-LW-${TYPE}-Dwaq-v0" \
    --log_project_name="LW_${TYPE}_locomotion_dwaq" \
    --logger=wandb \
    --num_envs=4096 \
    --headless \
    # --resume