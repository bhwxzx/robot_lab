#!/bin/bash

# 设置默认值
type="leg"

# 解析传入的命名参数
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --type) TYPE="$2"; shift ;;
        *) echo "未知参数: $1"; exit 1 ;;
    esac
    shift
done

python scripts/reinforcement_learning/rsl_rl/train.py \
    --task="RobotLab-Isaac-BeyondMimic-Flat-LW-${TYPE}-v0" \
    --log_project_name="LW_${TYPE}_beyondmimic" \
    --logger=wandb \
    --num_envs=4096 \
    --headless \
    # --resume