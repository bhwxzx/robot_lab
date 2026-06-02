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

python scripts/reinforcement_learning/rsl_rl/play_beyondmimic.py \
    --task="RobotLab-Isaac-BeyondMimic-Flat-LW-${TYPE}-v0" \
    --num_envs=1 \
    # --checkpoint="/home/young/liufengrong/robot_lab/logs/rsl_rl/LW_wheel_beyongdmimic/2026-04-24_22-26-19/model_45000.pt"