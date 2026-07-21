#!/bin/bash

# 设置默认值
TERRAIN="Flat"
type="leg"
METHOD="Dwaq"

# 解析传入的命名参数
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --terrain) TERRAIN="$2"; shift ;;
        --type) TYPE="$2"; shift ;;
        --method) METHOD="$2"; shift ;;
        *) echo "未知参数: $1"; exit 1 ;;
    esac
    shift
done

python scripts/reinforcement_learning/rsl_rl/play.py \
    --task="RobotLab-Isaac-Velocity-${TERRAIN}-LW-${TYPE}-${METHOD}-Play-v0" \
    --num_envs=500 \
    # --keyboard \
    # --checkpoint=/home/young/liufengrong/robot_lab/logs/rsl_rl/LW_leg_flat_amp_dwaq/2026-06-17_14-57-09/model_49999.pt
    # --keyboard \
    # --checkpoint=/home/young/liufengrong/robot_lab/logs/rsl_rl/LW_leg_rough_amp_dwaq/2026-02-26_16-03-20/model_49999.pt
    # --real-time \
    # --video \
    # --video_length=2000 

