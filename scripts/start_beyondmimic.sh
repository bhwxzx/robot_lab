#!/bin/bash

# 设置默认值
TYPE="leg"

# 解析传入的命名参数
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --type)
            if [[ "$#" -lt 2 ]]; then
                echo "--type 需要提供 leg 或 wheel" >&2
                exit 1
            fi
            case "$2" in
                leg|wheel) TYPE="$2" ;;
                *) echo "无效的 --type: $2（仅支持 leg 或 wheel）" >&2; exit 1 ;;
            esac
            shift
            ;;
        *) echo "未知参数: $1"; exit 1 ;;
    esac
    shift
done

conda run --no-capture-output -n isaacsim-5.1 python scripts/reinforcement_learning/rsl_rl/train.py \
    --task="RobotLab-Isaac-BeyondMimic-Flat-LW-${TYPE}-v0" \
    --log_project_name="LW_${TYPE}_beyondmimic" \
    --logger=wandb \
    --num_envs=4096 \
    --headless
