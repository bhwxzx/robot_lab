python scripts/reinforcement_learning/rsl_rl/train.py \
    --task=RobotLab-Isaac-Velocity-Rough-LW-leg-student-v0 \
    --log_project_name=LW_leg_locomotion_student \
    --logger=wandb \
    --num_envs=4096 \
    --headless \
    --load_run=2025-12-13_21-59-44 \
    --checkpoint=model_65999.pt