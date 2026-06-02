python scripts/reinforcement_learning/rsl_rl/train.py \
    --task=RobotLab-Isaac-Velocity-Rough-LW-leg-student-v0 \
    --log_project_name=LW_leg_locomotion_student \
    --logger=wandb \
    --num_envs=4096 \
    --headless \
    --load_run=2026-01-08_22-31-56 \
    --checkpoint=model_18000.pt \
