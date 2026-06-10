python scripts/reinforcement_learning/rsl_rl/train.py \
    --task=RobotLab-Isaac-Velocity-Flat-LW-leg-Amp-Roa-v0 \
    --log_project_name=LW_leg_locomotion_amp_roa \
    --logger=wandb \
    --num_envs=4096 \
    --max_iterations=50000 \
    --headless \
    "$@"