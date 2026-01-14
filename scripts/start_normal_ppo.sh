python scripts/reinforcement_learning/rsl_rl/train.py \
    --task=RobotLab-Isaac-Velocity-Flat-LW-leg-normal-ppo-v0 \
    --log_project_name=LW_leg_locomotion_ppo \
    --logger=wandb \
    --num_envs=4096 \
    --headless \
    # --resume \