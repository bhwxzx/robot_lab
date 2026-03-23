# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Script to play a BeyondMimic RL agent from RSL-RL and export standard PPO policy."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Play a BeyondMimic RL agent and export policy.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during playback.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate (default: 1 for clear vis).")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent config.")
parser.add_argument("--use_pretrained_checkpoint", action="store_true", help="Use pretrained checkpoint.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()

if args_cli.video:
    args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import time
import torch

from rsl_rl.runners import OnPolicyRunner, DistillationRunner

from isaaclab.envs import ManagerBasedRLEnvCfg, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
# 引入导出用的官方标准库
from isaaclab_rl.rsl_rl import (
    RslRlBaseRunnerCfg, 
    RslRlVecEnvWrapper, 
    export_policy_as_jit, 
    export_policy_as_onnx
)
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import robot_lab.tasks  # noqa: F401  # isort: skip


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with RSL-RL agent."""
    task_name = args_cli.task.split(":")[-1]

    # override configurations with non-hydra CLI arguments
    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else 1
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # ==========================================
    # BeyondMimic 专属环境参数覆盖 (Overrides)
    # ==========================================
    
    # 1. 关闭策略观测值加噪
    env_cfg.observations.policy.enable_corruption = False
    
    # 2. 关闭各种领域随机化 (Domain Randomizations)
    if hasattr(env_cfg, "events"):
        env_cfg.events.push_robot = None
        env_cfg.events.randomize_apply_external_force_torque = None
        env_cfg.events.add_joint_default_pos = None
        env_cfg.events.base_com = None
        if hasattr(env_cfg.events, "physics_material"):
            env_cfg.events.physics_material = None

    # 3. 开启参考动作可视化 & 取消起始位姿噪声
    if hasattr(env_cfg.commands, "motion"):
        # 强制开启可视化 (显示参考动作坐标系/鬼影)
        env_cfg.commands.motion.debug_vis = True
        
        # 将所有的初始化随机范围设为 0，保证每次重置都完美从动作的第一帧对齐开始
        env_cfg.commands.motion.pose_range = {}
        env_cfg.commands.motion.velocity_range = {}
        env_cfg.commands.motion.joint_position_range = (0.0, 0.0)
    # ==========================================

    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", task_name)
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # 仅保留标准 PPO 和 Distillation Runner
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
            
    runner.load(resume_path)

    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    try:
        policy_nn = runner.alg.policy
    except AttributeError:
        policy_nn = runner.alg.actor_critic

    # ==========================================
    # 标准 PPO 导出逻辑 (JIT 和 ONNX)
    # ==========================================
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    os.makedirs(export_model_dir, exist_ok=True)
    print(f"[INFO] Exporting standard PPO policy to: {export_model_dir}")

    # 获取对应的 Normalizer (观测值归一化层，部署时必须包含)
    if hasattr(policy_nn, "actor_obs_normalizer"):
        normalizer = policy_nn.actor_obs_normalizer
    elif hasattr(policy_nn, "student_obs_normalizer"):
        normalizer = policy_nn.student_obs_normalizer
    else:
        normalizer = None

    # 调用官方接口执行导出
    export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
    export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")
    print(f"[SUCCESS] Policy exported successfully!")
    # ==========================================

    dt = env.unwrapped.step_dt
    obs = env.get_observations()

    # 提取 BeyondMimic 相关的指令拦截器
    isaac_env = env.unwrapped
    motion_cmd = None
    if "motion" in isaac_env.command_manager.active_terms:
        motion_cmd = isaac_env.command_manager.get_term("motion")
        
        # ==================== 终极防崩溃修复 ====================
        # 定义一个强制将起始帧归零的函数
        def force_zero_start(env_ids):
            motion_cmd.time_steps[env_ids] = 0
            
        # 动态替换掉原版充满随机性的自适应采样算法
        motion_cmd._adaptive_sampling = force_zero_start
        
        # 初始全局归零
        motion_cmd.time_steps[:] = 0
        # ========================================================

    obs, _ = env.reset()

    timestep = 0
    print("[INFO] Starting playback loop...")
    while simulation_app.is_running():
        start_time = time.time()
        
        with torch.inference_mode():
            # # 获取当前帧的参考位置和速度
            # ref_pos = motion_cmd.joint_pos[0]  # 取第 0 个环境的参考位置
            # ref_vel = motion_cmd.joint_vel[0]  # 取第 0 个环境的参考速度
                    
            # print(f"Ref Pos : {ref_pos[:10].cpu().numpy()}")
            # print(f"Ref Vel : {ref_vel[:10].cpu().numpy()}")
            # print(f"Obs[command]: {obs['policy'][0, :20].cpu().numpy()}")
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
            policy_nn.reset(dones)

        if args_cli.video:
            timestep += 1
            if timestep == args_cli.video_length:
                break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if sleep_time > 0:
            time.sleep(sleep_time)

    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()