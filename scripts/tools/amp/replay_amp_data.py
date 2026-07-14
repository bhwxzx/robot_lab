"""
回放纯AMP训练数据
Usage:
    # 如果数据是相对角度 (default):
    python replay_amp_data.py --motion_file datasets/motion_amp_expert/walk.txt --data_type relative
    
    # 如果数据是绝对角度:
    python replay_amp_data.py --motion_file datasets/motion_amp_expert/walk.txt --data_type absolute
"""

import argparse
import numpy as np
import torch
import os
import json
import time as real_time_lib

from isaaclab.app import AppLauncher

# 第一阶段：解析参数
parser = argparse.ArgumentParser(description="Verify and playback processed AMP datasets.")
parser.add_argument("--motion_file", type=str, required=True, help="Path to processed amp dataset.")
parser.add_argument("--fps", type=float, default=30.0, help="Playback speed.")
parser.add_argument("--data_type", type=str, choices=['absolute', 'relative'], default='relative', 
                    help="Joint angle type in the file: 'absolute' or 'relative' (offset from default pose).")
parser.add_argument("--task", type=str, default="RobotLab-Isaac-Velocity-Flat-LW-leg-Amp-Play-v0")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# 第二阶段：启动 App
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# 第三阶段：导入库
from isaaclab.envs import ManagerBasedRLEnv
from robot_lab.tasks.manager_based.locomotion.velocity.config.LW.LW_Leg.flat_env_cfg import LWLegFlatAmpEnvCfg_Play
from rsl_rl.utils import AMPLoader 

# --- 配置区域：必须与生成数据时的 JOINTS_NUM 和顺序一致 ---
AMP_TARGET_JOINTS = [
    "right_hip_joint", "left_hip_joint",
    "right_thigh_joint", "left_thigh_joint",
    "right_shank_joint", "left_shank_joint",
    "right_foot_joint", "left_foot_joint"
]
JOINTS_NUM = len(AMP_TARGET_JOINTS)

def main():
    # 1. 环境初始化
    env_cfg = LWLegFlatAmpEnvCfg_Play()
    # 强制将回放数据的环境历史设为 0 (单帧)
    env_cfg.observations.amp.history_length = 0
    env_cfg.scene.num_envs = 1
    env_cfg.sim.disable_gravity = True
    env_cfg.scene.robot.spawn.rigid_props.disable_gravity = True
    # 确保在预览时关闭所有随机化
    env_cfg.events.randomize_apply_external_force_torque = None
    env_cfg.events.randomize_push_robot = None
    env_cfg.events.randomize_reset_base = None
    env_cfg.events.randomize_reset_joints = None
    env_cfg.events.push_robot_hard = None
    env_cfg.events.randomize_actuator_gains = None
    env_cfg.events.randomize_com_positions = None
    env_cfg.events.randomize_rigid_body_mass_base = None
    env_cfg.events.randomize_rigid_body_mass_others = None
    env_cfg.events.randomize_rigid_body_material = None
    
    env = ManagerBasedRLEnv(cfg=env_cfg)
    robot = env.scene["robot"]
    
    # 2. 关节映射逻辑
    joint_name_to_id = {name: i for i, name in enumerate(robot.joint_names)}
    try:
        target_indices = torch.tensor([joint_name_to_id[name] for name in AMP_TARGET_JOINTS], 
                                      device=env.device, dtype=torch.long)
    except KeyError as e:
        print(f"ERROR: Joint {e} not found in robot model.")
        return

    # 3. 初始化 AMPLoader
    amp_loader = AMPLoader(
        device=env.device,
        time_between_frames=1.0 / args_cli.fps,
        motion_files=[args_cli.motion_file],
        preload_transitions=False
    )
    
    motion_len = int(amp_loader.trajectory_num_frames[0])
    playback_dt = 1.0 / args_cli.fps

    print(f"Starting verification playback: {args_cli.motion_file}")
    print(f"Mode: {args_cli.data_type}")

    frame_cnt = 0

    # 4. 准备固定基座状态
    fixed_root_state = torch.zeros((1, 13), device=env.device, dtype=torch.float32)
    fixed_root_state[:, 2] = 0.8
    fixed_root_state[:, 3] = 1.0 

    while simulation_app.is_running():
        start_time = real_time_lib.time()
        
        current_time = (frame_cnt % motion_len) * playback_dt
        expert_frame = amp_loader.get_full_frame_at_time(0, current_time).float()

        # 5. 解包数据
        joint_pos_data = expert_frame[0 : JOINTS_NUM]
        joint_vel_data = expert_frame[JOINTS_NUM : 2 * JOINTS_NUM]

        # 6. 构造全量写入张量
        full_pos = robot.data.default_joint_pos.clone()
        full_vel = torch.zeros_like(full_pos)
        
        # 根据数据类型处理关节位置
        if args_cli.data_type == "relative":
            # 相对模式：将文件中的值作为偏移量加到默认姿态上
            # 公式: q = q_default + q_relative
            full_pos[:, target_indices] += joint_pos_data
        else:
            # 绝对模式：直接覆盖默认姿态
            # 公式: q = q_absolute
            full_pos[:, target_indices] = joint_pos_data

        # 速度通常始终是绝对的，直接填入
        full_vel[:, target_indices] = joint_vel_data

        # 7. 写入仿真器
        robot.write_root_pose_to_sim(fixed_root_state[:, :7])
        robot.write_root_velocity_to_sim(fixed_root_state[:, 7:])
        robot.write_joint_state_to_sim(position=full_pos, velocity=full_vel)
        
        env.scene.write_data_to_sim()
        env.scene.update(dt=playback_dt)
        env.sim.render()

        frame_cnt += 1

        used_time = real_time_lib.time() - start_time
        if used_time < playback_dt:
            real_time_lib.sleep(playback_dt - used_time)

    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()