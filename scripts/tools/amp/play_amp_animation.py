"""
Script to play AMP animation and generate dataset using Manager-based Env.
"""

import argparse
import numpy as np
import torch
import os

from isaaclab.app import AppLauncher

"""
输入可视化数据格式：  [robot_pos(xyz) robot_rot(euler xyz) joint_pos robot_lin_vel robot_ang_vel joint_vel]
输出amp训练数据格式: [joint_pos joint_vel end_effector_pos]
"""

# 添加参数
parser = argparse.ArgumentParser(description="Play AMP animation with Manager-based Env.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments.")
parser.add_argument("--fps", type=float, default=30.0, help="Target recording fps.")
parser.add_argument("--save_path", type=str, default=None, help="Path to save the recording.")

# Append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# Launch App
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Imports after launching app
from isaaclab.envs import ManagerBasedRLEnv
# 可视化全省，用 Display 版
from rsl_rl.utils import AMPLoaderDisplay 
from scipy.spatial.transform import Rotation
from robot_lab.envs.lw_leg.walking_cfg import LWLegAmpEnvCfg

JOINTS_NUM = 10
def main():

    env_cfg = LWLegAmpEnvCfg() 
    
    # 强制修改配置以适应“录制模式”
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.scene.terrain.terrain_type = "plane" # 强制平面
    env_cfg.scene.terrain.terrain_generator = None
    # 关闭所有随机化事件
    if hasattr(env_cfg, "events"):
        env_cfg.events = None 
    if hasattr(env_cfg, "domain_rand"):
        env_cfg.domain_rand = None
        
    # 2. 创建环境
    env = ManagerBasedRLEnv(cfg=env_cfg)
    
    # 3. 获取机器人句柄
    # 假设你的机器人实体名字叫 "robot"
    robot = env.scene["robot"]
    print(f"Robot initialized with {robot.num_joints} joints.")

    # 4. 初始化 AMP Loader
    # 这里的路径需要指向你的原始数据
    motion_files = ["source/robot_lab/datasets/LW_Leg/motion_visualization/walk.txt"] # 修改这里
    # 注意：AMPLoaderDisplay 需要你之前修改内部参数
    amp_loader = AMPLoaderDisplay(
        device=env.device,
        time_between_frames=env.step_dt, 
        motion_files=motion_files
    )
    motion_len = amp_loader.trajectory_num_frames[0]
    
    # 5. 准备录制
    frame_cnt = 0
    all_frames = []
    dt = 1.0 / args_cli.fps
    
    print("Starting playback...")
    
    # 仿真循环
    while simulation_app.is_running():
        # 计算时间
        time = (frame_cnt % motion_len) * dt
        
        # A. 从 Loader 获取专家数据 (Raw Expert Data)
        # get_full_frame_at_time 返回的是原始的一帧数据
        expert_frame = amp_loader.get_full_frame_at_time(0, time)
        
        # B. 解析数据并应用到机器人 (Force State)
        # -------------------------------------------------
        # 1. Root Position
        root_pos = expert_frame[0:3].clone()
        root_pos[2] += 0.55 # 抬高一点，防止脚陷地 (根据模型调整)
        
        # 2. Root Rotation (Euler -> Quat)
        # expert_frame[3:6] 是欧拉角
        euler = expert_frame[3:6].cpu().numpy()
        quat_xyzw = Rotation.from_euler("XYZ", euler, degrees=False).as_quat()
        quat_wxyz = torch.tensor([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], device=env.device)
        
        # 3. Joint Positions
        # 假设关节数据从 index 6 开始
        joint_pos_target = expert_frame[6 : 6 + JOINTS_NUM]
        
        # 4. Velocities
        lin_vel = expert_frame[6 + JOINTS_NUM : 9 + JOINTS_NUM]
        ang_vel = expert_frame[9 + JOINTS_NUM : 12 + JOINTS_NUM]
        # 假设关节速度从 32 开始 (根据你的 loader 逻辑调整)
        joint_vel_target = expert_frame[12 + JOINTS_NUM: 12 + JOINTS_NUM + JOINTS_NUM] 

        # C. 写入仿真器 (Write to Sim)
        # 构建 Root State [N, 13] -> (pos, quat, lin_vel, ang_vel)
        root_state = torch.zeros((args_cli.num_envs, 13), device=env.device)
        root_state[:, 0:3] = root_pos
        root_state[:, 3:7] = quat_wxyz
        root_state[:, 7:10] = lin_vel
        root_state[:, 10:13] = ang_vel
        
        # 写入状态
        robot.write_root_pose_to_sim(root_state[:, :7])
        robot.write_root_velocity_to_sim(root_state[:, 7:])
        robot.write_joint_state_to_sim(
            position=joint_pos_target.repeat(args_cli.num_envs, 1),
            velocity=joint_vel_target.repeat(args_cli.num_envs, 1)
        )
        
        # D. 刷新物理引擎
        # 必须调用 write_data_to_sim 才能生效
        env.scene.write_data_to_sim()
        # 执行一步物理模拟 (不进行 RL 逻辑)
        env.sim.step()
        # 更新场景状态 (包括传感器数据)
        env.scene.update(dt=env.step_dt)
        
        # E. 计算 AMP 观测 (Capture Obs)
        # -------------------------------------------------
        # 这是 Manager-based 最强大的地方：直接让 ObservationManager 计算
        # 需要定义了名为 "amp" 的观测组
        if args_cli.save_path:
            # 这一步会自动计算关节位置、速度和末端位置
            amp_obs = env.observation_manager.compute_group("amp")
            
            # 保存这一帧
            # 只保存第一个环境的数据
            frame_data = amp_obs[0].cpu().numpy()
            all_frames.append(frame_data)

        frame_cnt += 1
        
        # 检查是否结束
        if frame_cnt >= (motion_len - 1):
            if args_cli.save_path:
                print("Recording finished.")
                break
            else:
                frame_cnt = 0 # 循环播放

    # 6. 保存文件
    if args_cli.save_path:
        save_dataset(args_cli.save_path, all_frames, 1.0 / args_cli.fps)
        
    env.close()

def save_dataset(path, frames, frame_duration):
    """Helper to save frames in AMP JSON format."""
    import numpy as np
    
    all_frames_np = np.stack(frames, axis=0)
    # 先存临时 txt
    np.savetxt(path, all_frames_np, fmt='%f', delimiter=', ')

    with open(path, 'r') as f:
        frames_data = f.readlines()

    with open(path, 'w') as f:
        f.write('{\n')
        f.write('"LoopMode": "Wrap",\n')
        f.write(f'"FrameDuration": {frame_duration:.3f},\n')
        f.write('"EnableCycleOffsetPosition": true,\n')
        f.write('"EnableCycleOffsetRotation": true,\n')
        f.write('"MotionWeight": 0.5,\n\n')
        f.write('"Frames":\n[\n')

        for i, line in enumerate(frames_data):
            line_start_str = '  ['
            # 移除最后一行可能的逗号
            suffix = ']\n' if i == len(frames_data) - 1 else '],\n'
            f.write(line_start_str + line.rstrip() + suffix)

        f.write(']\n}')
    print(f"✅ Successfully saved to {path}")

if __name__ == "__main__":
    main()
    simulation_app.close()