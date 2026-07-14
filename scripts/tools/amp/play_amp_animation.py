"""
Script to play AMP animation and generate dataset using Manager-based Env.
"""

import argparse
import numpy as np
import torch
import os
import json
import time as real_time_lib

from isaaclab.app import AppLauncher
from isaaclab.utils.math import quat_apply, quat_from_euler_xyz, quat_slerp

"""
输入可视化数据格式：  [robot_pos(xyz) robot_rot(euler xyz) joint_pos robot_lin_vel robot_ang_vel joint_vel]
输出amp训练数据格式: [joint_pos joint_vel end_effector_pos]
"""
"""
使用示例:
    # 1. 回放原始数据
    python play_amp_animation.py --motion_file motion_01.txt

    # 2. 自动生成路径录制 (保存到原文件目录平级的 motion_amp_expert 文件夹下)
    python play_amp_animation.py --motion_file datasets/raw/walk.txt --save_path

    # 3. 指定特定路径录制
    python play_amp_animation.py --motion_file motion_01.txt --save_path /tmp/new_amp_data.txt
"""

# 添加参数
parser = argparse.ArgumentParser(description="Play AMP animation with Manager-based Env.")
parser.add_argument("--motion_file", type=str, required=True, help="Path to the original expert motion file (e.g. motion.txt).")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments.")
parser.add_argument("--fps", type=float, default=30.0, help="Target recording fps.")
parser.add_argument(
    "--save_path", 
    type=str, 
    nargs='?',    # 表示参数可选（0个或1个）
    const='auto', # 如果写了标志但没写值，赋予此常量
    default=None, # 如果完全没写标志，保持 None
    help="Path to save the recording. If provided without a value, generates path automatically."
)

# Append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

is_recording = args_cli.save_path is not None
if is_recording:
    print("[Info] Running in HEADLESS mode (no GUI).")
    args_cli.headless = True

# Launch App
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Imports after launching app
from isaaclab.envs import ManagerBasedRLEnv
# 可视化全省，用 Display 版
from rsl_rl.utils import AMPLoaderDisplay # 内部配置需要修改正确
from scipy.spatial.transform import Rotation
from robot_lab.tasks.manager_based.locomotion.velocity.config.LW.LW_Leg.flat_env_cfg import LWLegFlatAmpEnvCfg_Play

# --- 配置区域 ---
AMP_TARGET_JOINTS = [
    "right_hip_joint", "left_hip_joint",
    "right_thigh_joint", "left_thigh_joint",
    "right_shank_joint", "left_shank_joint",
    "right_foot_joint", "left_foot_joint"
]

# 动态计算关节数量
JOINTS_NUM = len(AMP_TARGET_JOINTS)

# 定义偏移量常量 (根据输入数据的结构)
# 数据结构: [Pos(3) Rot(3) JointPos(N) LinVel(3) AngVel(3) JointVel(N)]
IDX_ROOT_POS = 0
IDX_ROOT_ROT = 3
IDX_JOINT_POS = 6
IDX_LIN_VEL = 6 + JOINTS_NUM
IDX_ANG_VEL = 6 + JOINTS_NUM + 3
IDX_JOINT_VEL = 6 + JOINTS_NUM + 3 + 3

print(f"DEBUG: Original JOINT_POS_SIZE in library: {AMPLoaderDisplay.JOINT_POS_SIZE}")

# 强制设定关节数量
AMPLoaderDisplay.JOINT_POS_SIZE = JOINTS_NUM  # 设为 8
AMPLoaderDisplay.JOINT_VEL_SIZE = JOINTS_NUM  # 设为 8

# 重新计算索引 (完全复制类定义中的计算逻辑)
AMPLoaderDisplay.JOINT_POSE_START_IDX = 6
AMPLoaderDisplay.JOINT_POSE_END_IDX = 6 + JOINTS_NUM

# 跳过中间的 Root LinVel(3) + AngVel(3) = 6
AMPLoaderDisplay.JOINT_VEL_START_IDX = AMPLoaderDisplay.JOINT_POSE_END_IDX + 6 
AMPLoaderDisplay.JOINT_VEL_END_IDX = AMPLoaderDisplay.JOINT_VEL_START_IDX + JOINTS_NUM

print(f"DEBUG: Patched JOINT_VEL_START_IDX: {AMPLoaderDisplay.JOINT_VEL_START_IDX}") 
print(f"DEBUG: Patched JOINT_VEL_END_IDX: {AMPLoaderDisplay.JOINT_VEL_END_IDX}")   

# 原AMPLoaderDisplay的 blend_frame_pose 会把 Root 信息扔掉，导致返回维度变小
# 禁用 Loader 内部的自动插值，改为手动处理
def no_op_blend(self, frame0, frame1, blend):
    # 暂时返回 frame0，我们在 main 循环中手动做高质量插值
    return frame0, frame1, blend 

AMPLoaderDisplay.blend_frame_pose = no_op_blend

# 将补丁应用到类上
AMPLoaderDisplay.blend_frame_pose = no_op_blend
print("DEBUG: Successfully patched blend_frame_pose to return full frames.")

def get_quat_from_data(euler_vec, device):
    """
    将数据中的欧拉角转为 Isaac 要求的 WXYZ 四元数。
    """
    # 假设 TXT 里的顺序是 Roll, Pitch, Yaw (对应的轴是 X, Y, Z)
    # degrees=False 表示输入是弧度
    r = Rotation.from_euler('xyz', euler_vec.cpu().numpy(), degrees=False)
    q = r.as_quat() # 返回 [x, y, z, w]
    return torch.tensor([q[3], q[0], q[1], q[2]], device=device, dtype=torch.float32)

def main():
    # 确定路径和文件名
    input_file = args_cli.motion_file
    input_filename = os.path.basename(input_file)
    
    # 只有在录制模式下才处理路径
    if is_recording:
        # 如果用户只写了 --save_path 没写值，或者显式写了 'auto'
        if args_cli.save_path == 'auto':
            current_dir = os.path.dirname(input_file)
            parent_dir = os.path.dirname(current_dir)
            save_dir = os.path.join(parent_dir, "motion_amp_expert")
            
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            
            final_save_path = os.path.join(save_dir, input_filename)
        else:
            # 使用用户提供的具体路径
            final_save_path = args_cli.save_path
    else:
        final_save_path = None

    print(f"Target Save Path: {final_save_path}")

    # 提取原始文件的 MotionWeight
    print(f"Reading original metadata from: {input_file}")
    with open(input_file, 'r') as f:
        original_json = json.load(f)
        motion_weight = original_json.get("MotionWeight", 0.5)
        print(f"Detected MotionWeight: {motion_weight}")

    env_cfg = LWLegFlatAmpEnvCfg_Play()
    
    # 强制将生成数据的环境历史设为 0 (单帧)，保证输出单帧维度特征
    env_cfg.observations.amp.history_length = 0 
    
    # 强制修改配置
    env_cfg.sim.disable_gravity = True
    env_cfg.scene.robot.spawn.rigid_props.disable_gravity = True
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.scene.terrain.terrain_type = "plane"
    env_cfg.scene.terrain.terrain_generator = None
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
        
    # 创建环境
    env = ManagerBasedRLEnv(cfg=env_cfg)
    robot = env.scene["robot"]
    print(f"Robot initialized with {robot.num_joints} joints.")

    # 获取关节索引
    joint_name_to_id = {name: i for i, name in enumerate(robot.joint_names)}
    
    # 使用列表推导式一次性生成 Tensor
    try:
        target_indices = torch.tensor([joint_name_to_id[name] for name in AMP_TARGET_JOINTS], 
                                      device=env.device, dtype=torch.long)
    except KeyError as e:
        raise ValueError(f"关节 {e} 在机器人模型中不存在！可用关节: {robot.joint_names}")

    # 打印一行简洁的确认信息
    print(f"[Map] Mapped {len(target_indices)} joints from datasets to Robot indices: {target_indices.tolist()}")
    
    if len(target_indices) != JOINTS_NUM:
        raise ValueError(f"Error: Expected {JOINTS_NUM} joints from list, but found {len(target_indices)} in robot model. Check joint names.")
    
    # 初始化 AMP Loader
    amp_loader = AMPLoaderDisplay(
        device=env.device,
        time_between_frames=env.step_dt, 
        motion_files=[input_file]
    )
    motion_len = amp_loader.trajectory_num_frames[0]

    # 验证数据维度是否正确加载
    print(f"Loaded Trajectory Shape: {amp_loader.trajectories[0].shape}")
    
    frame_cnt = 0
    all_frames = []
    dt = 1.0 / args_cli.fps
    
    print("Starting playback...")
    
    while simulation_app.is_running():
        start_time = real_time_lib.time()
        # 计算时间进度
        current_time = (frame_cnt % motion_len) * dt
        # 手动计算插值
        p = current_time / amp_loader.trajectory_lens[0]
        n = motion_len

        raw_idx_low = np.floor(p * n)
        raw_idx_high = np.ceil(p * n)
        
        # 显式转换为整数，并进行边界裁剪
        idx_low = int(np.clip(raw_idx_low, 0, n - 1))
        idx_high = int(np.clip(raw_idx_high, 0, n - 1))
        
        # 此时再进行索引就不会报错了
        f0 = amp_loader.trajectories_full[0][idx_low]
        f1 = amp_loader.trajectories_full[0][idx_high]
        blend = p * n - idx_low

        # 1. 插值 Root Position
        pos0, pos1 = f0[IDX_ROOT_POS:IDX_ROOT_POS+3], f1[IDX_ROOT_POS:IDX_ROOT_POS+3]
        root_pos = (1.0 - blend) * pos0 + blend * pos1
        root_pos[2] += 0.1

        # 2. 使用四元数 Slerp 处理旋转，杜绝方向跳变
        q0 = get_quat_from_data(f0[IDX_ROOT_ROT:IDX_ROOT_ROT+3], env.device)
        q1 = get_quat_from_data(f1[IDX_ROOT_ROT:IDX_ROOT_ROT+3], env.device)
        
        # quat_slerp 内部会处理 q1 和 q2 的点积
        quat_wxyz = quat_slerp(q0, q1, blend) 
        
        # 确保归一化
        quat_wxyz = torch.nn.functional.normalize(quat_wxyz, p=2, dim=-1)
        
        # 3. 插值关节和其他向量 (线性插值)
        joint_pos = (1.0 - blend) * f0[IDX_JOINT_POS : IDX_JOINT_POS + JOINTS_NUM] + \
                    blend * f1[IDX_JOINT_POS : IDX_JOINT_POS + JOINTS_NUM]
        
        joint_vel = (1.0 - blend) * f0[IDX_JOINT_VEL : IDX_JOINT_VEL + JOINTS_NUM] + \
                    blend * f1[IDX_JOINT_VEL : IDX_JOINT_VEL + JOINTS_NUM]
        
        lin_vel_b = (1.0 - blend) * f0[IDX_LIN_VEL : IDX_LIN_VEL + 3] + \
                    blend * f1[IDX_LIN_VEL : IDX_LIN_VEL + 3]
        
        ang_vel_b = (1.0 - blend) * f0[IDX_ANG_VEL : IDX_ANG_VEL + 3] + \
                    blend * f1[IDX_ANG_VEL : IDX_ANG_VEL + 3]

        # 4. 填充全量 Buffer
        batch_size = args_cli.num_envs
        full_pos = robot.data.default_joint_pos.clone().repeat(batch_size, 1)
        full_vel = torch.zeros((batch_size, robot.num_joints), device=env.device)
        full_pos[:, target_indices] = joint_pos
        full_vel[:, target_indices] = joint_vel

        # 5. 设置 Root State
        root_state = torch.zeros((batch_size, 13), device=env.device, dtype=torch.float32)
        root_state[:, 0:3] = root_pos.float()
        root_state[:, 3:7] = quat_wxyz.float() # 自动广播到所有 env
        
        # 将速度转到世界系
        # q_in: [1, 4], v_in: [1, 3]
        q_in = quat_wxyz.float().unsqueeze(0)
        v_lin_in = lin_vel_b.float().unsqueeze(0)
        v_ang_in = ang_vel_b.float().unsqueeze(0)
        
        # 应用旋转并 repeat 到所有环境
        root_state[:, 7:10] = quat_apply(q_in, v_lin_in).repeat(batch_size, 1)
        root_state[:, 10:13] = quat_apply(q_in, v_ang_in).repeat(batch_size, 1)

        # 6. 写入并渲染
        robot.write_root_pose_to_sim(root_state[:, :7])
        robot.write_root_velocity_to_sim(root_state[:, 7:])
        robot.write_joint_state_to_sim(position=full_pos, velocity=full_vel)
        
        env.scene.write_data_to_sim()
        env.scene.update(dt=env.step_dt)
        
        if is_recording:
            amp_obs = env.observation_manager.compute_group("amp")
            all_frames.append(amp_obs[0].cpu().numpy())

        # 只渲染但不进行物理交互仿真
        env.sim.render()

        frame_cnt += 1

        # 如果不是录制模式，为了肉眼观察舒适，控制循环速度匹配 FPS
        if not is_recording and not args_cli.headless:
            used_time = real_time_lib.time() - start_time
            if used_time < dt:
                real_time_lib.sleep(dt - used_time)

        if is_recording and frame_cnt >= (motion_len - 1):
            break

    if is_recording:
         save_dataset(final_save_path, all_frames, 1.0 / args_cli.fps, motion_weight)
        
    env.close()

def save_dataset(path, frames, frame_duration, weight):
    """Helper to save frames in AMP JSON format."""
    all_frames_np = np.stack(frames, axis=0)
    temp_txt = path + ".tmp"
    np.savetxt(temp_txt, all_frames_np, fmt='%f', delimiter=', ')

    with open(temp_txt, 'r') as f:
        frames_data = f.readlines()

    with open(path, 'w') as f:
        f.write('{\n')
        f.write('"LoopMode": "Wrap",\n')
        f.write(f'"FrameDuration": {frame_duration:.3f},\n')
        f.write('"EnableCycleOffsetPosition": true,\n')
        f.write('"EnableCycleOffsetRotation": true,\n')
        f.write(f'"MotionWeight": {weight},\n\n')
        f.write('"Frames":\n[\n')

        for i, line in enumerate(frames_data):
            line_start_str = '  ['
            suffix = ']\n' if i == len(frames_data) - 1 else '],\n'
            f.write(line_start_str + line.rstrip() + suffix)

        f.write(']\n}')
    
    if os.path.exists(temp_txt):
        os.remove(temp_txt)
    print(f"✅ Successfully converted and saved to {path}")

if __name__ == "__main__":
    main()
    simulation_app.close()