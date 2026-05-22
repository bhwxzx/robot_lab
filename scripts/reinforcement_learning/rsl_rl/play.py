# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument("--keyboard", action="store_true", default=False, help="Whether to use keyboard.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import time
import torch
import torch.nn as nn 

from rsl_rl.runners import DistillationRunner, OnPolicyRunner, OnPolicyRunnerDwaq, OnPolicyRunnerAmp, OnPolicyRunnerAmpDwaq

from isaaclab.devices import Se2Keyboard, Se2KeyboardCfg
from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import robot_lab.tasks  # noqa: F401  # isort: skip

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from rl_utils import camera_follow

# PLACEHOLDER: Extension template (do not remove this comment)

# --- DWAQ 专用导出包装器 ---
class DWAQDeploymentWrapper(nn.Module):
    """将 DWAQ 模型包装为单 Tensor 输入格式以供部署。"""
    def __init__(self, policy_nn, num_obs):
        super().__init__()
        # 核心修改：只引用部署所需的子模块，这样 JIT 就不会导出巨大的 Critic 权重
        self.encoder_backbone = policy_nn.encoder_backbone
        self.encode_mean_vel = policy_nn.encode_mean_vel
        self.encode_mean_latent = policy_nn.encode_mean_latent
        self.actor = policy_nn.actor
        self.actor_obs_normalizer = policy_nn.actor_obs_normalizer
        
        self.num_obs = num_obs

    def forward(self, obs_history_flat: torch.Tensor):
        # 1. 提取当前帧
        policy_obs = obs_history_flat[:, -self.num_obs:]
        
        # 2. 确定性推理逻辑 (不采样)
        feat = self.encoder_backbone(obs_history_flat)
        mu_v = self.encode_mean_vel(feat)
        mu_l = self.encode_mean_latent(feat)
        latent_code = torch.cat((mu_v, mu_l), dim=-1)
        
        # 3. 拼接并归一化
        combined_obs = torch.cat((latent_code, policy_obs), dim=-1)
        combined_obs = self.actor_obs_normalizer(combined_obs)
        
        # 4. 输出动作
        actions = self.actor(combined_obs)
        # return torch.nan_to_num(actions, nan=0.0)
        return actions

@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with RSL-RL agent."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]

    # override configurations with non-hydra CLI arguments
    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else 64

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # spawn the robot randomly in the grid (instead of their terrain levels)
    env_cfg.scene.terrain.max_init_terrain_level = None
    # reduce the number of terrains to save memory
    if env_cfg.scene.terrain.terrain_generator is not None:
        env_cfg.scene.terrain.terrain_generator.num_rows = 5
        env_cfg.scene.terrain.terrain_generator.num_cols = 5
        env_cfg.scene.terrain.terrain_generator.curriculum = False

    # disable randomization for play
    env_cfg.observations.policy.enable_corruption = False
    # remove random pushing
    env_cfg.events.randomize_apply_external_force_torque = None
    env_cfg.events.push_robot = None
    env_cfg.curriculum.command_levels_lin_vel = None
    env_cfg.curriculum.command_levels_ang_vel = None

    if args_cli.keyboard:
        env_cfg.scene.num_envs = 1
        env_cfg.terminations.time_out = None
        env_cfg.commands.base_velocity.debug_vis = False
        config = Se2KeyboardCfg(
            v_x_sensitivity=env_cfg.commands.base_velocity.ranges.lin_vel_x[1],
            v_y_sensitivity=env_cfg.commands.base_velocity.ranges.lin_vel_y[1],
            omega_z_sensitivity=env_cfg.commands.base_velocity.ranges.ang_vel_z[1],
        )
        controller = Se2Keyboard(config)
        env_cfg.observations.policy.velocity_commands = ObsTerm(
            func=lambda env: torch.tensor(controller.advance(), dtype=torch.float32).unsqueeze(0).to(env.device),
        )

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "OnPolicyRunnerDwaq": # 新增 DWAQ 支持
        runner = OnPolicyRunnerDwaq(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "OnPolicyRunnerAmp": # [新增] AMP 支持
        runner = OnPolicyRunnerAmp(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "OnPolicyRunnerAmpDwaq":
        runner = OnPolicyRunnerAmpDwaq(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)

    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # 提前获取一次观察值，用于确定维度和后续播放
    obs = env.get_observations() 

    # extract the neural network module
    # we do this in a try-except to maintain backwards compatibility.
    try:
        # version 2.3 onwards
        policy_nn = runner.alg.policy
    except AttributeError:
        # version 2.2 and below
        policy_nn = runner.alg.actor_critic

     # --- DWAQ 专用导出逻辑 ---
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    os.makedirs(export_model_dir, exist_ok=True)

    if agent_cfg.class_name in ["OnPolicyRunnerDwaq", "OnPolicyRunnerAmpDwaq"]:
        print(f"[INFO] Detecting {agent_cfg.class_name}. Exporting with deployment wrapper...")
        # --- 1. 维度解析 (适配 3D 张量) ---
        policy_tensor = obs["policy"] # [Batch, Time, Dim]
        num_obs_single = policy_tensor.shape[-1]
        history_length = policy_tensor.shape[1]
        num_policy_total_flat = num_obs_single * history_length # 展平后的总长度

        print(f"[INFO] DWAQ Dimensions -> Single_Obs: {num_obs_single}, History_Len: {history_length}")

        # 重要：将模型先移动到 CPU
        policy_nn.cpu() 

        deployment_model = DWAQDeploymentWrapper(policy_nn, num_obs_single).to("cpu")
        deployment_model.eval()
        
        # 准备样例输入用于 Trace [Batch=1, HistoryLen * ObsDim]
        example_input = torch.zeros(1, num_policy_total_flat).to("cpu")
        
        # 导出为 TorchScript (JIT)
        traced_model = torch.jit.trace(deployment_model, example_input)
        traced_model.save(os.path.join(export_model_dir, "policy.pt"))
        print(f"[SUCCESS] DWAQ JIT model exported to: {export_model_dir}/policy.pt")
        
        # 导出为 ONNX (可选)
        torch.onnx.export(deployment_model, example_input, os.path.join(export_model_dir, "policy.onnx"))

        # ---------- 使用 onnxsim 简化模型 ----------
        print("[INFO] 正在使用 onnxsim 优化并消除冗余计算...")
        try:
            import onnx
            from onnxsim import simplify

            # 加载刚导出的臃肿模型
            model_onnx = onnx.load(os.path.join(export_model_dir, "policy.onnx"))
            
            # 执行核心简化逻辑
            model_simp, check = simplify(model_onnx)
            
            if check:
                # 覆盖保存为优化后的模型
                sim_onnx_path = os.path.join(export_model_dir, "policy.onnx")
                onnx.save(model_simp, sim_onnx_path)
                print(f"[SUCCESS] 优化后的 ONNX 已保存至: {sim_onnx_path}")
            else:
                print("[WARNING] onnxsim 简化失败，模型结构验证未通过。")
        except ImportError:
            print("[WARNING] 未安装 onnxsim。强烈建议运行 'pip install onnxsim' 以获得极致的推理速度！")

        print("\n" + "="*50)
        print("[验证] 开始导出模型数值一致性检查...")
        
        # --- 验证逻辑 ---
        print("\n" + "="*50)
        print("[验证] 开始导出模型数值一致性检查...")
        
        # A. 获取仿真器当前吐出的数据 (Term-First)
        test_policy_raw = obs["policy"].to("cpu")
        
        # B. 运行原始模型推理
        # policy_nn.act_inference 内部会自动处理 Term-First 转 Time-First 并切出当前帧
        with torch.no_grad():
            actions_orig = policy_nn.act_inference(obs.to("cpu"))
            
        # C. 模拟 C++ 部署端输入：手动展平 3D 张量
        # flatten(1, 2) 会把 Time 和 Dim 合并，且保持 Time-First 顺序
        test_flat_input = test_policy_raw.flatten(1, 2)
        
        # D. 运行 JIT 模型
        with torch.no_grad():
            actions_jit = traced_model(test_flat_input)
            
        # 计算误差
        diff = torch.max(torch.abs(actions_orig - actions_jit))
        print(f"[验证] 原始动作样例: {actions_orig[0, :3]}")
        print(f"[验证] JIT 动作样例:  {actions_jit[0, :3]}")
        print(f"[验证] 最大绝对误差: {diff.item():.8e}")

        # 把模型移回原来的设备，以免影响后续的 Play 可视化
        policy_nn.to(agent_cfg.device)
    else:
        # 标准 PPO 导出逻辑
        if hasattr(policy_nn, "actor_obs_normalizer"):
            normalizer = policy_nn.actor_obs_normalizer
        elif hasattr(policy_nn, "student_obs_normalizer"):
            normalizer = policy_nn.student_obs_normalizer
        else:
            normalizer = None

        export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
        export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")

    dt = env.unwrapped.step_dt

    # reset environment
    # obs = env.get_observations()  # 获取观测移到了前面
    timestep = 0
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            # actions = torch.zeros_like(actions)
            # env stepping
            obs, _, dones, _ = env.step(actions)
            # reset recurrent states for episodes that have terminated
            policy_nn.reset(dones)
        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        if args_cli.keyboard:
            camera_follow(env)

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
