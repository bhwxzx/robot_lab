"""This script demonstrates how to use the interactive scene interface to setup a scene with multiple prims.

.. code-block:: bash

    # Usage
    python replay_npz_LW.py --motion_file source/robot_lab/robot_lab/datasets/LW_Leg/motion_beyongdmimic/leg_to_wheel_transform_60hz.npz
    --robot LW_Leg
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import numpy as np
import torch

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Replay converted motions.")
parser.add_argument("--motion_file", type=str, help="The name of the input npz motion file.")
parser.add_argument("--robot", type=str, required=True, choices=["LW_Leg", "LW_Wheel"], help="The name of the robot.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

##
# Pre-defined configs
##
from robot_lab.assets.LW import LW_LEG_CFG, LW_WHEEL_CFG
from robot_lab.tasks.manager_based.beyondmimic.mdp import MotionLoader


@configclass
class ReplayMotionsSceneCfg(InteractiveSceneCfg):
    """Configuration for a replay motions scene."""

    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )

    # articulation
    if args_cli.robot=="LW_Leg":
        robot: ArticulationCfg = LW_LEG_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    elif args_cli.robot=="LW_Wheel":
        robot: ArticulationCfg = LW_WHEEL_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    # Extract scene entities
    robot: Articulation = scene["robot"]
    # # ==================== [核心：打印底层真实顺序] ====================
    # print("\n" + "="*50)
    # print("🎯 .npz 文件中 joint_pos/vel 的真实存储顺序 (Isaac Sim 原生顺序):")
    # for i, name in enumerate(robot.joint_names):
    #     print(f"  [{i}]: {name}")
        
    # print("\n🎯 .npz 文件中 body_pos/quat/vel 的真实存储顺序 (Isaac Sim 原生连杆顺序):")
    # for i, name in enumerate(robot.body_names):
    #     print(f"  [{i}]: {name}")
    # print("="*50 + "\n")
    # # =================================================================
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()

    # registry_name = args_cli.registry_name
    # if ":" not in registry_name:  # Check if the registry name includes alias, if not, append ":latest"
    #     registry_name += ":latest"
    # import pathlib

    # import wandb

    # api = wandb.Api()
    # artifact = api.artifact(registry_name)
    motion_file = args_cli.motion_file

    motion = MotionLoader(
        motion_file,
        torch.tensor([0], dtype=torch.long, device=sim.device),
        sim.device,
    )
    time_steps = torch.zeros(scene.num_envs, dtype=torch.long, device=sim.device)

    # Simulation loop
    while simulation_app.is_running():
        time_steps += 1
        reset_ids = time_steps >= motion.time_step_total
        time_steps[reset_ids] = 0

        root_states = robot.data.default_root_state.clone()
        root_states[:, :3] = motion.body_pos_w[time_steps][:, 0] + scene.env_origins[:, None, :]
        root_states[:, 3:7] = motion.body_quat_w[time_steps][:, 0]
        root_states[:, 7:10] = motion.body_lin_vel_w[time_steps][:, 0]
        root_states[:, 10:] = motion.body_ang_vel_w[time_steps][:, 0]

        robot.write_root_state_to_sim(root_states)
        robot.write_joint_state_to_sim(motion.joint_pos[time_steps], motion.joint_vel[time_steps])
        scene.write_data_to_sim()
        sim.render()  # We don't want physic (sim.step())
        scene.update(sim_dt)

        pos_lookat = root_states[0, :3].cpu().numpy()
        # sim.set_camera_view(pos_lookat + np.array([2.0, 2.0, 0.5]), pos_lookat)


def main():
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim_cfg.dt = 0.02
    sim = SimulationContext(sim_cfg)

    scene_cfg = ReplayMotionsSceneCfg(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    # Run the simulator
    run_simulator(sim, scene)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
