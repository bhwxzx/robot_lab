"""Ground-contact termination regressions; use --sim for a bounded 4-env PhysX smoke test.

CPU: conda run -n isaacsim-5.1 python -B scripts/tests/test_thigh_shank_ground_contact_termination.py
Sim: append --sim (headless, persistent settings disabled, eight control steps).
USD filter check: --filter-probe, then --filter-probe --unfiltered for the positive control.
"""

from __future__ import annotations

import ast
import math
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest

import torch


REPO = Path(__file__).resolve().parents[2]
TASK = REPO / "source/robot_lab/robot_lab/tasks/manager_based/beyondmimic"
BODIES = ("right_thigh_link", "left_thigh_link", "right_shank_link", "left_shank_link")
BODY_IDS = [2, 3, 4, 5]
SENSOR_CFG = SimpleNamespace(name="contact_forces", body_ids=BODY_IDS)


def load_term():
    path = TASK / "mdp/terminations.py"
    nodes = [node for node in ast.parse(path.read_text()).body if (
        isinstance(node, ast.ImportFrom) and node.module == "__future__"
    ) or (isinstance(node, ast.FunctionDef) and node.name == "thigh_shank_ground_contact")]
    namespace = {"math": math, "torch": torch}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace["thigh_shank_ground_contact"]


TERM = load_term()


def make_env():
    scene = {
        "contact_forces": SimpleNamespace(data=SimpleNamespace(
            net_forces_w_history=torch.zeros(3, 4, 11, 3),
        ))
    }
    return SimpleNamespace(num_envs=3, device="cpu", cfg=SimpleNamespace(decimation=4), scene=scene)


class GroundContactTests(unittest.TestCase):
    def test_no_contact_is_not_termination(self):
        self.assertEqual(TERM(make_env(), SENSOR_CFG).tolist(), [False, False, False])

    def test_each_of_four_links_can_terminate_only_its_own_environment(self):
        for body_id in BODY_IDS:
            with self.subTest(body_id=body_id):
                env = make_env()
                env.scene["contact_forces"].data.net_forces_w_history[1, 0, body_id, 2] = 1.01
                self.assertEqual(TERM(env, SENSOR_CFG).tolist(), [False, True, False])

    def test_one_newton_boundary_and_short_contact_at_first_physics_substep(self):
        env = make_env()
        history = env.scene["contact_forces"].data.net_forces_w_history
        history[0, 3, BODY_IDS[0], 2] = 1.0
        history[1, 3, BODY_IDS[0], 2] = 1.01  # Oldest of four samples; latest sample is zero.
        history[2, 0, BODY_IDS[0], 2] = 0.99
        self.assertEqual(TERM(env, SENSOR_CFG).tolist(), [False, True, False])

    def test_reset_clears_contact_for_selected_environment_without_state_leak(self):
        env = make_env()
        data = env.scene["contact_forces"].data
        data.net_forces_w_history[:2, 1, BODY_IDS[3], 2] = 10.0
        self.assertEqual(TERM(env, SENSOR_CFG).tolist(), [True, True, False])
        data.net_forces_w_history[1] = 0.0
        self.assertEqual(TERM(env, SENSOR_CFG).tolist(), [True, False, False])

    def test_contacts_on_unselected_links_do_not_terminate(self):
        env = make_env()
        other_ids = [i for i in range(11) if i not in BODY_IDS]
        env.scene["contact_forces"].data.net_forces_w_history[:, :, other_ids] = 10000.0
        self.assertEqual(TERM(env, SENSOR_CFG).tolist(), [False, False, False])

    def test_inputs_are_not_modified(self):
        env = make_env()
        data = env.scene["contact_forces"].data
        data.net_forces_w_history[1, 0, BODY_IDS[0], 2] = 4.0
        before = data.net_forces_w_history.clone()
        TERM(env, SENSOR_CFG)
        torch.testing.assert_close(data.net_forces_w_history, before)

    def test_incomplete_contact_history_fails_explicitly(self):
        for shape in ((3, 3, 11, 3), (3, 11, 3)):
            with self.subTest(shape=shape):
                env = make_env()
                env.scene["contact_forces"].data.net_forces_w_history = torch.zeros(shape)
                with self.assertRaises(RuntimeError):
                    TERM(env, SENSOR_CFG)

    def test_invalid_threshold_is_rejected(self):
        for threshold in (0.0, -1.0, math.nan, math.inf):
            with self.subTest(threshold=threshold), self.assertRaises(ValueError):
                TERM(make_env(), SENSOR_CFG, threshold)


def run_sim_smoke(filter_probe=False, unfiltered=False):
    """Verify USD filters or control-step ground termination and reset isolation."""
    import json

    from isaaclab.app import AppLauncher

    app = AppLauncher(headless=True, device="cuda:0", kit_args="--/app/settings/persistent=false").app
    env = None
    try:
        import gymnasium as gym
        import robot_lab.tasks  # noqa: F401
        from isaaclab_tasks.utils import parse_env_cfg
        from robot_lab.tasks.manager_based.beyondmimic.config.LW_Leg.flat_env_cfg import LWLegBeyondMimicFlatEnvCfg
        from robot_lab.tasks.manager_based.beyondmimic import mdp

        assert not hasattr(LWLegBeyondMimicFlatEnvCfg().terminations, "thigh_shank_ground_contact")
        cfg = parse_env_cfg("RobotLab-Isaac-BeyondMimic-Flat-LW-wheel-v0", device="cuda:0", num_envs=4)
        cfg.seed = 42
        cfg.commands.motion.debug_vis = False
        cfg.commands.motion.pose_range = {}
        cfg.commands.motion.velocity_range = {}
        cfg.commands.motion.joint_position_range = (0.0, 0.0)
        cfg.observations.policy.enable_corruption = False
        cfg.scene.contact_forces.debug_vis = False
        for name in ("physics_material", "add_joint_default_pos", "base_com", "push_robot",
                     "randomize_rigid_body_mass_base", "randomize_rigid_body_mass_others", "randomize_actuator_gains"):
            if hasattr(cfg.events, name):
                setattr(cfg.events, name, None)
        # Isolate the approved termination; other terms are covered by existing tests.
        for name in tuple(vars(cfg.terminations)):
            if filter_probe or name != "thigh_shank_ground_contact":
                setattr(cfg.terminations, name, None)
        filter_groups = {}
        if filter_probe:
            cfg.scene.env_spacing = 10.0
            # Test-only geometry deliberately overlaps nonadjacent robot links.
            # Leave the USD assets and their persistent layers unchanged.
            original_spawn = cfg.scene.robot.spawn.func

            def spawn_overlap(*args, **kwargs):
                from pxr import UsdGeom, UsdPhysics, PhysxSchema
                from isaaclab.sim import SimulationContext

                prim = original_spawn(*args, **kwargs)
                stage = SimulationContext.instance().stage
                # Author before InteractiveScene clones and replicates env_0.
                for i in (0,):
                    root = f"/World/envs/env_{i}/Robot"
                    assert PhysxSchema.PhysxArticulationAPI(
                        stage.GetPrimAtPath(root + "/base_link")
                    ).GetEnabledSelfCollisionsAttr().Get() is True
                    for name in ("GroupBase", "GroupFoot", "GroupWheel", "GroupOthers"):
                        group = UsdPhysics.CollisionGroup(stage.GetPrimAtPath(root + "/" + name))
                        paths = group.GetFilteredGroupsRel().GetTargets()
                        assert paths
                        filter_groups[f"{i}/{name}"] = [str(p) for p in paths]
                        if unfiltered:
                            group.GetFilteredGroupsRel().SetTargets([])
                    for body in (*BODIES, "base_link"):
                        sphere = UsdGeom.Sphere.Define(stage, root + "/" + body + "/filter_test_sphere")
                        sphere.CreateRadiusAttr(0.7)
                        UsdPhysics.CollisionAPI.Apply(sphere.GetPrim())
                return prim

            cfg.scene.robot.spawn.func = spawn_overlap
        env = gym.make("RobotLab-Isaac-BeyondMimic-Flat-LW-wheel-v0", cfg=cfg).unwrapped
        command = env.command_manager.get_term("motion")
        command._adaptive_sampling = lambda ids: command.time_steps.__setitem__(ids, 0)
        env.reset()
        if filter_probe:
            robot = env.scene["robot"]
            state = robot.data.root_state_w.clone()
            state[:, 2] += 10.0  # No ground contact in this controlled self-contact test.
            state[:, 7:] = 0.0
            robot.write_root_state_to_sim(state)
            ids, _ = env.scene["contact_forces"].find_bodies(list(BODIES), preserve_order=True)
            peaks = torch.zeros(4, 4, device=env.device)
            for _ in range(8):
                env.step(torch.zeros(4, env.action_manager.total_action_dim, device=env.device))
                forces = env.scene["contact_forces"].data.net_forces_w_history[:, :, ids]
                peaks = torch.maximum(peaks, forces.norm(dim=-1).amax(dim=1))
            if unfiltered:
                assert (peaks > 1.0).all(), peaks
            else:
                assert (peaks < 0.001).all(), peaks
            print("GROUND_CONTACT_SMOKE=" + json.dumps({
                "kind": "usd_collision_filter_probe", "filters_cleared_in_memory": unfiltered,
                "num_envs": 4, "control_steps": 8, "peak_self_contact_n": peaks.tolist(),
                "authored_filter_groups": filter_groups,
            }), flush=True)
            return
        assert cfg.terminations.thigh_shank_ground_contact.time_out is False
        assert not any(name.startswith("termination_self_") for name in env.scene.sensors)
        sensor_cfg = env.termination_manager.get_term_cfg("thigh_shank_ground_contact").params["sensor_cfg"]
        assert set(sensor_cfg.body_names) == set(BODIES)
        assert env.scene["contact_forces"].data.net_forces_w_history.shape == (4, 4, 11, 3)
        assert env.scene["contact_forces"].contact_physx_view.filter_count == 0
        record = {"contact_env_ids": [], "reset_histories_cleared": True, "normal_wheel_support_seen": False,
                  "num_envs": 4, "control_steps": 8, "sensor_names": list(env.scene.sensors)}
        original_reset = env._reset_idx

        def reset_and_check(ids):
            flags = mdp.thigh_shank_ground_contact(env, sensor_cfg).clone()
            assert flags[ids].all()
            record["contact_env_ids"].extend(ids.tolist())
            original_reset(ids)
            # Inspect buffers without refreshing pre-reset PhysX cached contacts.
            assert not env.scene["contact_forces"]._data.net_forces_w_history[ids].any()

        env._reset_idx = reset_and_check
        robot = env.scene["robot"]
        state = robot.data.root_state_w.clone()
        state[[1, 3], 2] -= 0.40  # Deliberate thigh/shank ground intersection in two environments.
        robot.write_root_state_to_sim(state)
        robot.set_joint_position_target(robot.data.joint_pos.clone())
        wheel_ids, _ = env.scene["contact_forces"].find_bodies(["right_wheel_link", "left_wheel_link"])
        for _ in range(8):
            _, _, terminated, truncated, _ = env.step(torch.zeros(4, env.action_manager.total_action_dim, device=env.device))
            assert not truncated.any()
            assert not terminated[[0, 2]].any()
            forces = env.scene["contact_forces"].data.net_forces_w[0, wheel_ids]
            record["normal_wheel_support_seen"] |= bool((forces.norm(dim=-1) > 1.0).any())
        assert set(record["contact_env_ids"]) == {1, 3}, record
        assert len(record["contact_env_ids"]) == 2, record
        assert record["normal_wheel_support_seen"], record
        print("GROUND_CONTACT_SMOKE=" + json.dumps(record), flush=True)
    except BaseException:
        # SimulationApp.close() can exit before Python prints a pending traceback.
        import traceback

        traceback.print_exc()
        sys.stderr.flush()
        raise
    finally:
        if env is not None:
            env.close()
        app.close()


if __name__ == "__main__":
    if "--sim" in sys.argv or "--filter-probe" in sys.argv:
        filter_probe = "--filter-probe" in sys.argv
        unfiltered = "--unfiltered" in sys.argv
        for arg in ("--sim", "--filter-probe", "--unfiltered"):
            if arg in sys.argv:
                sys.argv.remove(arg)
        run_sim_smoke(filter_probe, unfiltered)
    else:
        unittest.main()
