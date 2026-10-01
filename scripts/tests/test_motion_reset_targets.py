"""Motion-reset regressions using real target setters and DelayedPDActuator buffers.

Run with: conda run -n isaacsim-5.1 python scripts/tests/test_motion_reset_targets.py
No physics environment is created; the articulation state writes are stubbed.
"""

from types import SimpleNamespace
import unittest
from unittest.mock import patch

from isaaclab.app import AppLauncher

app = AppLauncher(headless=True).app

import omni.kit.app
import torch
from isaaclab.actuators import DelayedPDActuator, DelayedPDActuatorCfg
from isaaclab.assets import Articulation
from isaaclab.utils.types import ArticulationActions

from robot_lab.tasks.manager_based.beyondmimic.config.LW_Leg.flat_env_cfg import LWLegBeyondMimicFlatEnvCfg
from robot_lab.tasks.manager_based.beyondmimic.mdp.commands import MotionCommand, MotionCommandCfg


NATIVE_NAMES = [
    "left_hip_joint", "right_hip_joint", "left_thigh_joint", "right_thigh_joint",
    "left_shank_joint", "right_shank_joint", "left_foot_joint", "left_wheel_joint",
    "right_foot_joint", "right_wheel_joint",
]


class ResetRobot:
    """Stub only the PhysX state writes; use IsaacLab's actual target setters."""

    set_joint_position_target = Articulation.set_joint_position_target
    set_joint_velocity_target = Articulation.set_joint_velocity_target
    set_joint_effort_target = Articulation.set_joint_effort_target

    def __init__(self):
        self._data = SimpleNamespace(
            joint_pos=torch.full((3, 10), 0.1),
            joint_vel=torch.zeros(3, 10),
            joint_pos_target=torch.zeros(3, 10),
            joint_vel_target=torch.full((3, 10), 2.0),
            joint_effort_target=torch.full((3, 10), 3.0),
            soft_joint_pos_limits=torch.tensor([-2.3, 2.3]).expand(3, 10, 2).clone(),
        )
        self.data = self._data
        self.actuators = {}
        for name, ids, stiffness, damping, limit in (
            ("legs", [0, 1, 2, 3, 4, 5], 90.0, 3.0, 120.0),
            ("foots", [6, 8], 28.0, 1.4, 27.0),
            ("wheels", [7, 9], 0.0, 0.5, 40.0),
        ):
            cfg = DelayedPDActuatorCfg(
                joint_names_expr=[NATIVE_NAMES[i] for i in ids],
                stiffness=stiffness, damping=damping, effort_limit=limit,
                min_delay=0, max_delay=3,
            )
            actuator = DelayedPDActuator(
                cfg, joint_names=[NATIVE_NAMES[i] for i in ids], joint_ids=torch.tensor(ids),
                num_envs=3, device="cpu",
            )
            for buffer in self.delay_buffers(actuator):
                buffer.set_time_lag(torch.tensor([0, 1, 3]))
            self.actuators[name] = actuator

    @staticmethod
    def delay_buffers(actuator):
        return (actuator.positions_delay_buffer, actuator.velocities_delay_buffer, actuator.efforts_delay_buffer)

    def write_joint_state_to_sim(self, joint_pos, joint_vel, *, env_ids):
        self.data.joint_pos[env_ids] = joint_pos
        self.data.joint_vel[env_ids] = joint_vel

    def write_root_state_to_sim(self, state, *, env_ids):
        pass

    def write_actuators(self):
        computed = torch.zeros_like(self.data.joint_pos)
        for actuator in self.actuators.values():
            ids = actuator.joint_indices
            actuator.compute(
                ArticulationActions(
                    joint_positions=self.data.joint_pos_target[:, ids],
                    joint_velocities=self.data.joint_vel_target[:, ids],
                    joint_efforts=self.data.joint_effort_target[:, ids],
                ),
                joint_pos=self.data.joint_pos[:, ids], joint_vel=self.data.joint_vel[:, ids],
            )
            computed[:, ids] = actuator.computed_effort
        return computed


class MotionResetTargetTests(unittest.TestCase):
    def setUp(self):
        self.robot = ResetRobot()
        self.command = object.__new__(MotionCommand)
        self.command._debug_vis_handle = None
        self.command._env = SimpleNamespace(
            device="cpu", num_envs=3, scene=SimpleNamespace(env_origins=torch.zeros(3, 3)),
        )
        self.command.robot = self.robot
        self.command.cfg = SimpleNamespace(
            initialize_reset_targets=True, pose_range={}, velocity_range={}, joint_position_range=(0.0, 0.0),
        )
        self.command.body_indexes = [0]
        self.command.motion_anchor_body_index = 0
        self.command.time_steps = torch.zeros(3, dtype=torch.long)
        self.command.motion = SimpleNamespace(
            joint_pos=torch.tensor([[0.03, -0.02, 0.4363, -0.4363, 2.234, -2.234, 0.4712, 0.0, -0.4712, 0.0]]),
            joint_vel=torch.tensor([[0.0, 0.0, 0.02, -0.02, -0.2569, 0.2571, 0.01, 0.5, -0.01, -0.5]]),
            body_pos_w=torch.zeros(1, 1, 3), body_quat_w=torch.tensor([[[1.0, 0.0, 0.0, 0.0]]]),
            body_lin_vel_w=torch.zeros(1, 1, 3), body_ang_vel_w=torch.zeros(1, 1, 3),
        )
        self.sampling = patch.object(self.command, "_adaptive_sampling", return_value=None).start()
        self.addCleanup(patch.stopall)

    def prime_stale_history(self):
        for step in range(5):
            self.robot.data.joint_pos_target.fill_(step * 0.1)
            self.robot.data.joint_vel_target.fill_(2.0 + step)
            self.robot.data.joint_effort_target.fill_(3.0 + step)
            self.robot.write_actuators()

    def test_opt_in_is_scoped_to_lw_leg(self):
        self.assertFalse(MotionCommandCfg().initialize_reset_targets)
        self.assertTrue(LWLegBeyondMimicFlatEnvCfg().commands.motion.initialize_reset_targets)

    def test_resample_uses_sampled_clipped_native_pose_and_zero_desired_velocity(self):
        self.command.cfg.joint_position_range = (0.2, 0.2)
        env_ids = torch.tensor([0, 2])
        untouched = {name: getattr(self.robot.data, name)[1].clone() for name in (
            "joint_pos", "joint_vel", "joint_pos_target", "joint_vel_target", "joint_effort_target",
        )}
        self.command._resample_command(env_ids)
        expected = (self.command.motion.joint_pos + 0.2).clamp(-2.3, 2.3).expand(2, 10)
        torch.testing.assert_close(self.robot.data.joint_pos[env_ids], expected)
        torch.testing.assert_close(self.robot.data.joint_pos_target[env_ids], expected)
        self.assertAlmostEqual(self.robot.data.joint_pos_target[0, 4].item(), 2.3, places=5)
        torch.testing.assert_close(
            self.robot.data.joint_vel[env_ids], self.command.motion.joint_vel.expand(2, 10),
        )
        self.assertEqual(torch.count_nonzero(self.robot.data.joint_vel_target[env_ids]).item(), 0)
        self.assertEqual(torch.count_nonzero(self.robot.data.joint_effort_target[env_ids]).item(), 0)
        for name, value in untouched.items():
            torch.testing.assert_close(getattr(self.robot.data, name)[1], value, rtol=0, atol=0)

    def test_partial_history_reset_preserves_other_envs_lags_pointer_and_rng(self):
        self.prime_stale_history()
        snapshots = []
        for actuator in self.robot.actuators.values():
            for buffer in self.robot.delay_buffers(actuator):
                circular = buffer._circular_buffer
                snapshots.append((buffer, buffer.time_lags.clone(), circular._pointer,
                                  circular._num_pushes.clone(), circular._buffer[:, 1].clone()))
        rng = torch.random.get_rng_state().clone()
        self.command._initialize_reset_targets([0, 2], self.command.motion.joint_pos.expand(2, 10))
        torch.testing.assert_close(torch.random.get_rng_state(), rng, rtol=0, atol=0)
        for buffer, lags, pointer, pushes, untouched in snapshots:
            circular = buffer._circular_buffer
            torch.testing.assert_close(buffer.time_lags, lags, rtol=0, atol=0)
            self.assertEqual(circular._pointer, pointer)
            self.assertEqual(circular._num_pushes[0].item(), 0)
            self.assertEqual(circular._num_pushes[2].item(), 0)
            self.assertEqual(circular._num_pushes[1].item(), pushes[1].item())
            torch.testing.assert_close(circular._buffer[:, 1], untouched, rtol=0, atol=0)

    def test_startup_hold_is_damping_only_and_normal_policy_delay_is_retained(self):
        self.prime_stale_history()
        self.command._resample_command([0, 1, 2])
        bootstrap = self.robot.write_actuators()  # Initial env.reset() actuator write, before policy.
        damping = torch.tensor([3.0] * 6 + [1.4, 0.5, 1.4, 0.5])
        baseline = -damping * self.robot.data.joint_vel
        torch.testing.assert_close(bootstrap, baseline)
        self.assertLess(bootstrap[:, [4, 5]].abs().max().item(), 1.0)
        self.robot.data.joint_pos_target[:, :6] += 0.1
        self.robot.data.joint_vel_target[:, [7, 9]] = 4.0
        for substep in range(1, 5):
            computed = self.robot.write_actuators()
            active = (substep >= torch.tensor([0, 1, 3]) + 1).float()
            torch.testing.assert_close(computed[:, 4], baseline[:, 4] + 9.0 * active)
            torch.testing.assert_close(computed[:, 7], baseline[:, 7] + 2.0 * active)

    def test_auto_reset_without_bootstrap_uses_first_policy_target(self):
        self.prime_stale_history()
        self.command._resample_command([2])
        self.robot.data.joint_pos_target[2, :6] += 0.1
        self.robot.data.joint_vel_target[2, [7, 9]] = 4.0
        computed = self.robot.write_actuators()
        self.assertAlmostEqual(computed[2, 4].item(), 9.0 - 3.0 * self.command.motion.joint_vel[0, 4].item(), places=4)
        self.assertAlmostEqual(computed[2, 7].item(), 0.5 * (4.0 - 0.5), places=5)

    def test_disabled_and_empty_resamples_preserve_target_history(self):
        self.prime_stale_history()
        before = self.robot.data.joint_pos_target.clone()
        self.command.cfg.initialize_reset_targets = False
        self.command._resample_command([0, 2])
        torch.testing.assert_close(self.robot.data.joint_pos_target, before, rtol=0, atol=0)
        self.command.cfg.initialize_reset_targets = True
        self.command._resample_command([])
        self.sampling.assert_called_once()
        for actuator in self.robot.actuators.values():
            for buffer in self.robot.delay_buffers(actuator):
                torch.testing.assert_close(buffer._circular_buffer._num_pushes, torch.full((3,), 5), rtol=0, atol=0)


if __name__ == "__main__":
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(MotionResetTargetTests))
    exit_code = 0 if result.wasSuccessful() else 1
    omni.kit.app.get_app().post_quit(exit_code)
    app.close()
    raise SystemExit(exit_code)
