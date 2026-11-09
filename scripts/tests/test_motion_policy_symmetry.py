"""CPU tests for LW policy reflection and telemetry-history reconstruction.

Run with isaacsim-5.1 Python -B. No simulator or training configuration imports.
PPO interface tests use synthetic data and block optimizer updates.
"""

from __future__ import annotations

import ast
import contextlib
import copy
import importlib.util
import io
import math
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

import numpy as np
import torch
from tensordict import TensorDict

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/reinforcement_learning/rsl_rl"))
import diagnose_motion_policy_symmetry as diagnostic

mirror = diagnostic.mirror


def rotation(axis, angle):
    axis = np.asarray(axis, dtype=float)
    axis /= np.linalg.norm(axis)
    x, y, z = axis
    skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + math.sin(angle)*skew + (1-math.cos(angle))*(skew@skew)


class MirrorTests(unittest.TestCase):
    def test_involution_and_no_input_mutation_for_actor_critic_and_actions(self):
        torch.manual_seed(12)
        for width, func in ((590, mirror.mirror_policy_observations),
                            (146, mirror.mirror_critic_observations),
                            (164, mirror.mirror_critic_observations), (10, mirror.mirror_actions)):
            original = torch.randn(2, 3, width, dtype=torch.float64)
            before = original.clone()
            reflected = func(original)
            torch.testing.assert_close(func(reflected), original, rtol=0, atol=0)
            torch.testing.assert_close(original, before, rtol=0, atol=0)
            self.assertEqual(reflected.dtype, original.dtype)

    def test_explicit_joint_order_and_alternate_physical_order(self):
        q = torch.arange(1, 11, dtype=torch.float64)
        expected = torch.tensor([-2, -1, -4, -3, -6, -5, -8, -7, -10, -9], dtype=torch.float64)
        torch.testing.assert_close(mirror.mirror_actions(q), expected)
        names = tuple(reversed(mirror.JOINT_NAMES))
        torch.testing.assert_close(mirror.mirror_joints(q.flip(0), names).flip(0), expected)

    def test_history_time_and_environment_are_preserved(self):
        x = torch.arange(2*590, dtype=torch.float64).reshape(2, 590)
        y = mirror.mirror_policy_observations(x)
        for b in range(2):
            for h in range(10):
                # Commands have q and dq blocks in each history slot.
                self.assertEqual(y[b, h*20].item(), -x[b, h*20+1].item())
                self.assertEqual(y[b, h*20+10].item(), -x[b, h*20+11].item())
                # Angular velocity begins after command and relative-orientation histories.
                torch.testing.assert_close(y[b, 260+3*h:263+3*h],
                                           x[b, 260+3*h:263+3*h]*torch.tensor([-1, 1, -1]))
                self.assertEqual(y[b, 490+h*10].item(), -x[b, 490+h*10+1].item())

    def test_rotation6d_matches_geometric_reflection_with_nonzero_rpy(self):
        reflection = np.diag([1, -1, 1])
        for angles in ((.5, -.7, 1.2), (-.9, .3, -.6), (0, 0, 0)):
            roll, pitch, yaw = angles
            r = rotation([0, 0, 1], yaw) @ rotation([0, 1, 0], pitch) @ rotation([1, 0, 0], roll)
            actual = mirror.mirror_rotation_6d(torch.tensor(r[:, :2].reshape(6)))
            expected = torch.tensor((reflection@r@reflection)[:, :2].reshape(6))
            torch.testing.assert_close(actual, expected)

    def test_critic_body_swap_with_polar_and_axial_signs(self):
        x = torch.arange(146, dtype=torch.float64)[None, :]
        y = mirror.mirror_critic_observations(x)
        # Body positions start at 29; base remains first, right hip receives left hip.
        torch.testing.assert_close(y[0, 29:32], x[0, 29:32]*torch.tensor([1, -1, 1]))
        torch.testing.assert_close(y[0, 32:35], x[0, 35:38]*torch.tensor([1, -1, 1]))
        torch.testing.assert_close(y[0, 110:113], x[0, 110:113]*torch.tensor([1, -1, 1]))
        torch.testing.assert_close(y[0, 113:116], x[0, 113:116]*torch.tensor([-1, 1, -1]))

    def test_axial_signs_follow_cross_product_not_polar_rule(self):
        a, b = torch.tensor([.2, -.3, .4]), torch.tensor([-.8, .7, .1])
        polar = torch.tensor([1, -1, 1])
        axial = torch.tensor([-1, 1, -1])
        torch.testing.assert_close(torch.linalg.cross(a*polar, b*polar), torch.linalg.cross(a, b)*axial)

    def test_wheel_critic_swaps_wheel_positions_and_orientations(self):
        x = torch.arange(164, dtype=torch.float64)[None, :]
        y = mirror.mirror_critic_observations(x)
        # 11 bodies: positions start at 29, orientations at 62; wheels are slots 9 and 10.
        for target, source in ((9, 10), (10, 9)):
            torch.testing.assert_close(y[0, 29+3*target:32+3*target],
                                       x[0, 29+3*source:32+3*source]*torch.tensor([1, -1, 1]))
            torch.testing.assert_close(y[0, 62+6*target:68+6*target],
                                       x[0, 62+6*source:68+6*source]*torch.tensor([1, -1, -1, 1, 1, -1]))
        torch.testing.assert_close(y[0, 128:131], x[0, 128:131]*torch.tensor([1, -1, 1]))
        torch.testing.assert_close(y[0, 131:134], x[0, 131:134]*torch.tensor([-1, 1, -1]))

    def test_mixed_action_targets_commute_when_offsets_are_also_reflected(self):
        # Includes asymmetric startup offsets: the *mirrored environment* must use mirrored defaults.
        a = torch.linspace(-2, 3, 10)
        offset = torch.tensor([.01, -.02, .73, -.72, -.18, .16, .003, -.007, 0, 0])
        scale = torch.tensor([.25]*8+[1.0]*2)
        torch.testing.assert_close(mirror.mirror_actions(a)*scale+mirror.mirror_joints(offset),
                                   mirror.mirror_joints(a*scale+offset))
        self.assertFalse(torch.allclose(mirror.mirror_actions(a)*scale+offset,
                                        mirror.mirror_joints(a*scale+offset)))

    def test_mirror_preserves_gradients_and_norm(self):
        x = torch.linspace(-1, 2, 590, requires_grad=True)
        y = mirror.mirror_policy_observations(x)
        y.square().sum().backward()
        torch.testing.assert_close(x.grad, 2*x.detach())
        torch.testing.assert_close(y.square().sum(), x.detach().square().sum())

    def test_invalid_contract_is_rejected(self):
        for value in (torch.zeros(589), torch.zeros(590, dtype=torch.int64), torch.full((590,), float('nan'))):
            with self.assertRaises(ValueError):
                mirror.mirror_policy_observations(value)
        with self.assertRaises(ValueError):
            mirror.mirror_policy_observations(torch.zeros(590), 0)
        with self.assertRaises(ValueError):
            mirror.mirror_joints(torch.zeros(10), ('left_hip_joint',)*10)

    def test_urdf_forward_kinematics_under_mirrored_joint_angles(self):
        urdf = ET.parse(ROOT / 'source/robot_lab/data/Robots/LW/LW_description/LW.urdf').getroot()
        joints = urdf.findall('joint')
        def fk(q):
            poses = {'base_link': (np.zeros(3), np.eye(3))}
            remaining = joints.copy()
            while remaining:
                for joint in remaining.copy():
                    parent = joint.find('parent').get('link')
                    if parent not in poses:
                        continue
                    origin = joint.find('origin')
                    xyz = np.fromstring(origin.get('xyz', '0 0 0'), sep=' ')
                    r, p, y = np.fromstring(origin.get('rpy', '0 0 0'), sep=' ')
                    orient = rotation([0, 0, 1], y) @ rotation([0, 1, 0], p) @ rotation([1, 0, 0], r)
                    axis = joint.find('axis')
                    if axis is not None:
                        orient = orient @ rotation(np.fromstring(axis.get('xyz'), sep=' '), q[joint.get('name')])
                    pp, pr = poses[parent]
                    poses[joint.find('child').get('link')] = (pp+pr@xyz, pr@orient)
                    remaining.remove(joint)
            return poses
        rng = np.random.default_rng(42)
        reflect = np.diag([1, -1, 1])
        for _ in range(20):
            q = torch.tensor(rng.uniform(-.8, .8, 10))
            a = fk(dict(zip(mirror.JOINT_NAMES, q.tolist())))
            b = fk(dict(zip(mirror.JOINT_NAMES, mirror.mirror_joints(q).tolist())))
            for name in a:
                other = 'left_'+name[6:] if name.startswith('right_') else 'right_'+name[5:] if name.startswith('left_') else name
                np.testing.assert_allclose(b[other][0], reflect@a[name][0], atol=2e-6, rtol=0)
                np.testing.assert_allclose(b[other][1], reflect@a[name][1]@reflect, atol=2e-6, rtol=0)


def training_symmetry_cfg():
    """Read the actual task switches without importing Isaac Sim."""
    path = ROOT / 'source/robot_lab/robot_lab/tasks/manager_based/beyondmimic/config/LW_Wheel/agents/rsl_rl_ppo_cfg.py'
    tree = ast.parse(path.read_text())
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Name) and node.func.id == 'RslRlSymmetryCfg']
    if len(calls) != 1:
        raise AssertionError('Expected one LW training symmetry configuration.')
    return {keyword.arg: ast.literal_eval(keyword.value) for keyword in calls[0].keywords}


def fake_training_env(with_wheels=False):
    class Env:
        pass
    env = Env()
    critic_terms = mirror.WHEEL_CRITIC_TERMS if with_wheels else mirror.CRITIC_TERMS
    body_names = mirror.WHEEL_BODY_NAMES if with_wheels else mirror.BODY_NAMES
    groups = {'policy': (mirror.POLICY_TERMS, 10), 'critic': (critic_terms, 1)}
    env.observation_manager = SimpleNamespace(
        active_terms={group: [name for name, _ in terms] for group, (terms, _) in groups.items()},
        group_obs_concatenate=dict.fromkeys(groups, True),
        group_obs_dim={group: (sum(width for _, width in terms)*history,)
                       for group, (terms, history) in groups.items()},
        group_obs_term_dim={group: [(width*history,) for _, width in terms]
                            for group, (terms, history) in groups.items()},
    )
    observations = {}
    for group in groups:
        terms = {name: SimpleNamespace(params={'asset_cfg': SimpleNamespace(
            joint_names=list(mirror.JOINT_NAMES), preserve_order=True)}) for name in ('joint_pos', 'joint_vel')}
        observations[group] = SimpleNamespace(**terms)
    actions = {name: SimpleNamespace(joint_names=list(names), preserve_order=True, scale=scale, use_default_offset=True)
               for name, names, scale in (('joint_pos', mirror.JOINT_NAMES[:8], .25),
                                          ('joint_vel', mirror.JOINT_NAMES[8:], 1.0))}
    env.cfg = SimpleNamespace(
        observations=SimpleNamespace(**observations), actions=SimpleNamespace(**actions),
        commands=SimpleNamespace(motion=SimpleNamespace(
            joint_names=list(mirror.JOINT_NAMES), body_names=list(body_names))),
    )
    env.action_manager = SimpleNamespace(active_terms=['joint_pos', 'joint_vel'])
    return env


class TrainingSymmetryTests(unittest.TestCase):
    def test_callback_handles_both_critic_layouts_without_cache_aliasing(self):
        for width in (164, 146, 164):
            env = fake_training_env(with_wheels=width == 164)
            critic = torch.randn(2, width, requires_grad=True)
            obs = TensorDict({'policy': torch.randn(2, 590), 'critic': critic}, batch_size=[2])
            aug, _ = mirror.compute_symmetric_states(env, obs=obs)
            torch.testing.assert_close(aug['critic'][:2], critic, rtol=0, atol=0)
            torch.testing.assert_close(aug['critic'][2:], mirror.mirror_critic_observations(critic), rtol=0, atol=0)
            aug['critic'][2:].square().sum().backward()
            torch.testing.assert_close(critic.grad, 2*critic.detach())
            wrong_width = 146 if width == 164 else 164
            mismatched = obs.clone().set('critic', torch.zeros(2, wrong_width))
            with self.assertRaises(ValueError):
                mirror.compute_symmetric_states(env, obs=mismatched)

    def test_wheel_environment_rejects_old_dimensions_and_reordered_wheels(self):
        for bad_order in (False, True):
            env = fake_training_env(with_wheels=True)
            if bad_order:
                env.cfg.commands.motion.body_names[-2:] = ['left_wheel_link', 'right_wheel_link']
            else:
                env.observation_manager.group_obs_dim['critic'] = (146,)
            with self.assertRaises(ValueError):
                mirror.compute_symmetric_states(env, actions=torch.zeros(2, 10))

    def test_callback_original_first_and_matches_geometric_maps(self):
        for dtype in (torch.float32, torch.float64):
            obs = TensorDict({'policy': torch.randn(3, 590, dtype=dtype),
                              'critic': torch.randn(3, 146, dtype=dtype)}, batch_size=[3], device='cpu')
            actions = torch.randn(3, 10, dtype=dtype)
            before, actions_before = obs.clone(), actions.clone()
            env = fake_training_env()
            aug, act = mirror.compute_symmetric_states(SimpleNamespace(unwrapped=env), obs, actions)
            self.assertEqual(aug.batch_size, torch.Size([6]))
            self.assertEqual(aug.device, obs.device)
            for group, reflect in (('policy', mirror.mirror_policy_observations),
                                   ('critic', mirror.mirror_critic_observations)):
                torch.testing.assert_close(aug[group][:3], obs[group], rtol=0, atol=0)
                torch.testing.assert_close(aug[group][3:], reflect(obs[group]), rtol=0, atol=0)
                torch.testing.assert_close(obs[group], before[group], rtol=0, atol=0)
            torch.testing.assert_close(act[:3], actions, rtol=0, atol=0)
            torch.testing.assert_close(act[3:], mirror.mirror_actions(actions), rtol=0, atol=0)
            torch.testing.assert_close(actions, actions_before, rtol=0, atol=0)
            twice, twice_actions = mirror.compute_symmetric_states(env, aug[3:], act[3:])
            self.assertTrue((twice[3:] == obs).all())
            torch.testing.assert_close(twice_actions[3:], actions, rtol=0, atol=0)

    def test_callback_optional_inputs_and_gradients(self):
        policy = torch.randn(2, 590, requires_grad=True)
        critic = torch.randn(2, 146, requires_grad=True)
        obs = TensorDict({'policy': policy, 'critic': critic}, batch_size=[2])
        aug, missing = mirror.compute_symmetric_states(None, obs=obs)
        self.assertIsNone(missing)
        (aug['policy'][2:].square().sum()+aug['critic'][2:].square().sum()).backward()
        torch.testing.assert_close(policy.grad, 2*policy.detach())
        torch.testing.assert_close(critic.grad, 2*critic.detach())
        actions = torch.randn(2, 10, requires_grad=True)
        missing, aug_actions = mirror.compute_symmetric_states(None, actions=actions)
        self.assertIsNone(missing)
        aug_actions[2:].square().sum().backward()
        torch.testing.assert_close(actions.grad, 2*actions.detach())
        self.assertEqual(mirror.compute_symmetric_states(None), (None, None))

    def test_callback_rejects_incomplete_or_mismatched_batches(self):
        valid = TensorDict({'policy': torch.zeros(2, 590), 'critic': torch.zeros(2, 146)}, batch_size=[2])
        invalid = [valid.exclude('critic'),
                   TensorDict({'policy': torch.zeros(2, 589), 'critic': torch.zeros(2, 146)}, batch_size=[2]),
                   valid.clone().set('unmapped', torch.zeros(2, 1))]
        for obs in invalid:
            with self.assertRaises(ValueError):
                mirror.compute_symmetric_states(None, obs=obs)
        for actions in (torch.zeros(2, 9), torch.zeros(3, 10), torch.zeros(2, 10, dtype=torch.long)):
            with self.assertRaises(ValueError):
                mirror.compute_symmetric_states(None, valid, actions)

    def test_runtime_contract_rejects_semantic_reordering(self):
        for case in ('observation_terms', 'history', 'joint_names', 'body_names', 'action_order', 'action_scale'):
            env = fake_training_env()
            if case == 'observation_terms':
                env.observation_manager.active_terms['policy'].reverse()
            elif case == 'history':
                env.observation_manager.group_obs_term_dim['policy'][0] = (20,)
            elif case == 'joint_names':
                env.cfg.observations.critic.joint_pos.params['asset_cfg'].joint_names.reverse()
            elif case == 'body_names':
                env.cfg.commands.motion.body_names.reverse()
            elif case == 'action_order':
                env.action_manager.active_terms.reverse()
            else:
                env.cfg.actions.joint_pos.scale = .5
            with self.subTest(case=case), self.assertRaises(ValueError):
                mirror.compute_symmetric_states(env, actions=torch.zeros(2, 10))

    def test_task_config_enables_and_resolves_lw_callback(self):
        from rsl_rl.utils import string_to_callable
        cfg = training_symmetry_cfg()
        self.assertIs(cfg['use_data_augmentation'], True)
        self.assertIs(cfg['use_mirror_loss'], True)
        self.assertTrue(math.isfinite(cfg['mirror_loss_coeff']) and cfg['mirror_loss_coeff'] > 0)
        module, name = cfg['data_augmentation_func'].split(':')
        self.assertEqual(module, 'robot_lab.tasks.manager_based.beyondmimic.mdp.symmetry')
        with patch.dict(sys.modules, {module: mirror}):
            self.assertIs(string_to_callable(cfg['data_augmentation_func']), mirror.compute_symmetric_states)

    def test_local_ppo_modes_and_mirror_loss_gradient_without_updates(self):
        for width in (146, 164):
            with self.subTest(critic_width=width):
                self._check_ppo_modes_without_updates(width)

    def _check_ppo_modes_without_updates(self, critic_width):
        from rsl_rl.algorithms import PPO
        from rsl_rl.modules import ActorCritic
        from rsl_rl.modules.symmetry import resolve_symmetry_config

        class OneBatch:
            def __init__(self, batch):
                self.batch = batch
            def mini_batch_generator(self, *_):
                yield self.batch
            def clear(self):
                pass

        def flatten_grad(params, grads):
            return torch.cat([(torch.zeros_like(p) if g is None else g).flatten() for p, g in zip(params, grads)])

        cfg = training_symmetry_cfg()
        results = {}
        for do_aug, do_loss in ((False, False), (False, True), (True, False), (True, True)):
            with self.subTest(augmentation=do_aug, mirror_loss=do_loss):
                torch.manual_seed(11)
                obs = TensorDict({'policy': torch.randn(4, 590), 'critic': torch.randn(4, critic_width)}, batch_size=[4])
                symmetry = dict(cfg, use_data_augmentation=do_aug, use_mirror_loss=do_loss,
                                data_augmentation_func=mirror.compute_symmetric_states)
                with contextlib.redirect_stdout(io.StringIO()):
                    policy = ActorCritic(obs, {'policy': ['policy'], 'critic': ['critic']}, 10,
                                         actor_hidden_dims=[32, 16], critic_hidden_dims=[32, 16])
                    ppo = PPO(policy, num_learning_epochs=1, num_mini_batches=1, max_grad_norm=1e9,
                              **resolve_symmetry_config({'symmetry_cfg': symmetry},
                                                        fake_training_env(with_wheels=critic_width == 164)))
                with torch.no_grad():
                    actions = policy.act(obs).clone()
                    old_log = policy.get_actions_log_prob(actions).unsqueeze(-1).clone()
                    old_mu, old_sigma = policy.action_mean.clone(), policy.action_std.clone()
                    values = policy.evaluate(obs).clone()
                    advantages, returns = torch.randn(4, 1), values+torch.randn(4, 1)
                params = list(policy.parameters())
                mirrored_obs = TensorDict({'policy': mirror.mirror_policy_observations(obs['policy']),
                                           'critic': mirror.mirror_critic_observations(obs['critic'])}, batch_size=[4])
                expected_loss = torch.nn.functional.mse_loss(
                    policy.act_inference(mirrored_obs), mirror.mirror_actions(policy.act_inference(obs)).detach())
                expected_grad = flatten_grad(params, torch.autograd.grad(expected_loss, params, allow_unused=True))
                before = {key: value.detach().clone() for key, value in policy.state_dict().items()}
                ppo.storage = OneBatch((obs, actions, values, advantages, returns, old_log, old_mu, old_sigma,
                                        (None, None), None))
                with patch.object(ppo.optimizer, 'step', return_value=None) as blocked_step:
                    losses = ppo.update()
                    self.assertEqual(blocked_step.call_count, 1)
                self.assertTrue(all(math.isfinite(value) for value in losses.values()))
                self.assertAlmostEqual(losses['symmetry'], expected_loss.item(), places=7)
                for key, value in policy.state_dict().items():
                    torch.testing.assert_close(value, before[key], rtol=0, atol=0)
                results[do_aug, do_loss] = flatten_grad(params, [p.grad for p in params])
        for do_aug in (False, True):
            torch.testing.assert_close(results[do_aug, True]-results[do_aug, False],
                                       cfg['mirror_loss_coeff']*expected_grad, atol=2e-6, rtol=2e-4)


def fake_telemetry():
    names = list(reversed(mirror.JOINT_NAMES))
    frames = 30
    q = np.zeros((frames, 10))
    q[:, names.index('right_hip_joint')] = np.arange(frames)*.1
    quat = np.tile([1., 0, 0, 0], (frames, 1, 1))
    motion = dict(joint_pos=q, joint_vel=q*.2, body_quat_w=quat)
    samples = []
    for ep in range(2):
        for frame in range(15):
            samples.append(dict(episode_id=ep, frame_index=frame, step=len(samples), done=frame==14,
                                reference_joint_position=q[frame].tolist(), reference_joint_velocity=(q[frame]*.2).tolist(),
                                root_quaternion_w=[1, 0, 0, 0], root_angular_velocity_b=[ep, frame, 0],
                                joint_position=np.full(10, ep+frame*.01).tolist(), joint_velocity=np.zeros(10).tolist(),
                                action=np.full(10, ep+frame).tolist()))
    expected = [n for n, _ in mirror.POLICY_TERMS]
    t = dict(joint_names=names, samples=samples, step_dt_seconds=.02, motion_diagnostics=dict(
        observation_history=dict(term_order={'policy': expected}, input_shapes={'policy': [1, 590]},
                                 history_lengths={'policy': dict.fromkeys(expected, 10)},
                                 flatten_order='per-term history, oldest to newest'),
        action_sampling={'mode': 'mean'}, fps=50., sampling_phase='post_physics_pre_reset',
        initial_runtime={'body_names': ['base_link']}))
    env = {'scene': {'robot': {'init_state': {'joint_pos': dict.fromkeys(mirror.JOINT_NAMES, 0.)}}}}
    return t, motion, env


class ReconstructionTests(unittest.TestCase):
    def test_next_reference_frame_history_and_reset_exclusion(self):
        t, motion, env = fake_telemetry()
        before = copy.deepcopy(t)
        obs, actions, labels = diagnostic.reconstruct_observations(t, motion, env)
        self.assertEqual(obs.shape, (10, 590))
        self.assertEqual([(r['episode_id'], r['frame_index']) for r in labels],
                         [(ep, frame) for ep in range(2) for frame in range(10, 15)])
        self.assertAlmostEqual(obs[0, 0].item(), .1, places=6)  # command frame 1, not 0
        self.assertAlmostEqual(obs[0, 180].item(), 1., places=6)  # newest command frame 10
        self.assertEqual(obs[0, 490].item(), 0.)
        self.assertEqual(obs[0, 580].item(), 9.)  # previous action, not action 10
        self.assertEqual(obs[5, 490].item(), 1.)  # no preceding episode in history
        self.assertEqual(actions[0, 0], 10.)
        self.assertEqual(t, before)

    def test_missing_control_sample_does_not_silently_compress_time(self):
        t, motion, env = fake_telemetry()
        t['samples'].pop(3)
        _, _, labels = diagnostic.reconstruct_observations(t, motion, env)
        self.assertEqual([r['frame_index'] for r in labels if r['episode_id']==0], [14])

    def test_nominal_action_replay_failure_is_fatal(self):
        actor = torch.nn.Linear(590, 10)
        torch.nn.init.zeros_(actor.weight)
        torch.nn.init.zeros_(actor.bias)
        obs = torch.zeros(3, 590)
        labels = [dict(phase=p, episode_id=0) for p in ('takeoff', 'landing', 'other')]
        with self.assertRaises(ValueError):
            diagnostic.diagnose(actor, obs, np.ones((3, 10)), labels, True, 1e-4)
        result, _, _ = diagnostic.diagnose(actor, obs, np.ones((3, 10)), labels, False, 1e-4)
        self.assertFalse(result['recorded_action_comparison']['exact_replay_claimed'])

    def test_exactly_equivariant_linear_policy_has_zero_error(self):
        class EquivariantActor(torch.nn.Module):
            def forward(self, obs):
                return obs[..., -10:]
        obs = torch.randn(3, 590)
        labels = [dict(phase=p, episode_id=0) for p in ('takeoff', 'landing', 'other')]
        result, _, _ = diagnostic.diagnose(EquivariantActor(), obs, obs[:, -10:].numpy(), labels, True, 1e-4)
        self.assertEqual(result['all']['mse_raw_action'], 0.)


if __name__ == '__main__':
    unittest.main()
