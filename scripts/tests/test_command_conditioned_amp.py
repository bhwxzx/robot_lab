"""CPU checks for velocity-conditioned AMP, independent of the local LW dataset."""
import ast
import io
import json
import unittest
from contextlib import contextmanager, redirect_stdout
from itertools import product
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from tensordict import TensorDict

from rsl_rl.modules import ActorCritic, Discriminator
from rsl_rl.runners import OnPolicyRunnerAmp
from rsl_rl.storage import ReplayBuffer
from rsl_rl.utils.command_amp_loader import CommandConditionedAMPLoader


def load_reward_terms(*names, **dependencies):
    """Load the requested reward functions without starting Isaac Sim for the CPU tests."""
    path = Path(__file__).resolve().parents[2] / (
        "source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/mdp/rewards.py"
    )
    tree = ast.parse(path.read_text(), filename=str(path))
    terms = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    namespace = {
        "torch": torch,
        "ManagerBasedRLEnv": SimpleNamespace,
        "SceneEntityCfg": lambda name: SimpleNamespace(name=name),
        **dependencies,
    }
    exec(compile(ast.Module(body=terms, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


@contextmanager
def expert_files(duplicate=False, *, heading_velocity=False):
    points = list(product((-.4, 0., .3, .4), (-.25, 0., .15, .25)))
    if duplicate:
        points.append((.3, .15))
    data = {}
    for i, (vx, wz) in enumerate(points):
        stand = (vx, wz) == (0., 0.)
        motion = {
            "Command": [vx, 0., wz], "MotionWeight": 3 if duplicate and i == len(points) - 1 else 1,
            "FrameDuration": .02, "ClipType": "stand" if stand else "episode",
            "CruiseStart": .08, "BrakeStart": .36,
            "Frames": [[i * 100 + (0 if stand else frame)] * 22 for frame in range(24)],
        }
        if heading_velocity:
            motion.update(
                RootVelocityFrame="heading", RootVelocityPoint="base_link",
                FrameLayout=["base_linear_velocity_heading", "base_angular_velocity_heading",
                             "joint_position_offset", "joint_velocity", "foot_position_base"],
            )
            motion["Frames"] = [
                ([0.] * 6 if stand else [.9 * vx, t * .01, -.02, .03, -.04, .8 * wz]) + joints
                for t, joints in enumerate(motion["Frames"])
            ]
        data[f"clip_{i:02}.txt"] = json.dumps(motion).encode()
    with patch("rsl_rl.utils.command_amp_loader.resolve_motion_files", return_value=list(data)), patch.object(
        Path, "read_bytes", autospec=True, side_effect=lambda path: data[str(path)]
    ):
        yield torch.tensor(points)


def expert_loader(history=3, dt=0.02, duplicate=False, stand_threshold=0.0):
    with expert_files(duplicate) as points:
        loader = CommandConditionedAMPLoader("cpu", dt, ["*.txt"], history, stand_threshold=stand_threshold)
    return loader, points


class SyntheticEnv:
    num_envs, num_actions, device, step_dt = 3, 2, "cpu", .02

    def __init__(self, amp_frame_dim=22):
        self.amp_frame_dim = amp_frame_dim
        self.commands = torch.tensor([[.34, 0, .17], [-.4, 0, 0], [0., 0, 0]])

    def get_observations(self):
        return TensorDict({
            "policy": torch.randn(3, 5), "critic": torch.randn(3, 5),
            "amp": torch.randn(3, 3, self.amp_frame_dim), "amp_command": self.commands,
        }, batch_size=[3])

    def step(self, actions):
        return self.get_observations(), torch.ones(3), torch.zeros(3, dtype=torch.long), {}


class HistoryEnv(SyntheticEnv):
    widths = {"policy": (2, 3), "critic": (1, 4)}

    def __init__(self, time_first):
        super().__init__()
        self.time_first = time_first
        self.observation_manager = SimpleNamespace(
            active_terms={key: ["a", "b"] for key in self.widths},
            group_obs_term_dim={key: [(3, w) for w in widths] for key, widths in self.widths.items()},
        )

    @classmethod
    def term_first(cls, obs):
        obs = obs.clone()
        for key, widths in cls.widths.items():
            obs[key] = torch.cat([term.flatten(1) for term in obs[key].split(widths, dim=-1)], dim=1)
        return obs

    def get_observations(self):
        obs = super().get_observations()
        for key, widths in self.widths.items():
            obs[key] = torch.randn(self.num_envs, 3, sum(widths))
        return obs if self.time_first else self.term_first(obs)


def make_runner(steps=6, *, plain=False, history_window=True, env=None, normalize=False, heading_velocity=False):
    env = SyntheticEnv() if env is None else env
    cfg = {
        "num_steps_per_env": steps, "save_interval": 1000, "amp_history_length": 3,
        "obs_groups": {k: [k] for k in ("policy", "critic", "amp")},
        "policy": {"class_name": "ActorCritic", "actor_hidden_dims": [8], "critic_hidden_dims": [8],
                   "actor_obs_normalization": normalize, "critic_obs_normalization": normalize},
        "algorithm": {"class_name": "CommandConditionedAMPPPO", "num_learning_epochs": 1, "num_mini_batches": 1},
        "amp_motion_files": [], "amp_discriminator_history_window": history_window, "amp_discr_hidden_dims": [8],
        "amp_reward_coef": 1., "amp_task_reward_lerp": .7, "amp_replay_buffer_size": 32,
        "min_normalized_std": [0.] * 2, "amp_dataset_version": "test-grid",
    }
    with redirect_stdout(io.StringIO()):
        if plain:
            dataset = Path(__file__).resolve().parents[2] / "source/robot_lab/robot_lab/datasets/LW/motion_amp_expert"
            cfg.update(amp_motion_files=[str(dataset / "motion_*.txt")], amp_num_preload_transitions=32)
            cfg["algorithm"]["class_name"] = "AMPPPO"
            return OnPolicyRunnerAmp(env, cfg)
        cfg["amp_stand_threshold"] = 0.1
        with expert_files(heading_velocity=heading_velocity):
            return OnPolicyRunnerAmp(env, cfg)


class CommandAmpTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(153)

    def test_heading_velocity_experts_and_legacy_online_layout(self):
        runner = make_runner(env=SyntheticEnv(28), heading_velocity=True)
        legacy = make_runner(heading_velocity=True)
        data = runner.alg.amp_data
        torch.testing.assert_close(data.frames[:, 6:], legacy.alg.amp_data.frames)
        self.assertEqual(data.frame_dim, 28)
        self.assertEqual(runner.alg.amp_normalizer.mean.shape, (28,))
        self.assertEqual(runner.alg.discriminator.input_dim, 3 * 28 + 2)
        self.assertEqual(legacy.alg.discriminator.input_dim, 3 * 22 + 2)

        command = torch.tensor([[.3, .15]])
        before, after = (window.reshape(3, 28) for window in data.sample_for_commands(command))
        torch.testing.assert_close(before[1:], after[:-1])
        expected = torch.tensor([.27, 0., -.02, .03, -.04, .12]).repeat(3, 1)
        expected[:, 1] = after[:, 6].remainder(100) * .01  # Frame index within the chosen clip.
        torch.testing.assert_close(after[:, :6], expected)
        with self.assertRaisesRegex(ValueError, "online AMP requests 28.*expert provides 22"):
            make_runner(env=SyntheticEnv(28))

    def test_grid_endpoints_local_probabilities_and_standing(self):
        loader, points = expert_loader(duplicate=True)
        exact = points[loader.sample_clip_ids(points)]
        torch.testing.assert_close(exact, points)
        queries = torch.tensor([[.34, .17]]).repeat(30000, 1)
        ids = loader.sample_clip_ids(queries)
        chosen = points[ids]
        for query, expected in [((.3, .15), .48), ((.3, .25), .12), ((.4, .15), .32), ((.4, .25), .08)]:
            mask = (chosen == torch.tensor(query)).all(dim=1)
            self.assertAlmostEqual(mask.float().mean().item(), expected, delta=.012)
            if query == (.3, .15):
                self.assertAlmostEqual((ids[mask] == len(points) - 1).float().mean().item(), .75, delta=.02)
        axes = torch.tensor([[0., -.001], [0., .001], [.0001, 0], [-.4, 0], [0, 0]])
        expected = torch.tensor([[0., -.25], [0., .15], [.3, 0], [-.4, 0], [0, 0]])
        torch.testing.assert_close(points[loader.sample_clip_ids(axes)], expected)
        for invalid in ([.41, 0], [0, float("nan")]):
            with self.assertRaisesRegex(ValueError, "grid"):
                loader.sample_for_commands(torch.tensor([invalid]))

    def test_entire_transition_stays_inside_one_cruise_segment(self):
        for dt in (.02, .03):
            loader, points = expert_loader(dt=dt)
            queries = points[(points != 0).any(dim=1)].repeat(40, 1)
            curr, nxt = loader.sample_for_commands(queries)
            curr, nxt = curr.reshape(-1, 3, 22), nxt.reshape(-1, 3, 22)
            self.assertTrue(torch.equal(curr[:, 1:], nxt[:, :-1]))
            values = torch.cat((curr[:, :, 0], nxt[:, -1:, 0]), dim=1)
            torch.testing.assert_close(points[(values[:, 0] // 100).long()], queries)
            self.assertTrue((values.remainder(100) >= 4).all() and (values.remainder(100) < 18).all())
            torch.testing.assert_close(values.diff(dim=1), torch.full_like(values[:, 1:], dt / .02))

    def test_standing_radius_selects_only_stand_without_changing_the_query(self):
        loader, points = expert_loader(stand_threshold=0.1)
        commands = torch.tensor([
            [0., 0.], [-.1, 0.], [0., .1], [.06, -.08], [.03, .04],
            [.08, .08], [-.10001, 0.], [0., -.10001],
        ]).repeat(128, 1)
        expected_stand = torch.tensor([True] * 5 + [False] * 3).repeat(128)
        original = commands.clone()
        selected = points[loader.sample_clip_ids(commands)]
        self.assertTrue(torch.equal((selected == 0).all(dim=1), expected_stand))
        self.assertTrue(torch.equal(commands, original))
        for invalid in (-0.1, float("nan"), float("inf")):
            with self.assertRaisesRegex(ValueError, "standing command threshold"):
                expert_loader(stand_threshold=invalid)

    def test_standing_rewards_use_the_same_inclusive_command_radius(self):
        rewards = load_reward_terms(
            "stand_still", "stop_motion",
            mdp=SimpleNamespace(joint_deviation_l1=lambda env, cfg: env.deviation.clone()),
        )
        commands = torch.tensor([
            [0., 0., 0.], [-.1, 0., 0.], [0., 0., .1], [.06, 0., -.08],
            [0., -.1, 0.], [.03, .04, .02], [.08, 0., .08], [-.2, 0., 0.],
            [0., 0., -.2], [0., .2, 0.], [.06, .08, .001],
        ])
        data = SimpleNamespace(
            root_lin_vel_w=torch.tensor([[5., 0., 0.]]).repeat(len(commands), 1),
            root_ang_vel_w=torch.tensor([[0., 0., -4.]]).repeat(len(commands), 1),
        )
        data.root_lin_vel_w[1] = torch.tensor([3., 4., 1.])
        data.root_lin_vel_w[2] = torch.tensor([0., -5., -1.])
        env = SimpleNamespace(scene={"robot": SimpleNamespace(data=data)}, deviation=torch.full((len(commands),), .25),
                              command_manager=SimpleNamespace(get_command=lambda name: commands))
        standing = torch.tensor([True] * 6 + [False] * 5)
        torch.testing.assert_close(rewards["stop_motion"](env), 9. * standing)
        torch.testing.assert_close(rewards["stand_still"](env, "base_velocity"), .25 * standing)

    def test_tracking_rewards_target_rest_within_the_command_radius(self):
        linear_terms = ("track_lin_vel_xy_exp", "track_lin_vel_xy_yaw_frame_exp")
        yaw_terms = ("track_ang_vel_z_exp", "track_ang_vel_z_world_exp")
        # All base orientations in this test are the identity.
        rewards = load_reward_terms(
            "_velocity_tracking_target", *linear_terms, *yaw_terms,
            yaw_quat=lambda quat: quat, quat_apply_inverse=lambda quat, vec: vec,
        )
        commands = torch.tensor([
            [0., 0., 0.], [0., 0., .1], [-.1, 0., 0.], [.06, 0., -.08], [0., -.1, 0.],
            [.03, .04, .02], [.08, 0., .08], [.06, .08, .001], [-.10001, 0., 0.], [0., 0., -.10001],
        ])
        original = commands.clone()
        target = commands.clone()
        target[:6] = 0.
        measured = commands.clone()
        measured[::2] = 0.
        data = SimpleNamespace(
            root_quat_w=torch.tensor([[1., 0., 0., 0.]]).repeat(len(commands), 1),
            root_lin_vel_b=measured, root_lin_vel_w=measured,
            root_ang_vel_b=measured, root_ang_vel_w=measured,
        )
        env = SimpleNamespace(scene={"robot": SimpleNamespace(data=data)},
                              command_manager=SimpleNamespace(get_command=lambda name: commands))
        std = .3
        error = ((measured - target) / std).square()
        for name in linear_terms:
            torch.testing.assert_close(rewards[name](env, command_name="base_velocity", std=std),
                                       torch.exp(-error[:, :2].sum(dim=1)))
        for name in yaw_terms:
            torch.testing.assert_close(rewards[name](env, command_name="base_velocity", std=std),
                                       torch.exp(-error[:, 2]))
        torch.testing.assert_close(commands, original)

    def test_replay_keeps_labels_when_wrapping_and_after_oversized_insert(self):
        buf = ReplayBuffer(1, 4, "cpu", command_dim=2)
        for x in ([0., 1., 2.], [3., 4., 5., 6., 7., 8., 9.]):
            s = torch.tensor(x)[:, None]
            buf.insert(s, s + 10, torch.cat((s, -s), dim=1))
        self.assertEqual(set(buf.states[:, 0].tolist()), {6., 7., 8., 9.})
        state, nxt, command = next(buf.feed_forward_generator(1, 32))
        torch.testing.assert_close(nxt, state + 10)
        torch.testing.assert_close(command, torch.cat((state, -state), dim=1))

    def test_pre_step_command_switch_reset_timeout_and_shared_label(self):
        runner = make_runner()
        alg, env = runner.alg, runner.env
        rewards = torch.ones(3)
        env.commands[0] = torch.tensor([.06, 0., .08])
        old = env.commands[:, (0, 2)].clone()
        with torch.inference_mode():
            for t in range(5):
                obs = env.get_observations()
                alg.act(obs, obs["amp"].flatten(1))
                dones = torch.tensor([False, t == 3, t == 3])
                valid = torch.tensor([True, t != 3, True])
                bonus = alg.gamma * alg.transition.values[1, 0].clone()
                if t == 3:
                    # The environment mutates the pre-step observation's backing command tensor.
                    env.commands[0] = torch.tensor([-.4, 0., -.25])
                post = env.get_observations()
                alg.process_env_step(post, rewards, dones, {"time_outs": dones}, post["amp"].flatten(1), valid)
                if t < 3 or t == 4:
                    torch.testing.assert_close(alg.storage.rewards[t, :, 0], .7 * rewards)
                else:
                    torch.testing.assert_close(alg.storage.rewards[t, 1, 0], .7 + bonus)
        self.assertEqual(alg.amp_storage.num_samples, 2)
        torch.testing.assert_close(alg.amp_storage.commands[:2], old[[0, 2]])
        with patch.object(alg.amp_data, "sample_for_commands", wraps=alg.amp_data.sample_for_commands) as sampler:
            policy, expert, query = next(alg._amp_batches(1, 16))
            self.assertIs(sampler.call_args.args[0], query)
            torch.testing.assert_close(expert[0], expert[1])  # Both stored queries select the static expert.
        with patch.object(alg.discriminator, "prepare_input", wraps=alg.discriminator.prepare_input) as inputs:
            alg._discriminator_losses(policy, expert, query)
            self.assertEqual(len(inputs.call_args_list), 2)
            self.assertTrue(all(call.args[2] is query for call in inputs.call_args_list))

    def test_reward_uses_command_and_penalty_only_uses_motion_gradient(self):
        d = Discriminator(5, 1., [1], "cpu", use_history_window=True, command_scale=[.4, .25])
        with torch.no_grad():
            d.trunk[0].weight.zero_()
            d.trunk[0].weight[0, -2] = 1.
            d.trunk[0].bias.zero_()
            d.amp_linear.weight.fill_(1.)
            d.amp_linear.bias.zero_()
        h = torch.ones(2, 3)
        c = torch.tensor([[.4, 0], [-.4, 0]])
        reward, logits = d.predict_amp_reward(h, h, torch.zeros(2), command=c)
        torch.testing.assert_close(logits[:, 0], torch.tensor([1., 0.]))
        self.assertGreater(reward[0], reward[1])
        self.assertEqual(d.compute_grad_pen(h, h, command=c).item(), 0.)

    def test_shared_expert_score_matches_separate_forwards_and_parameter_gradients(self):
        alg = make_runner().alg
        disc = alg.discriminator
        width = alg.amp_data.observation_dim
        policy = (torch.randn(8, width), torch.randn(8, width))
        expert = (torch.randn(8, width), torch.randn(8, width))
        command = torch.rand(8, 2, requires_grad=True)

        with patch.object(disc, "forward", wraps=disc.forward) as forward:
            loss, penalty, policy_score, expert_score = alg._discriminator_losses(policy, expert, command)
            self.assertEqual(forward.call_count, 2)
        params = tuple(disc.parameters())
        *actual_grads, command_grad = torch.autograd.grad(loss + penalty, (*params, command), allow_unused=True)
        self.assertIsNone(command_grad)

        policy_motion = disc.normalize_amp_observation(policy[1], alg.amp_normalizer)
        expert_motion = disc.normalize_amp_observation(expert[1], alg.amp_normalizer)
        expected_policy = disc(disc.prepare_input(policy[0], policy_motion, command))
        expected_expert = disc(disc.prepare_input(expert[0], expert_motion, command))
        expected_loss = 0.5 * ((expected_expert - 1).square().mean() + (expected_policy + 1).square().mean())
        expert_motion.requires_grad_(True)
        separate_score = disc(disc.prepare_input(expert[0], expert_motion, command))
        (motion_grad,) = torch.autograd.grad(separate_score.sum(), expert_motion, create_graph=True)
        expected_penalty = 10 * motion_grad.square().sum(dim=1).mean()
        torch.testing.assert_close((policy_score, expert_score), (expected_policy, expected_expert))
        torch.testing.assert_close((loss, penalty), (expected_loss, expected_penalty))
        torch.testing.assert_close(tuple(actual_grads), torch.autograd.grad(expected_loss + expected_penalty, params))

    def test_mirror_loss_reuses_actor_only_for_an_already_augmented_batch(self):
        @torch.no_grad()
        def mirror(env, obs=None, actions=None):
            augmented = None
            if obs is not None:
                groups = obs.select("policy", "critic")
                augmented = torch.cat((groups, groups.apply(torch.neg)), dim=0)
            return augmented, None if actions is None else torch.cat((actions, -actions))

        for augment, forwards in ((True, 1), (False, 2)):
            with self.subTest(augment=augment):
                runner = make_runner(steps=2)
                alg, env = runner.alg, runner.env
                alg.symmetry = {
                    "use_data_augmentation": augment, "use_mirror_loss": True, "mirror_loss_coeff": 1.,
                    "data_augmentation_func": mirror, "_env": env,
                }
                alg.value_loss_coef = alg.entropy_coef = 0.
                with torch.inference_mode():
                    obs = env.get_observations()
                    for _ in range(runner.num_steps_per_env):
                        actions = alg.act(obs, obs["amp"].flatten(1))
                        obs, rewards, dones, extras = env.step(actions)
                        alg.process_env_step(obs, rewards, dones, extras, obs["amp"].flatten(1))
                    alg.compute_returns(obs)
                    alg.storage.advantages.zero_()
                    both, _ = mirror(env, obs=alg.storage.observations.flatten(0, 1))
                    original, mirrored = alg.policy.act_inference(both).chunk(2)
                    expected_loss = (mirrored + original).square().mean().item()

                before = alg.policy.actor[-1].weight.detach().clone()
                with patch.object(alg.policy.actor, "forward", wraps=alg.policy.actor.forward) as forward:
                    stats = alg.update()
                self.assertEqual(forward.call_count, forwards)
                self.assertAlmostEqual(stats["symmetry"], expected_loss, places=6)
                self.assertFalse(torch.equal(alg.policy.actor[-1].weight, before))

    def test_runner_terminal_windows_full_compact_and_missing(self):
        for terminal_rows in (3, 1, 0):
            runner = make_runner(steps=4)
            calls = 0

            def step(actions):
                nonlocal calls
                calls += 1
                dones = torch.tensor([False, False, calls == 4])
                extras = {}
                if calls == 4 and terminal_rows:
                    extras["terminal_obs"] = {"amp": torch.full((terminal_rows, 3, 22), 123.)}
                return runner.env.get_observations(), torch.ones(3), dones, extras

            with patch.object(runner.env, "step", side_effect=step):
                runner.learn(1)
            buf = runner.alg.amp_storage
            self.assertEqual(buf.num_samples, 3 if terminal_rows else 2)
            if terminal_rows:
                self.assertTrue((buf.next_states[2] == 123).all())

    def test_episode_return_logging_excludes_timeout_bootstrap_and_unfinished_episodes(self):
        runner = make_runner(steps=3)
        runner.log_dir, runner.writer, runner.logger_type = "unused", Mock(), "tensorboard"
        with torch.no_grad():
            # D=1 gives a fixed style reward of 0.02 per valid window.
            runner.alg.discriminator.amp_linear.weight.zero_()
            runner.alg.discriminator.amp_linear.bias.fill_(1.0)
            runner.alg.policy.critic[-1].weight.zero_()
            runner.alg.policy.critic[-1].bias.fill_(5.0)

        steps = iter(range(1, 7))

        def step(actions):
            t = next(steps)
            dones = torch.tensor([t in (2, 6), t == 6, False])
            extras = {"time_outs": torch.tensor([t == 2, False, False])}
            if t == 4:
                runner.env.commands[1] = torch.tensor([0.3, 0.0, 0.15])
            obs = runner.env.get_observations()
            if t == 6:
                extras["terminal_obs"] = {"amp": obs["amp"][dones]}
            return obs, torch.tensor([2.0, -1.0, 500.0]), dones, extras

        with patch.object(runner.env, "step", side_effect=step), patch.object(runner, "save"), patch(
            "rsl_rl.runners.on_policy_runner_amp.store_code_state", return_value=[]
        ), redirect_stdout(io.StringIO()):
            runner.learn(2)

        scalars = {}
        for call in runner.writer.add_scalar.call_args_list:
            tag, value, iteration = call.args
            scalars[tag, iteration] = value
        # First episode ends during warmup. The next two each earn one valid style step.
        expected_returns = [(4.0, 0.0), ((4.0 + 8.0 - 6.0) / 3, 2 * 0.006 / 3)]
        for it, (task, style) in enumerate(expected_returns):
            self.assertAlmostEqual(scalars["Train/mean_reward", it], task)
            self.assertAlmostEqual(scalars["Train/mean_task_reward", it], task)
            self.assertAlmostEqual(scalars["Train/mean_weighted_style_reward", it], style)
            self.assertAlmostEqual(scalars["Train/mean_mixed_reward", it], 0.7 * task + style, places=6)

    def test_warmup_update_and_checkpoint_roundtrip(self):
        runner = make_runner(steps=2)
        d = runner.alg.discriminator
        before = d.amp_linear.weight.detach().clone()
        runner.learn(1)
        self.assertEqual(runner.alg.amp_storage.num_samples, 0)
        self.assertTrue(torch.equal(before, d.amp_linear.weight))
        runner.learn(2)
        self.assertGreater(runner.alg.amp_storage.num_samples, 0)
        self.assertFalse(torch.equal(before, d.amp_linear.weight))
        self.assertEqual(runner.alg.amp_normalizer.mean.shape, (22,))
        runner.logger_type = "tensorboard"
        fp = io.BytesIO()
        runner.save(fp)
        other = make_runner()
        fp.seek(0)
        other.load(fp)
        self.assertEqual(other.amp_conditioning, runner.amp_conditioning)
        torch.testing.assert_close(other.alg.discriminator.amp_linear.weight, d.amp_linear.weight)
        fp.seek(0)
        checkpoint = torch.load(fp, weights_only=False)
        self.assertEqual(checkpoint["amp_conditioning"].pop("stand_threshold"), .1)
        legacy = io.BytesIO()
        torch.save(checkpoint, legacy)
        legacy.seek(0)
        with self.assertWarnsRegex(UserWarning, "threshold was 0.0"):
            other.load(legacy)
        checkpoint["amp_conditioning"]["command_axes"] = ("wz", "vx")
        invalid = io.BytesIO()
        torch.save(checkpoint, invalid)
        invalid.seek(0)
        with self.assertRaisesRegex(ValueError, "layout"):
            other.load(invalid)

    def test_mlp_joins_features_per_frame_and_preserves_batch_axes(self):
        history = torch.arange(2 * 3 * 4 * 5, dtype=torch.float32).reshape(2, 3, 4, 5)
        groups = {"policy": ["left", "right"], "critic": ["right", "left"]}
        for values in (history, history[:, :, -1]):
            left, right = values.split((2, 3), dim=-1)
            obs = TensorDict({"left": left, "right": right}, batch_size=[2, 3])
            with redirect_stdout(io.StringIO()):
                policy = ActorCritic(obs[0], groups, num_actions=2, actor_hidden_dims=[8], critic_hidden_dims=[8])
            torch.testing.assert_close(policy.get_actor_obs(obs), values.flatten(2))
            torch.testing.assert_close(policy.get_critic_obs(obs), torch.cat((right, left), dim=-1).flatten(2))
            self.assertEqual(policy.act_inference(obs).shape, (2, 3, 2))

    def test_legacy_history_migrates_policy_value_normalizers_and_adam_once(self):
        old_runner = make_runner(env=HistoryEnv(time_first=False), normalize=True)
        old_runner.learn(1)
        old_runner.logger_type = "tensorboard"
        checkpoint = io.BytesIO()
        old_runner.save(checkpoint)
        checkpoint.seek(0)
        saved = torch.load(checkpoint, weights_only=False)
        del saved["amp_conditioning"]["actor_critic_history"]
        legacy = io.BytesIO()
        torch.save(saved, legacy)

        env = HistoryEnv(time_first=True)
        obs = env.get_observations()
        flat = env.term_first(obs)
        new_runner = make_runner(env=env, normalize=True)
        old, new = old_runner.alg.policy, new_runner.alg.policy
        for load_optimizer in (False, True):
            legacy.seek(0)
            new_runner.load(legacy, load_optimizer=load_optimizer)
            self.assertTrue(new_runner.migrated_term_first_history)
            torch.testing.assert_close(new.get_actor_obs(obs), obs["policy"].flatten(1))
            torch.testing.assert_close(new.get_critic_obs(obs), obs["critic"].flatten(1))
            torch.testing.assert_close(new.act_inference(obs), old.act_inference(flat))
            torch.testing.assert_close(new.evaluate(obs), old.evaluate(flat))

        for role, network in (("policy", "actor"), ("critic", "critic")):
            old_weight, new_weight = getattr(old, network)[0].weight, getattr(new, network)[0].weight
            old_state = old_runner.alg.optimizer.state[old_weight]
            new_state = new_runner.alg.optimizer.state[new_weight]
            for key in ("exp_avg", "exp_avg_sq"):
                # Convert the new columns back to the old term blocks as an independent oracle.
                frames = new_state[key].reshape(new_weight.shape[0], 3, -1)
                blocks = frames.split(env.widths[role], dim=-1)
                torch.testing.assert_close(torch.cat([x.flatten(1) for x in blocks], dim=1), old_state[key])
            torch.testing.assert_close(old_state["step"], new_state["step"])

        for runner, inputs in ((old_runner, flat), (new_runner, obs)):
            policy, optimizer = runner.alg.policy, runner.alg.optimizer
            optimizer.zero_grad()
            (policy.act_inference(inputs).square().mean() + policy.evaluate(inputs).square().mean()).backward()
            optimizer.step()
        torch.testing.assert_close(new.act_inference(obs), old.act_inference(flat))
        torch.testing.assert_close(new.evaluate(obs), old.evaluate(flat))

        new_runner.logger_type = "tensorboard"
        converted = io.BytesIO()
        new_runner.save(converted)
        converted.seek(0)
        restored = make_runner(env=env, normalize=True)
        restored.load(converted)
        self.assertFalse(restored.migrated_term_first_history)
        self.assertEqual(restored.amp_conditioning["actor_critic_history"]["policy"]["order"], "time_first")
        torch.testing.assert_close(restored.alg.policy.actor[0].weight, new.actor[0].weight, rtol=0, atol=0)
        torch.testing.assert_close(restored.alg.policy.critic[0].weight, new.critic[0].weight, rtol=0, atol=0)
        converted.seek(0)
        with self.assertRaisesRegex(ValueError, "history layout"):
            old_runner.load(converted)

    def test_plain_amp_keeps_global_sampling_and_legacy_input_dimensions(self):
        for history_window in (True, False):
            runner = make_runner(steps=2, plain=True, history_window=history_window)
            alg = runner.alg
            self.assertEqual(alg.discriminator.input_dim, 66 if history_window else 132)
            self.assertNotIn("command_scale", alg.discriminator.state_dict())
            before = alg.discriminator.amp_linear.weight.detach().clone()
            with patch.object(alg.amp_data, "feed_forward_generator", wraps=alg.amp_data.feed_forward_generator) as sample:
                runner.learn(1)
                sample.assert_called_once()
            self.assertIsNone(alg.amp_storage.commands)
            self.assertFalse(torch.equal(before, alg.discriminator.amp_linear.weight))


if __name__ == "__main__":
    unittest.main()
