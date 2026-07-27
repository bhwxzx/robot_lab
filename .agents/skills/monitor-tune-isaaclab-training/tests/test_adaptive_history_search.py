#!/usr/bin/env python3
"""Bounded local-W&B priors and deterministic adaptive-round tests."""

from __future__ import annotations

import json
import sys
import unittest
from datetime import datetime
from pathlib import Path

import wandb


SKILL = Path(__file__).resolve().parents[1]
SCRIPTS = SKILL / "scripts"
TESTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(TESTS))

from build_trial_plan import (  # noqa: E402
    build_plan,
    extend_adaptive_plan,
    validate_trial_plan,
)
from execute_trial_plan import (  # noqa: E402
    adopt_expanded_plan,
    initialize_state,
)
from index_local_wandb_history import build_history_index  # noqa: E402
from merge_historical_priors import merge_history_indexes  # noqa: E402
import test_fixed_single_seed as fixed_tests  # noqa: E402
from validate_session_spec import SpecError, validate_spec  # noqa: E402


class AdaptiveHistorySearchTests(unittest.TestCase):
    def setUp(self) -> None:
        helper = fixed_tests.FixedSingleSeedTests(methodName="runTest")
        helper.setUp()
        self.helper = helper
        self.temp = helper.temp
        self.root = helper.root
        self.session = helper.session
        self.wandb_root = self.root / "wandb"
        self.session["tuning"]["allowed_parameters"][0] = {
            "path": "agent.learning_rate",
            "values": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
            "baseline": 0.1,
        }
        self.session["tuning"]["max_trials"] = 5
        self.session["history_prior"] = {
            "enabled": True,
            "source": "local_wandb",
            "wandb_project": "adaptive-history-test",
            "lookback_days": 30,
            "max_selected_runs": 2,
            "max_points_per_run": 3,
            "include_failed_runs": False,
            "max_first_round_fraction": 0.5,
            "explicit_run_ids": [],
            "config_path_map": {
                "agent.learning_rate": "agent.learning_rate",
            },
            "metric_key_map": {
                "score": "score",
                "unsafe": "unsafe",
            },
            "worker_roots": {"local": str(self.wandb_root)},
        }
        self.session["adaptive_search"] = {
            "enabled": True,
            "max_rounds": 2,
            "trials_per_round": 2,
            "exploration_fraction": 0.5,
        }

    def tearDown(self) -> None:
        self.helper.tearDown()

    def _wandb_run(
        self,
        run_id: str,
        learning_rate: float,
        score: float,
    ) -> None:
        run = wandb.init(
            project="adaptive-history-test",
            id=run_id,
            name=run_id,
            dir=str(self.root),
            mode="offline",
            config={"agent": {"learning_rate": learning_rate}},
            settings=wandb.Settings(console="off"),
        )
        for offset in range(4):
            run.log({"score": score + offset, "unsafe": 0.0})
        run.finish(exit_code=0)

    def test_bounded_history_builds_and_extends_adaptive_plan(self) -> None:
        self._wandb_run("hist-a", 0.2, 10.0)
        self._wandb_run("hist-b", 0.3, 20.0)
        self._wandb_run("hist-c", 0.4, 30.0)
        spec = validate_spec(self.session)
        index = build_history_index(
            spec,
            "local",
            now=datetime.now().astimezone(),
        )
        self.assertEqual(index["max_selected_runs"], 2)
        self.assertEqual(index["candidate_read_limit"], 4)
        self.assertEqual(len(index["selected_runs"]), 2)
        self.assertTrue(
            all(
                max(run["retained_points"].values()) <= 3
                for run in index["selected_runs"]
            )
        )
        prior = merge_history_indexes(spec, [index])
        self.assertEqual(prior["selected_run_count"], 2)
        plan = build_plan(spec, prior)
        self.assertEqual(plan["version"], 5)
        self.assertEqual(plan["adaptive"]["rounds"][0][
            "history_influenced_trial_count"
        ], 1)
        historical = {
            run["overrides_sha256"] for run in prior["selected_runs"]
        }
        self.assertTrue(
            all(
                trial["trial_id"] == "baseline"
                or __import__("hashlib").sha256(
                    json.dumps(
                        trial["overrides"],
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode()
                ).hexdigest()
                not in historical
                for trial in plan["trials"]
            )
        )
        results = [
            {
                "run_id": run["run_id"],
                "trial_id": run["trial_id"],
                "seed": 42,
                "status": "completed",
                "metrics": {
                    "score": float(index + 10),
                    "unsafe": 0.0,
                },
            }
            for index, run in enumerate(plan["runs"])
        ]
        expanded = extend_adaptive_plan(spec, plan, results)
        self.assertEqual(len(expanded["adaptive"]["rounds"]), 2)
        self.assertEqual(len(expanded["trials"]), 5)
        validate_trial_plan(spec, expanded)
        tampered = json.loads(json.dumps(expanded))
        tampered["trials"][-1]["overrides"]["agent.learning_rate"] = 0.2
        with self.assertRaisesRegex(
            SpecError,
            "deterministic authorized expansion",
        ):
            validate_trial_plan(spec, tampered)

        oversized_index = json.loads(json.dumps(index))
        oversized_index["selected_runs"][0]["retained_points"]["score"] = 4
        unsigned = dict(oversized_index)
        unsigned.pop("index_sha256")
        oversized_index["index_sha256"] = __import__("hashlib").sha256(
            json.dumps(
                unsigned,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        with self.assertRaisesRegex(SpecError, "bounded schema"):
            merge_history_indexes(spec, [oversized_index])

        session_path = self.root / "adaptive-session.json"
        plan_path = self.root / "adaptive-plan-1.json"
        expanded_path = self.root / "adaptive-plan-2.json"
        session_path.write_text(json.dumps(spec), encoding="utf-8")
        plan_path.write_text(json.dumps(plan), encoding="utf-8")
        expanded_path.write_text(json.dumps(expanded), encoding="utf-8")
        state_path, state = initialize_state(
            spec,
            session_path,
            plan,
            plan_path,
        )
        for record in state["runs"].values():
            record["status"] = "completed"
        state_path.write_text(json.dumps(state), encoding="utf-8")
        _, adopted = adopt_expanded_plan(
            spec,
            session_path,
            expanded,
            expanded_path,
        )
        self.assertEqual(adopted["stage"], "screening")
        self.assertEqual(
            sum(run["status"] == "pending" for run in adopted["runs"].values()),
            2,
        )

    def test_history_limits_and_fixed_seed_are_hard_gates(self) -> None:
        too_many = json.loads(json.dumps(self.session))
        too_many["history_prior"]["max_selected_runs"] = 7
        with self.assertRaisesRegex(SpecError, "between 1 and 6"):
            validate_spec(too_many)

        too_much_influence = json.loads(json.dumps(self.session))
        too_much_influence["history_prior"]["max_first_round_fraction"] = 0.75
        with self.assertRaisesRegex(SpecError, "between 0 and 0.5"):
            validate_spec(too_much_influence)

        multi_seed = json.loads(json.dumps(self.session))
        multi_seed["tuning"]["seed_strategy"] = {
            "mode": "robust_multi_seed",
            "screening_seeds": [42],
            "confirmation_seeds": [42, 43],
            "confirmation_top_k": 1,
        }
        multi_seed["tuning"]["seeds"] = [42, 43]
        multi_seed["tuning"]["ranking"]["minimum_final_training_seeds"] = 2
        with self.assertRaisesRegex(SpecError, "requires fixed_single_seed"):
            validate_spec(multi_seed)


if __name__ == "__main__":
    unittest.main()
