"""Smoke for Z-stage curriculum + grid3d eval (Section C+D).

Covers the Z-axis additions to train_rl:
- Z_RANDOMIZATION_STAGES: Z0=±0, Z1=0..5cm, Z2=0..10cm, Z3=0..15cm
- ZRandomizationStageRef carries the current stage like RandomizationStageRef
- next_zrand_stage returns target stage based on range_success_rate
- apply_zrand_stage_to_env applies the z range via env.set_pedestal_range
- evaluate_policy_grid3d runs 3×3×3 (xy × pedestal) episodes

Run from the repo root:

    uv run python -m sim.smoke_z_curriculum
"""

from __future__ import annotations

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from sim.env import RageCageEnv
from sim.train_rl import (
    PEDESTAL_GRID_HEIGHTS,
    Z_RANDOMIZATION_STAGES,
    ZRandomizationStageRef,
    apply_zrand_stage_to_env,
    evaluate_policy_grid3d,
    next_zrand_stage,
)


def test_z_stage_table_shape() -> None:
    assert set(Z_RANDOMIZATION_STAGES) == {0, 1, 2, 3}
    assert Z_RANDOMIZATION_STAGES[0]["z_range"] == (0.0, 0.0)
    assert Z_RANDOMIZATION_STAGES[1]["z_range"][1] == 0.05
    assert Z_RANDOMIZATION_STAGES[2]["z_range"][1] == 0.10
    assert Z_RANDOMIZATION_STAGES[3]["z_range"][1] == 0.15
    assert Z_RANDOMIZATION_STAGES[0]["promote_at"] == 0.5
    assert Z_RANDOMIZATION_STAGES[1]["promote_at"] == 0.4
    assert Z_RANDOMIZATION_STAGES[2]["promote_at"] == 0.3
    assert Z_RANDOMIZATION_STAGES[3]["promote_at"] is None
    print("OK Z_RANDOMIZATION_STAGES")


def test_next_zrand_stage_promotes() -> None:
    assert next_zrand_stage(0, {"success_rate": 0.50}) == 1
    assert next_zrand_stage(0, {"success_rate": 0.49}) == 0
    assert next_zrand_stage(1, {"success_rate": 0.40}) == 2
    assert next_zrand_stage(2, {"success_rate": 0.30}) == 3
    assert next_zrand_stage(3, {"success_rate": 1.0}) == 3
    print("OK next_zrand_stage promotion thresholds")


def test_apply_zrand_stage_to_env() -> None:
    env = RageCageEnv(randomize_cup=True, reward_stage=3)
    apply_zrand_stage_to_env(env, 2)
    assert env.pedestal_z_range == (0.0, 0.10), f"got {env.pedestal_z_range}"
    apply_zrand_stage_to_env(env, 0)
    assert env.pedestal_z_range == (0.0, 0.0)
    apply_zrand_stage_to_env(env, 3)
    assert env.pedestal_z_range == (0.0, 0.15)
    print("OK apply_zrand_stage_to_env")


def test_pedestal_grid_heights() -> None:
    assert PEDESTAL_GRID_HEIGHTS == (0.0, 0.075, 0.15)
    print(f"OK PEDESTAL_GRID_HEIGHTS={PEDESTAL_GRID_HEIGHTS}")


def test_evaluate_policy_grid3d_shape() -> None:
    base_env = RageCageEnv(randomize_cup=True, reward_stage=3)
    env = DummyVecEnv([lambda: base_env])
    env = VecNormalize.load("models/random_pos_cup_thrower_v1/vecnormalize.pkl", env)
    env.training = False
    env.norm_reward = False
    env.obs_rms.mean[20] = 0.0
    env.obs_rms.var[20] = 1.0
    model = PPO.load("models/random_pos_cup_thrower_v1/policy.zip", env=env)

    aggregate, rows = evaluate_policy_grid3d(model, reward_stage=3, seed=0)
    assert len(rows) == 27, f"expected 27 cells, got {len(rows)}"
    cells = {(round(r["cup_x"], 4), round(r["cup_y"], 4), round(r["pedestal_height"], 4)) for r in rows}
    assert len(cells) == 27, f"duplicate cells: {len(cells)} unique vs 27 rows"
    pedestals = {round(r["pedestal_height"], 4) for r in rows}
    assert pedestals == {0.0, 0.075, 0.15}, f"got pedestals {pedestals}"
    assert "success_rate" in aggregate
    print(f"OK grid3d eval n_cells=27 success_rate={aggregate['success_rate']:.3f}")


if __name__ == "__main__":
    test_z_stage_table_shape()
    test_next_zrand_stage_promotes()
    test_apply_zrand_stage_to_env()
    test_pedestal_grid_heights()
    test_evaluate_policy_grid3d_shape()
    print("\nAll Z curriculum smoke checks passed.")
