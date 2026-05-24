"""Smoke-test the cup_eval_mode dispatch in evaluate_policy_metrics.

Verifies all three modes:

- ``fixed``  — every episode places the cup at NOMINAL_CUP_XY.
- ``range``  — every episode samples the cup uniformly within the supplied
  cup_range box; with a non-trivial range we expect both axes to vary.
- ``grid``   — episodes match the canonical 3x3 grid spanning ±10cm of
  NOMINAL_CUP_XY; per-cell info is exposed via evaluate_policy_grid.

Uses an untrained PPO model since we only care about exercising the eval
plumbing, not the policy quality. Episode lengths are short enough that
9-25 episodes runs in <30s.

Run via:  uv run python -m sim.smoke_eval_modes
"""

from __future__ import annotations

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from sim.env import NOMINAL_CUP_XY, RageCageEnv
from sim.train_rl import GRID_OFFSETS, evaluate_policy_grid, evaluate_policy_metrics


def _make_dummy_model() -> PPO:
    env = make_vec_env(RageCageEnv, n_envs=1, env_kwargs={"reward_stage": 1}, vec_env_cls=DummyVecEnv)
    env = VecNormalize(env, norm_obs=True, norm_reward=True)
    return PPO("MlpPolicy", env, n_steps=64, batch_size=32, verbose=0)


def main() -> None:
    model = _make_dummy_model()

    fixed_metrics = evaluate_policy_metrics(
        model, reward_stage=3, episodes=4, seed=0, cup_eval_mode="fixed"
    )
    fixed_xys = fixed_metrics.pop("_episode_cup_xys")
    fixed_arr = np.stack(fixed_xys)
    assert np.allclose(fixed_arr, NOMINAL_CUP_XY), f"fixed mode varied: {fixed_arr}"
    print(f"fixed_mode_ok n={len(fixed_xys)} cup_xy={fixed_arr[0].tolist()}")

    range_metrics = evaluate_policy_metrics(
        model,
        reward_stage=3,
        episodes=12,
        seed=0,
        cup_eval_mode="range",
        cup_range=((1.00, 1.20), (-0.10, 0.10)),
    )
    range_xys = np.stack(range_metrics.pop("_episode_cup_xys"))
    assert range_xys[:, 0].std() > 0.01, f"range mode x-std too low: {range_xys[:, 0].std()}"
    assert range_xys[:, 1].std() > 0.01, f"range mode y-std too low: {range_xys[:, 1].std()}"
    assert (range_xys[:, 0] >= 1.00 - 1e-6).all() and (range_xys[:, 0] <= 1.20 + 1e-6).all()
    assert (range_xys[:, 1] >= -0.10 - 1e-6).all() and (range_xys[:, 1] <= 0.10 + 1e-6).all()
    print(
        f"range_mode_ok n={len(range_xys)} "
        f"x=[{range_xys[:,0].min():.3f},{range_xys[:,0].max():.3f}] "
        f"y=[{range_xys[:,1].min():.3f},{range_xys[:,1].max():.3f}]"
    )

    expected_cells = [
        (NOMINAL_CUP_XY[0] + dx, NOMINAL_CUP_XY[1] + dy)
        for dx in GRID_OFFSETS
        for dy in GRID_OFFSETS
    ]
    grid_metrics, grid_rows = evaluate_policy_grid(model, reward_stage=3, seed=0)
    assert len(grid_rows) == 9, f"expected 9 grid rows, got {len(grid_rows)}"
    actual_cells = [(round(r["cup_x"], 4), round(r["cup_y"], 4)) for r in grid_rows]
    expected_rounded = [(round(x, 4), round(y, 4)) for x, y in expected_cells]
    assert sorted(actual_cells) == sorted(expected_rounded), (
        f"grid cells mismatch:\n actual={sorted(actual_cells)}\n expected={sorted(expected_rounded)}"
    )
    required = {"success", "closest_cup_dist", "valid_bounce", "cup_x", "cup_y"}
    missing = required - set(grid_rows[0].keys())
    assert not missing, f"grid row missing fields: {missing}"
    print(f"grid_mode_ok n={len(grid_rows)} cells={sorted(actual_cells)}")
    print(f"grid_aggregate success_rate={grid_metrics['success_rate']:.3f}")

    print("smoke_eval_modes OK")


if __name__ == "__main__":
    main()
