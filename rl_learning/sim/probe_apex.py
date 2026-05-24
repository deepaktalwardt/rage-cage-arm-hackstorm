"""Probe: max ball apex achievable by v34 across the (xy × pedestal) grid.

The stacked-cup design (Section E) targets pedestal_height ∈ [0, 0.15m],
which puts the cup mouth at z=0.27m. v34's training only covered cups at
z=0.12m. The question this probe answers: is the joint dynamics' max
ball apex even close to 27cm? If v34 can't get above ~25cm at any cell,
the upper bound has to come down before training kicks off.

The v34 policy is loaded with a surgical reset of obs_rms[20] (the
former cup_count slot, repurposed as pedestal_height in this branch)
so the policy receives ~0 normalized in that slot — matching its
training distribution where cup_count was always 1.

Run:

    uv run python -m sim.probe_apex
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from sim.env import NOMINAL_CUP_XY, RageCageEnv
from sim.train_rl import GRID_OFFSETS

V34_DIR = Path("models/random_pos_cup_thrower_v1")


def grid_cells() -> list[tuple[float, float]]:
    return [
        (float(NOMINAL_CUP_XY[0] + dx), float(NOMINAL_CUP_XY[1] + dy))
        for dx in GRID_OFFSETS
        for dy in GRID_OFFSETS
    ]


def rollout_apex(
    model: PPO,
    env: VecNormalize,
    base_env: RageCageEnv,
    cup_xy: tuple[float, float],
    pedestal: float,
) -> tuple[float, bool, float]:
    base_env.set_next_cup(np.asarray(cup_xy, dtype=np.float32))
    base_env.set_next_pedestal(pedestal)
    obs = env.reset()
    max_z = float(base_env._ball_pos()[2])
    done = [False]
    info: dict = {}
    while not done[0]:
        action, _ = model.predict(obs, deterministic=True)
        obs, _r, done, infos = env.step(action)
        info = infos[0]
        max_z = max(max_z, float(base_env._ball_pos()[2]))
        for row in info.get("passive_info_rows", []) or []:
            pass
    return max_z, bool(info.get("success", False)), float(info.get("closest_post_bounce_cup_dist", float("inf")))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pedestal-heights", type=str, default="0.0,0.05,0.10,0.15")
    args = parser.parse_args()

    pedestals = [float(s) for s in args.pedestal_heights.split(",")]
    base_env = RageCageEnv(randomize_cup=True, reward_stage=3)
    env = DummyVecEnv([lambda: base_env])
    env = VecNormalize.load(str(V34_DIR / "vecnormalize.pkl"), env)
    env.training = False
    env.norm_reward = False
    env.obs_rms.mean[20] = 0.0
    env.obs_rms.var[20] = 1.0
    model = PPO.load(str(V34_DIR / "policy.zip"), env=env)
    print(
        f"loaded v34 with surgical reset of obs_rms[20] "
        f"(was mean=0.1, var≈0; now mean=0, var=1)"
    )

    cells = grid_cells()
    rows: list[dict[str, float]] = []
    for pedestal in pedestals:
        cup_top_z = pedestal + 0.12
        per_cell_max: list[float] = []
        per_cell_success = 0
        per_cell_closest: list[float] = []
        for cup_xy in cells:
            apex, succ, closest = rollout_apex(model, env, base_env, cup_xy, pedestal)
            per_cell_max.append(apex)
            if succ:
                per_cell_success += 1
            per_cell_closest.append(closest)
        agg_max = max(per_cell_max)
        agg_min = min(per_cell_max)
        agg_med = float(np.median(per_cell_max))
        margin = agg_max - cup_top_z
        rows.append(
            {
                "pedestal": pedestal,
                "cup_top_z": cup_top_z,
                "min_apex": agg_min,
                "median_apex": agg_med,
                "max_apex": agg_max,
                "successes": per_cell_success,
                "median_closest": float(np.median(per_cell_closest)),
                "margin": margin,
            }
        )
        print(
            f"pedestal={pedestal:.3f}m  cup_top_z={cup_top_z:.3f}m  "
            f"apex(min/med/max)=({agg_min:.3f}, {agg_med:.3f}, {agg_max:.3f})  "
            f"successes={per_cell_success}/9  "
            f"median_closest={float(np.median(per_cell_closest)):.3f}  "
            f"margin(max_apex - cup_top)={margin:+.3f}m"
        )

    print()
    fail_rows = [r for r in rows if r["max_apex"] < r["cup_top_z"] + 0.05]
    if fail_rows:
        print(
            "VERDICT: max apex falls short of cup-top + 5cm at "
            f"{[r['pedestal'] for r in fail_rows]}m. Either the upper bound "
            "must come down or the new policy needs a meaningfully higher "
            "arc than v34 produces."
        )
    else:
        print(
            "VERDICT: max apex clears all probed pedestals (with ≥5cm margin) — "
            "physical reachability is fine. Training can proceed at "
            f"pedestal up to {max(pedestals):.3f}m."
        )


if __name__ == "__main__":
    main()
