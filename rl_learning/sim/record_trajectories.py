"""Record successful policy trajectories at a (cup_xy x pedestal) grid for replay on the real arm.

Loads a saved policy, sweeps a 3x3x3 grid of (cup_xy, pedestal) cells, runs
deterministic rollouts (with retry-on-fail), and writes one CSV per cell
containing per-step joint targets and joint actuals. Also writes a manifest
CSV summarizing each cell's outcome. The trajectories cover the windup phase
only (45 control steps from reset to ball release); real-arm replay should
issue gripper-open after the last recorded step.

Run from the repo root:

    uv run python -m sim.record_trajectories \
        --model models/random_stack_cup_thrower_no_ball_obs_smooth_verB \
        --out experiments/no_ball_obs_smooth_verB/replay_trajectories

Output layout:

    <out>/manifest.csv                                    # one row per grid cell
    <out>/cup_<x>_<y>_z<ped>.csv                          # one CSV per trajectory
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from sim.env import NOMINAL_CUP_XY, RageCageEnv, env_kwargs_from_training_json


PPO_INFERENCE_CUSTOM_OBJECTS = {
    "lr_schedule": lambda _: 0.0,
    "learning_rate": 0.0,
}

GRID_OFFSETS = (-0.10, 0.0, 0.10)
PEDESTAL_HEIGHTS = (0.0, 0.075, 0.15)
CONTROL_DT = 0.02  # 50Hz; matches RageCageEnv.control_dt default


def grid_cells() -> list[tuple[float, float, float]]:
    return [
        (float(NOMINAL_CUP_XY[0] + dx), float(NOMINAL_CUP_XY[1] + dy), float(z))
        for dx in GRID_OFFSETS
        for dy in GRID_OFFSETS
        for z in PEDESTAL_HEIGHTS
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Model directory containing policy.zip + vecnormalize.pkl + training.json.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output directory for per-trajectory CSVs and manifest.csv.",
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=45,
        help="Number of policy steps to record per trajectory (windup only). Default 45 = full pre-release windup.",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=5,
        help="Per-cell attempts before giving up. Each attempt uses a different seed.",
    )
    parser.add_argument(
        "--seed-base",
        type=int,
        default=0,
        help="Base seed; each (cell, attempt) uses seed_base + cell_idx*100 + attempt.",
    )
    return parser.parse_args()


def _resolve_paths(model_dir: Path) -> tuple[Path, Path, Path]:
    return model_dir / "policy.zip", model_dir / "vecnormalize.pkl", model_dir / "training.json"


def _label(cup_x: float, cup_y: float, pedestal: float) -> str:
    return (
        f"cup_{cup_x:.3f}_{cup_y:+.3f}_z{pedestal:.3f}"
        .replace("+", "p")
        .replace("-", "m")
    )


def _run_one(
    model: PPO,
    env: VecNormalize,
    underlying: RageCageEnv,
    cup_xy: tuple[float, float],
    pedestal_height: float,
    seed: int,
    n_steps: int,
) -> tuple[list[dict[str, Any]], bool, dict[str, Any]]:
    """Run one rollout at a fixed (cup_xy, pedestal). Returns (rows, success, info)."""
    env.seed(seed)
    underlying.set_next_cup(np.asarray(cup_xy, dtype=np.float32))
    underlying.set_next_pedestal(float(pedestal_height))
    obs = env.reset()

    rows: list[dict[str, Any]] = []
    final_info: dict[str, Any] = {}

    for step in range(n_steps):
        # Read joint qpos BEFORE stepping. This is the joint state at time
        # step * control_dt, before the policy issues its command for this step.
        joint_qpos = underlying.data.qpos[underlying.joint_qposadr].copy()

        action, _ = model.predict(obs, deterministic=True)
        obs, _reward, done, infos = env.step(action)

        # Read commanded joint target AFTER stepping. This is what the env
        # wrote to data.ctrl[actuator_ids] for this step — i.e. the move_js
        # target for time step * control_dt.
        arm_target = underlying.arm_target.copy()
        gripper_cmd = float(underlying.data.ctrl[underlying.gripper_actuator_id])
        ball_released = bool(underlying.ball_released)

        row: dict[str, Any] = {
            "step": step,
            "time_s": step * CONTROL_DT,
            "phase": "policy",
            "ball_released": ball_released,
            "gripper_cmd": gripper_cmd,
        }
        for i in range(6):
            row[f"q{i+1}_target"] = float(arm_target[i])
            row[f"q{i+1}_actual"] = float(joint_qpos[i])
        rows.append(row)

        final_info = infos[0]
        if done[0]:
            break

    success = bool(final_info.get("success", False))
    return rows, success, final_info


def main() -> int:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    policy_path, vecnorm_path, training_json_path = _resolve_paths(args.model)
    if not policy_path.is_file() or not vecnorm_path.is_file():
        print(f"error: policy.zip + vecnormalize.pkl not found under {args.model}", file=sys.stderr)
        return 1

    env_kwargs = env_kwargs_from_training_json(training_json_path)
    if env_kwargs:
        print(f"env_kwargs from training.json: {env_kwargs}")

    base_env = DummyVecEnv(
        [lambda: RageCageEnv(randomize_cup=False, reward_stage=3, **env_kwargs)]
    )
    env = VecNormalize.load(str(vecnorm_path), base_env)
    env.training = False
    env.norm_reward = False
    env.env_method("set_pedestal_range", (0.0, 0.15))

    model = PPO.load(str(policy_path), env=env, custom_objects=PPO_INFERENCE_CUSTOM_OBJECTS)

    underlying = base_env.envs[0]

    cells = grid_cells()
    manifest_rows: list[dict[str, Any]] = []

    for idx, (cx, cy, ped) in enumerate(cells):
        label = _label(cx, cy, ped)
        rows: list[dict[str, Any]] = []
        success = False
        info: dict[str, Any] = {}
        attempts_used = 0
        for attempt in range(args.max_attempts):
            attempts_used = attempt + 1
            seed = args.seed_base + idx * 100 + attempt
            rows, success, info = _run_one(
                model, env, underlying, (cx, cy), ped, seed, args.n_steps
            )
            if success:
                break

        csv_path = args.out / f"{label}.csv"
        if rows:
            with csv_path.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)

        closest = info.get("closest_post_bounce_cup_dist", float("inf"))
        manifest_rows.append(
            {
                "filename": csv_path.name,
                "cup_x": cx,
                "cup_y": cy,
                "pedestal_height": ped,
                "success": success,
                "n_steps_recorded": len(rows),
                "attempts_used": attempts_used,
                "closest_post_bounce_cup_dist": float(closest) if np.isfinite(closest) else "",
                "bounce_count": int(info.get("bounce_count", 0)),
            }
        )
        status = "OK  " if success else "FAIL"
        print(
            f"[{idx + 1:>2}/{len(cells)}] {status} {label} "
            f"attempts={attempts_used} closest={closest if np.isfinite(closest) else 'n/a'}"
        )

    manifest_path = args.out / "manifest.csv"
    with manifest_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys()))
        writer.writeheader()
        writer.writerows(manifest_rows)

    n_success = sum(1 for r in manifest_rows if r["success"])
    print()
    print(f"Recorded {n_success}/{len(cells)} successful trajectories.")
    print(f"Manifest: {manifest_path}")
    print(f"CSVs:     {args.out}/cup_*.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
