"""Render multi-cup, multi-pedestal rollouts from the latest training checkpoint.

Designed to run alongside an active ``train_rl`` run that doesn't have
``--train-rollout-viz-every`` enabled. Watches ``<run-dir>/checkpoints/``
for new files and, when one appears, renders one rollout per
(cup_xy × pedestal_height) combination using the saved policy. GIFs land
under ``<run-dir>/watch_rollouts/checkpoint_<steps>/``. Gives a live
progress signal that's independent of the training loop and the
auto-promotion cadence.

Run via:
  uv run python -m sim.watch_rollouts \
      --run-dir runs/v36_stacked_scratch \
      --reward-stage 3 \
      --pedestals 0,0.02,0.05,0.10,0.15 \
      --interval 60
"""

from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from sim.env import NOMINAL_CUP_XY, RageCageEnv
from sim.train_rl import GRID_OFFSETS, render_policy_rollout_at_cup


def _parse_pedestals(s: str) -> list[float]:
    parts = [chunk.strip() for chunk in s.split(",") if chunk.strip()]
    if not parts:
        raise argparse.ArgumentTypeError("--pedestals must be a non-empty comma list")
    return [float(p) for p in parts]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Run directory written by sim.train_rl. Reads checkpoints/, writes watch_rollouts/.",
    )
    parser.add_argument("--reward-stage", type=int, default=3)
    parser.add_argument("--interval", type=int, default=60, help="seconds between checkpoint scans")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument(
        "--cells",
        choices=("grid", "corners"),
        default="grid",
        help="grid = 3x3 across ±10cm (9 rollouts); corners = 4 corners + center (5 rollouts)",
    )
    parser.add_argument(
        "--pedestals",
        type=_parse_pedestals,
        default=[0.0, 0.02, 0.05, 0.10, 0.15],
        help="Comma-separated pedestal heights (m) to render at each cup_xy cell. "
        "Default '0.0,0.02,0.05,0.10,0.15' = 5 stack heights covering 1 to ~9 nested cups.",
    )
    return parser.parse_args()


_CHECKPOINT_RE = re.compile(r"checkpoint_(\d+)_steps\.zip$")


def _latest_checkpoint(checkpoints_dir: Path) -> tuple[int, Path, Path] | None:
    if not checkpoints_dir.exists():
        return None
    best: tuple[int, Path, Path] | None = None
    for entry in checkpoints_dir.iterdir():
        m = _CHECKPOINT_RE.match(entry.name)
        if m is None:
            continue
        steps = int(m.group(1))
        vec_path = checkpoints_dir / f"checkpoint_vecnormalize_{steps}_steps.pkl"
        if not vec_path.exists():
            continue
        if best is None or steps > best[0]:
            best = (steps, entry, vec_path)
    return best


def _load_model(model_path: Path, vec_path: Path, reward_stage: int) -> PPO:
    env = DummyVecEnv([lambda: RageCageEnv(randomize_cup=False, reward_stage=reward_stage)])
    env = VecNormalize.load(str(vec_path), env)
    env.training = False
    env.norm_reward = False
    return PPO.load(str(model_path), env=env)


def _render_for_checkpoint(
    steps: int,
    model_path: Path,
    vec_path: Path,
    out_root: Path,
    reward_stage: int,
    seed: int,
    max_steps: int,
    width: int,
    height: int,
    cells: list[tuple[float, float]],
    pedestals: list[float],
) -> Path:
    out_dir = out_root / f"checkpoint_{steps:09d}"
    if out_dir.exists():
        return out_dir
    model = _load_model(model_path, vec_path, reward_stage)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_lines: list[str] = []
    for cup_x, cup_y in cells:
        for ped in pedestals:
            label = (
                f"cup_{cup_x:.3f}_{cup_y:+.3f}_z{ped:.3f}"
                .replace("+", "p")
                .replace("-", "m")
            )
            gif_path, _csv, total_reward, final_info = render_policy_rollout_at_cup(
                model,
                out_dir=out_dir,
                label=label,
                seed=seed,
                cup_xy=(cup_x, cup_y),
                reward_stage=reward_stage,
                max_steps=max_steps,
                width=width,
                height=height,
                pedestal_height=ped,
            )
            success = bool(final_info.get("success", False))
            closest = float(final_info.get("closest_post_bounce_cup_dist", float("inf")))
            if not np.isfinite(closest):
                closest = float("inf")
            summary_lines.append(
                f"  cup=({cup_x:.3f},{cup_y:+.3f}) z={ped:.3f} "
                f"success={success} closest_post_bounce={closest:.3f} "
                f"reward={total_reward:.2f} gif={gif_path}"
            )
    print(f"watch_rollouts checkpoint={steps} cells={len(cells)} pedestals={len(pedestals)}")
    for line in summary_lines:
        print(line)
    return out_dir


def main() -> None:
    args = parse_args()
    checkpoints_dir = args.run_dir / "checkpoints"
    out_root = args.run_dir / "watch_rollouts"
    out_root.mkdir(parents=True, exist_ok=True)
    if args.cells == "grid":
        cells = [
            (float(NOMINAL_CUP_XY[0] + dx), float(NOMINAL_CUP_XY[1] + dy))
            for dx in GRID_OFFSETS
            for dy in GRID_OFFSETS
        ]
    else:
        cells = [
            (float(NOMINAL_CUP_XY[0] - 0.10), -0.10),
            (float(NOMINAL_CUP_XY[0] - 0.10), +0.10),
            (float(NOMINAL_CUP_XY[0] + 0.10), -0.10),
            (float(NOMINAL_CUP_XY[0] + 0.10), +0.10),
            (float(NOMINAL_CUP_XY[0]), 0.0),
        ]

    pedestals = list(args.pedestals)
    n_rollouts = len(cells) * len(pedestals)
    print(
        f"watching {checkpoints_dir}, interval={args.interval}s, "
        f"cells={len(cells)}, pedestals={pedestals}, rollouts/checkpoint={n_rollouts}"
    )
    last_steps: int | None = None
    while True:
        latest = _latest_checkpoint(checkpoints_dir)
        if latest is None:
            time.sleep(args.interval)
            continue
        steps, model_path, vec_path = latest
        if last_steps is None or steps > last_steps:
            try:
                _render_for_checkpoint(
                    steps,
                    model_path,
                    vec_path,
                    out_root,
                    args.reward_stage,
                    args.seed,
                    args.max_steps,
                    args.width,
                    args.height,
                    cells,
                    pedestals,
                )
                last_steps = steps
            except Exception as exc:
                print(f"watch_rollouts error checkpoint={steps}: {exc}")
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
