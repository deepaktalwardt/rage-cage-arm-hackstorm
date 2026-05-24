"""Run grid-mode evaluation against a saved policy and dump per-cell metrics.

Used as the C4 probe in the multi-position cup-thrower design — runs v22
(or any saved policy) across the 3x3 grid spanning ±10cm of NOMINAL and
records success_rate, closest_cup_dist, valid_bounce per cell.

Outputs a CSV at the path given by --out and prints a text heatmap.

Run via:
  uv run python -m sim.eval_grid \
      --model models/single_bounce_cup_thrower_v1/policy.zip \
      --vecnormalize models/single_bounce_cup_thrower_v1/vecnormalize.pkl \
      --out sim/_rl_eval/v22_grid.csv

Or, against a v35-style run dir (loads policy.zip + vecnormalize.pkl,
writes grid.csv inside the run dir):

  uv run python -m sim.eval_grid --run-dir runs/v35_stacked_warm
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from sim.env import RageCageEnv, env_kwargs_from_training_json
from sim.train_rl import (
    GRID_OFFSETS,
    RUN_ENV_KWARGS,
    evaluate_policy_grid,
    evaluate_policy_metrics,
)


PPO_INFERENCE_CUSTOM_OBJECTS = {
    "lr_schedule": lambda _: 0.0,
    "learning_rate": 0.0,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="v35-style run dir; loads policy.zip + vecnormalize.pkl from here, writes grid.csv inside.",
    )
    parser.add_argument("--model", type=Path, default=None, help="path to <policy>.zip (with --vecnormalize/--out)")
    parser.add_argument("--vecnormalize", type=Path, default=None, help="path to <policy>.vecnormalize.pkl")
    parser.add_argument("--out", type=Path, default=None, help="path to write per-cell CSV")
    parser.add_argument("--reward-stage", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--also-fixed",
        action="store_true",
        help="also run fixed-cup eval and print mean success_rate as a sanity check",
    )
    args = parser.parse_args()
    if args.run_dir is not None:
        if any(v is not None for v in (args.model, args.vecnormalize, args.out)):
            parser.error("--run-dir is mutually exclusive with --model/--vecnormalize/--out")
        args.model = args.run_dir / "policy.zip"
        args.vecnormalize = args.run_dir / "vecnormalize.pkl"
        args.out = args.run_dir / "grid.csv"
    else:
        missing = [
            name
            for name, val in (("--model", args.model), ("--vecnormalize", args.vecnormalize), ("--out", args.out))
            if val is None
        ]
        if missing:
            parser.error(f"--model/--vecnormalize/--out all required when --run-dir is not given (missing: {missing})")
    return args


def _resolve_training_json(model_path: Path) -> Path:
    """Return the training.json that lives next to a model artifact."""
    if model_path.is_dir():
        return model_path / "training.json"
    return model_path.with_suffix(".training.json")


def _load_model(model_path: Path, vecnormalize_path: Path, reward_stage: int) -> PPO:
    env_kwargs = env_kwargs_from_training_json(_resolve_training_json(model_path))
    env = DummyVecEnv([lambda: RageCageEnv(randomize_cup=False, reward_stage=reward_stage, **env_kwargs)])
    env = VecNormalize.load(str(vecnormalize_path), env)
    env.training = False
    env.norm_reward = False
    return PPO.load(str(model_path), env=env, custom_objects=PPO_INFERENCE_CUSTOM_OBJECTS)


def _print_text_heatmap(rows: list[dict[str, float]]) -> None:
    by_cell = {(round(r["cup_x"], 4), round(r["cup_y"], 4)): r for r in rows}
    xs = sorted({round(r["cup_x"], 4) for r in rows})
    ys = sorted({round(r["cup_y"], 4) for r in rows}, reverse=True)
    print("\nsuccess_rate heatmap (cup_y rows × cup_x cols, ±10cm of nominal):")
    header = "         " + "  ".join(f"{x:6.3f}" for x in xs)
    print(header)
    for y in ys:
        cells = []
        for x in xs:
            r = by_cell[(x, y)]
            cells.append(f"{r['success']:6.2f}")
        print(f"y={y:+.3f}  " + "  ".join(cells))
    print("\nclosest_cup_dist heatmap (m, lower is better):")
    print(header)
    for y in ys:
        cells = []
        for x in xs:
            r = by_cell[(x, y)]
            cells.append(f"{r['closest_cup_dist']:6.3f}")
        print(f"y={y:+.3f}  " + "  ".join(cells))


def main() -> None:
    args = parse_args()
    # Populate RUN_ENV_KWARGS from the model's training.json so all
    # env constructions inside train_rl helpers (evaluate_policy_grid,
    # evaluate_policy_metrics) match what the policy was trained on.
    RUN_ENV_KWARGS.update(
        env_kwargs_from_training_json(_resolve_training_json(args.model))
    )
    model = _load_model(args.model, args.vecnormalize, args.reward_stage)

    if args.also_fixed:
        fixed = evaluate_policy_metrics(
            model, reward_stage=args.reward_stage, episodes=8, seed=args.seed, cup_eval_mode="fixed"
        )
        print(f"fixed_eval episodes={fixed['episodes']:.0f} success_rate={fixed['success_rate']:.3f} mean_reward={fixed['mean_reward']:.2f}")

    aggregate, rows = evaluate_policy_grid(model, reward_stage=args.reward_stage, seed=args.seed)
    print(
        f"grid_eval cells={len(rows)} "
        f"mean_success={aggregate['success_rate']:.3f} "
        f"valid_bounce_rate={aggregate['valid_bounce_rate']:.3f} "
        f"median_closest_cup_dist={aggregate['median_closest_cup_dist']:.3f}"
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["cup_x", "cup_y", "success", "closest_cup_dist", "valid_bounce", "exact_one_bounce", "bounce_target_hit", "total_reward"]
    with args.out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r[k] for k in fieldnames})
    print(f"wrote {args.out}")

    _print_text_heatmap(rows)


if __name__ == "__main__":
    main()
