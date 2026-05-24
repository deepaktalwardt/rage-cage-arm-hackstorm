"""Live MuJoCo-viewer rollout of RageCageEnv with a trained or random policy.

Run from the repo root with the ``-m`` form so ``sim.env`` resolves:

    uv run mjpython -m sim.play_policy
    uv run mjpython -m sim.play_policy --model runs/rage_v33_15M_seed1_best_R3
    uv run mjpython -m sim.play_policy --model runs/rage_v33_15M_seed1_best_R3 --cup-grid
    uv run mjpython -m sim.play_policy --model runs/rage_v33_15M_seed1_best_R3 --cup-xy 0.95,0.10
    uv run mjpython -m sim.play_policy --model runs/rage_v33_15M_seed1_best_R3 --rand-stage 3
    uv run mjpython -m sim.play_policy --model random --no-randomize-cup --speed 0.5

macOS note: ``mujoco.viewer.launch_passive`` requires the main thread for Cocoa,
so the script must be launched with ``mjpython`` (installed alongside the
``mujoco`` pip package). Plain ``python`` will deadlock on launch.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Callable

import mujoco
import mujoco.viewer
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from sim.env import NOMINAL_CUP_XY, RageCageEnv


def _resolve_model_paths(model_arg: str) -> tuple[Path, Path]:
    """Return (policy.zip, vecnormalize.pkl) for a model argument.

    Accepts a directory (``policy.zip`` + ``vecnormalize.pkl`` inside),
    a stem (``<stem>.zip`` + ``<stem>.vecnormalize.pkl`` — matches what
    ``train_rl.py`` writes for best snapshots), or a direct ``.zip`` path
    (we look for the matching ``.vecnormalize.pkl`` next to it).
    """
    p = Path(model_arg)
    if p.is_dir():
        return p / "policy.zip", p / "vecnormalize.pkl"
    if p.suffix == ".zip":
        return p, p.with_suffix(".vecnormalize.pkl")
    return p.with_suffix(".zip"), p.with_suffix(".vecnormalize.pkl")


def _load_obs_normalizer(
    vecnorm_pkl: Path, base_env: RageCageEnv
) -> tuple[Any, float, float]:
    """Load only the obs-normalization parameters from a VecNormalize pickle.

    We deliberately avoid using the VecNormalize wrapper in the playback loop
    because ``DummyVecEnv.step_wait`` auto-resets on done, which clobbers the
    terminal MuJoCo state before we can pause on it. By extracting just
    ``obs_rms`` / ``clip_obs`` / ``epsilon`` we keep the normalization but step
    the base env directly so terminal frames stay visible for ``--pause-between``.
    """
    dummy = DummyVecEnv([lambda: base_env])
    vec_norm = VecNormalize.load(str(vecnorm_pkl), dummy)
    return vec_norm.obs_rms, float(vec_norm.clip_obs), float(vec_norm.epsilon)


def _parse_cup_xy(arg: str) -> tuple[float, float]:
    parts = arg.split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(f"--cup-xy expects 'x,y', got {arg!r}")
    return float(parts[0]), float(parts[1])


# Sampled cup positions for cycling modes. Same 3x3 layout as the
# ``cup_eval_mode='grid'`` evaluator and watch_rollouts (corners of the
# ±10cm operational envelope plus center, in row-major order).
_GRID_OFFSETS = (-0.10, 0.0, 0.10)
_RAND_STAGE_HALF_WIDTHS = {0: 0.02, 1: 0.05, 2: 0.08, 3: 0.10}
_ZRAND_STAGE_RANGES = {0: (0.0, 0.0), 1: (0.0, 0.05), 2: (0.0, 0.10), 3: (0.0, 0.15)}
_PEDESTAL_GRID_HEIGHTS = (0.0, 0.075, 0.15)


def _grid_positions() -> list[tuple[float, float]]:
    return [
        (float(NOMINAL_CUP_XY[0]) + dx, float(NOMINAL_CUP_XY[1]) + dy)
        for dx in _GRID_OFFSETS
        for dy in _GRID_OFFSETS
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default="models/single_bounce_cup_thrower_v1",
        help="Model dir / stem / .zip path, or 'random' for random actions.",
    )
    parser.add_argument(
        "--reward-stage",
        type=int,
        choices=(1, 2, 3, 4),
        default=3,
        help="Curriculum stage to run the env in. The shipped model trained "
        "to its peak in stage 3, so 3 matches train-time conditions.",
    )
    cup_group = parser.add_mutually_exclusive_group()
    cup_group.add_argument(
        "--no-randomize-cup",
        action="store_true",
        help="Pin the cup at NOMINAL_CUP_XY each reset (no randomization).",
    )
    cup_group.add_argument(
        "--cup-xy",
        type=_parse_cup_xy,
        default=None,
        help="Pin the cup at a specific (x, y) each reset, e.g. '--cup-xy 0.95,0.10'. "
        "Useful for inspecting policy behavior at a single workspace position.",
    )
    cup_group.add_argument(
        "--cup-grid",
        action="store_true",
        help="Cycle through the 3x3 ±10cm grid (9 positions) one per episode. "
        "Mirrors the 'grid' eval mode used by sim.eval_grid.",
    )
    cup_group.add_argument(
        "--rand-stage",
        type=int,
        choices=tuple(_RAND_STAGE_HALF_WIDTHS),
        default=None,
        help="Randomize cup within the specified R-stage box (R0=±2cm, R1=±5cm, "
        "R2=±8cm, R3=±10cm) each reset.",
    )
    pedestal_group = parser.add_mutually_exclusive_group()
    pedestal_group.add_argument(
        "--pedestal",
        type=float,
        default=None,
        help="Pin the pedestal at a specific height (m) each reset, e.g. '--pedestal 0.10' "
        "for a 10cm stack. Default: pedestal=0 (cup on table).",
    )
    pedestal_group.add_argument(
        "--pedestal-grid",
        action="store_true",
        help="Cycle pedestal through {0, 7.5cm, 15cm} per episode (mirrors grid3d eval). "
        "Combine with --cup-grid to walk the full 27-cell xy×z grid.",
    )
    pedestal_group.add_argument(
        "--rand-stage-z",
        type=int,
        choices=tuple(_ZRAND_STAGE_RANGES),
        default=None,
        help="Randomize pedestal within Z-stage range each reset (Z0=0, Z1=0-5cm, "
        "Z2=0-10cm, Z3=0-15cm).",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Wallclock playback speed multiplier (1.0 = real-time, 0.25 = "
        "quarter speed for the windup).",
    )
    parser.add_argument(
        "--pause-between",
        type=float,
        default=2.0,
        help="Seconds to freeze on the terminal frame before resetting.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=0,
        help="Stop after this many episodes (0 = run until viewer is closed).",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    # Cup-position mode: which env knob to set, and what to do per reset.
    pin_cup = args.no_randomize_cup or args.cup_xy is not None or args.cup_grid
    base_env = RageCageEnv(
        randomize_cup=not pin_cup,
        reward_stage=args.reward_stage,
    )
    if args.rand_stage is not None:
        half = _RAND_STAGE_HALF_WIDTHS[args.rand_stage]
        x_range = (float(NOMINAL_CUP_XY[0]) - half, float(NOMINAL_CUP_XY[0]) + half)
        y_range = (float(NOMINAL_CUP_XY[1]) - half, float(NOMINAL_CUP_XY[1]) + half)
        base_env.set_cup_range(x_range, y_range)
        print(f"Randomizing cup within R{args.rand_stage}: x={x_range} y={y_range}")
    elif args.cup_xy is not None:
        print(f"Pinning cup at {args.cup_xy}")
    elif args.cup_grid:
        grid = _grid_positions()
        print(f"Cycling cup through {len(grid)} grid positions: {grid}")
    elif args.no_randomize_cup:
        print(f"Pinning cup at NOMINAL_CUP_XY={tuple(NOMINAL_CUP_XY.tolist())}")

    if args.rand_stage_z is not None:
        z_range = _ZRAND_STAGE_RANGES[args.rand_stage_z]
        base_env.set_pedestal_range(z_range)
        print(f"Randomizing pedestal within Z{args.rand_stage_z}: {z_range}")
    elif args.pedestal is not None:
        print(f"Pinning pedestal at {args.pedestal:.3f}m")
    elif args.pedestal_grid:
        print(f"Cycling pedestal through {_PEDESTAL_GRID_HEIGHTS}")

    grid_positions = _grid_positions() if args.cup_grid else []

    def cup_for_reset(episode_idx: int) -> tuple[float, float] | None:
        if args.cup_xy is not None:
            return args.cup_xy
        if args.cup_grid:
            return grid_positions[episode_idx % len(grid_positions)]
        return None

    def pedestal_for_reset(episode_idx: int) -> float | None:
        if args.pedestal is not None:
            return float(args.pedestal)
        if args.pedestal_grid:
            return float(_PEDESTAL_GRID_HEIGHTS[episode_idx % len(_PEDESTAL_GRID_HEIGHTS)])
        return None

    select_action: Callable[[np.ndarray], np.ndarray]
    if args.model == "random":
        print("Using random policy (action_space.sample())")
        select_action = lambda _obs: base_env.action_space.sample()  # noqa: E731
    else:
        policy_zip, vecnorm_pkl = _resolve_model_paths(args.model)
        for path, label in ((policy_zip, "policy zip"), (vecnorm_pkl, "vecnormalize stats")):
            if not path.exists():
                raise FileNotFoundError(f"{label} not found at {path}")
        obs_rms, clip_obs, epsilon = _load_obs_normalizer(vecnorm_pkl, base_env)
        ppo = PPO.load(str(policy_zip))

        def select_action(obs: np.ndarray) -> np.ndarray:
            normalized = np.clip(
                (obs - obs_rms.mean) / np.sqrt(obs_rms.var + epsilon),
                -clip_obs,
                clip_obs,
            ).astype(np.float32)
            action, _ = ppo.predict(normalized, deterministic=True)
            return action

        print(f"Loaded {policy_zip} (with normalization stats from {vecnorm_pkl.name})")

    # One env.step covers control_steps MuJoCo substeps. Wallclock target per
    # step keeps the windup at real-time × speed; post-release the same step_dt
    # is used inside the passive_step_callback below so the ball's flight
    # animates instead of snapping to the terminal frame.
    step_dt = base_env.model.opt.timestep * base_env.control_steps / max(args.speed, 1e-6)
    print(
        f"  reward_stage={args.reward_stage}  randomize_cup={not args.no_randomize_cup}  "
        f"step_dt={step_dt:.4f}s  speed={args.speed}x"
    )

    initial_cup = cup_for_reset(0)
    if initial_cup is not None:
        base_env.set_next_cup(np.asarray(initial_cup, dtype=np.float32))
    initial_pedestal = pedestal_for_reset(0)
    if initial_pedestal is not None:
        base_env.set_next_pedestal(initial_pedestal)
    obs, _info = base_env.reset(seed=args.seed)
    ep_reward = 0.0
    episode_idx = 0
    successes = 0

    with mujoco.viewer.launch_passive(base_env.model, base_env.data) as viewer:
        # Hook: env's post-release passive loop calls this between control
        # steps so the viewer can sync mid-flight. Without it the ball would
        # teleport from the gripper to the terminal state in one frame.
        last_passive_sync = [time.time()]

        def passive_step_callback() -> None:
            if not viewer.is_running():
                return
            viewer.sync()
            elapsed = time.time() - last_passive_sync[0]
            if elapsed < step_dt:
                time.sleep(step_dt - elapsed)
            last_passive_sync[0] = time.time()

        base_env.passive_step_callback = passive_step_callback

        while viewer.is_running():
            t0 = time.time()
            action = select_action(obs)
            last_passive_sync[0] = time.time()  # reset timer for the callback
            obs, reward, terminated, truncated, info = base_env.step(action)
            ep_reward += float(reward)
            viewer.sync()

            if terminated or truncated:
                episode_idx += 1
                if info.get("success"):
                    successes += 1
                cup_xy = info.get("cup_xy", np.array([np.nan, np.nan]))
                closest = float(info.get("closest_post_bounce_cup_dist", float("inf")))
                cup_dist = float(info.get("cup_dist", float("nan")))
                bounces = int(info.get("table_bounce_count", 0))
                success = bool(info.get("success"))
                print(
                    f"ep {episode_idx:>3}  "
                    f"success={'YES' if success else ' no'}  "
                    f"bounces={bounces}  "
                    f"closest={closest:.3f}m  "
                    f"final_cup_dist={cup_dist:.3f}m  "
                    f"cup=({cup_xy[0]:.3f},{cup_xy[1]:+.3f})  "
                    f"reward={ep_reward:+.1f}  "
                    f"running_success_rate={successes / episode_idx:.2f}"
                )

                if args.episodes and episode_idx >= args.episodes:
                    break

                # Hold the terminal frame so the user can actually see the
                # outcome before reset snaps the arm/ball home.
                end_t = time.time() + args.pause_between
                while time.time() < end_t and viewer.is_running():
                    viewer.sync()
                    time.sleep(0.02)

                ep_reward = 0.0
                next_cup = cup_for_reset(episode_idx)
                if next_cup is not None:
                    base_env.set_next_cup(np.asarray(next_cup, dtype=np.float32))
                next_pedestal = pedestal_for_reset(episode_idx)
                if next_pedestal is not None:
                    base_env.set_next_pedestal(next_pedestal)
                obs, _info = base_env.reset()
                continue

            elapsed = time.time() - t0
            if elapsed < step_dt:
                time.sleep(step_dt - elapsed)


if __name__ == "__main__":
    main()
