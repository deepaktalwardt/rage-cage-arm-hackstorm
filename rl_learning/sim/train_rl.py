"""Train the first state-based PPO thrower policy."""

from __future__ import annotations

import argparse
import csv
import json
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch.nn as nn
from PIL import Image, ImageDraw
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from sim.env import NOMINAL_CUP_XY, RageCageEnv, bounce_score

# 3x3 grid of cup positions used by ``cup_eval_mode='grid'`` and the
# heatmap viz. Spans the full ±10cm operational envelope regardless of
# the current randomization sub-stage; this is the metric we judge
# end-to-end generalization on.
GRID_OFFSETS: tuple[float, ...] = (-0.10, 0.0, 0.10)


def _grid_cells() -> list[tuple[float, float]]:
    return [
        (float(NOMINAL_CUP_XY[0] + dx), float(NOMINAL_CUP_XY[1] + dy))
        for dx in GRID_OFFSETS
        for dy in GRID_OFFSETS
    ]


def _parse_latency_range(s: str) -> tuple[int, int]:
    """Parse a 'lo,hi' CLI string into a validated (int, int) tuple."""
    parts = [p.strip() for p in s.split(",")]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(
            f"--action-latency-range must be 'lo,hi'; got {s!r}"
        )
    lo, hi = int(parts[0]), int(parts[1])
    if lo < 0 or hi < lo:
        raise argparse.ArgumentTypeError(
            f"--action-latency-range must satisfy 0 <= lo <= hi; got ({lo}, {hi})"
        )
    return (lo, hi)

NET_ARCHS: dict[str, dict[str, list[int]]] = {
    "default": {},
    "medium": {"pi": [256, 256], "vf": [256, 256]},
    "large": {"pi": [512, 512], "vf": [512, 512]},
    "deep": {"pi": [256, 256, 128], "vf": [256, 256, 128]},
}

ACTIVATIONS = {
    "tanh": nn.Tanh,
    "relu": nn.ReLU,
}

# Run-wide RageCageEnv overrides set by main() from CLI args. Read by
# the env-construction helpers so training, eval, and rollout-viz envs
# all use the same action_delta / action_filter_alpha /
# action_latency_range / obs_joint_pos_noise_std / history lens. Empty
# dict = constructor defaults (back-compat).
# Public for cross-module callers (e.g., watch_rollouts.py populates
# this from the run's training.json before invoking rendering helpers).
RUN_ENV_KWARGS: dict[str, Any] = {}


@dataclass
class RewardStageRef:
    current: int


@dataclass
class RandomizationStageRef:
    current: int


@dataclass
class ZRandomizationStageRef:
    current: int


# Cup-position randomization curriculum. Sits inside reward stage 3
# (the working v22 reward); reward weights are unchanged across R0..R3.
# Each stage widens the per-episode cup-randomization box. Promotion is
# gated on success_rate measured at the *current* stage's range — a
# policy proficient on R1 can be R2-promoted without first matching R1's
# absolute number, since the harder task should not be required to meet
# easier-task thresholds. R3 has promote_at=None and is terminal.
RANDOMIZATION_STAGES: dict[int, dict[str, Any]] = {
    0: {
        "x_range": (float(NOMINAL_CUP_XY[0]) - 0.02, float(NOMINAL_CUP_XY[0]) + 0.02),
        "y_range": (-0.02, 0.02),
        "promote_at": 0.5,
    },
    1: {
        "x_range": (float(NOMINAL_CUP_XY[0]) - 0.05, float(NOMINAL_CUP_XY[0]) + 0.05),
        "y_range": (-0.05, 0.05),
        "promote_at": 0.4,
    },
    2: {
        "x_range": (float(NOMINAL_CUP_XY[0]) - 0.08, float(NOMINAL_CUP_XY[0]) + 0.08),
        "y_range": (-0.08, 0.08),
        "promote_at": 0.3,
    },
    3: {
        "x_range": (float(NOMINAL_CUP_XY[0]) - 0.10, float(NOMINAL_CUP_XY[0]) + 0.10),
        "y_range": (-0.10, 0.10),
        "promote_at": None,
    },
}


def next_rand_stage(current: int, metrics: dict[str, Any]) -> int | None:
    if current not in RANDOMIZATION_STAGES:
        raise ValueError(f"unknown rand_stage={current}")
    threshold = RANDOMIZATION_STAGES[current]["promote_at"]
    if threshold is None:
        return None
    if float(metrics.get("success_rate", 0.0)) >= threshold:
        return current + 1
    return None


def apply_rand_stage_to_env(env: Any, stage: int) -> None:
    cfg = RANDOMIZATION_STAGES[stage]
    if hasattr(env, "set_cup_range"):
        env.set_cup_range(cfg["x_range"], cfg["y_range"])
    else:
        env.env_method("set_cup_range", cfg["x_range"], cfg["y_range"])


# Pedestal-height curriculum, parallel to RANDOMIZATION_STAGES. Sits
# inside reward stage 3 + R-stage 3 (warm-start case from v34). Each
# stage widens the per-episode pedestal range. Z0=0 only doubles as a
# warm-start sanity check (pedestal=0 should match v34 baseline). Z3 is
# terminal. Promotion thresholds mirror R-stage's 0.5/0.4/0.3.
Z_RANDOMIZATION_STAGES: dict[int, dict[str, Any]] = {
    0: {"z_range": (0.0, 0.0), "promote_at": 0.5},
    1: {"z_range": (0.0, 0.05), "promote_at": 0.4},
    2: {"z_range": (0.0, 0.10), "promote_at": 0.3},
    3: {"z_range": (0.0, 0.15), "promote_at": None},
}

# Pedestal heights probed by the 3×3×3 grid eval. Spans the full
# Z3 envelope; matches the design's `grid3d` mode.
PEDESTAL_GRID_HEIGHTS: tuple[float, ...] = (0.0, 0.075, 0.15)


def next_zrand_stage(current: int, metrics: dict[str, Any]) -> int | None:
    if current not in Z_RANDOMIZATION_STAGES:
        raise ValueError(f"unknown zrand_stage={current}")
    threshold = Z_RANDOMIZATION_STAGES[current]["promote_at"]
    if threshold is None:
        return current
    if metrics.get("success_rate", 0.0) >= threshold:
        return min(current + 1, max(Z_RANDOMIZATION_STAGES))
    return current


def apply_zrand_stage_to_env(env: Any, stage: int) -> None:
    cfg = Z_RANDOMIZATION_STAGES[stage]
    if hasattr(env, "set_pedestal_range"):
        env.set_pedestal_range(cfg["z_range"])
    else:
        env.env_method("set_pedestal_range", cfg["z_range"])


def linear_schedule(initial_lr: float):
    def schedule(progress_remaining: float) -> float:
        return progress_remaining * initial_lr

    return schedule


def learning_rate_arg(initial_lr: float, schedule: str):
    if schedule == "constant":
        return initial_lr
    if schedule == "linear":
        return linear_schedule(initial_lr)
    raise ValueError(f"unsupported learning rate schedule: {schedule}")


def policy_kwargs(net_arch: str, activation: str) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"activation_fn": ACTIVATIONS[activation]}
    if NET_ARCHS[net_arch]:
        kwargs["net_arch"] = NET_ARCHS[net_arch]
    return kwargs


def _make_render_env(
    source_env: VecNormalize | None,
    seed: int,
    fixed_cup: bool,
    reward_stage: int,
    width: int,
    height: int,
) -> VecNormalize:
    env = DummyVecEnv(
        [
            lambda: RageCageEnv(
                randomize_cup=not fixed_cup,
                reward_stage=reward_stage,
                render_mode="rgb_array",
                image_width=width,
                image_height=height,
                **RUN_ENV_KWARGS,
            )
        ]
    )
    env.seed(seed)
    render_env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)
    if source_env is not None:
        render_env.obs_rms = deepcopy(source_env.obs_rms)
        render_env.ret_rms = deepcopy(source_env.ret_rms)
        render_env.clip_obs = source_env.clip_obs
        render_env.gamma = source_env.gamma
        render_env.epsilon = source_env.epsilon
    render_env.training = False
    render_env.norm_reward = False
    return render_env


def _annotate_frame(frame: np.ndarray, lines: list[str]) -> Image.Image:
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image, "RGBA")
    line_height = 16
    box_height = 12 + line_height * len(lines)
    draw.rectangle((8, 8, 320, box_height), fill=(0, 0, 0, 150))
    for idx, line in enumerate(lines):
        draw.text((16, 14 + idx * line_height), line, fill=(255, 255, 255, 255))
    return image


def render_policy_rollout(
    model: PPO,
    out_dir: Path,
    label: str,
    seed: int,
    fixed_cup: bool,
    reward_stage: int,
    max_steps: int,
    width: int,
    height: int,
) -> tuple[Path | None, Path, float, dict[str, Any]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    env = _make_render_env(
        model.get_vec_normalize_env(),
        seed=seed,
        fixed_cup=fixed_cup,
        reward_stage=reward_stage,
        width=width,
        height=height,
    )

    obs = env.reset()
    done = np.array([False])
    frames: list[Image.Image] = []
    rows: list[dict[str, Any]] = []
    total_reward = 0.0
    final_info: dict[str, Any] = {}
    step = 0

    # Capture the post-reset state as frame 0 so GIFs always have at
    # least one frame even if the episode terminates before any control
    # step completes (early-training arm flailing fires
    # motion_limit_violated immediately).
    initial_frame = env.env_method("render")[0]
    if initial_frame is not None:
        frames.append(
            _annotate_frame(
                initial_frame,
                [f"{label}", "step=0 reset"],
            )
        )

    while not done[0] and step < max_steps:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, infos = env.step(action)
        step_reward = float(reward[0])
        total_reward += step_reward
        final_info = infos[0]

        row = {
            "step": step,
            "phase": "policy",
            "sim_step": final_info.get("step_count", step),
            "reward": step_reward,
            "cumulative_reward": total_reward,
            "reward_so_far_within_release": np.nan,
            "success": final_info.get("success", False),
            "bounce_count": final_info.get("bounce_count", 0),
            "table_bounce_count": final_info.get("table_bounce_count", final_info.get("bounce_count", 0)),
            "invalid_bounce_count": final_info.get("invalid_bounce_count", 0),
            "cup_dist": final_info.get("cup_dist", np.nan),
            "closest_post_bounce_cup_dist": final_info.get("closest_post_bounce_cup_dist", np.nan),
            "second_table_bounce_cup_dist": final_info.get("second_table_bounce_cup_dist", np.nan),
            "ball_released": final_info.get("ball_released", False),
            "ball_entered_cup": final_info.get("ball_entered_cup", False),
            "settled_in_cup": final_info.get("settled_in_cup", False),
            "robot_table_contact": final_info.get("robot_table_contact", False),
            "robot_cup_contact": final_info.get("robot_cup_contact", False),
            "ball_contacted_floor": final_info.get("ball_contacted_floor", False),
            "ball_contacted_robot": final_info.get("ball_contacted_robot", False),
            "max_joint_vel": final_info.get("max_joint_vel", np.nan),
            "max_joint_acc": final_info.get("max_joint_acc", np.nan),
            "max_joint_jerk": final_info.get("max_joint_jerk", np.nan),
            "motion_limit_violated": final_info.get("motion_limit_violated", False),
            "reward_components": json.dumps(final_info.get("reward_components", {}), sort_keys=True),
        }
        rows.append(row)

        passive_frames = final_info.get("passive_render_frames", [])
        passive_rows = final_info.get("passive_info_rows", [])
        for passive_idx, passive_frame in enumerate(passive_frames):
            passive_info = passive_rows[passive_idx] if passive_idx < len(passive_rows) else {}
            passive_row = {
                "step": step,
                "phase": "passive",
                "sim_step": passive_info.get("step_count", np.nan),
                "reward": np.nan,
                "cumulative_reward": total_reward,
                "reward_so_far_within_release": passive_info.get("reward_so_far", np.nan),
                "success": passive_info.get("success", False),
                "bounce_count": passive_info.get("bounce_count", 0),
                "table_bounce_count": passive_info.get("table_bounce_count", 0),
                "invalid_bounce_count": passive_info.get("invalid_bounce_count", 0),
                "cup_dist": passive_info.get("cup_dist", np.nan),
                "closest_post_bounce_cup_dist": passive_info.get("closest_post_bounce_cup_dist", np.nan),
                "second_table_bounce_cup_dist": passive_info.get("second_table_bounce_cup_dist", np.nan),
                "ball_released": passive_info.get("ball_released", True),
                "ball_entered_cup": passive_info.get("ball_entered_cup", False),
                "settled_in_cup": passive_info.get("settled_in_cup", False),
                "robot_table_contact": passive_info.get("robot_table_contact", False),
                "robot_cup_contact": passive_info.get("robot_cup_contact", False),
                "ball_contacted_floor": passive_info.get("ball_contacted_floor", False),
                "ball_contacted_robot": passive_info.get("ball_contacted_robot", False),
                "max_joint_vel": passive_info.get("max_joint_vel", np.nan),
                "max_joint_acc": passive_info.get("max_joint_acc", np.nan),
                "max_joint_jerk": passive_info.get("max_joint_jerk", np.nan),
                "motion_limit_violated": passive_info.get("motion_limit_violated", False),
                "reward_components": "",
            }
            rows.append(passive_row)
            frames.append(
                _annotate_frame(
                    passive_frame,
                    [
                        f"{label}",
                        f"policy_step={step} passive_step={passive_row['sim_step']}",
                        f"total={total_reward:.3f} bounces={passive_row['bounce_count']}",
                        f"cup_dist={float(passive_row['cup_dist']):.3f} success={passive_row['success']}",
                    ],
                )
            )

        # Skip the post-step env.render() if the episode terminated this
        # step. SB3's VecEnv auto-resets on done, so env.render() now
        # shows the NEXT episode's t=0 state with the cup back at default.
        # Capturing that frame produces the visible "cup snaps back to
        # nominal" artifact when the GIF loops.
        if not done[0]:
            frame = env.env_method("render")[0]
            if frame is not None:
                frames.append(
                    _annotate_frame(
                        frame,
                        [
                            f"{label}",
                            f"step={step} reward={step_reward:.3f}",
                            f"total={total_reward:.3f} bounces={row['bounce_count']}",
                            f"cup_dist={float(row['cup_dist']):.3f} success={row['success']}",
                        ],
                    )
                )
        step += 1

    csv_path = out_dir / f"{label}.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "step",
                "phase",
                "sim_step",
                "reward",
                "cumulative_reward",
                "reward_so_far_within_release",
                "success",
                "bounce_count",
                "table_bounce_count",
                "invalid_bounce_count",
                "cup_dist",
                "closest_post_bounce_cup_dist",
                "second_table_bounce_cup_dist",
                "ball_released",
                "ball_entered_cup",
                "settled_in_cup",
                "robot_table_contact",
                "robot_cup_contact",
                "ball_contacted_floor",
                "ball_contacted_robot",
                "max_joint_vel",
                "max_joint_acc",
                "max_joint_jerk",
                "motion_limit_violated",
                "reward_components",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    gif_path = None
    if frames:
        gif_path = out_dir / f"{label}.gif"
        frames[0].save(
            gif_path,
            save_all=True,
            append_images=frames[1:],
            duration=20,
            loop=0,
        )

    env.close()
    return gif_path, csv_path, total_reward, final_info


def render_policy_rollout_at_cup(
    model: PPO,
    out_dir: Path,
    label: str,
    seed: int,
    cup_xy: tuple[float, float],
    reward_stage: int,
    max_steps: int,
    width: int,
    height: int,
    pedestal_height: float | None = None,
) -> tuple[Path | None, Path, float, dict[str, Any]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    env = _make_render_env(
        model.get_vec_normalize_env(),
        seed=seed,
        fixed_cup=True,
        reward_stage=reward_stage,
        width=width,
        height=height,
    )
    env.env_method("set_next_cup", np.asarray(cup_xy, dtype=np.float32))
    if pedestal_height is not None:
        env.env_method("set_next_pedestal", float(pedestal_height))

    obs = env.reset()
    done = np.array([False])
    frames: list[Image.Image] = []
    rows: list[dict[str, Any]] = []
    total_reward = 0.0
    final_info: dict[str, Any] = {}
    step = 0

    # Capture the post-reset state as frame 0 so GIFs always have at
    # least one frame even if the episode terminates before any control
    # step completes (early-training arm flailing fires
    # motion_limit_violated immediately).
    initial_frame = env.env_method("render")[0]
    if initial_frame is not None:
        frames.append(
            _annotate_frame(
                initial_frame,
                [f"{label}", "step=0 reset"],
            )
        )

    while not done[0] and step < max_steps:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, infos = env.step(action)
        step_reward = float(reward[0])
        total_reward += step_reward
        final_info = infos[0]

        cup_xy_now = final_info.get("cup_xy", np.asarray(cup_xy, dtype=np.float32))
        rows.append(
            {
                "step": step,
                "sim_step": final_info.get("step_count", step),
                "reward": step_reward,
                "cumulative_reward": total_reward,
                "success": final_info.get("success", False),
                "table_bounce_count": final_info.get("table_bounce_count", 0),
                "cup_x": float(cup_xy_now[0]),
                "cup_y": float(cup_xy_now[1]),
                "cup_dist": final_info.get("cup_dist", np.nan),
                "closest_post_bounce_cup_dist": final_info.get("closest_post_bounce_cup_dist", np.nan),
                "ball_released": final_info.get("ball_released", False),
                "ball_entered_cup": final_info.get("ball_entered_cup", False),
            }
        )

        passive_frames = final_info.get("passive_render_frames", [])
        for passive_frame in passive_frames:
            frames.append(
                _annotate_frame(
                    passive_frame,
                    [
                        f"{label}",
                        f"cup=({cup_xy_now[0]:.3f},{cup_xy_now[1]:+.3f})",
                        f"total={total_reward:.3f}",
                    ],
                )
            )
        if not done[0]:
            frame = env.env_method("render")[0]
            if frame is not None:
                frames.append(
                    _annotate_frame(
                        frame,
                        [
                            f"{label}",
                            f"cup=({cup_xy_now[0]:.3f},{cup_xy_now[1]:+.3f})",
                            f"step={step} success={final_info.get('success', False)}",
                        ],
                    )
                )
        step += 1

    csv_path = out_dir / f"{label}.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "step",
                "sim_step",
                "reward",
                "cumulative_reward",
                "success",
                "table_bounce_count",
                "cup_x",
                "cup_y",
                "cup_dist",
                "closest_post_bounce_cup_dist",
                "ball_released",
                "ball_entered_cup",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    gif_path = None
    if frames:
        gif_path = out_dir / f"{label}.gif"
        frames[0].save(
            gif_path,
            save_all=True,
            append_images=frames[1:],
            duration=20,
            loop=0,
        )

    env.close()
    return gif_path, csv_path, total_reward, final_info


def _make_eval_env(
    source_env: VecNormalize | None,
    seed: int,
    reward_stage: int,
    randomize_cup: bool = False,
    cup_range: tuple[tuple[float, float], tuple[float, float]] | None = None,
    pedestal_range: tuple[float, float] | None = None,
) -> VecNormalize:
    env = DummyVecEnv([lambda: RageCageEnv(randomize_cup=randomize_cup, reward_stage=reward_stage, **RUN_ENV_KWARGS)])
    env.seed(seed)
    eval_env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)
    if source_env is not None:
        eval_env.obs_rms = deepcopy(source_env.obs_rms)
        eval_env.ret_rms = deepcopy(source_env.ret_rms)
        eval_env.clip_obs = source_env.clip_obs
        eval_env.gamma = source_env.gamma
        eval_env.epsilon = source_env.epsilon
    if cup_range is not None:
        eval_env.env_method("set_cup_range", cup_range[0], cup_range[1])
    if pedestal_range is not None:
        eval_env.env_method("set_pedestal_range", pedestal_range)
    eval_env.training = False
    eval_env.norm_reward = False
    return eval_env


def _run_eval_episodes(
    model: PPO,
    reward_stage: int,
    seed: int,
    cup_eval_mode: str,
    episodes: int,
    cup_range: tuple[tuple[float, float], tuple[float, float]] | None,
    pedestal_range: tuple[float, float] | None = None,
) -> list[dict[str, Any]]:
    pedestal_overrides: list[float | None]
    if cup_eval_mode == "fixed":
        randomize_cup = False
        cup_overrides: list[np.ndarray | None] = [None] * episodes
        pedestal_overrides = [None] * episodes
        applied_range = None
        applied_pedestal_range: tuple[float, float] | None = None
    elif cup_eval_mode == "range":
        if cup_range is None:
            raise ValueError("cup_eval_mode='range' requires cup_range argument")
        randomize_cup = True
        cup_overrides = [None] * episodes
        pedestal_overrides = [None] * episodes
        applied_range = cup_range
        applied_pedestal_range = pedestal_range
    elif cup_eval_mode == "grid":
        randomize_cup = False
        cells = _grid_cells()
        cup_overrides = [np.asarray(cell, dtype=np.float32) for cell in cells]
        pedestal_overrides = [None] * len(cup_overrides)
        episodes = len(cup_overrides)
        applied_range = None
        applied_pedestal_range = None
    elif cup_eval_mode == "grid3d":
        randomize_cup = False
        cells = _grid_cells()
        cup_overrides = []
        pedestal_overrides = []
        for cell in cells:
            for ped in PEDESTAL_GRID_HEIGHTS:
                cup_overrides.append(np.asarray(cell, dtype=np.float32))
                pedestal_overrides.append(float(ped))
        episodes = len(cup_overrides)
        applied_range = None
        applied_pedestal_range = None
    else:
        raise ValueError(f"unsupported cup_eval_mode={cup_eval_mode!r}")

    env = _make_eval_env(
        model.get_vec_normalize_env(),
        seed=seed,
        reward_stage=reward_stage,
        randomize_cup=randomize_cup,
        cup_range=applied_range,
        pedestal_range=applied_pedestal_range,
    )

    rows: list[dict[str, Any]] = []
    for i in range(episodes):
        if cup_overrides[i] is not None:
            env.env_method("set_next_cup", cup_overrides[i])
        if pedestal_overrides[i] is not None:
            env.env_method("set_next_pedestal", pedestal_overrides[i])
        obs = env.reset()
        done = np.array([False])
        total_reward = 0.0
        final_info: dict[str, Any] = {}
        while not done[0]:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, infos = env.step(action)
            total_reward += float(reward[0])
            final_info = infos[0]

        cup_xy = np.asarray(final_info.get("cup_xy"), dtype=np.float64)
        pedestal_height = float(final_info.get("pedestal_height", 0.0))
        table_bounce = int(final_info.get("table_bounce_count", final_info.get("bounce_count", 0)))
        invalid = int(final_info.get("invalid_bounce_count", 0))
        first_bounce_xy = final_info.get("first_table_bounce_xy")
        if first_bounce_xy is not None:
            # Score in [0, 1] using the throw-frame elliptical reward
            # geometry. Diagnostic threshold: bounce_target_hit means
            # the bounce landed inside the unit ellipse (score > 0).
            bounce_target_score = bounce_score(np.asarray(first_bounce_xy), cup_xy)
            bounce_target_hit = float(bounce_target_score > 0.0)
        else:
            bounce_target_hit = 0.0
        closest = float(final_info.get("closest_post_bounce_cup_dist", np.inf))
        if not np.isfinite(closest):
            closest = 10.0
        second = float(final_info.get("second_table_bounce_cup_dist", np.inf))
        if not np.isfinite(second):
            second = 10.0
        rows.append(
            {
                "cup_x": float(cup_xy[0]),
                "cup_y": float(cup_xy[1]),
                "pedestal_height": pedestal_height,
                "total_reward": total_reward,
                "success": float(final_info.get("success", False)),
                "valid_bounce": float(table_bounce >= 1 and invalid == 0),
                "exact_one_bounce": float(table_bounce == 1 and invalid == 0),
                "invalid_contact": float(invalid > 0),
                "bounce_target_hit": bounce_target_hit,
                "closest_cup_dist": closest,
                "second_bounce_cup_dist": second,
            }
        )

    env.close()
    return rows


def _aggregate_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"episodes": 0.0}
    cols = {
        key: np.array([r[key] for r in rows], dtype=np.float64)
        for key in rows[0]
        if key not in {"cup_x", "cup_y", "pedestal_height"}
    }
    return {
        "episodes": float(len(rows)),
        "mean_reward": float(cols["total_reward"].mean()),
        "success_rate": float(cols["success"].mean()),
        "valid_bounce_rate": float(cols["valid_bounce"].mean()),
        "exact_one_bounce_rate": float(cols["exact_one_bounce"].mean()),
        "bounce_target_rate": float(cols["bounce_target_hit"].mean()),
        "median_closest_cup_dist": float(np.median(cols["closest_cup_dist"])),
        "median_second_bounce_cup_dist": float(np.median(cols["second_bounce_cup_dist"])),
        "invalid_contact_rate": float(cols["invalid_contact"].mean()),
        "_episode_cup_xys": [(r["cup_x"], r["cup_y"]) for r in rows],
    }


def evaluate_policy_metrics(
    model: PPO,
    reward_stage: int,
    episodes: int,
    seed: int,
    cup_eval_mode: str = "fixed",
    cup_range: tuple[tuple[float, float], tuple[float, float]] | None = None,
    pedestal_range: tuple[float, float] | None = None,
) -> dict[str, Any]:
    rows = _run_eval_episodes(
        model,
        reward_stage=reward_stage,
        seed=seed,
        cup_eval_mode=cup_eval_mode,
        episodes=episodes,
        cup_range=cup_range,
        pedestal_range=pedestal_range,
    )
    return _aggregate_metrics(rows)


def evaluate_policy_grid(
    model: PPO,
    reward_stage: int,
    seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows = _run_eval_episodes(
        model,
        reward_stage=reward_stage,
        seed=seed,
        cup_eval_mode="grid",
        episodes=len(_grid_cells()),
        cup_range=None,
    )
    return _aggregate_metrics(rows), rows


def evaluate_policy_grid3d(
    model: PPO,
    reward_stage: int,
    seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows = _run_eval_episodes(
        model,
        reward_stage=reward_stage,
        seed=seed,
        cup_eval_mode="grid3d",
        episodes=len(_grid_cells()) * len(PEDESTAL_GRID_HEIGHTS),
        cup_range=None,
    )
    return _aggregate_metrics(rows), rows


class TrainingRolloutVizCallback(BaseCallback):
    def __init__(
        self,
        out_dir: Path,
        every_timesteps: int,
        seed: int,
        fixed_cup: bool,
        reward_stage_ref: RewardStageRef,
        max_steps: int,
        width: int,
        height: int,
    ) -> None:
        super().__init__()
        self.out_dir = out_dir
        self.every_timesteps = every_timesteps
        self.seed = seed
        self.fixed_cup = fixed_cup
        self.reward_stage_ref = reward_stage_ref
        self.max_steps = max_steps
        self.width = width
        self.height = height
        self.next_timestep = every_timesteps

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        if self.every_timesteps <= 0 or self.num_timesteps < self.next_timestep:
            return
        label = f"train_rollout_{self.num_timesteps:09d}"
        gif_path, csv_path, total_reward, final_info = render_policy_rollout(
            self.model,
            self.out_dir,
            label=label,
            seed=self.seed + self.num_timesteps,
            fixed_cup=self.fixed_cup,
            reward_stage=self.reward_stage_ref.current,
            max_steps=self.max_steps,
            width=self.width,
            height=self.height,
        )
        print(
            f"training_rollout timestep={self.num_timesteps} reward={total_reward:.2f} "
            f"success={final_info.get('success')} bounce_count={final_info.get('bounce_count')} "
            f"cup_dist={final_info.get('cup_dist'):.3f} gif={gif_path} csv={csv_path}"
        )
        while self.next_timestep <= self.num_timesteps:
            self.next_timestep += self.every_timesteps


class CurriculumCallback(BaseCallback):
    def __init__(
        self,
        stage_ref: RewardStageRef,
        mode: str,
        eval_every_timesteps: int,
        eval_episodes: int,
        seed: int,
        log_path: Path,
        stage_rollout_dir: Path | None = None,
        stage_rollout_fixed_cup: bool = True,
        stage_rollout_max_steps: int = 300,
        stage_rollout_width: int = 640,
        stage_rollout_height: int = 480,
        best_model_save_stem: Path | None = None,
        rand_stage_ref: RandomizationStageRef | None = None,
        range_eval_episodes: int = 16,
        grid_log_path: Path | None = None,
        zrand_stage_ref: ZRandomizationStageRef | None = None,
    ) -> None:
        super().__init__()
        self.stage_ref = stage_ref
        self.mode = mode
        self.eval_every_timesteps = eval_every_timesteps
        self.eval_episodes = eval_episodes
        self.seed = seed
        self.log_path = log_path
        self.stage_rollout_dir = stage_rollout_dir
        self.stage_rollout_fixed_cup = stage_rollout_fixed_cup
        self.stage_rollout_max_steps = stage_rollout_max_steps
        self.stage_rollout_width = stage_rollout_width
        self.stage_rollout_height = stage_rollout_height
        self.next_eval_timestep = eval_every_timesteps
        # Best-snapshot tracking. With rand_stage_ref / zrand_stage_ref
        # provided we save one best snapshot per curriculum stage so a
        # late-stage destabilization doesn't lose the prior stage's peak
        # (the mitigation we wished we had after v21's stage-4 collapse).
        # Without these refs we keep legacy single-global-best behavior.
        self.best_model_save_stem = best_model_save_stem
        self._best_success_rate = -1.0
        self._best_mean_reward = float("-inf")
        self._best_timestep: int | None = None
        self._best_per_rand_stage: dict[int, float] = {}
        self._best_per_rand_stage_reward: dict[int, float] = {}
        self._best_per_zrand_stage: dict[int, float] = {}
        self._best_per_zrand_stage_reward: dict[int, float] = {}
        # Randomization-curriculum state.
        self.rand_stage_ref = rand_stage_ref
        self.range_eval_episodes = range_eval_episodes
        self.grid_log_path = grid_log_path
        self.zrand_stage_ref = zrand_stage_ref

    def _on_step(self) -> bool:
        return True

    def _on_training_start(self) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_path.write_text(
            "timesteps,stage,rand_stage,zrand_stage,mean_reward,success_rate,valid_bounce_rate,"
            "exact_one_bounce_rate,bounce_target_rate,median_closest_cup_dist,"
            "median_second_bounce_cup_dist,invalid_contact_rate,"
            "range_success_rate,grid_success_rate,grid3d_success_rate,"
            "promoted_to,rand_promoted_to,zrand_promoted_to\n"
        )
        if self.grid_log_path is not None:
            self.grid_log_path.parent.mkdir(parents=True, exist_ok=True)
            self.grid_log_path.write_text(
                "timesteps,rand_stage,zrand_stage,cup_x,cup_y,pedestal_height,success,closest_cup_dist,valid_bounce\n"
            )
        self._set_training_env_stage(self.stage_ref.current)
        if self.rand_stage_ref is not None:
            self._apply_rand_stage_to_training_env(self.rand_stage_ref.current)
        if self.zrand_stage_ref is not None:
            self._apply_zrand_stage_to_training_env(self.zrand_stage_ref.current)

    def _on_rollout_end(self) -> None:
        if self.mode != "auto" or self.eval_every_timesteps <= 0:
            return
        if self.num_timesteps < self.next_eval_timestep:
            return
        evaluated_stage = self.stage_ref.current
        evaluated_rand = self.rand_stage_ref.current if self.rand_stage_ref is not None else None
        evaluated_zrand = self.zrand_stage_ref.current if self.zrand_stage_ref is not None else None
        fixed_metrics = evaluate_policy_metrics(
            self.model,
            reward_stage=evaluated_stage,
            episodes=self.eval_episodes,
            seed=self.seed + self.num_timesteps,
            cup_eval_mode="fixed",
        )

        range_metrics: dict[str, Any] | None = None
        grid_aggregate: dict[str, Any] | None = None
        grid_rows: list[dict[str, Any]] | None = None
        grid3d_aggregate: dict[str, Any] | None = None
        if self.rand_stage_ref is not None:
            rand_cfg = RANDOMIZATION_STAGES[evaluated_rand]
            zrand_cfg = (
                Z_RANDOMIZATION_STAGES[evaluated_zrand]
                if self.zrand_stage_ref is not None
                else None
            )
            range_metrics = evaluate_policy_metrics(
                self.model,
                reward_stage=evaluated_stage,
                episodes=self.range_eval_episodes,
                seed=self.seed + self.num_timesteps + 1,
                cup_eval_mode="range",
                cup_range=(rand_cfg["x_range"], rand_cfg["y_range"]),
                pedestal_range=zrand_cfg["z_range"] if zrand_cfg is not None else None,
            )
            if self.zrand_stage_ref is not None:
                # When the Z curriculum is active, the 3×3×3 grid is the
                # generalization metric we judge best snapshots on. The
                # 9-cell z=0 layer is sub-aggregated for backwards-compat
                # display alongside it.
                grid3d_aggregate, grid3d_rows = evaluate_policy_grid3d(
                    self.model,
                    reward_stage=evaluated_stage,
                    seed=self.seed + self.num_timesteps + 2,
                )
                grid_rows = grid3d_rows
                z0_rows = [r for r in grid3d_rows if r["pedestal_height"] == 0.0]
                grid_aggregate = _aggregate_metrics(z0_rows) if z0_rows else None
            else:
                grid_aggregate, grid_rows = evaluate_policy_grid(
                    self.model,
                    reward_stage=evaluated_stage,
                    seed=self.seed + self.num_timesteps + 2,
                )
            if self.grid_log_path is not None and grid_rows is not None:
                self._append_grid_log(grid_rows, evaluated_rand, evaluated_zrand)

        # Reward-stage promotion (1→2→3) always runs. When already at
        # stage 3 (warm-start case), _maybe_promote returns None because
        # 3→4 promotion is disabled. The randomization curricula below
        # are independent — they advance R0..R3 and Z0..Z3 once we're at
        # reward stage 3. All three curricula can be active in the same
        # run; in v35 R is pinned at R3 and only Z advances.
        promoted_reward = self._maybe_promote(fixed_metrics)

        promoted_rand = None
        if self.rand_stage_ref is not None and range_metrics is not None:
            target = next_rand_stage(self.rand_stage_ref.current, range_metrics)
            if target is not None and target != self.rand_stage_ref.current:
                self._render_rand_stage_end_rollout(self.rand_stage_ref.current)
                self.rand_stage_ref.current = target
                self._apply_rand_stage_to_training_env(target)
                promoted_rand = target

        promoted_zrand = None
        if self.zrand_stage_ref is not None and range_metrics is not None:
            target = next_zrand_stage(self.zrand_stage_ref.current, range_metrics)
            if target is not None and target != self.zrand_stage_ref.current:
                self._render_zrand_stage_end_rollout(self.zrand_stage_ref.current)
                self.zrand_stage_ref.current = target
                self._apply_zrand_stage_to_training_env(target)
                promoted_zrand = target

        # Best-snapshot saving. With Z curriculum, judge by grid3d
        # success_rate (full 3×3×3 generalization). With R curriculum
        # only, by 3×3 grid. Otherwise legacy fixed-cup.
        if self.zrand_stage_ref is not None and grid3d_aggregate is not None:
            saved_best = self._maybe_save_best_per_zrand_stage(grid3d_aggregate, evaluated_zrand)
        elif self.rand_stage_ref is not None and grid_aggregate is not None:
            saved_best = self._maybe_save_best_per_rand_stage(grid_aggregate, evaluated_rand)
        else:
            saved_best = self._maybe_save_best(fixed_metrics)

        self._append_log(
            fixed_metrics,
            evaluated_stage,
            evaluated_rand,
            evaluated_zrand,
            range_metrics,
            grid_aggregate,
            grid3d_aggregate,
            promoted_reward,
            promoted_rand,
            promoted_zrand,
        )

        self.logger.record("rage/eval/stage", evaluated_stage)
        if evaluated_rand is not None:
            self.logger.record("rage/eval/rand_stage", evaluated_rand)
        if evaluated_zrand is not None:
            self.logger.record("rage/eval/zrand_stage", evaluated_zrand)
        self.logger.record("rage/eval/mean_reward", fixed_metrics["mean_reward"])
        self.logger.record("rage/eval/success_rate", fixed_metrics["success_rate"])
        self.logger.record("rage/eval/valid_bounce_rate", fixed_metrics["valid_bounce_rate"])
        self.logger.record("rage/eval/exact_one_bounce_rate", fixed_metrics["exact_one_bounce_rate"])
        self.logger.record("rage/eval/bounce_target_rate", fixed_metrics["bounce_target_rate"])
        self.logger.record("rage/eval/median_closest_cup_dist", fixed_metrics["median_closest_cup_dist"])
        self.logger.record("rage/eval/median_second_bounce_cup_dist", fixed_metrics["median_second_bounce_cup_dist"])
        self.logger.record("rage/eval/invalid_contact_rate", fixed_metrics["invalid_contact_rate"])
        if range_metrics is not None:
            self.logger.record("rage/eval/range_success_rate", range_metrics["success_rate"])
            self.logger.record("rage/eval/range_valid_bounce_rate", range_metrics["valid_bounce_rate"])
        if grid_aggregate is not None:
            self.logger.record("rage/eval/grid_success_rate", grid_aggregate["success_rate"])
            self.logger.record("rage/eval/grid_valid_bounce_rate", grid_aggregate["valid_bounce_rate"])
        if grid3d_aggregate is not None:
            self.logger.record("rage/eval/grid3d_success_rate", grid3d_aggregate["success_rate"])
            self.logger.record("rage/eval/grid3d_valid_bounce_rate", grid3d_aggregate["valid_bounce_rate"])

        rand_str = (
            f" rand_stage=R{evaluated_rand} "
            f"range_success={range_metrics['success_rate']:.2f} "
            f"grid_success={grid_aggregate['success_rate']:.2f} "
            f"rand_promoted_to={('R' + str(promoted_rand)) if promoted_rand is not None else '-'}"
            if range_metrics is not None and grid_aggregate is not None
            else ""
        )
        zrand_str = (
            f" zrand_stage=Z{evaluated_zrand} "
            f"grid3d_success={grid3d_aggregate['success_rate']:.2f} "
            f"zrand_promoted_to={('Z' + str(promoted_zrand)) if promoted_zrand is not None else '-'}"
            if evaluated_zrand is not None and grid3d_aggregate is not None
            else ""
        )
        print(
            "curriculum_eval "
            f"timestep={self.num_timesteps} evaluated_stage={evaluated_stage} "
            f"valid_bounce_rate={fixed_metrics['valid_bounce_rate']:.2f} "
            f"bounce_target_rate={fixed_metrics['bounce_target_rate']:.2f} "
            f"median_closest_cup_dist={fixed_metrics['median_closest_cup_dist']:.3f} "
            f"fixed_success_rate={fixed_metrics['success_rate']:.2f}"
            f"{rand_str}{zrand_str} "
            f"reward_promoted_to={promoted_reward or '-'} "
            f"new_best={'yes' if saved_best else 'no'}"
        )
        while self.next_eval_timestep <= self.num_timesteps:
            self.next_eval_timestep += self.eval_every_timesteps

    def _maybe_promote(self, metrics: dict[str, float]) -> int | None:
        current = self.stage_ref.current
        next_stage = current
        if current == 1 and metrics["valid_bounce_rate"] >= 0.75:
            next_stage = 2
        elif (
            current == 2
            and metrics["valid_bounce_rate"] >= 0.70
            and metrics["bounce_target_rate"] >= 0.60
        ):
            next_stage = 3
        # Stage 3 → 4 auto-promotion is disabled. In v21 the stage-4 reward
        # weights (cup_dist halved 10→5, cup_entry tripled 30→100) destabilized
        # the policy that had been hitting success_rate=0.75 in stage 3 — the
        # dense gradient (cup_dist) the policy was riding got cut while the
        # rare-event signals got amplified, and the policy never recovered the
        # working trajectory. Until stage 4's weight schedule is redesigned,
        # stage 3 is the last stage and training just runs longer there.
        # REWARD_WEIGHTS[4] is still defined and reachable via --reward-stage 4
        # for explicit experimentation.

        if next_stage == current:
            return None
        self._render_stage_end_rollout(current)
        self.stage_ref.current = next_stage
        self._set_training_env_stage(next_stage)
        return next_stage

    def _maybe_save_best(self, metrics: dict[str, Any]) -> bool:
        if self.best_model_save_stem is None:
            return False
        success = metrics["success_rate"]
        if success <= 0.0:
            return False
        is_better = success > self._best_success_rate or (
            success == self._best_success_rate
            and metrics["mean_reward"] > self._best_mean_reward
        )
        if not is_better:
            return False
        self._best_success_rate = success
        self._best_mean_reward = metrics["mean_reward"]
        self._best_timestep = self.num_timesteps
        zip_path = self.best_model_save_stem.with_suffix(".zip")
        vec_path = self.best_model_save_stem.with_suffix(".vecnormalize.pkl")
        zip_path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(str(self.best_model_save_stem))
        env = self.model.get_vec_normalize_env()
        if env is not None:
            env.save(str(vec_path))
        print(
            f"best_model_saved timestep={self.num_timesteps} "
            f"success_rate={success:.3f} mean_reward={metrics['mean_reward']:.2f} "
            f"closest_cup_dist={metrics['median_closest_cup_dist']:.3f} "
            f"path={zip_path}"
        )
        return True

    def _maybe_save_best_per_rand_stage(
        self, grid_metrics: dict[str, Any], rand_stage: int
    ) -> bool:
        if self.best_model_save_stem is None:
            return False
        success = grid_metrics["success_rate"]
        if success <= 0.0:
            return False
        prior_success = self._best_per_rand_stage.get(rand_stage, -1.0)
        prior_reward = self._best_per_rand_stage_reward.get(rand_stage, float("-inf"))
        is_better = success > prior_success or (
            success == prior_success and grid_metrics["mean_reward"] > prior_reward
        )
        if not is_better:
            return False
        self._best_per_rand_stage[rand_stage] = success
        self._best_per_rand_stage_reward[rand_stage] = grid_metrics["mean_reward"]
        stem = self.best_model_save_stem.with_name(
            f"{self.best_model_save_stem.name}_R{rand_stage}"
        )
        zip_path = stem.with_suffix(".zip")
        vec_path = stem.with_suffix(".vecnormalize.pkl")
        zip_path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(str(stem))
        env = self.model.get_vec_normalize_env()
        if env is not None:
            env.save(str(vec_path))
        print(
            f"best_model_saved_per_rand_stage R{rand_stage} timestep={self.num_timesteps} "
            f"grid_success_rate={success:.3f} mean_reward={grid_metrics['mean_reward']:.2f} "
            f"median_closest_cup_dist={grid_metrics['median_closest_cup_dist']:.3f} "
            f"path={zip_path}"
        )
        return True

    def _maybe_save_best_per_zrand_stage(
        self, grid3d_metrics: dict[str, Any], zrand_stage: int
    ) -> bool:
        if self.best_model_save_stem is None:
            return False
        success = grid3d_metrics["success_rate"]
        if success <= 0.0:
            return False
        prior_success = self._best_per_zrand_stage.get(zrand_stage, -1.0)
        prior_reward = self._best_per_zrand_stage_reward.get(zrand_stage, float("-inf"))
        is_better = success > prior_success or (
            success == prior_success and grid3d_metrics["mean_reward"] > prior_reward
        )
        if not is_better:
            return False
        self._best_per_zrand_stage[zrand_stage] = success
        self._best_per_zrand_stage_reward[zrand_stage] = grid3d_metrics["mean_reward"]
        stem = self.best_model_save_stem.with_name(
            f"{self.best_model_save_stem.name}_Z{zrand_stage}"
        )
        zip_path = stem.with_suffix(".zip")
        vec_path = stem.with_suffix(".vecnormalize.pkl")
        zip_path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(str(stem))
        env = self.model.get_vec_normalize_env()
        if env is not None:
            env.save(str(vec_path))
        print(
            f"best_model_saved_per_zrand_stage Z{zrand_stage} timestep={self.num_timesteps} "
            f"grid3d_success_rate={success:.3f} mean_reward={grid3d_metrics['mean_reward']:.2f} "
            f"median_closest_cup_dist={grid3d_metrics['median_closest_cup_dist']:.3f} "
            f"path={zip_path}"
        )
        return True

    def _render_stage_end_rollout(self, reward_stage: int) -> None:
        if self.stage_rollout_dir is None:
            return
        label = f"reward_stage_{reward_stage}_end_{self.num_timesteps:09d}"
        gif_path, csv_path, total_reward, final_info = render_policy_rollout(
            self.model,
            self.stage_rollout_dir,
            label=label,
            seed=self.seed + self.num_timesteps,
            fixed_cup=self.stage_rollout_fixed_cup,
            reward_stage=reward_stage,
            max_steps=self.stage_rollout_max_steps,
            width=self.stage_rollout_width,
            height=self.stage_rollout_height,
        )
        print(
            f"stage_end_rollout stage={reward_stage} timestep={self.num_timesteps} "
            f"reward={total_reward:.2f} success={final_info.get('success')} "
            f"bounce_count={final_info.get('bounce_count')} "
            f"second_bounce_cup_dist={final_info.get('second_table_bounce_cup_dist')} "
            f"gif={gif_path} csv={csv_path}"
        )

    def _set_training_env_stage(self, reward_stage: int) -> None:
        training_env = self.model.get_env()
        if training_env is not None:
            training_env.env_method("set_reward_stage", reward_stage)

    def _apply_rand_stage_to_training_env(self, rand_stage: int) -> None:
        training_env = self.model.get_env()
        if training_env is not None:
            cfg = RANDOMIZATION_STAGES[rand_stage]
            training_env.env_method("set_cup_range", cfg["x_range"], cfg["y_range"])

    def _apply_zrand_stage_to_training_env(self, zrand_stage: int) -> None:
        training_env = self.model.get_env()
        if training_env is not None:
            cfg = Z_RANDOMIZATION_STAGES[zrand_stage]
            training_env.env_method("set_pedestal_range", cfg["z_range"])

    def _render_rand_stage_end_rollout(self, rand_stage: int) -> None:
        if self.stage_rollout_dir is None:
            return
        cfg = RANDOMIZATION_STAGES[rand_stage]
        cells = [
            (cfg["x_range"][0], cfg["y_range"][0]),
            (cfg["x_range"][0], cfg["y_range"][1]),
            (cfg["x_range"][1], cfg["y_range"][0]),
            (cfg["x_range"][1], cfg["y_range"][1]),
            (float(NOMINAL_CUP_XY[0]), float(NOMINAL_CUP_XY[1])),
        ]
        out_dir = self.stage_rollout_dir / f"rand_R{rand_stage}_end_{self.num_timesteps:09d}"
        for cup_x, cup_y in cells:
            label = f"cup_{cup_x:.3f}_{cup_y:+.3f}".replace("+", "p").replace("-", "m")
            render_policy_rollout_at_cup(
                self.model,
                out_dir=out_dir,
                label=label,
                seed=self.seed + self.num_timesteps,
                cup_xy=(cup_x, cup_y),
                reward_stage=self.stage_ref.current,
                max_steps=self.stage_rollout_max_steps,
                width=self.stage_rollout_width,
                height=self.stage_rollout_height,
            )
        print(
            f"rand_stage_end_rollout rand_stage=R{rand_stage} "
            f"timestep={self.num_timesteps} cells={len(cells)} dir={out_dir}"
        )

    def _render_zrand_stage_end_rollout(self, zrand_stage: int) -> None:
        if self.stage_rollout_dir is None:
            return
        cfg = Z_RANDOMIZATION_STAGES[zrand_stage]
        # Render the 3×3 cup-xy grid at the just-completed stage's max
        # pedestal — that's the hardest configuration the policy passed
        # before promotion. One folder per Z-stage end.
        pedestal = float(cfg["z_range"][1])
        cells = _grid_cells()
        out_dir = self.stage_rollout_dir / f"zrand_Z{zrand_stage}_end_{self.num_timesteps:09d}"
        for cup_x, cup_y in cells:
            label = (
                f"cup_{cup_x:.3f}_{cup_y:+.3f}_z{pedestal:.3f}"
                .replace("+", "p")
                .replace("-", "m")
            )
            render_policy_rollout_at_cup(
                self.model,
                out_dir=out_dir,
                label=label,
                seed=self.seed + self.num_timesteps,
                cup_xy=(cup_x, cup_y),
                reward_stage=self.stage_ref.current,
                max_steps=self.stage_rollout_max_steps,
                width=self.stage_rollout_width,
                height=self.stage_rollout_height,
                pedestal_height=pedestal,
            )
        print(
            f"zrand_stage_end_rollout zrand_stage=Z{zrand_stage} "
            f"pedestal={pedestal:.3f} timestep={self.num_timesteps} "
            f"cells={len(cells)} dir={out_dir}"
        )

    def _append_grid_log(
        self,
        rows: list[dict[str, Any]],
        rand_stage: int,
        zrand_stage: int | None,
    ) -> None:
        if self.grid_log_path is None:
            return
        z_str = "" if zrand_stage is None else str(zrand_stage)
        with self.grid_log_path.open("a") as f:
            for r in rows:
                f.write(
                    f"{self.num_timesteps},{rand_stage},{z_str},"
                    f"{r['cup_x']:.4f},{r['cup_y']:.4f},{r['pedestal_height']:.4f},"
                    f"{r['success']:.0f},{r['closest_cup_dist']:.4f},{r['valid_bounce']:.0f}\n"
                )

    def _append_log(
        self,
        metrics: dict[str, Any],
        evaluated_stage: int,
        evaluated_rand: int | None,
        evaluated_zrand: int | None,
        range_metrics: dict[str, Any] | None,
        grid_aggregate: dict[str, Any] | None,
        grid3d_aggregate: dict[str, Any] | None,
        promoted_to: int | None,
        rand_promoted_to: int | None,
        zrand_promoted_to: int | None,
    ) -> None:
        rand_str = "" if evaluated_rand is None else str(evaluated_rand)
        zrand_str = "" if evaluated_zrand is None else str(evaluated_zrand)
        range_success = "" if range_metrics is None else f"{range_metrics['success_rate']:.6f}"
        grid_success = "" if grid_aggregate is None else f"{grid_aggregate['success_rate']:.6f}"
        grid3d_success = "" if grid3d_aggregate is None else f"{grid3d_aggregate['success_rate']:.6f}"
        rand_promoted = "" if rand_promoted_to is None else str(rand_promoted_to)
        zrand_promoted = "" if zrand_promoted_to is None else str(zrand_promoted_to)
        with self.log_path.open("a") as f:
            f.write(
                f"{self.num_timesteps},{evaluated_stage},{rand_str},{zrand_str},"
                f"{metrics['mean_reward']:.6f},{metrics['success_rate']:.6f},"
                f"{metrics['valid_bounce_rate']:.6f},{metrics['exact_one_bounce_rate']:.6f},"
                f"{metrics['bounce_target_rate']:.6f},"
                f"{metrics['median_closest_cup_dist']:.6f},"
                f"{metrics['median_second_bounce_cup_dist']:.6f},"
                f"{metrics['invalid_contact_rate']:.6f},"
                f"{range_success},{grid_success},{grid3d_success},"
                f"{promoted_to or ''},{rand_promoted},{zrand_promoted}\n"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timesteps", type=int, default=50_000)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--n-steps", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--lr-schedule", choices=("constant", "linear"), default="constant")
    parser.add_argument("--net-arch", choices=tuple(NET_ARCHS), default="default")
    parser.add_argument("--activation", choices=tuple(ACTIVATIONS), default="tanh")
    parser.add_argument("--reward-stage", type=int, choices=(1, 2, 3, 4), default=1)
    parser.add_argument("--curriculum", choices=("manual", "auto"), default="manual")
    parser.add_argument("--curriculum-eval-every", type=int, default=100_000)
    parser.add_argument("--curriculum-eval-episodes", type=int, default=8)
    parser.add_argument("--curriculum-log", type=Path)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("sim/_rl_out/ppo_thrower"),
        help=(
            "Run directory. All artifacts (policy.zip, vecnormalize.pkl, "
            "best_R*.zip, curriculum.csv, grid.csv, checkpoints/, "
            "train_rollouts/, tb/, training.json) are written inside."
        ),
    )
    parser.add_argument("--train-rollout-viz-every", type=int, default=0)
    parser.add_argument("--train-rollout-viz-steps", type=int, default=300)
    parser.add_argument("--train-rollout-viz-fixed-cup", action="store_true")
    parser.add_argument("--train-rollout-viz-width", type=int, default=640)
    parser.add_argument("--train-rollout-viz-height", type=int, default=480)
    # Periodic checkpoint saves during training so we can recover any point.
    # SB3 measures save_freq in env-step counts (n_envs * n_calls), so with
    # n_envs=16 and the default 500_000, a 10M-step run produces ~20 snapshots
    # of ~2MB each. Set to 0 to disable.
    parser.add_argument("--checkpoint-freq", type=int, default=500_000)
    parser.add_argument(
        "--rand-stage",
        type=int,
        choices=tuple(RANDOMIZATION_STAGES),
        default=None,
        help="Initial randomization sub-stage R0..R3. When set, training enables the cup-position randomization curriculum (auto-promotion via curriculum eval).",
    )
    parser.add_argument(
        "--rand-eval-episodes",
        type=int,
        default=16,
        help="Episodes per range-mode eval (drives R-stage auto-promotion).",
    )
    parser.add_argument(
        "--z-stage",
        type=int,
        choices=tuple(Z_RANDOMIZATION_STAGES),
        default=None,
        help="Initial pedestal-height sub-stage Z0..Z3. When set, training enables the pedestal-height randomization curriculum (cup elevated by 0..15cm). Requires --rand-stage to also be set so the range-eval covers the (cup_xy × pedestal) box.",
    )
    parser.add_argument(
        "--grid-log",
        type=Path,
        default=None,
        help="Optional path for the per-cell grid CSV. Defaults to <out>/grid.csv when --rand-stage is set.",
    )
    parser.add_argument(
        "--warm-start-policy",
        type=Path,
        default=None,
        help="Optional path to a saved <policy>.zip. When set, weights are loaded into a freshly-built PPO model before training begins.",
    )
    parser.add_argument(
        "--warm-start-vecnormalize",
        type=Path,
        default=None,
        help="Optional path to a saved <policy>.vecnormalize.pkl. Required when --warm-start-policy is set.",
    )
    parser.add_argument(
        "--surgical-reset-obs-rms-slots",
        type=str,
        default="",
        help=(
            "Comma-separated obs_rms slot indices to reset to (mean=0, var=1) "
            "after warm-start load. Use this when warm-loading a policy whose "
            "vecnormalize was trained with a constant value at some slot whose "
            "semantic has changed in the new env (e.g., '20' for the cup_count "
            "slot now holding pedestal_height in the v35 stacked-cup task)."
        ),
    )
    parser.add_argument(
        "--ent-coef",
        type=float,
        default=0.01,
        help="PPO entropy coefficient. v22 used 0.01; warm-start runs may want a small bump (0.015) for cup-conditioned exploration.",
    )
    parser.add_argument(
        "--vec-env",
        choices=("subproc", "dummy"),
        default="subproc",
        help="DummyVecEnv runs in-process (single CPU; only useful for smoke runs in sandboxed shells where subprocess IPC is blocked).",
    )
    parser.add_argument(
        "--action-delta",
        type=float,
        default=0.06,
        help="Per-step joint-target delta (rad) at action=1. v36/no_ball_obs_v1 used 0.06; the smooth_v1 design uses 0.05.",
    )
    parser.add_argument(
        "--action-filter-alpha",
        type=float,
        default=1.0,
        help="First-order low-pass filter coefficient on the (post-latency) action. 1.0 = no filter (default, back-compat). smooth_v1 uses 0.6.",
    )
    parser.add_argument(
        "--action-latency-range",
        type=str,
        default="0,0",
        help="Per-episode-sampled latency range as 'lo,hi' (inclusive). Each reset() draws an int from [lo, hi] uniformly and resizes the action queue. '0,0' = no latency (default, back-compat). latency_robust_v1 uses '2,4'.",
    )
    parser.add_argument(
        "--obs-joint-pos-noise-std",
        type=float,
        default=0.0,
        help="Std (rad) of gaussian noise added to joint_pos values entering the obs history buffer. Physics state stays clean. 0.0 = no noise (default). latency_robust_v1 uses 0.001.",
    )
    parser.add_argument(
        "--joint-pos-history-len",
        type=int,
        default=1,
        help="Number of joint_pos frames to expose in the obs (current + N-1 previous). 1 = current only (default, matches no_ball_obs minus joint_vel). latency_robust_v1 uses 4.",
    )
    parser.add_argument(
        "--action-history-len",
        type=int,
        default=0,
        help="Number of previous raw policy actions to expose in the obs. 0 = no action history (default). latency_robust_v1 uses 4.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    # Populate run-wide env override dict before any env is constructed.
    # All env-construction helpers splat this in; defaults stay the
    # constructor's (action_delta=0.06, alpha=1.0, latency=0) when args
    # match those.
    RUN_ENV_KWARGS.update(
        action_delta=args.action_delta,
        action_filter_alpha=args.action_filter_alpha,
        action_latency_range=_parse_latency_range(args.action_latency_range),
        obs_joint_pos_noise_std=args.obs_joint_pos_noise_std,
        joint_pos_history_len=args.joint_pos_history_len,
        action_history_len=args.action_history_len,
    )
    stage_ref = RewardStageRef(args.reward_stage)
    rand_stage_ref = RandomizationStageRef(args.rand_stage) if args.rand_stage is not None else None
    zrand_stage_ref = (
        ZRandomizationStageRef(args.z_stage) if args.z_stage is not None else None
    )
    if zrand_stage_ref is not None and rand_stage_ref is None:
        raise ValueError("--z-stage requires --rand-stage to be set; the range eval is over (R-box × Z-range)")
    curriculum_log = args.curriculum_log or args.out / "curriculum.csv"
    grid_log_path = args.grid_log
    if grid_log_path is None and rand_stage_ref is not None:
        grid_log_path = args.out / "grid.csv"
    train_rollout_viz_dir = args.out / "train_rollouts"
    train_rollouts_enabled = args.train_rollout_viz_every > 0

    # SubprocVecEnv runs each env in a separate process — true CPU parallelism.
    # make_vec_env defaults to DummyVecEnv (serial in trainer process), which
    # caps fps at ~1/n_envs of theoretical max because envs share the GIL with
    # PPO updates. SubprocVecEnv on n_envs >= 4 gives 2-3x throughput.
    vec_env_cls = SubprocVecEnv if args.vec_env == "subproc" else DummyVecEnv
    env = make_vec_env(
        RageCageEnv,
        n_envs=args.n_envs,
        seed=args.seed,
        env_kwargs={"reward_stage": stage_ref.current, **RUN_ENV_KWARGS},
        vec_env_cls=vec_env_cls,
    )
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    if args.warm_start_policy is not None:
        if args.warm_start_vecnormalize is None:
            raise ValueError("--warm-start-policy requires --warm-start-vecnormalize")
        # Load the prior run's running normalization stats into the freshly
        # built training env so the warm-loaded policy sees obs in the same
        # representation it was trained on. New training updates these stats
        # online as the cup distribution widens via the R-curriculum.
        loaded_norm = VecNormalize.load(str(args.warm_start_vecnormalize), env.venv)
        env.obs_rms = loaded_norm.obs_rms
        env.ret_rms = loaded_norm.ret_rms
        env.clip_obs = loaded_norm.clip_obs
        env.gamma = loaded_norm.gamma
        env.epsilon = loaded_norm.epsilon
        if args.surgical_reset_obs_rms_slots:
            slots = [int(s) for s in args.surgical_reset_obs_rms_slots.split(",") if s.strip()]
            for slot in slots:
                env.obs_rms.mean[slot] = 0.0
                env.obs_rms.var[slot] = 1.0
            print(f"surgical_reset_obs_rms slots={slots}")

    if args.warm_start_policy is not None:
        # custom_objects overrides the saved obs/action spaces with the
        # current env's spaces. The v34 obs_space had different bounds
        # for the cup_count slot (now repurposed as pedestal_height in
        # v35); SB3's strict space-equality check would otherwise reject
        # the load even though the dim and dtype match.
        model = PPO.load(
            str(args.warm_start_policy),
            env=env,
            learning_rate=learning_rate_arg(args.lr, args.lr_schedule),
            ent_coef=args.ent_coef,
            tensorboard_log=str(args.out / "tb"),
            custom_objects={
                "observation_space": env.observation_space,
                "action_space": env.action_space,
            },
        )
        print(
            f"warm_start_loaded policy={args.warm_start_policy} "
            f"vecnormalize={args.warm_start_vecnormalize}"
        )
    else:
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=learning_rate_arg(args.lr, args.lr_schedule),
            policy_kwargs=policy_kwargs(args.net_arch, args.activation),
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            gamma=0.99,
            gae_lambda=0.95,
            ent_coef=args.ent_coef,
            verbose=1,
            tensorboard_log=str(args.out / "tb"),
            seed=args.seed,
        )
    callbacks: list[BaseCallback] = [
        CurriculumCallback(
            stage_ref=stage_ref,
            mode=args.curriculum,
            eval_every_timesteps=args.curriculum_eval_every,
            eval_episodes=args.curriculum_eval_episodes,
            seed=args.seed,
            log_path=curriculum_log,
            stage_rollout_dir=train_rollout_viz_dir if train_rollouts_enabled else None,
            stage_rollout_fixed_cup=args.train_rollout_viz_fixed_cup,
            stage_rollout_max_steps=args.train_rollout_viz_steps,
            stage_rollout_width=args.train_rollout_viz_width,
            stage_rollout_height=args.train_rollout_viz_height,
            best_model_save_stem=args.out / "best",
            rand_stage_ref=rand_stage_ref,
            range_eval_episodes=args.rand_eval_episodes,
            grid_log_path=grid_log_path,
            zrand_stage_ref=zrand_stage_ref,
        )
    ]
    if args.checkpoint_freq > 0:
        # Periodic snapshots independent of eval. SB3's CheckpointCallback
        # saves both the model and the VecNormalize stats together so the
        # snapshot is self-contained for inference.
        callbacks.append(
            CheckpointCallback(
                save_freq=max(args.checkpoint_freq // args.n_envs, 1),
                save_path=str(args.out / "checkpoints"),
                name_prefix="checkpoint",
                save_vecnormalize=True,
            )
        )
    if train_rollouts_enabled:
        callbacks.append(
            TrainingRolloutVizCallback(
                out_dir=train_rollout_viz_dir,
                every_timesteps=args.train_rollout_viz_every,
                seed=args.seed,
                fixed_cup=args.train_rollout_viz_fixed_cup,
                reward_stage_ref=stage_ref,
                max_steps=args.train_rollout_viz_steps,
                width=args.train_rollout_viz_width,
                height=args.train_rollout_viz_height,
            )
        )
    callback = CallbackList(callbacks)

    model.learn(
        total_timesteps=args.timesteps,
        log_interval=args.log_interval,
        callback=callback,
    )

    if train_rollouts_enabled:
        label = f"train_rollout_final_{model.num_timesteps:09d}"
        gif_path, csv_path, total_reward, final_info = render_policy_rollout(
            model,
            train_rollout_viz_dir,
            label=label,
            seed=args.seed,
            fixed_cup=args.train_rollout_viz_fixed_cup,
            reward_stage=stage_ref.current,
            max_steps=args.train_rollout_viz_steps,
            width=args.train_rollout_viz_width,
            height=args.train_rollout_viz_height,
        )
        print(
            f"training_rollout_final timestep={model.num_timesteps} reward={total_reward:.2f} "
            f"success={final_info.get('success')} bounce_count={final_info.get('bounce_count')} "
            f"cup_dist={final_info.get('cup_dist'):.3f} gif={gif_path} csv={csv_path}"
        )

    policy_path = args.out / "policy.zip"
    vecnorm_path = args.out / "vecnormalize.pkl"
    metadata_path = args.out / "training.json"
    model.save(str(policy_path))
    env.save(str(vecnorm_path))
    metadata_path.write_text(
        json.dumps(
            {
                "curriculum": args.curriculum,
                "initial_reward_stage": args.reward_stage,
                "final_reward_stage": stage_ref.current,
                "curriculum_log": str(curriculum_log),
                "timesteps": model.num_timesteps,
                "action_delta": args.action_delta,
                "action_filter_alpha": args.action_filter_alpha,
                "action_latency_range": list(_parse_latency_range(args.action_latency_range)),
                "obs_joint_pos_noise_std": args.obs_joint_pos_noise_std,
                "joint_pos_history_len": args.joint_pos_history_len,
                "action_history_len": args.action_history_len,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    env.close()
    print(f"wrote {policy_path}")
    print(f"wrote {metadata_path}")


if __name__ == "__main__":
    main()
