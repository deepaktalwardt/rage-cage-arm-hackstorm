"""Behavior tests for the action filter and latency knobs added in the
no_ball_obs_smooth_v1 design (see docs/plans/2026-05-23-no-ball-obs-smooth-design.md)
and extended for latency_robust_v1 (see docs/plans/2026-05-24-latency-robust-history-conditioned-design.md).

Run via:  uv run python -m sim.test_action_filter_latency

Asserts:
  1. With action_filter_alpha=1.0 + action_latency_range=(0,0), env
     behavior is unchanged: filtered_action equals the input action.
  2. With action_filter_alpha=0.6 + action_latency_range=(0,0),
     filtered_action follows the first-order low-pass:
     f_t = 0.6*a_t + 0.4*f_{t-1}, with f_0 = 0.
  3. With action_latency_range=(1,1) + action_filter_alpha=1.0, the
     applied action at step t is the action emitted at step t-1.
  4. With action_latency_range=(2,2) + action_filter_alpha=1.0, the
     applied action at step t is the action emitted at step t-2.
  5. reset() zeros filtered_action and refills the action_queue with
     N zero vectors so behavior is identical across episodes.
  6. action_latency_range=(2,4) samples a fresh int per episode and the
     queue length stays constant within each episode.
"""
from __future__ import annotations

import sys

import numpy as np

from sim.env import RageCageEnv


TOL = 1e-6


def _make_env(alpha: float, latency: int) -> RageCageEnv:
    return RageCageEnv(
        action_filter_alpha=alpha,
        action_latency_range=(latency, latency),
        randomize_cup=False,
        reward_stage=1,
    )


def _check(label: str, ok: bool) -> None:
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {label}")
    if not ok:
        global FAILED
        FAILED = True


FAILED = False


def test_default_behavior_unchanged() -> None:
    print("test_default_behavior_unchanged (alpha=1.0, latency=0)")
    env = _make_env(alpha=1.0, latency=0)
    env.reset(seed=0)
    action = np.full(6, 0.5, dtype=np.float32)
    env.step(action)
    _check(
        "filtered_action equals input action with alpha=1.0",
        np.allclose(env.filtered_action, action, atol=TOL),
    )
    _check(
        "action_queue is empty with latency=0",
        len(env.action_queue) == 0,
    )


def test_filter_alpha_math() -> None:
    print("test_filter_alpha_math (alpha=0.6, latency=0)")
    env = _make_env(alpha=0.6, latency=0)
    env.reset(seed=0)
    a = np.full(6, 1.0, dtype=np.float32)
    env.step(a)
    expected_step1 = 0.6 * a + 0.4 * np.zeros(6, dtype=np.float32)
    _check(
        "filtered_action after step 1 = 0.6*a + 0.4*0",
        np.allclose(env.filtered_action, expected_step1, atol=TOL),
    )
    env.step(a)
    expected_step2 = 0.6 * a + 0.4 * expected_step1
    _check(
        "filtered_action after step 2 = 0.6*a + 0.4*(prev filtered)",
        np.allclose(env.filtered_action, expected_step2, atol=TOL),
    )


def test_latency_one_step() -> None:
    print("test_latency_one_step (alpha=1.0, latency=1)")
    env = _make_env(alpha=1.0, latency=1)
    env.reset(seed=0)
    a1 = np.full(6, 0.3, dtype=np.float32)
    env.step(a1)
    _check(
        "step 1 applied action equals zeros (queue init)",
        np.allclose(env.filtered_action, np.zeros(6, dtype=np.float32), atol=TOL),
    )
    a2 = np.full(6, -0.5, dtype=np.float32)
    env.step(a2)
    _check(
        "step 2 applied action equals step 1's emitted action",
        np.allclose(env.filtered_action, a1, atol=TOL),
    )


def test_latency_two_steps() -> None:
    print("test_latency_two_steps (alpha=1.0, latency=2)")
    env = _make_env(alpha=1.0, latency=2)
    env.reset(seed=0)
    a1 = np.full(6, 0.4, dtype=np.float32)
    a2 = np.full(6, -0.2, dtype=np.float32)
    a3 = np.full(6, 0.7, dtype=np.float32)
    env.step(a1)
    _check(
        "step 1 applied action equals zeros",
        np.allclose(env.filtered_action, np.zeros(6, dtype=np.float32), atol=TOL),
    )
    env.step(a2)
    _check(
        "step 2 applied action equals zeros",
        np.allclose(env.filtered_action, np.zeros(6, dtype=np.float32), atol=TOL),
    )
    env.step(a3)
    _check(
        "step 3 applied action equals step 1's emitted action",
        np.allclose(env.filtered_action, a1, atol=TOL),
    )


def test_state_resets_between_episodes() -> None:
    print("test_state_resets_between_episodes (alpha=0.6, latency=2)")
    env = _make_env(alpha=0.6, latency=2)
    env.reset(seed=0)
    env.step(np.full(6, 1.0, dtype=np.float32))
    env.step(np.full(6, 1.0, dtype=np.float32))
    env.step(np.full(6, 1.0, dtype=np.float32))
    # Mid-episode: filter and queue should be non-trivial.
    _check(
        "filtered_action is non-zero before reset",
        np.linalg.norm(env.filtered_action) > 0.0,
    )
    env.reset(seed=1)
    _check(
        "filtered_action is zero after reset",
        np.allclose(env.filtered_action, np.zeros(6, dtype=np.float32), atol=TOL),
    )
    _check(
        "action_queue length equals latency_steps after reset",
        len(env.action_queue) == 2,
    )
    _check(
        "action_queue contents are zero after reset",
        all(np.allclose(q, np.zeros(6, dtype=np.float32), atol=TOL) for q in env.action_queue),
    )


def test_action_latency_range_samples_per_episode() -> None:
    print("test_action_latency_range_samples_per_episode (range=(2,4))")
    env = RageCageEnv(action_latency_range=(2, 4), randomize_cup=False, reward_stage=1)
    seen: set[int] = set()
    for ep in range(20):
        env.reset(seed=ep)
        seen.add(env.current_action_latency)
        initial_len = len(env.action_queue)
        env.step(np.zeros(6, dtype=np.float32))
        _check(
            f"queue length unchanged within ep {ep}",
            len(env.action_queue) == initial_len,
        )
    _check(
        "saw at least 2 distinct latency values across 20 resets",
        len(seen) >= 2,
    )
    _check(
        f"all sampled latencies in {{2,3,4}} (saw {sorted(seen)})",
        seen.issubset({2, 3, 4}),
    )


def test_action_latency_range_default_is_no_op() -> None:
    print("test_action_latency_range_default_is_no_op (default kwargs)")
    env = RageCageEnv(randomize_cup=False, reward_stage=1)
    env.reset(seed=0)
    _check(
        "current_action_latency == 0 by default",
        env.current_action_latency == 0,
    )
    _check(
        "action_queue is empty by default",
        env.action_queue == [],
    )


def test_joint_pos_history_default_len_one() -> None:
    print("test_joint_pos_history_default_len_one (default kwargs)")
    env = RageCageEnv(randomize_cup=False, reward_stage=1)
    env.reset(seed=0)
    _check(
        "joint_pos_history_len == 1 by default",
        env.joint_pos_history_len == 1,
    )
    _check(
        "joint_pos_history has one slot",
        len(env.joint_pos_history) == 1,
    )
    _check(
        "history slot equals reset pose",
        np.allclose(env.joint_pos_history[0], env.data.qpos[env.joint_qposadr], atol=1e-9),
    )


def test_joint_pos_history_initial_state_is_repeat_reset_pose() -> None:
    print("test_joint_pos_history_initial_state_is_repeat_reset_pose (len=4)")
    env = RageCageEnv(joint_pos_history_len=4, randomize_cup=False, reward_stage=1)
    env.reset(seed=0)
    reset_pose = env.data.qpos[env.joint_qposadr].copy()
    _check(
        "history has 4 slots after reset",
        len(env.joint_pos_history) == 4,
    )
    for i, slot in enumerate(env.joint_pos_history):
        _check(
            f"slot[{i}] equals reset pose",
            np.allclose(slot, reset_pose, atol=1e-9),
        )


def test_joint_pos_history_shifts_on_step() -> None:
    print("test_joint_pos_history_shifts_on_step (len=4)")
    env = RageCageEnv(joint_pos_history_len=4, randomize_cup=False, reward_stage=1)
    env.reset(seed=0)
    pose_at_step_entry = env.data.qpos[env.joint_qposadr].copy()
    env.step(np.full(6, 0.5, dtype=np.float32))
    # slot[0] = pose seen at step entry (= reset pose, since step records
    # before physics advances). slot[1..3] = reset pose still.
    _check(
        "slot[0] equals pose at step entry",
        np.allclose(env.joint_pos_history[0], pose_at_step_entry, atol=1e-9),
    )
    for i in range(1, 4):
        _check(
            f"slot[{i}] still equals reset pose",
            np.allclose(env.joint_pos_history[i], pose_at_step_entry, atol=1e-9),
        )
    pose_after_step1 = env.data.qpos[env.joint_qposadr].copy()
    env.step(np.full(6, 0.5, dtype=np.float32))
    _check(
        "slot[0] equals pose at step-2 entry (= post-step-1 pose)",
        np.allclose(env.joint_pos_history[0], pose_after_step1, atol=1e-9),
    )


def test_action_history_default_len_zero() -> None:
    print("test_action_history_default_len_zero (default kwargs)")
    env = RageCageEnv(randomize_cup=False, reward_stage=1)
    env.reset(seed=0)
    _check(
        "action_history_len == 0 by default",
        env.action_history_len == 0,
    )
    _check(
        "action_history is empty by default",
        env.action_history == [],
    )


def test_action_history_initial_state_is_zeros() -> None:
    print("test_action_history_initial_state_is_zeros (len=4)")
    env = RageCageEnv(action_history_len=4, randomize_cup=False, reward_stage=1)
    env.reset(seed=0)
    _check(
        "action_history has 4 slots after reset",
        len(env.action_history) == 4,
    )
    for i, slot in enumerate(env.action_history):
        _check(
            f"slot[{i}] is zeros",
            np.allclose(slot, np.zeros(6, dtype=np.float32), atol=1e-9),
        )


def test_action_history_shifts_on_step() -> None:
    print("test_action_history_shifts_on_step (len=4)")
    env = RageCageEnv(action_history_len=4, randomize_cup=False, reward_stage=1)
    env.reset(seed=0)
    a1 = np.full(6, 0.3, dtype=np.float32)
    env.step(a1)
    _check(
        "after step 1, slot[0] == a1",
        np.allclose(env.action_history[0], a1, atol=1e-6),
    )
    _check(
        "after step 1, slot[1] == zeros",
        np.allclose(env.action_history[1], np.zeros(6), atol=1e-9),
    )
    a2 = np.full(6, -0.2, dtype=np.float32)
    env.step(a2)
    _check(
        "after step 2, slot[0] == a2",
        np.allclose(env.action_history[0], a2, atol=1e-6),
    )
    _check(
        "after step 2, slot[1] == a1",
        np.allclose(env.action_history[1], a1, atol=1e-6),
    )
    _check(
        "after step 2, slot[2] == zeros",
        np.allclose(env.action_history[2], np.zeros(6), atol=1e-9),
    )


def test_obs_joint_pos_noise_default_clean() -> None:
    print("test_obs_joint_pos_noise_default_clean (jpos_hist=4, noise=0.0)")
    env = RageCageEnv(joint_pos_history_len=4, randomize_cup=False, reward_stage=1)
    env.reset(seed=0)
    reset_pose = env.data.qpos[env.joint_qposadr].copy()
    for i, slot in enumerate(env.joint_pos_history):
        _check(
            f"reset slot[{i}] is exactly reset pose (no noise applied at reset)",
            np.allclose(slot, reset_pose, atol=1e-9),
        )


def test_obs_joint_pos_noise_applied_to_history_only() -> None:
    print("test_obs_joint_pos_noise_applied_to_history_only (noise=0.05)")
    env = RageCageEnv(
        joint_pos_history_len=2,
        obs_joint_pos_noise_std=0.05,
        randomize_cup=False,
        reward_stage=1,
    )
    env.reset(seed=0)
    reset_pose = env.data.qpos[env.joint_qposadr].copy()
    _check(
        "reset history slot[0] is clean (no noise at reset)",
        np.allclose(env.joint_pos_history[0], reset_pose, atol=1e-9),
    )
    env.step(np.zeros(6, dtype=np.float32))
    pos_now = env.data.qpos[env.joint_qposadr].copy()
    diff = env.joint_pos_history[0] - pos_now
    _check(
        "after step, slot[0] differs from current pos by gaussian noise",
        np.max(np.abs(diff)) > 0.0,
    )


def test_obs_joint_pos_noise_std_matches_param() -> None:
    print("test_obs_joint_pos_noise_std_matches_param (noise=0.01)")
    env = RageCageEnv(
        joint_pos_history_len=1,
        obs_joint_pos_noise_std=0.01,
        randomize_cup=False,
        reward_stage=1,
    )
    diffs = []
    for ep in range(50):
        env.reset(seed=ep)
        env.step(np.zeros(6, dtype=np.float32))
        diff = env.joint_pos_history[0] - env.data.qpos[env.joint_qposadr]
        diffs.append(diff)
    diffs_arr = np.stack(diffs)
    measured_std = float(np.std(diffs_arr))
    _check(
        f"measured std {measured_std:.4f} in (0.005, 0.020)",
        0.005 < measured_std < 0.020,
    )


def test_obs_layout_with_histories() -> None:
    print("test_obs_layout_with_histories (jpos=4, act=4 -> 52 dim)")
    env = RageCageEnv(
        joint_pos_history_len=4,
        action_history_len=4,
        randomize_cup=False,
        reward_stage=1,
    )
    env.reset(seed=0)
    obs = env._get_obs()
    _check(
        f"obs shape == (52,); got {obs.shape}",
        obs.shape == (52,),
    )
    _check(
        f"observation_space.shape == (52,); got {env.observation_space.shape}",
        env.observation_space.shape == (52,),
    )
    _check(
        "obs[0:2] == cup_xy",
        np.allclose(obs[0:2], env.cup_xy, atol=1e-6),
    )
    _check(
        "obs[2] == pedestal_height",
        np.allclose(obs[2:3], [env.pedestal_height], atol=1e-6),
    )
    _check(
        "obs[3] == release_countdown",
        np.allclose(obs[3:4], [env._release_countdown()], atol=1e-6),
    )
    for i in range(4):
        _check(
            f"obs joint_pos slot {i} matches history[{i}]",
            np.allclose(obs[4 + i * 6 : 4 + (i + 1) * 6], env.joint_pos_history[i], atol=1e-6),
        )
    for i in range(4):
        _check(
            f"obs action slot {i} matches history[{i}]",
            np.allclose(obs[28 + i * 6 : 28 + (i + 1) * 6], env.action_history[i], atol=1e-6),
        )


def test_obs_layout_default_is_legacy_minus_vel() -> None:
    print("test_obs_layout_default_is_legacy_minus_vel (defaults -> 10 dim)")
    env = RageCageEnv(randomize_cup=False, reward_stage=1)
    env.reset(seed=0)
    obs = env._get_obs()
    _check(
        f"obs shape == (10,); got {obs.shape}",
        obs.shape == (10,),
    )
    _check(
        f"observation_space.shape == (10,); got {env.observation_space.shape}",
        env.observation_space.shape == (10,),
    )


def main() -> int:
    test_default_behavior_unchanged()
    test_filter_alpha_math()
    test_latency_one_step()
    test_latency_two_steps()
    test_state_resets_between_episodes()
    test_action_latency_range_samples_per_episode()
    test_action_latency_range_default_is_no_op()
    test_joint_pos_history_default_len_one()
    test_joint_pos_history_initial_state_is_repeat_reset_pose()
    test_joint_pos_history_shifts_on_step()
    test_action_history_default_len_zero()
    test_action_history_initial_state_is_zeros()
    test_action_history_shifts_on_step()
    test_obs_joint_pos_noise_default_clean()
    test_obs_joint_pos_noise_applied_to_history_only()
    test_obs_joint_pos_noise_std_matches_param()
    test_obs_layout_with_histories()
    test_obs_layout_default_is_legacy_minus_vel()
    if FAILED:
        print("\nFAIL: one or more checks failed")
        return 1
    print("\nPASS: all action-filter / latency checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
