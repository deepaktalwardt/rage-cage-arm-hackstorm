"""State-machine tests for `ThrowController`.

These tests load the real (small) policy via the controller's __init__
to keep mocking out of scope — policy outputs aren't checked here, only
state transitions, tick counts, gripper trigger, and the homing-trajectory
shape. The model is loaded once per module via a fixture.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from real.controller import GRIPPER_HOLD_M, GRIPPER_OPEN_M, ThrowController, ThrowState

MODEL_DIR = (
    Path(__file__).resolve().parents[2]
    / "models"
    / "random_stack_cup_thrower_no_ball_obs_v1"
)

HOME_QPOS = np.array([0.0, 1.57, -1.3485, 0.0, 0.0, 0.0], dtype=np.float32)


@pytest.fixture
def controller() -> ThrowController:
    return ThrowController(model_dir=MODEL_DIR)


def test_initial_state_is_idle(controller: ThrowController) -> None:
    assert controller.state == ThrowState.IDLE


def test_start_cycle_transitions_idle_to_homing(controller: ThrowController) -> None:
    controller.start_cycle(current_joint_pos=np.zeros(6, dtype=np.float32))
    assert controller.state == ThrowState.HOMING


def test_homing_first_tick_steps_one_percent_toward_home(controller: ThrowController) -> None:
    """2s homing @ 50Hz = 100 ticks. First tick should move arm_target 1/100 of
    the way from start_q toward HOME_QPOS, on the straight line between them."""
    start_q = np.array([1.0, 0.5, -1.0, 0.5, 0.5, 0.5], dtype=np.float32)
    controller.start_cycle(current_joint_pos=start_q)

    result = controller.tick(
        joint_pos=start_q,
        joint_vel=np.zeros(6, dtype=np.float32),
        cup_top_xyz=np.array([0.85, 0.0, 0.12], dtype=np.float32),
    )

    assert result.state == ThrowState.HOMING
    expected = start_q + (HOME_QPOS - start_q) / 100.0
    assert result.arm_target is not None
    assert np.allclose(result.arm_target, expected, atol=1e-5)
    assert result.gripper_position == GRIPPER_HOLD_M
    assert result.done is False


def _drive(controller: ThrowController, n: int, start_q: NDArray[np.float32]) -> object:
    """Helper: tick N times with constant inputs; return the last TickResult."""
    result = None
    for _ in range(n):
        result = controller.tick(
            joint_pos=start_q,
            joint_vel=np.zeros(6, dtype=np.float32),
            cup_top_xyz=np.array([0.85, 0.0, 0.12], dtype=np.float32),
        )
    return result


def test_settle_home_begins_after_100_homing_ticks(controller: ThrowController) -> None:
    """After the 100 homing interp ticks, the state machine transitions to
    SETTLE_HOME and holds arm_target at HOME_QPOS."""
    start_q = np.array([1.0, 0.5, -1.0, 0.5, 0.5, 0.5], dtype=np.float32)
    controller.start_cycle(current_joint_pos=start_q)

    result = _drive(controller, n=101, start_q=start_q)

    assert result.state == ThrowState.SETTLE_HOME
    assert np.allclose(result.arm_target, HOME_QPOS, atol=1e-5)


def test_throwing_begins_after_settle_home(controller: ThrowController) -> None:
    """SETTLE_HOME runs for 15 ticks (300ms @ 50Hz). On tick 116 (100 HOMING +
    15 SETTLE_HOME + 1) the state machine enters THROWING."""
    start_q = HOME_QPOS.copy()
    controller.start_cycle(current_joint_pos=start_q)

    result = _drive(controller, n=116, start_q=start_q)

    assert result.state == ThrowState.THROWING


def test_release_fires_at_throwing_tick_45(controller: ThrowController) -> None:
    """The 45th THROWING tick is the release: gripper_cmd='open' AND state
    transitions to SETTLE_RELEASE on the same tick. Total ticks to get there:
    100 HOMING + 15 SETTLE_HOME + 45 THROWING = 160."""
    start_q = HOME_QPOS.copy()
    controller.start_cycle(current_joint_pos=start_q)

    result = _drive(controller, n=160, start_q=start_q)

    assert result.gripper_position == GRIPPER_OPEN_M
    assert result.state == ThrowState.SETTLE_RELEASE


def test_full_cycle_completes_in_210_ticks(controller: ThrowController) -> None:
    """Full cycle: 100 HOMING + 15 SETTLE_HOME + 45 THROWING + 50 SETTLE_RELEASE
    = 210 ticks. On tick 210 the state machine returns to IDLE with done=True."""
    start_q = HOME_QPOS.copy()
    controller.start_cycle(current_joint_pos=start_q)

    result = _drive(controller, n=210, start_q=start_q)

    assert result.done is True
    assert result.state == ThrowState.IDLE


def test_home_only_mode_skips_throwing(controller: ThrowController) -> None:
    """start_home_only() runs HOMING + SETTLE_HOME and returns to IDLE — never
    enters THROWING. 100 HOMING + 15 SETTLE_HOME + 1 transition = 116 ticks."""
    start_q = np.array([1.0, 0.5, -1.0, 0.5, 0.5, 0.5], dtype=np.float32)
    controller.start_home_only(current_joint_pos=start_q)

    result = _drive(controller, n=116, start_q=start_q)

    assert result.state == ThrowState.IDLE
    assert result.done is True


def test_replay_mode_plays_each_row(controller: ThrowController) -> None:
    """start_replay() with a 3-row trajectory runs HOMING (100) + SETTLE_HOME
    (15) + REPLAY (3) + SETTLE_RELEASE (50) → IDLE. On REPLAY tick i, the
    arm_target should equal trajectory[i-1] exactly (open loop).
    """
    start_q = HOME_QPOS.copy()
    traj = np.array(
        [
            [0.10, 1.60, -1.30, 0.05, 0.05, 0.05],
            [0.20, 1.65, -1.25, 0.10, 0.10, 0.10],
            [0.30, 1.70, -1.20, 0.15, 0.15, 0.15],
        ],
        dtype=np.float32,
    )
    controller.start_replay(current_joint_pos=start_q, trajectory=traj)

    # First REPLAY tick: 100 HOMING + 15 SETTLE_HOME + 1 = 116 ticks.
    r = _drive(controller, n=116, start_q=start_q)
    assert r.state == ThrowState.REPLAY
    assert np.allclose(r.arm_target, traj[0], atol=1e-6)
    assert r.gripper_position == GRIPPER_HOLD_M

    # Second REPLAY tick.
    r = _drive(controller, n=1, start_q=start_q)
    assert r.state == ThrowState.REPLAY
    assert np.allclose(r.arm_target, traj[1], atol=1e-6)

    # Third (last) REPLAY tick fires release and transitions to SETTLE_RELEASE.
    r = _drive(controller, n=1, start_q=start_q)
    assert r.state == ThrowState.SETTLE_RELEASE
    assert np.allclose(r.arm_target, traj[2], atol=1e-6)
    assert r.gripper_position == GRIPPER_OPEN_M


def test_replay_hold_ticks_stretches_per_row(controller: ThrowController) -> None:
    """hold_ticks=3 means each trajectory row is held for 3 internal ticks
    before advancing. A 2-row trajectory therefore takes 6 REPLAY ticks
    total (= 100 HOMING + 15 SETTLE_HOME + 6 REPLAY = 121 to land on the
    last REPLAY tick which fires release).
    """
    start_q = HOME_QPOS.copy()
    traj = np.array(
        [
            [0.10, 1.60, -1.30, 0.05, 0.05, 0.05],
            [0.20, 1.65, -1.25, 0.10, 0.10, 0.10],
        ],
        dtype=np.float32,
    )
    controller.start_replay(
        current_joint_pos=start_q, trajectory=traj, hold_ticks=3,
    )

    # Ticks 1, 2, 3 of REPLAY → row 0.
    r = _drive(controller, n=116, start_q=start_q)
    assert r.state == ThrowState.REPLAY
    assert np.allclose(r.arm_target, traj[0], atol=1e-6)
    r = _drive(controller, n=2, start_q=start_q)  # ticks 2, 3 of REPLAY
    assert r.state == ThrowState.REPLAY
    assert np.allclose(r.arm_target, traj[0], atol=1e-6)

    # Tick 4 of REPLAY → row 1.
    r = _drive(controller, n=1, start_q=start_q)
    assert r.state == ThrowState.REPLAY
    assert np.allclose(r.arm_target, traj[1], atol=1e-6)

    # Tick 5 of REPLAY → still row 1.
    r = _drive(controller, n=1, start_q=start_q)
    assert r.state == ThrowState.REPLAY
    assert np.allclose(r.arm_target, traj[1], atol=1e-6)

    # Tick 6 of REPLAY → last tick of last row, fires release.
    r = _drive(controller, n=1, start_q=start_q)
    assert r.state == ThrowState.SETTLE_RELEASE
    assert np.allclose(r.arm_target, traj[1], atol=1e-6)
    assert r.gripper_position == GRIPPER_OPEN_M


def test_replay_release_row_fires_early(controller: ThrowController) -> None:
    """release_row=1 (in a 4-row trajectory) means gripper opens at the
    start of row 1 and stays open. Arm continues following rows 2 and 3.
    Transition to SETTLE_RELEASE still happens at end-of-trajectory.
    """
    start_q = HOME_QPOS.copy()
    traj = np.array(
        [
            [0.10, 1.60, -1.30, 0.05, 0.05, 0.05],
            [0.20, 1.65, -1.25, 0.10, 0.10, 0.10],
            [0.30, 1.70, -1.20, 0.15, 0.15, 0.15],
            [0.40, 1.75, -1.15, 0.20, 0.20, 0.20],
        ],
        dtype=np.float32,
    )
    controller.start_replay(
        current_joint_pos=start_q, trajectory=traj, release_row=1,
    )

    # REPLAY tick 1 → row 0 → gripper still HOLD.
    r = _drive(controller, n=116, start_q=start_q)
    assert r.state == ThrowState.REPLAY
    assert np.allclose(r.arm_target, traj[0], atol=1e-6)
    assert r.gripper_position == GRIPPER_HOLD_M

    # Tick 2 → row 1 = release_row → gripper OPEN, still REPLAY.
    r = _drive(controller, n=1, start_q=start_q)
    assert r.state == ThrowState.REPLAY
    assert np.allclose(r.arm_target, traj[1], atol=1e-6)
    assert r.gripper_position == GRIPPER_OPEN_M

    # Tick 3 → row 2, still past release → still OPEN, still REPLAY.
    r = _drive(controller, n=1, start_q=start_q)
    assert r.state == ThrowState.REPLAY
    assert np.allclose(r.arm_target, traj[2], atol=1e-6)
    assert r.gripper_position == GRIPPER_OPEN_M

    # Tick 4 → last row, end of trajectory → transitions to SETTLE_RELEASE.
    r = _drive(controller, n=1, start_q=start_q)
    assert r.state == ThrowState.SETTLE_RELEASE
    assert np.allclose(r.arm_target, traj[3], atol=1e-6)
    assert r.gripper_position == GRIPPER_OPEN_M


def test_replay_freeze_joint1_holds_throughout(controller: ThrowController) -> None:
    """freeze_joints=[1] (1-indexed) keeps joint1 at its start_q value
    across HOMING (no rotation toward trajectory[0][0]), SETTLE_HOME,
    and REPLAY (recorded trajectory[*][0] ignored). Other joints follow
    the trajectory normally.
    """
    # Pre-replay pose has joint1 at 0.5 rad; trajectory wants joint1 at 0.1.
    # Without freeze, HOMING would interp joint1 from 0.5 → 0.1.
    start_q = np.array([0.5, 1.57, -1.3485, 0.0, 0.0, 0.0], dtype=np.float32)
    traj = np.array(
        [
            [0.10, 1.60, -1.30, 0.05, 0.05, 0.05],
            [0.20, 1.65, -1.25, 0.10, 0.10, 0.10],
        ],
        dtype=np.float32,
    )
    controller.start_replay(
        current_joint_pos=start_q, trajectory=traj, freeze_joints=[1],
    )

    # Mid-HOMING: joint1 should still be at start_q[0] = 0.5 (not interpolating).
    r = _drive(controller, n=50, start_q=start_q)
    assert r.state == ThrowState.HOMING
    assert r.arm_target[0] == pytest.approx(0.5, abs=1e-6)
    # Joints 2..6 should be interpolating toward trajectory[0].
    assert r.arm_target[1] != pytest.approx(start_q[1], abs=1e-4)

    # End of HOMING / SETTLE_HOME: joint1 still pinned.
    r = _drive(controller, n=66, start_q=start_q)  # ticks 51..116
    assert r.state == ThrowState.REPLAY
    assert r.arm_target[0] == pytest.approx(0.5, abs=1e-6)
    # Joint2 should be at traj[0][1] = 1.60 (REPLAY tick 1 → row 0).
    assert r.arm_target[1] == pytest.approx(1.60, abs=1e-6)

    # REPLAY tick 2 → row 1: joint1 STILL frozen; joint2..6 follow row 1.
    r = _drive(controller, n=1, start_q=start_q)
    assert r.state == ThrowState.SETTLE_RELEASE  # last row of 2-row traj
    assert r.arm_target[0] == pytest.approx(0.5, abs=1e-6)
    assert r.arm_target[1] == pytest.approx(1.65, abs=1e-6)


def test_replay_freeze_joints_rejects_bad_index(controller: ThrowController) -> None:
    start_q = HOME_QPOS.copy()
    traj = np.tile(HOME_QPOS, (3, 1)).astype(np.float32)
    with pytest.raises(ValueError, match="freeze_joints"):
        controller.start_replay(
            current_joint_pos=start_q, trajectory=traj, freeze_joints=[0],  # 0-indexed not allowed
        )
    with pytest.raises(ValueError, match="freeze_joints"):
        controller.start_replay(
            current_joint_pos=start_q, trajectory=traj, freeze_joints=[7],
        )


def test_replay_joint1_override_pins_to_value(controller: ThrowController) -> None:
    """joint1_override (in radians) overrides both the recorded trajectory's
    joint1 AND any freeze_joints=[1] pin. Other joints follow the
    trajectory normally.
    """
    start_q = np.array([0.5, 1.57, -1.3485, 0.0, 0.0, 0.0], dtype=np.float32)
    traj = np.array(
        [
            [0.10, 1.60, -1.30, 0.05, 0.05, 0.05],
            [0.20, 1.65, -1.25, 0.10, 0.10, 0.10],
        ],
        dtype=np.float32,
    )
    target_j1 = 1.234  # rad — neither start_q[0] nor any traj value
    controller.start_replay(
        current_joint_pos=start_q,
        trajectory=traj,
        joint1_override=target_j1,
    )

    # Mid-HOMING: joint1 already pinned at the override value (home_target
    # is overridden in start_replay).
    r = _drive(controller, n=50, start_q=start_q)
    assert r.state == ThrowState.HOMING
    # alpha=0.5 between start_q[0]=0.5 and home_target[0]=1.234 → 0.867
    expected_homing = 0.5 + 0.5 * (target_j1 - 0.5)
    assert r.arm_target[0] == pytest.approx(expected_homing, abs=1e-4)

    # End of HOMING / start of REPLAY: joint1 lands exactly at override,
    # ignoring trajectory[0][0] = 0.10.
    r = _drive(controller, n=66, start_q=start_q)  # ticks 51..116
    assert r.state == ThrowState.REPLAY
    assert r.arm_target[0] == pytest.approx(target_j1, abs=1e-5)
    # Joint2 follows trajectory[0][1] = 1.60.
    assert r.arm_target[1] == pytest.approx(1.60, abs=1e-6)


def test_replay_joint1_override_wins_over_freeze(controller: ThrowController) -> None:
    """When freeze_joints=[1] AND joint1_override are both set, the override
    wins (overrides the start_q pin)."""
    start_q = np.array([0.5, 1.57, -1.3485, 0.0, 0.0, 0.0], dtype=np.float32)
    traj = np.tile(HOME_QPOS, (2, 1)).astype(np.float32)
    controller.start_replay(
        current_joint_pos=start_q,
        trajectory=traj,
        freeze_joints=[1],
        joint1_override=2.0,
    )

    # First REPLAY tick.
    r = _drive(controller, n=116, start_q=start_q)
    assert r.state == ThrowState.REPLAY
    assert r.arm_target[0] == pytest.approx(2.0, abs=1e-5)


def test_replay_full_cycle_completes(controller: ThrowController) -> None:
    """Full replay cycle: 100 HOMING + 15 SETTLE_HOME + N REPLAY + 50
    SETTLE_RELEASE. For N=45 (matches our recorded trajectories) that's
    210 ticks — same as a policy throw. Returns to IDLE with done=True.
    """
    start_q = HOME_QPOS.copy()
    traj = np.tile(HOME_QPOS, (45, 1)).astype(np.float32)
    controller.start_replay(current_joint_pos=start_q, trajectory=traj)

    result = _drive(controller, n=210, start_q=start_q)

    assert result.done is True
    assert result.state == ThrowState.IDLE


def test_replay_rejects_bad_shape(controller: ThrowController) -> None:
    start_q = HOME_QPOS.copy()
    # Wrong column count.
    with pytest.raises(ValueError, match="must be"):
        controller.start_replay(
            current_joint_pos=start_q,
            trajectory=np.zeros((3, 5), dtype=np.float32),
        )
    # Empty.
    with pytest.raises(ValueError, match="empty"):
        controller.start_replay(
            current_joint_pos=start_q,
            trajectory=np.zeros((0, 6), dtype=np.float32),
        )


# ---------- Dance mode ----------


def test_dance_transitions_through_homing_to_dance(controller: ThrowController) -> None:
    """start_dance() runs HOMING (100) + SETTLE_HOME (15) + DANCE → IDLE.
    On tick 116 (100 + 15 + 1) the state machine enters DANCE.
    """
    start_q = HOME_QPOS.copy()
    controller.start_dance(
        current_joint_pos=start_q,
        target_rad=0.175,  # ~10 deg
        duration_s=1.0,    # 50 ticks
        sweep_period_s=1.0,
        amplitude_rad=0.262,  # ~15 deg
    )

    result = _drive(controller, n=116, start_q=start_q)
    assert result.state == ThrowState.DANCE


def test_dance_last_tick_lands_at_target(controller: ThrowController) -> None:
    """On the final DANCE tick (progress=1), damp=0 so joint1 = target_rad
    exactly. Joints 2-6 stay at HOME_QPOS throughout the dance.
    """
    start_q = HOME_QPOS.copy()
    target_rad = 0.175  # ~10 deg
    controller.start_dance(
        current_joint_pos=start_q,
        target_rad=target_rad,
        duration_s=1.0,    # 50 ticks
        sweep_period_s=1.0,
        amplitude_rad=0.262,
    )

    # 100 HOMING + 15 SETTLE_HOME + 50 DANCE = 165 ticks lands on the last DANCE tick.
    result = _drive(controller, n=165, start_q=start_q)
    assert result.state == ThrowState.DANCE
    assert result.arm_target[0] == pytest.approx(target_rad, abs=1e-5)
    # Joints 2-6 held at HOME_QPOS for the whole dance.
    assert np.allclose(result.arm_target[1:], HOME_QPOS[1:], atol=1e-6)
    assert result.gripper_position == GRIPPER_HOLD_M


def test_dance_returns_to_idle_after_total_ticks(controller: ThrowController) -> None:
    """Tick immediately after the final DANCE tick transitions to IDLE
    with done=True. Total: 100 HOMING + 15 SETTLE_HOME + 50 DANCE + 1 = 166.
    """
    start_q = HOME_QPOS.copy()
    controller.start_dance(
        current_joint_pos=start_q,
        target_rad=0.0,
        duration_s=1.0,
        sweep_period_s=1.0,
        amplitude_rad=0.262,
    )

    result = _drive(controller, n=166, start_q=start_q)
    assert result.state == ThrowState.IDLE
    assert result.done is True


def test_dance_full_amplitude_during_first_60_percent(controller: ThrowController) -> None:
    """During the first 60% of the dance (progress < 0.6), damping is 1
    so joint1 = amplitude * sin(phase). The 60% boundary on a 50-tick
    dance is tick 30; at tick 25 (progress=0.5) we're still in full
    amplitude. sin(2π * 25 / 50) = sin(π) = 0, but at tick 12.5 (use 13)
    we should see a non-zero swing.
    """
    start_q = HOME_QPOS.copy()
    amplitude = 0.262
    controller.start_dance(
        current_joint_pos=start_q,
        target_rad=0.5,
        duration_s=1.0,
        sweep_period_s=1.0,
        amplitude_rad=amplitude,
    )

    # 100 HOMING + 15 SETTLE_HOME + 13 DANCE = 128 ticks lands on DANCE tick 13.
    # progress=0.26 (< 0.6 → damp=1). phase = 2π * 13/50 = 1.634. sin ≈ 0.998.
    result = _drive(controller, n=128, start_q=start_q)
    assert result.state == ThrowState.DANCE
    expected_j1 = amplitude * float(np.sin(2.0 * np.pi * 13.0 / 50.0))
    assert result.arm_target[0] == pytest.approx(expected_j1, abs=1e-5)
    # Joints 2-6 held at HOME the whole time.
    assert np.allclose(result.arm_target[1:], HOME_QPOS[1:], atol=1e-6)


def test_dance_homes_joints_2_to_6_from_arbitrary_start(controller: ThrowController) -> None:
    """When start_q is not at HOME_QPOS, the HOMING phase brings joints 2-6
    to HOME before DANCE begins. Verify on the first DANCE tick.
    """
    start_q = np.array([0.5, 0.5, -0.5, 0.5, 0.5, 0.5], dtype=np.float32)
    controller.start_dance(
        current_joint_pos=start_q,
        target_rad=0.0,
        duration_s=1.0,
        sweep_period_s=1.0,
        amplitude_rad=0.262,
    )

    # First DANCE tick: 100 HOMING + 15 SETTLE_HOME + 1 = 116 ticks.
    result = _drive(controller, n=116, start_q=start_q)
    assert result.state == ThrowState.DANCE
    # Joints 2-6 should be at HOME_QPOS, not anywhere near start_q values.
    assert np.allclose(result.arm_target[1:], HOME_QPOS[1:], atol=1e-6)


def test_start_dance_rejects_bad_duration(controller: ThrowController) -> None:
    start_q = HOME_QPOS.copy()
    with pytest.raises(ValueError, match="duration"):
        controller.start_dance(
            current_joint_pos=start_q,
            target_rad=0.0,
            duration_s=0.0,
            sweep_period_s=1.0,
            amplitude_rad=0.262,
        )


def test_start_dance_rejects_bad_period(controller: ThrowController) -> None:
    start_q = HOME_QPOS.copy()
    with pytest.raises(ValueError, match="period"):
        controller.start_dance(
            current_joint_pos=start_q,
            target_rad=0.0,
            duration_s=1.0,
            sweep_period_s=0.0,
            amplitude_rad=0.262,
        )
