"""Equivalence test: ThrowController.tick during THROWING must produce the
same arm_target sequence as a reference loop that invokes the policy and
VecNormalize stats directly.

The reference loop here exists only as a test fixture. It mirrors the
inference glue from `sim/play_policy.py` (the existing sim-side rollout
tool) but stays pure-numpy so we can compare bit-identically.

Inputs are held constant across ticks. With deterministic=True, the
policy produces the same action each tick, and arm_target grows linearly
until joint limits clip. That's not realistic motion — but the goal is
to verify obs construction, normalization, prediction, and action
integration, not policy behavior.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from real.controller import (
    ACTION_DELTA,
    GRIPPER_OPEN_M,
    HOME_QPOS,
    JOINT_HIGH,
    JOINT_LOW,
    RELEASE_STEP,
    SIM_CUP_HEIGHT,
    ThrowController,
    ThrowState,
)
from real.stub_env import StubEnv

MODEL_DIR = (
    Path(__file__).resolve().parents[2]
    / "models"
    / "random_stack_cup_thrower_no_ball_obs_v1"
)

# Constant inputs for the duration of the throw.
JOINT_POS = HOME_QPOS.copy()
JOINT_VEL = np.zeros(6, dtype=np.float32)
CUP_TOP_XYZ = np.array([0.85, 0.0, 0.12], dtype=np.float32)


def _build_obs(tick_idx_1based: int) -> NDArray[np.float32]:
    """Reference obs vector — must match what controller builds internally.

    Layout (16-d): joint_pos(6), joint_vel(6), cup_xy(2), pedestal(1), countdown(1).
    Matches `sim/env.py:_get_obs` on the no-ball-obs branch.
    """
    pedestal_h = max(CUP_TOP_XYZ[2] - SIM_CUP_HEIGHT, 0.0)
    countdown = max(RELEASE_STEP - tick_idx_1based, 0) / RELEASE_STEP
    return np.concatenate(
        [
            JOINT_POS,
            JOINT_VEL,
            CUP_TOP_XYZ[:2],
            np.array([pedestal_h], dtype=np.float32),
            np.array([countdown], dtype=np.float32),
        ]
    ).astype(np.float32)


@pytest.fixture(scope="module")
def reference_artifacts():
    """Load the policy and VecNormalize stats once for the whole module."""
    ppo = PPO.load(str(MODEL_DIR / "policy.zip"))
    dummy = DummyVecEnv([lambda: StubEnv()])
    vn = VecNormalize.load(str(MODEL_DIR / "vecnormalize.pkl"), dummy)
    return ppo, vn.obs_rms.mean.copy(), vn.obs_rms.var.copy(), float(vn.clip_obs), float(vn.epsilon)


def _drive_to_throwing(controller: ThrowController) -> None:
    """Advance the controller through HOMING + SETTLE_HOME so the *next*
    `tick()` is the 1st THROWING tick. 100 HOMING + 15 SETTLE_HOME = 115 ticks."""
    controller.start_cycle(current_joint_pos=JOINT_POS)
    for _ in range(115):
        controller.tick(JOINT_POS, JOINT_VEL, CUP_TOP_XYZ)
    assert controller.state == ThrowState.SETTLE_HOME


def test_throwing_arm_targets_match_reference_policy(reference_artifacts) -> None:
    ppo, obs_mean, obs_var, clip_obs, eps = reference_artifacts

    controller = ThrowController(model_dir=MODEL_DIR)
    _drive_to_throwing(controller)

    # Reference's running arm_target — same starting point the controller uses
    # (current joint_pos, reset on SETTLE_HOME → THROWING transition).
    ref_arm_target = JOINT_POS.astype(np.float32).copy()

    for throwing_tick in range(1, RELEASE_STEP + 1):
        # Controller computes its arm_target.
        result = controller.tick(JOINT_POS, JOINT_VEL, CUP_TOP_XYZ)

        # Reference: build obs, normalize, predict, integrate, clip.
        obs = _build_obs(throwing_tick)
        normalized = np.clip(
            (obs - obs_mean) / np.sqrt(obs_var + eps),
            -clip_obs, clip_obs,
        ).astype(np.float32)
        action, _ = ppo.predict(normalized, deterministic=True)
        action = np.clip(action, -1.0, 1.0)
        ref_arm_target = np.clip(
            ref_arm_target + action * ACTION_DELTA,
            JOINT_LOW, JOINT_HIGH,
        ).astype(np.float32)

        assert result.arm_target is not None
        assert np.allclose(result.arm_target, ref_arm_target, atol=1e-5), (
            f"throw tick {throwing_tick}: controller={result.arm_target} "
            f"reference={ref_arm_target}"
        )

    # Last throw tick fired the gripper and transitioned.
    assert result.gripper_position == GRIPPER_OPEN_M
    assert result.state == ThrowState.SETTLE_RELEASE
