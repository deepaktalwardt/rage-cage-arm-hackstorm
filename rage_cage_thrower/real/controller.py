"""ThrowController — Layer 1 of the rage_cage_thrower node.

State machine, model I/O, action integration, homing-trajectory generation.
Contains no rclpy imports so it can be exercised on the Mac.

See `docs/plans/2026-05-22-ros2-inference-node-design.md`.
"""

from __future__ import annotations

# Force CPU-only torch BEFORE stable_baselines3/torch get imported. The
# Jetson container has an old NVIDIA driver (12060) that segfaults inside
# torch's CUDA init when the saved policy was trained on GPU. Setting
# CUDA_VISIBLE_DEVICES="" makes torch see zero CUDA devices and skip the
# whole init path. PPO.load(device="cpu") alone wasn't enough — the lr
# schedule call still hit something that touched CUDA.
import os as _os

_os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

# stable-baselines3 unconditionally imports `common.atari_wrappers` which
# does a top-level `import cv2` AND calls things like
# `cv2.ocl.setUseOpenCL(False)` at module load time. The container base
# image's apt-installed cv2 was compiled against numpy 1.x and crashes
# on our numpy 2.x (needed to unpickle vecnormalize.pkl). We never use
# Atari wrappers, so stub cv2 with an object that swallows arbitrary
# attribute access and calls. Benign on the Mac (no real cv2 anyway).
import sys as _sys


class _CvStub:
    """Mock for cv2: any attribute returns another _CvStub; calling it
    returns another _CvStub. SB3's atari_wrappers only needs `import cv2`
    to succeed plus a no-op `cv2.ocl.setUseOpenCL(False)` at module load.
    """

    def __getattr__(self, _name: str) -> "_CvStub":
        return _CvStub()

    def __call__(self, *_args: object, **_kwargs: object) -> "_CvStub":
        return _CvStub()


_sys.modules.setdefault("cv2", _CvStub())

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from real.stub_env import StubEnv

# Joint limits from the AgileX PiPER MJCF (sim/mjcf/agilex_piper/piper.xml).
# Order is (joint1, joint2, joint3, joint4, joint5, joint6); must match what
# the policy was trained on and what the real /joint_states uses (verified
# by name remap at ingest).
ARM_JOINTS = ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6")
JOINT_LOW = np.array([-2.618, 0.0, -2.697, -1.832, -1.22, -3.14], dtype=np.float32)
JOINT_HIGH = np.array([2.618, 3.14, 0.0, 1.832, 1.22, 3.14], dtype=np.float32)

# Gripper opening in meters. The piper_ros driver expects the gripper as
# position[6] of the JointState command; range is 0 (closed) to 0.080
# (80 mm open). HOLD = 39.7 mm grip on the ball, OPEN = 50 mm release.
# Tuned on the real arm; sim used 0.022 / 0.035.
GRIPPER_HOLD_M = 0.0397
GRIPPER_OPEN_M = 0.050

# `rage_home` keyframe ctrl values from sim/mjcf/rage_cage.xml.
HOME_QPOS = np.array([0.0, 1.57, -1.3485, 0.0, 0.0, 0.0], dtype=np.float32)

# Tick cadence. Must match training.
CONTROL_DT = 0.02            # 50 Hz
ACTION_DELTA = 0.06          # max rad/tick per joint; sim/env.py RageCageEnv.action_delta
RELEASE_STEP = 45            # 45 THROWING ticks (0.9s @ 50Hz)

# Use SIM's cup height (0.12), NOT the real 0.115, when converting
# /cup_pose (top-of-cup) into pedestal_height. Keeps the obs distribution
# aligned with what the policy saw at training time. See design doc §2.
SIM_CUP_HEIGHT = 0.12

# Phase durations (in 50Hz ticks).
HOMING_DURATION_S = 2.0
N_HOMING_TICKS = int(HOMING_DURATION_S / CONTROL_DT)      # 100
SETTLE_HOME_DURATION_S = 0.3
N_SETTLE_HOME_TICKS = int(SETTLE_HOME_DURATION_S / CONTROL_DT)  # 15
SETTLE_RELEASE_DURATION_S = 1.0
N_SETTLE_RELEASE_TICKS = int(SETTLE_RELEASE_DURATION_S / CONTROL_DT)  # 50


class ThrowState(Enum):
    IDLE = "idle"
    HOMING = "homing"
    SETTLE_HOME = "settle_home"
    THROWING = "throwing"
    REPLAY = "replay"
    SETTLE_RELEASE = "settle_release"
    DANCE = "dance"


@dataclass
class TickResult:
    arm_target: NDArray[np.float32] | None
    # Gripper opening in meters, always set (gripper goes in every
    # JointState command — driver closes the gripper if absent).
    gripper_position: float
    state: ThrowState
    done: bool
    # Raw policy output (6 floats, [-1, 1]) when the policy ran this tick.
    # None for non-policy phases (HOMING/SETTLE_*) so logging can tell
    # commanded-from-policy apart from commanded-from-interp.
    action: NDArray[np.float32] | None = None


class ThrowController:
    """Pure-numpy controller: state machine + obs build + policy + integration.

    No ROS dependencies. The Layer 2 rclpy node owns subscribers/publishers and
    delegates per-tick work to this class via `tick()`.
    """

    def __init__(self, model_dir: Path) -> None:
        self._model_dir = model_dir
        # device="cpu" + CUDA_VISIBLE_DEVICES="" (set at module top) keeps
        # torch off the broken Jetson CUDA driver. custom_objects replaces
        # the pickled lr_schedule/clip_range with constants so SB3 never
        # unpickles those callables — at training time they're cloudpickled
        # with closures that include torch tensors, which segfault on
        # restore in this container. Inference doesn't use either, so the
        # placeholder values are never read.
        self._ppo = PPO.load(
            str(model_dir / "policy.zip"),
            device="cpu",
            custom_objects={
                "lr_schedule": lambda _: 1e-4,
                "clip_range": lambda _: 0.2,
            },
        )
        # Auto-detect which policy variant we're loading from its saved
        # observation space. The two we ship:
        #   16-dim → no_ball_obs_v1 (current default; trained without ball state)
        #   22-dim → v1 (older; expects ball_pos(3)+ball_vel(3) slots after joint_vel)
        # Anything else is unexpected and we refuse to guess.
        obs_dim = int(self._ppo.observation_space.shape[0])
        if obs_dim == 16:
            self._uses_ball_obs = False
        elif obs_dim == 22:
            self._uses_ball_obs = True
        else:
            raise ValueError(
                f"unexpected policy obs dim {obs_dim} from {model_dir}; "
                "expected 16 (no_ball_obs) or 22 (v1 with ball state)"
            )
        dummy = DummyVecEnv([lambda: StubEnv(obs_dim=obs_dim)])
        vn = VecNormalize.load(str(model_dir / "vecnormalize.pkl"), dummy)
        self._obs_mean = vn.obs_rms.mean.copy()
        self._obs_var = vn.obs_rms.var.copy()
        self._clip_obs = float(vn.clip_obs)
        self._epsilon = float(vn.epsilon)

        self.state = ThrowState.IDLE
        # "full_throw" runs HOMING → SETTLE_HOME → THROWING → SETTLE_RELEASE → IDLE.
        # "home_only" runs HOMING → SETTLE_HOME → IDLE (skips throw + release).
        # "replay" runs HOMING → SETTLE_HOME → REPLAY → SETTLE_RELEASE → IDLE,
        # where REPLAY plays back a recorded N×6 q-target trajectory open-loop.
        self._mode = "full_throw"
        self._start_q: NDArray[np.float32] | None = None
        # The pose HOMING interps toward (and SETTLE_HOME holds). For
        # full_throw / home_only this is HOME_QPOS. For replay this is
        # trajectory[0] so the first REPLAY tick doesn't have to teleport
        # the arm from HOME_QPOS to the recorded start pose — joints 2/5/6
        # typically differ by 3-4° on row 0, which is a hard jerk under
        # soft MIT gains.
        self._home_target: NDArray[np.float32] = HOME_QPOS.copy()
        self._tick_idx = 0
        self._arm_target: NDArray[np.float32] | None = None
        self._replay_traj: NDArray[np.float32] | None = None
        self._replay_hold_ticks: int = 1
        self._replay_release_row: int = -1
        self._replay_freeze_joints: list[int] = []  # 0-indexed; empty = no freeze
        self._replay_joint1_override: float = float("nan")
        # Dance-mode state: written by start_dance(), read by the DANCE
        # tick handler. None outside an active dance cycle.
        self._dance_target_rad: float = 0.0
        self._dance_total_ticks: int = 0
        self._dance_period_ticks: float = 1.0  # ticks per full sine cycle
        self._dance_amplitude_rad: float = 0.0

    def start_cycle(self, current_joint_pos: NDArray[np.float32]) -> None:
        """Begin a full HOMING → THROWING → SETTLE_RELEASE cycle."""
        self.state = ThrowState.HOMING
        self._mode = "full_throw"
        self._start_q = np.asarray(current_joint_pos, dtype=np.float32).copy()
        self._tick_idx = 0
        self._home_target = HOME_QPOS.copy()

    def start_home_only(self, current_joint_pos: NDArray[np.float32]) -> None:
        """Begin a HOMING + SETTLE_HOME cycle that returns to IDLE without
        running the throw. Used by the /home_arm service."""
        self.state = ThrowState.HOMING
        self._mode = "home_only"
        self._start_q = np.asarray(current_joint_pos, dtype=np.float32).copy()
        self._tick_idx = 0
        self._home_target = HOME_QPOS.copy()

    def start_replay(
        self,
        current_joint_pos: NDArray[np.float32],
        trajectory: NDArray[np.float32],
        hold_ticks: int = 1,
        release_row: int = -1,
        freeze_joints: list[int] | None = None,
        joint1_override: float = float("nan"),
    ) -> None:
        """Begin a HOMING → SETTLE_HOME → REPLAY → SETTLE_RELEASE cycle.

        trajectory: shape (N, 6) of per-tick joint targets, fed open-loop
        to the driver during REPLAY (no policy in the loop).

        HOMING interps to trajectory[0] (NOT HOME_QPOS) so the arm is
        already at the recorded start pose when REPLAY begins — avoids
        a step-input jerk on the first replay tick.

        hold_ticks: how many internal 50Hz ticks to hold each trajectory
        row before advancing. 1 = play at 50fps (recorded rate). 5 = play
        at 10fps (slow motion — 5× wall-clock). All rows are visited
        regardless of hold_ticks; only the per-row dwell time changes.

        release_row: trajectory row at which to open the gripper
        (zero-indexed). -1 (default) = release on the last row. The arm
        continues following the trajectory after release through to the
        end of the recording, then transitions to SETTLE_RELEASE — useful
        when the recorded follow-through matters for the throw.

        freeze_joints: 1-indexed joint numbers (e.g. [1] freezes joint1)
        that should NOT move during the entire replay cycle. The target
        for each frozen joint stays at the value the arm was at when
        start_replay() was called — HOMING leaves it alone, REPLAY
        overrides the recorded trajectory for that joint, SETTLE_RELEASE
        holds it. Useful for isolating which joints actually matter for
        the throw.

        joint1_override: if not NaN, joint1 is held at this absolute
        rad value throughout the cycle — HOMING interps the other 5
        joints to trajectory[0] but pins joint1 at this value, and
        REPLAY ignores the recorded joint1 column. Use to retarget a
        known-good trajectory to a different base rotation. Wins over
        freeze_joints=[1] when both are specified.
        """
        traj = np.asarray(trajectory, dtype=np.float32)
        if traj.ndim != 2 or traj.shape[1] != 6:
            raise ValueError(
                f"replay trajectory must be (N, 6), got shape {traj.shape}"
            )
        if len(traj) == 0:
            raise ValueError("replay trajectory is empty")
        # Convert 1-indexed external API to 0-indexed internal storage.
        freeze_zero_idx: list[int] = []
        for j in (freeze_joints or []):
            if j < 1 or j > 6:
                raise ValueError(
                    f"freeze_joints values must be 1..6, got {j}"
                )
            freeze_zero_idx.append(int(j) - 1)
        self.state = ThrowState.HOMING
        self._mode = "replay"
        self._start_q = np.asarray(current_joint_pos, dtype=np.float32).copy()
        self._tick_idx = 0
        self._home_target = traj[0].copy()
        # Frozen joints stay at their pre-replay pose throughout the cycle
        # by pinning home_target (and the per-tick REPLAY target) back to
        # the snapshot of start_q.
        for i in freeze_zero_idx:
            self._home_target[i] = self._start_q[i]
        # joint1_override (if set) wins over freeze for joint1 — sets the
        # absolute base-rotation target for the whole cycle.
        override_j1 = float(joint1_override)
        if not np.isnan(override_j1):
            self._home_target[0] = override_j1
        self._replay_joint1_override = override_j1
        self._replay_traj = traj
        self._replay_hold_ticks = max(int(hold_ticks), 1)
        self._replay_freeze_joints = freeze_zero_idx
        # Resolve release_row: -1 (or out of range) means "release on last row".
        rr = int(release_row)
        if rr < 0 or rr >= len(traj):
            rr = len(traj) - 1
        self._replay_release_row = rr

    def start_dance(
        self,
        current_joint_pos: NDArray[np.float32],
        target_rad: float,
        duration_s: float,
        sweep_period_s: float,
        amplitude_rad: float,
    ) -> None:
        """Begin a HOMING → SETTLE_HOME → DANCE → IDLE cycle.

        DANCE is a side-to-side sine sweep on joint1 only — joints 2-6
        are held at HOME_QPOS for the entire dance. The sweep keeps full
        amplitude for the first 60% of the duration then eases out with
        cos² damping over the final 40%, landing exactly at `target_rad`
        with zero velocity so the next sub-cycle (typically a replay
        with joint1_override=target_rad) hands off cleanly.

        target_rad: final joint1 angle (radians) at the end of DANCE.
        duration_s: dance duration in seconds; converted to 50Hz ticks.
        sweep_period_s: seconds per full sine cycle (sets the visual
            rhythm of the sweep — 2.0s gives ~3-5 cycles in a 7-10s dance).
        amplitude_rad: peak excursion of joint1 from 0 during the
            full-amplitude phase.
        """
        if duration_s <= 0:
            raise ValueError(f"dance duration must be > 0, got {duration_s}")
        if sweep_period_s <= 0:
            raise ValueError(f"dance sweep period must be > 0, got {sweep_period_s}")
        self.state = ThrowState.HOMING
        self._mode = "dance"
        self._start_q = np.asarray(current_joint_pos, dtype=np.float32).copy()
        self._tick_idx = 0
        self._home_target = HOME_QPOS.copy()
        self._dance_target_rad = float(target_rad)
        self._dance_total_ticks = max(int(round(duration_s / CONTROL_DT)), 1)
        self._dance_period_ticks = float(sweep_period_s) / CONTROL_DT
        self._dance_amplitude_rad = float(amplitude_rad)

    def tick(
        self,
        joint_pos: NDArray[np.float32],
        joint_vel: NDArray[np.float32],
        cup_top_xyz: NDArray[np.float32],
    ) -> TickResult:
        self._tick_idx += 1

        # ---- Transitions (run before the state's per-tick logic) ----
        if self.state == ThrowState.HOMING and self._tick_idx > N_HOMING_TICKS:
            self.state = ThrowState.SETTLE_HOME
            self._tick_idx = 1
        elif self.state == ThrowState.SETTLE_HOME and self._tick_idx > N_SETTLE_HOME_TICKS:
            if self._mode == "home_only":
                # /home_arm path: skip THROWING, return to IDLE on this tick.
                self.state = ThrowState.IDLE
            elif self._mode == "replay":
                self.state = ThrowState.REPLAY
                self._tick_idx = 1
                # Snapshot current pose as the held position; replay rows
                # write into _arm_target each tick below.
                self._arm_target = np.asarray(joint_pos, dtype=np.float32).copy()
            elif self._mode == "dance":
                self.state = ThrowState.DANCE
                self._tick_idx = 1
            else:
                self.state = ThrowState.THROWING
                self._tick_idx = 1
                # Crucial: integrate deltas from where the arm actually is,
                # not from a stale value.
                self._arm_target = np.asarray(joint_pos, dtype=np.float32).copy()
        elif self.state == ThrowState.SETTLE_RELEASE and self._tick_idx > N_SETTLE_RELEASE_TICKS:
            self.state = ThrowState.IDLE
        elif self.state == ThrowState.DANCE and self._tick_idx > self._dance_total_ticks:
            self.state = ThrowState.IDLE

        # ---- Per-state handlers ----
        if self.state == ThrowState.HOMING:
            assert self._start_q is not None
            alpha = self._tick_idx / N_HOMING_TICKS
            arm_target = self._start_q + alpha * (self._home_target - self._start_q)
            return TickResult(
                arm_target=arm_target.astype(np.float32),
                gripper_position=GRIPPER_HOLD_M,
                state=self.state,
                done=False,
            )

        if self.state == ThrowState.SETTLE_HOME:
            return TickResult(
                arm_target=self._home_target.copy(),
                gripper_position=GRIPPER_HOLD_M,
                state=self.state,
                done=False,
            )

        if self.state == ThrowState.THROWING:
            assert self._arm_target is not None
            obs = self._build_obs(joint_pos, joint_vel, cup_top_xyz, self._tick_idx)
            normalized = self._normalize(obs)
            action, _ = self._ppo.predict(normalized, deterministic=True)
            action = np.clip(action, -1.0, 1.0).astype(np.float32)
            self._arm_target = np.clip(
                self._arm_target + action * ACTION_DELTA,
                JOINT_LOW, JOINT_HIGH,
            ).astype(np.float32)

            gripper_position = GRIPPER_HOLD_M
            if self._tick_idx == RELEASE_STEP:
                gripper_position = GRIPPER_OPEN_M
                self.state = ThrowState.SETTLE_RELEASE
                self._tick_idx = 1  # release tick counts as 1st SETTLE_RELEASE tick

            return TickResult(
                arm_target=self._arm_target.copy(),
                gripper_position=gripper_position,
                state=self.state,
                done=False,
                action=action,
            )

        if self.state == ThrowState.SETTLE_RELEASE:
            assert self._arm_target is not None
            return TickResult(
                arm_target=self._arm_target.copy(),
                gripper_position=GRIPPER_OPEN_M,
                state=self.state,
                done=False,
            )

        if self.state == ThrowState.REPLAY:
            assert self._replay_traj is not None
            n_rows = len(self._replay_traj)
            hold = self._replay_hold_ticks
            total_ticks = n_rows * hold
            # Each row held for `hold` consecutive ticks before advancing.
            # _tick_idx is 1-based: ticks 1..hold → row 0, ticks hold+1..2*hold → row 1, etc.
            row_idx = (self._tick_idx - 1) // hold
            self._arm_target = self._replay_traj[row_idx].astype(np.float32)
            # Freeze override: pin specified joints back to their start_q
            # snapshot. Home_target was also pinned in start_replay() so
            # the HOMING phase didn't move these joints; we maintain that
            # here so the trajectory doesn't move them either.
            for i in self._replay_freeze_joints:
                self._arm_target[i] = self._start_q[i]
            # joint1_override (if set) wins over freeze — overrides any
            # joint1 freeze AND any recorded joint1 value with the
            # absolute target set in start_replay().
            if not np.isnan(self._replay_joint1_override):
                self._arm_target[0] = self._replay_joint1_override
            # Gripper opens once we're at or past the release row, and stays
            # open. The node's per-change gripper-publish logic ensures we
            # only actually send the open command once.
            if row_idx >= self._replay_release_row:
                gripper_position = GRIPPER_OPEN_M
            else:
                gripper_position = GRIPPER_HOLD_M
            if self._tick_idx == total_ticks:
                # End of trajectory — transition to SETTLE_RELEASE so the
                # arm holds the final pose while the ball flies out.
                self.state = ThrowState.SETTLE_RELEASE
                self._tick_idx = 1
            return TickResult(
                arm_target=self._arm_target.copy(),
                gripper_position=gripper_position,
                state=self.state,
                done=False,
            )

        if self.state == ThrowState.DANCE:
            # Sine sweep on joint1 with cos²-damped tail.
            # Full amplitude for 60% of duration; damp 1 → 0 over last 40%
            # so the swing winds down and joint1 lands exactly at target
            # with zero velocity. _tick_idx is 1-based: progress = 1 at the
            # final tick.
            progress = self._tick_idx / self._dance_total_ticks
            if progress < 0.6:
                damp = 1.0
            else:
                ease = (progress - 0.6) / 0.4
                damp = float(np.cos(0.5 * np.pi * ease) ** 2)
            phase = 2.0 * np.pi * self._tick_idx / self._dance_period_ticks
            joint1 = (
                damp * self._dance_amplitude_rad * float(np.sin(phase))
                + (1.0 - damp) * self._dance_target_rad
            )
            arm_target = HOME_QPOS.copy()
            arm_target[0] = joint1
            return TickResult(
                arm_target=arm_target.astype(np.float32),
                gripper_position=GRIPPER_HOLD_M,
                state=self.state,
                done=False,
            )

        if self.state == ThrowState.IDLE:
            return TickResult(
                arm_target=None,
                gripper_position=GRIPPER_HOLD_M,
                state=self.state,
                done=True,
            )

        raise NotImplementedError(f"tick in state {self.state} not implemented")

    def _build_obs(
        self,
        joint_pos: NDArray[np.float32],
        joint_vel: NDArray[np.float32],
        cup_top_xyz: NDArray[np.float32],
        tick_idx: int,
    ) -> NDArray[np.float32]:
        """Build the policy's obs in the exact order it was trained with.

        16-dim (no_ball_obs_v1):
          joint_pos(6), joint_vel(6), cup_xy(2), pedestal_height(1),
          release_countdown(1).

        22-dim (v1):
          joint_pos(6), joint_vel(6), ball_pos(3), ball_vel(3), cup_xy(2),
          pedestal_height(1), release_countdown(1).

        For the 22-dim variant on the real arm we have no ball estimate
        and pass zeros — out-of-distribution vs training (where the ball
        was welded to the gripper during the windup). Throw quality with
        v1 on the real arm is unpredictable; prefer no_ball_obs_v1.
        """
        pedestal_h = max(float(cup_top_xyz[2]) - SIM_CUP_HEIGHT, 0.0)
        countdown = max(RELEASE_STEP - tick_idx, 0) / RELEASE_STEP
        parts = [
            np.asarray(joint_pos, dtype=np.float32),
            np.asarray(joint_vel, dtype=np.float32),
        ]
        if self._uses_ball_obs:
            parts.append(np.zeros(3, dtype=np.float32))  # ball_pos placeholder
            parts.append(np.zeros(3, dtype=np.float32))  # ball_vel placeholder
        parts.extend([
            np.asarray(cup_top_xyz[:2], dtype=np.float32),
            np.array([pedestal_h], dtype=np.float32),
            np.array([countdown], dtype=np.float32),
        ])
        return np.concatenate(parts).astype(np.float32)

    def _normalize(self, obs: NDArray[np.float32]) -> NDArray[np.float32]:
        return np.clip(
            (obs - self._obs_mean) / np.sqrt(self._obs_var + self._epsilon),
            -self._clip_obs,
            self._clip_obs,
        ).astype(np.float32)
