"""Gymnasium environment for the first Rage Cage PPO experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces
from numpy.typing import NDArray

SCENE = Path(__file__).parent / "mjcf" / "rage_cage.xml"

ARM_JOINTS = tuple(f"joint{i}" for i in range(1, 7))
ARM_ACTUATORS = tuple(f"joint{i}" for i in range(1, 7))
GRIPPER_ACTUATOR = "gripper"
RESET_KEY = "rage_home"
BALL_GRIP_EQUALITY = "ball_grip"
GRIPPER_HOLD_CTRL = 0.022
GRIPPER_OPEN_CTRL = 0.035

CUP_RADIUS = 0.047
CUP_HEIGHT = 0.12
TABLE_Z = 0.0
NOMINAL_CUP_XY = np.array([0.85, 0.0], dtype=np.float32)
CUP_X_RANGE = (0.83, 0.87)
CUP_Y_RANGE = (-0.02, 0.02)
# Pedestal under the cup; height ∈ [0, 0.15m] simulates a stack of 1-9
# cups in real rage cage (Solo cups nest at ~1.8cm/cup). Clamped at a
# tiny min half-height so MuJoCo doesn't see a degenerate cylinder when
# pedestal_height=0 (matches v34 behavior — cup at table top).
PEDESTAL_RADIUS = 0.040
PEDESTAL_Z_MIN_HALF = 0.0001
DEFAULT_PEDESTAL_RANGE = (0.0, 0.0)
DISTANCE_REWARD_SCALE = 0.5
# Length scale (m) for the post-bounce cup-distance reward. The old linear ramp
# `1 - dist/0.5` saturates near the goal — at dist=0.06 (the v16 plateau) the
# score is already 0.88, so the policy gets ~88% of the available reward by
# stopping 6cm from the cup, with only 12% left for closing that final gap.
# v16 trained 10M steps and never closed it. With exp(-dist / 0.04) instead,
# score=0.22 at dist=0.06 and 1.0 at dist=0, so 78% of the reward is gated on
# the last 6cm — flipping the gradient so that pushing through the wall-grazing
# zone into the cup mouth is where most of the cup_dist reward lives.
CUP_DIST_REWARD_SCALE = 0.04
# Bounce-target geometry. The previous design used a fixed (-0.32, 0)
# offset behind the cup, which only made geometric sense for an on-axis
# cup at the v22 nominal (1.10, 0): for off-axis cups the rule placed
# the target on a line parallel to world-x, not on the actual throw
# axis from arm base to cup. The new design parametrizes:
#
#   target = BOUNCE_FRACTION * cup_xy
#
# along the line from the world origin (arm base, approximately) to the
# cup. BOUNCE_FRACTION = 0.7 reproduces (0.77, 0) at the v22 nominal —
# matching the working prior — and naturally adapts to any cup position
# under the ±10cm randomization sweep without retuning.
#
# Reward tolerance is elliptical in the throw-frame coordinates:
#   - SIGMA_LONG (along throw): 30cm — physical bounces tolerate
#     longitudinal offset because forward velocity / bounce angle
#     partly absorbs it.
#   - SIGMA_PERP (perpendicular): 8cm — perpendicular error translates
#     ~1:1 into a y-miss at the cup, so the reward is much tighter.
BOUNCE_FRACTION = 0.7
BOUNCE_SIGMA_LONG = 0.30
BOUNCE_SIGMA_PERP = 0.08


def _bounce_geometry(cup_xy: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Return (target, throw_dir, perp_dir) for the given cup position.

    target is BOUNCE_FRACTION along the line from world origin to cup.
    throw_dir is the unit base→cup vector; perp_dir is its 2-D normal.
    """
    cup64 = np.asarray(cup_xy, dtype=np.float64)
    norm = float(np.linalg.norm(cup64))
    if norm < 1e-9:
        throw_dir = np.array([1.0, 0.0], dtype=np.float64)
    else:
        throw_dir = cup64 / norm
    perp_dir = np.array([-throw_dir[1], throw_dir[0]], dtype=np.float64)
    target = BOUNCE_FRACTION * cup64
    return target, throw_dir, perp_dir


def bounce_score(bounce_xy: NDArray[np.float64], cup_xy: NDArray[np.float64]) -> float:
    """Elliptical bounce-target score in [0, 1].

    Returns 1.0 at the bounce target, decreases smoothly with throw-frame
    longitudinal/perpendicular error, and clips to 0.0 outside the unit
    ellipse defined by SIGMA_LONG / SIGMA_PERP.
    """
    target, throw_dir, perp_dir = _bounce_geometry(cup_xy)
    delta = np.asarray(bounce_xy, dtype=np.float64) - target
    long_err = float(abs(delta @ throw_dir))
    perp_err = float(abs(delta @ perp_dir))
    radial = float(np.sqrt((long_err / BOUNCE_SIGMA_LONG) ** 2 + (perp_err / BOUNCE_SIGMA_PERP) ** 2))
    return max(0.0, 1.0 - radial)

# Hard motion-safety limits. JOINT_VEL_LIMIT=8.0 gives the policy enough
# headroom to throw at action_delta=0.06 without firing the limit penalty
# on smooth motion. ACC/JERK are pathology backstops — smooth motion produces
# acc ~3.5 rad/s² (well below 80) so they only fire on actual oscillation.
# We tried action_delta=0.07 in v9-v11; the wider range produced catastrophic
# motions during early exploration and training went unstable. 0.06 is the
# working middle ground.
JOINT_VEL_LIMIT = 8.0
JOINT_ACC_LIMIT = 80.0
JOINT_JERK_LIMIT = 10000.0

# Standard deviation (rad) of Gaussian noise added to arm joint qpos at reset.
# 0.005 rad ≈ 0.3° per joint — comparable to real-arm repeatability error.
# Adds initial-state diversity so each training episode starts from a slightly
# different arm pose, giving the policy more lucky-discovery chances for the
# rare cup-entry trajectory. v17/v18 bumped this to 0.01 for more exploration
# but combined with action_delta=0.06 produced too much variance and slowed
# stage 1 promotion. Back to 0.005.
RESET_NOISE_STD = 0.005

# Coefficient of restitution applied to the ball after each ball-surface
# contact substep. MuJoCo's default contact dynamics give COR ≈ 0.6 with the
# stable parameters we need; real ping-pong on wood is 0.85-0.93. We apply a
# per-substep velocity correction in _apply_cor_correction so bounces are
# realistic without destabilizing the solver. Cup wall is lower because real
# plastic-cup walls are dampened (a real Solo cup feels like ~0.4-0.5 COR
# when you flick a ball at it). Higher COR_CUP makes the cup actively
# *deflect* incoming balls instead of catching them; 0.5 lets gravity win
# and the ball settles into the cup more often. Water is 0 — ball "splashes"
# and stops on contact.
COR_TABLE = 0.88
COR_CUP = 0.50
COR_WATER = 0.0

# Skip COR correction below this normal-velocity threshold so a ball resting
# on the table doesn't have its tiny noise-velocity flipped each substep,
# which would otherwise inject energy and cause jiggle.
BOUNCE_NORMAL_VEL_MIN = 0.10  # m/s

REWARD_WEIGHTS: dict[int, dict[str, float]] = {
    1: {
        "bounce": 15.0,
        "bounce_xy": 0.0,
        "cup_dist": 0.0,
        "second_bounce_cup": 0.0,
        "cup_entry": 0.0,
        "success": 0.0,
    },
    2: {
        "bounce": 10.0,
        "bounce_xy": 5.0,
        "cup_dist": 0.1,
        "second_bounce_cup": 0.0,
        "cup_entry": 0.0,
        "success": 0.0,
    },
    3: {
        "bounce": 5.0,
        "bounce_xy": 2.0,
        "cup_dist": 10.0,
        "second_bounce_cup": 0.0,
        "cup_entry": 30.0,
        "success": 100.0,
    },
    4: {
        "bounce": 2.0,
        "bounce_xy": 1.0,
        "cup_dist": 5.0,
        "second_bounce_cup": 0.0,
        "cup_entry": 100.0,
        "success": 150.0,
    },
}


class RageCageEnv(gym.Env[NDArray[np.float32], NDArray[np.float32]]):
    """Privileged-state MuJoCo task for bootstrapping PPO throw policies.

    Observation is a flat vector:
    joint_pos(6), joint_vel(6), ball_pos(3), ball_vel(3), cup_xy(2),
    pedestal_height(1), release_countdown(1).

    Action is six joint-target deltas in [-1, 1]. The ball is welded to the
    gripper and released automatically at a fixed control step (release_step).
    The release_countdown obs slot lets the policy time peak gripper velocity
    to coincide with the release moment.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}

    def __init__(
        self,
        scene: Path | str = SCENE,
        max_episode_steps: int = 150,
        control_dt: float = 0.02,
        action_delta: float = 0.06,
        release_step: int = 45,
        reward_stage: int = 3,
        randomize_cup: bool = True,
        render_mode: str | None = None,
        image_width: int = 128,
        image_height: int = 128,
        camera: str | None = None,
    ) -> None:
        super().__init__()
        self.model = mujoco.MjModel.from_xml_path(str(scene))
        self.data = mujoco.MjData(self.model)

        self.max_episode_steps = max_episode_steps
        self.control_steps = max(1, round(control_dt / self.model.opt.timestep))
        self.action_delta = action_delta
        self.release_step = release_step
        if reward_stage not in REWARD_WEIGHTS:
            raise ValueError(f"unsupported reward_stage={reward_stage}; expected one of {tuple(REWARD_WEIGHTS)}")
        self.set_reward_stage(reward_stage)
        self.randomize_cup = randomize_cup
        self.render_mode = render_mode
        self.image_width = image_width
        self.image_height = image_height
        self.camera = camera
        self.renderer: mujoco.Renderer | None = None

        self.joint_ids = np.array(
            [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in ARM_JOINTS],
            dtype=np.int32,
        )
        self.actuator_ids = np.array(
            [
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
                for name in ARM_ACTUATORS
            ],
            dtype=np.int32,
        )
        self.gripper_actuator_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, GRIPPER_ACTUATOR
        )
        self.key_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, RESET_KEY)
        self.cup_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "cup_free")
        self.ball_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "ball_free")
        self.ball_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "ball")
        self.cup_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "cup")
        # The cup_world weld pins the cup to its keyframe pose so it
        # doesn't slide when struck (see rage_cage.xml). Its relpose is
        # hard-coded to NOMINAL_CUP_XY in the MJCF; if we only rewrote
        # qpos at reset, the soft weld would pull the cup back to (1.10,
        # 0) within a few solver steps. We override eq_data per reset so
        # the weld actually anchors at the sampled cup_xy.
        self.cup_world_eq_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_EQUALITY, "cup_world"
        )
        self.cup_pedestal_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "cup_pedestal"
        )
        self.cup_pedestal_geom_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "cup_pedestal_geom"
        )
        self.ball_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "ball_geom")
        self.table_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "table")
        self.floor_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        self.water_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "cup_water")
        self.ball_grip_eq_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_EQUALITY, BALL_GRIP_EQUALITY
        )
        self.left_finger_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "link7")
        self.right_finger_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "link8")

        model_refs = list(zip(ARM_JOINTS, self.joint_ids, strict=True))
        model_refs.extend(zip(ARM_ACTUATORS, self.actuator_ids, strict=True))
        model_refs.extend(
            [
                (GRIPPER_ACTUATOR, self.gripper_actuator_id),
                (RESET_KEY, self.key_id),
                ("cup_free", self.cup_joint_id),
                ("ball_free", self.ball_joint_id),
                ("ball", self.ball_body_id),
                ("cup", self.cup_body_id),
                ("ball_geom", self.ball_geom_id),
                ("table", self.table_geom_id),
                ("floor", self.floor_geom_id),
                ("cup_water", self.water_geom_id),
                ("cup_pedestal", self.cup_pedestal_body_id),
                ("cup_pedestal_geom", self.cup_pedestal_geom_id),
                (BALL_GRIP_EQUALITY, self.ball_grip_eq_id),
                ("link7", self.left_finger_body_id),
                ("link8", self.right_finger_body_id),
            ]
        )
        missing = [name for name, idx in model_refs if idx < 0]
        if missing:
            raise ValueError(f"missing MuJoCo model names: {missing}")

        self.joint_qposadr = self.model.jnt_qposadr[self.joint_ids]
        self.joint_dofadr = self.model.jnt_dofadr[self.joint_ids]
        self.cup_qposadr = int(self.model.jnt_qposadr[self.cup_joint_id])
        self.cup_dofadr = int(self.model.jnt_dofadr[self.cup_joint_id])
        self.ball_qposadr = int(self.model.jnt_qposadr[self.ball_joint_id])
        self.ball_dofadr = int(self.model.jnt_dofadr[self.ball_joint_id])
        self.robot_body_ids = {
            body_id
            for body_id in range(1, self.model.nbody)
            if body_id not in {self.cup_body_id, self.ball_body_id, self.cup_pedestal_body_id}
        }
        # Geom IDs for everything attached to the cup body — cup walls, base,
        # and the water cylinder. Used by the COR correction to apply
        # COR_CUP / COR_WATER on ball-cup contacts. The water-vs-walls split
        # is enforced by ID equality before falling through to this set.
        self.cup_geom_ids = frozenset(
            g for g in range(self.model.ngeom)
            if self.model.geom_bodyid[g] == self.cup_body_id
        )

        self.joint_low = self.model.jnt_range[self.joint_ids, 0].astype(np.float32)
        self.joint_high = self.model.jnt_range[self.joint_ids, 1].astype(np.float32)
        # cup_xy obs bounds widened to comfortably cover the R3 ±10cm
        # randomization box plus a margin. The pedestal-height slot
        # (formerly cup_count, currently bound to pedestal_height in
        # metres ∈ [0, 0.15]) is bounded slightly wider to leave headroom.
        # The last obs slot is the release_countdown.
        obs_low = np.concatenate(
            [
                self.joint_low,
                np.full(6, -20.0, dtype=np.float32),
                np.array([-1.0, -1.0, -0.5], dtype=np.float32),
                np.full(3, -10.0, dtype=np.float32),
                np.array([0.70, -0.15], dtype=np.float32),
                np.array([0.0], dtype=np.float32),
                np.array([0.0], dtype=np.float32),
            ]
        )
        obs_high = np.concatenate(
            [
                self.joint_high,
                np.full(6, 20.0, dtype=np.float32),
                np.array([1.0, 1.0, 1.0], dtype=np.float32),
                np.full(3, 10.0, dtype=np.float32),
                np.array([1.00, 0.15], dtype=np.float32),
                np.array([0.20], dtype=np.float32),
                np.array([1.0], dtype=np.float32),
            ]
        )
        # Action: 6 arm joint deltas. Release fires automatically when
        # step_count reaches self.release_step.
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=obs_low,
            high=obs_high,
            dtype=np.float32,
        )

        self.step_count = 0
        self.cup_xy = NOMINAL_CUP_XY.copy()
        # Per-instance copies of the randomization box so the curriculum
        # can grow it during training (set_cup_range). Module-level
        # CUP_X_RANGE / CUP_Y_RANGE remain the canonical default.
        self.cup_x_range: tuple[float, float] = tuple(CUP_X_RANGE)
        self.cup_y_range: tuple[float, float] = tuple(CUP_Y_RANGE)
        # One-shot cup-position override consumed by the next reset().
        # Used by ``grid``-mode evaluation to land the cup at fixed
        # workspace cells. Set via set_next_cup; cleared after a single
        # reset so subsequent resets revert to randomization.
        self._next_cup_override: NDArray[np.float32] | None = None
        self.cup_count = 1
        # Pedestal under the cup. set_pedestal_range bounds the per-reset
        # uniform sample (only consulted when randomize_cup is also True);
        # set_next_pedestal supplies a one-shot override (for grid eval).
        # pedestal_height in [0, 0.15m] simulates a stack of 1-9 nested
        # Solo cups; 0 matches v34's table-top cup.
        self.pedestal_z_range: tuple[float, float] = tuple(DEFAULT_PEDESTAL_RANGE)
        self.pedestal_height: float = 0.0
        self._next_pedestal_override: float | None = None
        self.arm_target = np.zeros(6, dtype=np.float32)
        self.best_final_dist = np.inf
        self.bounced = False
        self.table_bounce_count = 0
        self.invalid_bounce_count = 0
        self.ball_was_touching_table = False
        self.first_table_bounce_xy: NDArray[np.float64] | None = None
        self.first_table_bounce_time: float | None = None
        self.second_table_bounce_xy: NDArray[np.float64] | None = None
        self.second_table_bounce_cup_dist = np.inf
        self.closest_post_bounce_cup_dist = np.inf
        self.ball_entered_cup = False
        self.settled_in_cup = False
        self.ball_contacted_table = False
        self.ball_contacted_cup = False
        self.ball_contacted_floor = False
        self.ball_contacted_robot = False
        self.ball_touched_water = False
        self.robot_table_contact = False
        self.robot_cup_contact = False
        self.bounce_bonus_given = False
        self.ball_released = False
        self.prev_joint_vel = np.zeros(6, dtype=np.float64)
        self.prev_joint_acc = np.zeros(6, dtype=np.float64)
        self.prev_action = np.zeros(6, dtype=np.float32)
        self.max_joint_vel = 0.0
        self.max_joint_acc = 0.0
        self.max_joint_jerk = 0.0
        self.motion_limit_violated = False
        self.last_reward_components: dict[str, float] = {}
        self.passive_render_frames: list[NDArray[np.uint8]] = []
        self.passive_info_rows: list[dict[str, Any]] = []
        # Optional hook fired once per control step inside the post-release
        # passive-flight loop. ``step()`` runs the ball's entire post-release
        # trajectory in a single call, so a live MuJoCo viewer would otherwise
        # see the ball "teleport" from the gripper to the terminal frame.
        # play_policy.py sets this to a viewer.sync()+sleep so the user can
        # actually watch the throw arc. Default None = no-op for training.
        self.passive_step_callback: Any = None

    def set_reward_stage(self, reward_stage: int) -> None:
        if reward_stage not in REWARD_WEIGHTS:
            raise ValueError(f"unsupported reward_stage={reward_stage}; expected one of {tuple(REWARD_WEIGHTS)}")
        self.reward_stage = reward_stage
        self.reward_weights = REWARD_WEIGHTS[reward_stage]

    def set_cup_range(
        self,
        x_range: tuple[float, float],
        y_range: tuple[float, float],
    ) -> None:
        self.cup_x_range = (float(x_range[0]), float(x_range[1]))
        self.cup_y_range = (float(y_range[0]), float(y_range[1]))

    def set_next_cup(self, cup_xy: NDArray[np.float32] | None) -> None:
        if cup_xy is None:
            self._next_cup_override = None
            return
        self._next_cup_override = np.asarray(cup_xy, dtype=np.float32).copy()

    def set_pedestal_range(self, z_range: tuple[float, float]) -> None:
        self.pedestal_z_range = (float(z_range[0]), float(z_range[1]))

    def set_next_pedestal(self, height: float | None) -> None:
        if height is None:
            self._next_pedestal_override = None
            return
        self._next_pedestal_override = float(height)

    def _bounce_target(self) -> NDArray[np.float64]:
        target, _, _ = _bounce_geometry(np.asarray(self.cup_xy, dtype=np.float64))
        return target

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[NDArray[np.float32], dict[str, Any]]:
        super().reset(seed=seed)
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.key_id)

        if self._next_cup_override is not None:
            self.cup_xy = self._next_cup_override.copy()
            self._next_cup_override = None
        elif self.randomize_cup:
            self.cup_xy = np.array(
                [
                    self.np_random.uniform(*self.cup_x_range),
                    self.np_random.uniform(*self.cup_y_range),
                ],
                dtype=np.float32,
            )
        else:
            self.cup_xy = NOMINAL_CUP_XY.copy()
        self.cup_count = int(self.np_random.integers(1, 2))

        if self._next_pedestal_override is not None:
            self.pedestal_height = float(self._next_pedestal_override)
            self._next_pedestal_override = None
        elif self.randomize_cup:
            self.pedestal_height = float(
                self.np_random.uniform(*self.pedestal_z_range)
            )
        else:
            self.pedestal_height = 0.0

        self.data.qpos[self.cup_qposadr : self.cup_qposadr + 7] = (
            self.cup_xy[0],
            self.cup_xy[1],
            self.pedestal_height,
            1.0,
            0.0,
            0.0,
            0.0,
        )
        self.data.qvel[self.cup_dofadr : self.cup_dofadr + 6] = 0.0
        # Re-anchor the cup_world weld to the sampled cup_xy / pedestal.
        # Without this, the MJCF's hard-coded relpose=(0.85, 0, 0) pulls
        # the cup back over the next few solver steps. Slot layout in
        # eq_data for a weld:
        # [anchor1(3) | relpose_xyz(3) | relquat(4) | torquescale].
        self.model.eq_data[self.cup_world_eq_id, 3] = float(self.cup_xy[0])
        self.model.eq_data[self.cup_world_eq_id, 4] = float(self.cup_xy[1])
        self.model.eq_data[self.cup_world_eq_id, 5] = float(self.pedestal_height)
        # Resize and reposition the pedestal cylinder to span z=0 to
        # z=pedestal_height directly under the cup. Half-height clamped
        # so MuJoCo doesn't see a degenerate cylinder when pedestal=0.
        # geom_aabb and geom_rbound must be updated alongside geom_size:
        # they're cached at compile time and the broadphase uses them to
        # cull contact pairs, so a stale aabb makes the ball tunnel.
        pedestal_half = max(self.pedestal_height / 2.0, PEDESTAL_Z_MIN_HALF)
        self.model.body_pos[self.cup_pedestal_body_id, 0] = float(self.cup_xy[0])
        self.model.body_pos[self.cup_pedestal_body_id, 1] = float(self.cup_xy[1])
        self.model.body_pos[self.cup_pedestal_body_id, 2] = pedestal_half
        self.model.geom_size[self.cup_pedestal_geom_id, 1] = pedestal_half
        self.model.geom_aabb[self.cup_pedestal_geom_id, 5] = pedestal_half
        self.model.geom_rbound[self.cup_pedestal_geom_id] = float(
            np.sqrt(PEDESTAL_RADIUS * PEDESTAL_RADIUS + pedestal_half * pedestal_half)
        )
        # Add tiny noise to arm joint positions at reset. ~0.3° per joint
        # (RESET_NOISE_STD=0.005 rad). Each episode now starts from a slightly
        # different arm pose → slightly different release point → different
        # trajectory, which gives the policy more chances to stumble into
        # rare cup-entry trajectories. Real robotic arms have repeatability
        # error of similar magnitude, so this is also a sim-to-real prior.
        self.data.qpos[self.joint_qposadr] += self.np_random.normal(
            0.0, RESET_NOISE_STD, size=6
        )
        self.arm_target = self.data.qpos[self.joint_qposadr].astype(np.float32)
        self.data.ctrl[self.actuator_ids] = self.arm_target
        self.data.ctrl[self.gripper_actuator_id] = GRIPPER_HOLD_CTRL
        self.data.eq_active[self.ball_grip_eq_id] = 1
        mujoco.mj_forward(self.model, self.data)

        self.step_count = 0
        self.best_final_dist = np.inf
        self.bounced = False
        self.table_bounce_count = 0
        self.invalid_bounce_count = 0
        self.ball_was_touching_table = False
        self.first_table_bounce_xy = None
        self.first_table_bounce_time = None
        self.second_table_bounce_xy = None
        self.second_table_bounce_cup_dist = np.inf
        self.closest_post_bounce_cup_dist = np.inf
        self.ball_entered_cup = False
        self.settled_in_cup = False
        self.ball_contacted_table = False
        self.ball_contacted_cup = False
        self.ball_contacted_floor = False
        self.ball_contacted_robot = False
        self.ball_touched_water = False
        self.robot_table_contact = False
        self.robot_cup_contact = False
        self.bounce_bonus_given = False
        self.ball_released = False
        self.prev_joint_vel = self.data.qvel[self.joint_dofadr].copy()
        self.prev_joint_acc = np.zeros(6, dtype=np.float64)
        self.prev_action = np.zeros(6, dtype=np.float32)
        self.max_joint_vel = 0.0
        self.max_joint_acc = 0.0
        self.max_joint_jerk = 0.0
        self.motion_limit_violated = False
        self.last_reward_components = {}
        self.passive_render_frames = []
        self.passive_info_rows = []
        return self._get_obs(), self._get_info()

    def step(
        self, action: NDArray[np.float32]
    ) -> tuple[NDArray[np.float32], float, bool, bool, dict[str, Any]]:
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, self.action_space.low, self.action_space.high)
        if self.ball_released:
            raise RuntimeError("step() called after release; reset the environment after done=True")
        self.passive_render_frames = []
        self.passive_info_rows = []

        self.arm_target = np.clip(
            self.arm_target + action * self.action_delta,
            self.joint_low,
            self.joint_high,
        )
        self.data.ctrl[self.actuator_ids] = self.arm_target
        self.data.ctrl[self.gripper_actuator_id] = GRIPPER_HOLD_CTRL

        reward_components: dict[str, float] = {}
        self._advance_control_step(reward_components, action=action)

        self.step_count += 1
        # Fixed-step release: once step_count reaches release_step the
        # gripper opens and the env runs the entire post-release passive
        # flight inside this step() call.
        release_now = self.step_count >= self.release_step
        if release_now:
            self._release_ball()
            self._hold_last_arm_target()
            while True:
                success = self._success()
                terminated = success or self._terminal_failure()
                truncated = self.step_count >= self.max_episode_steps
                if terminated or truncated:
                    break
                self._advance_control_step(reward_components, action=None)
                self.step_count += 1
                self._capture_passive_step(reward_components)
                if self.passive_step_callback is not None:
                    self.passive_step_callback()
        else:
            success = self._success()
            terminated = self._terminal_failure()
            truncated = self.step_count >= self.max_episode_steps

        if success:
            self._add_reward(reward_components, "success", self.reward_weights["success"])

        # Capture a render frame of the terminal state into
        # passive_render_frames so the rollout viz can show where the
        # episode ended. Without this, episodes that terminate before
        # the first env.render() call (e.g., motion_limit_violated on
        # step 1) produce GIFs containing only the initial reset frame.
        # The rollout loop skips env.render() after done because SB3's
        # VecEnv has already auto-reset the env, so any post-step
        # render shows the wrong state — capturing inside step() runs
        # before the auto-reset.
        if (terminated or truncated) and self.render_mode == "rgb_array":
            self._capture_passive_step(reward_components)

        reward = float(sum(reward_components.values()))
        self.last_reward_components = reward_components
        return self._get_obs(), reward, terminated, truncated, self._get_info(success=success)

    def render(self) -> NDArray[np.uint8] | None:
        if self.render_mode != "rgb_array":
            return None
        if self.renderer is None:
            self.renderer = mujoco.Renderer(
                self.model,
                height=self.image_height,
                width=self.image_width,
            )
        if self.camera is None:
            self.renderer.update_scene(self.data)
        else:
            self.renderer.update_scene(self.data, camera=self.camera)
        return self.renderer.render()

    def close(self) -> None:
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None

    def _get_obs(self) -> NDArray[np.float32]:
        return np.concatenate(
            [
                self.data.qpos[self.joint_qposadr],
                self.data.qvel[self.joint_dofadr],
                self._ball_pos(),
                self._ball_vel(),
                self.cup_xy,
                np.array([self.pedestal_height], dtype=np.float32),
                np.array([self._release_countdown()], dtype=np.float32),
            ]
        ).astype(np.float32)

    def _get_info(self, success: bool = False) -> dict[str, Any]:
        ball_pos = self._ball_pos()
        cup_dist = float(np.linalg.norm(ball_pos[:2] - self.cup_xy))
        return {
            "success": success,
            "step_count": self.step_count,
            "cup_dist": cup_dist,
            "bounced": self.bounced,
            "bounce_count": self.table_bounce_count,
            "table_bounce_count": self.table_bounce_count,
            "invalid_bounce_count": self.invalid_bounce_count,
            "first_table_bounce_xy": (
                None
                if self.first_table_bounce_xy is None
                else self.first_table_bounce_xy.copy()
            ),
            "closest_post_bounce_cup_dist": self.closest_post_bounce_cup_dist,
            "second_table_bounce_xy": (
                None
                if self.second_table_bounce_xy is None
                else self.second_table_bounce_xy.copy()
            ),
            "second_table_bounce_cup_dist": self.second_table_bounce_cup_dist,
            "ball_entered_cup": self.ball_entered_cup,
            "settled_in_cup": self.settled_in_cup,
            "ball_released": self.ball_released,
            "release_countdown": self._release_countdown(),
            "cup_xy": self.cup_xy.copy(),
            "cup_count": self.cup_count,
            "pedestal_height": float(self.pedestal_height),
            "robot_table_contact": self.robot_table_contact,
            "robot_cup_contact": self.robot_cup_contact,
            "ball_contacted_floor": self.ball_contacted_floor,
            "ball_contacted_robot": self.ball_contacted_robot,
            "ball_touched_water": self.ball_touched_water,
            "max_joint_vel": self.max_joint_vel,
            "max_joint_acc": self.max_joint_acc,
            "max_joint_jerk": self.max_joint_jerk,
            "motion_limit_violated": self.motion_limit_violated,
            "reward_components": self.last_reward_components.copy(),
            "passive_render_frames": self.passive_render_frames,
            "passive_info_rows": self.passive_info_rows,
        }

    def _success(self) -> bool:
        ball_pos = self._ball_pos()
        ball_vel = self._ball_vel()
        cup_dist = float(np.linalg.norm(ball_pos[:2] - self.cup_xy))
        self.best_final_dist = min(self.best_final_dist, cup_dist)

        inside_cup_xy = cup_dist < CUP_RADIUS
        inside_cup_z = (
            self.pedestal_height + 0.015 < ball_pos[2] < self.pedestal_height + CUP_HEIGHT
        )
        slow_enough = np.linalg.norm(ball_vel) < 0.75
        exactly_one_table_bounce = self.table_bounce_count == 1
        # Two paths to "in cup": settled inside the cup volume (slow + xy/z
        # within bounds) OR water-touch. The water cylinder sits inside the
        # cup at z ∈ [0.003, 0.033]; the only way to contact it is from above
        # through the cup mouth, so a water touch unambiguously means the ball
        # made it into the cup. Either path counts as success as long as the
        # ball took exactly one valid table bounce first.
        settled = inside_cup_xy and inside_cup_z and slow_enough
        self.settled_in_cup = bool(
            exactly_one_table_bounce
            and not self._has_invalid_ball_contact()
            and (settled or self.ball_touched_water)
        )
        return self.settled_in_cup

    def _ball_pos(self) -> NDArray[np.float64]:
        return self.data.xpos[self.ball_body_id].copy()

    def _ball_vel(self) -> NDArray[np.float64]:
        return self.data.qvel[self.ball_dofadr : self.ball_dofadr + 3].copy()

    def _gripper_ball_pos(self) -> NDArray[np.float64]:
        left_pos = self.data.xpos[self.left_finger_body_id]
        right_pos = self.data.xpos[self.right_finger_body_id]
        return ((left_pos + right_pos) * 0.5).copy()

    def _release_ball(self) -> None:
        self.data.eq_active[self.ball_grip_eq_id] = 0
        self.data.ctrl[self.gripper_actuator_id] = GRIPPER_OPEN_CTRL
        self.ball_released = True
        mujoco.mj_forward(self.model, self.data)

    def _hold_last_arm_target(self) -> None:
        self.data.ctrl[self.actuator_ids] = self.arm_target
        self.data.ctrl[self.gripper_actuator_id] = GRIPPER_OPEN_CTRL

    def _release_countdown(self) -> float:
        if self.release_step <= 0:
            return 0.0
        return max(self.release_step - self.step_count, 0) / self.release_step

    def _advance_control_step(
        self,
        reward_components: dict[str, float],
        action: NDArray[np.float32] | None,
    ) -> None:
        if self.ball_released:
            self._hold_last_arm_target()

        # Per-substep contact tracking. xinyi's original code classified
        # contacts only once at the end of the control step (after 10 mj
        # substeps). With our COR correction giving fast bounces (1-2
        # substeps of contact), the end-of-control-step check usually saw
        # "not touching" and missed the transition entirely — diagnostic
        # showed ~75% of bounces undercounted on flat skip-bounce
        # trajectories. We now classify per-substep, accumulate transitions
        # and any-substep contact flags, and emit rewards once per control
        # step in _update_contacts_and_events using the accumulated state.
        bounce_events: list[tuple[NDArray[np.float64], float, NDArray[np.float64]]] = []
        any_contact = {
            "ball_table": False,
            "ball_cup": False,
            "ball_water": False,
            "ball_floor": False,
            "ball_robot": False,
            "robot_table": False,
            "robot_cup": False,
        }
        for _ in range(self.control_steps):
            ball_vel_pre = self.data.qvel[
                self.ball_dofadr : self.ball_dofadr + 3
            ].copy()
            mujoco.mj_step(self.model, self.data)
            if self.ball_released:
                self._apply_cor_correction(ball_vel_pre)

            c = self._classify_contacts()
            for k in any_contact:
                any_contact[k] |= c[k]

            touching_table_now = c["ball_table"]
            if (
                self.ball_released
                and touching_table_now
                and not self.ball_was_touching_table
            ):
                bounce_events.append(
                    (
                        self._ball_pos()[:2].copy(),
                        float(self._ball_pos()[2]),
                        self._ball_vel().copy(),
                    )
                )
            self.ball_was_touching_table = touching_table_now

        self._update_motion_limits(reward_components, action=action)
        self._update_contacts_and_events(reward_components, bounce_events, any_contact)
        self._add_dense_reward(reward_components, action=action)

    def _apply_cor_correction(self, ball_vel_pre: NDArray[np.float64]) -> None:
        """Walk the contact list after a substep and dial ball-surface bounces
        to realistic ping-pong COR. We replace the ball's outgoing normal-velocity
        with -v_in_normal · COR (table 0.88, cup wall 0.70, water 0), leaving
        tangential velocity untouched.

        - Skip if |v_in_normal| < BOUNCE_NORMAL_VEL_MIN: a ball at rest doesn't
          get its tiny noise velocity flipped, which would otherwise inject
          energy and cause jiggle.
        - Apply at most one correction per substep. Multiple ball-surface
          contacts in a single substep usually mean the solver is double-
          counting a single bounce (e.g. ball touching both cup wall and
          water at the same instant).
        """
        applied = False
        for j in range(self.data.ncon):
            c = self.data.contact[j]
            g1, g2 = int(c.geom1), int(c.geom2)
            if self.ball_geom_id not in (g1, g2):
                continue
            other = g2 if g1 == self.ball_geom_id else g1

            if other == self.table_geom_id:
                cor = COR_TABLE
            elif other == self.water_geom_id:
                cor = COR_WATER
            elif other in self.cup_geom_ids:
                cor = COR_CUP
            else:
                continue

            if applied:
                continue

            normal = c.frame[:3].copy()
            if g1 == self.ball_geom_id:
                normal = -normal  # frame normal points away from ball; flip it.

            v_in_normal = float(np.dot(ball_vel_pre, normal))
            if v_in_normal >= -BOUNCE_NORMAL_VEL_MIN:
                continue

            ball_vel_now = self.data.qvel[
                self.ball_dofadr : self.ball_dofadr + 3
            ]
            v_out_normal = float(np.dot(ball_vel_now, normal))
            target_v_out_normal = -v_in_normal * cor
            v_tangent = ball_vel_now - v_out_normal * normal
            self.data.qvel[
                self.ball_dofadr : self.ball_dofadr + 3
            ] = v_tangent + target_v_out_normal * normal
            applied = True

    def _update_motion_limits(
        self,
        reward_components: dict[str, float],
        action: NDArray[np.float32] | None,
    ) -> None:
        joint_vel = self.data.qvel[self.joint_dofadr].copy()
        joint_acc = (joint_vel - self.prev_joint_vel) / (self.control_steps * self.model.opt.timestep)
        joint_jerk = (joint_acc - self.prev_joint_acc) / (self.control_steps * self.model.opt.timestep)

        vel_norm = float(np.linalg.norm(joint_vel))
        acc_norm = float(np.linalg.norm(joint_acc))
        jerk_norm = float(np.linalg.norm(joint_jerk))
        self.max_joint_vel = max(self.max_joint_vel, vel_norm)
        self.max_joint_acc = max(self.max_joint_acc, acc_norm)
        self.max_joint_jerk = max(self.max_joint_jerk, jerk_norm)

        if action is not None:
            self._add_reward(reward_components, "joint_vel_penalty", -0.01 * vel_norm)
            self._add_reward(reward_components, "joint_acc_penalty", -0.0005 * acc_norm)
            self._add_reward(reward_components, "joint_jerk_penalty", -0.000002 * jerk_norm)

            action_delta = float(np.linalg.norm(action - self.prev_action))
            self._add_reward(reward_components, "action_delta_penalty", -0.005 * action_delta)
            self.prev_action = action.copy()

            if vel_norm > JOINT_VEL_LIMIT:
                self.motion_limit_violated = True
                self._add_reward(reward_components, "joint_vel_limit_penalty", -10.0)
            if acc_norm > JOINT_ACC_LIMIT:
                self.motion_limit_violated = True
                self._add_reward(reward_components, "joint_acc_limit_penalty", -10.0)
            if jerk_norm > JOINT_JERK_LIMIT:
                self.motion_limit_violated = True
                self._add_reward(reward_components, "joint_jerk_limit_penalty", -10.0)

        self.prev_joint_vel = joint_vel
        self.prev_joint_acc = joint_acc

    def _update_contacts_and_events(
        self,
        reward_components: dict[str, float],
        bounce_events: list[tuple[NDArray[np.float64], float, NDArray[np.float64]]],
        any_contact: dict[str, bool],
    ) -> None:
        """Consume per-substep contact data accumulated by ``_advance_control_step``
        and emit rewards / update sticky state. Called once per control step.

        Inputs
        ------
        bounce_events : list of (xy, z, vel) tuples — one per ball-table
            transition (rising edge) detected across all substeps. Position
            and velocity are captured at the substep where the transition
            fired, after COR correction. Length 0 means no bounce this control
            step; >1 means multiple physical bounces happened (skip-bounce).
        any_contact : dict — sticky-OR flags for each ball/robot/surface pair
            across all substeps in this control step. Used for the
            ball_contacted_* / robot_*_contact bookkeeping that xinyi's
            original logic accumulated.
        """
        self.ball_contacted_table = self.ball_contacted_table or any_contact["ball_table"]
        self.ball_contacted_cup = self.ball_contacted_cup or any_contact["ball_cup"]
        # Ball-floor and ball-robot contacts are only meaningful AFTER
        # release. Pre-release the ball is welded to the gripper and any
        # solver-compliance jitter that lets it briefly clip a finger
        # geom (link7/link8) is a physics artifact, not a real failure.
        # Counting those would silently terminate the episode without
        # the -25 invalid_ball_contact_penalty (which is gated on
        # ball_released), giving PPO a credit-assignment hole.
        if self.ball_released:
            self.ball_contacted_floor = self.ball_contacted_floor or any_contact["ball_floor"]
            self.ball_contacted_robot = self.ball_contacted_robot or any_contact["ball_robot"]
        self.ball_touched_water = self.ball_touched_water or any_contact["ball_water"]
        self.robot_table_contact = self.robot_table_contact or any_contact["robot_table"]
        self.robot_cup_contact = self.robot_cup_contact or any_contact["robot_cup"]

        if self.robot_table_contact:
            self._add_reward(reward_components, "robot_table_contact_penalty", -50.0)
        if self.robot_cup_contact:
            self._add_reward(reward_components, "robot_cup_contact_penalty", -50.0)

        if self.ball_released:
            for bounce_xy, _bounce_z, _bounce_vel in bounce_events:
                self.table_bounce_count += 1
                self.bounced = self.table_bounce_count > 0
                if self.table_bounce_count == 1:
                    self.first_table_bounce_xy = bounce_xy
                    self.first_table_bounce_time = float(self.data.time)
                    self._add_reward(
                        reward_components,
                        "valid_table_bounce_bonus",
                        self.reward_weights["bounce"],
                    )
                    bounce_score_value = bounce_score(self.first_table_bounce_xy, self.cup_xy)
                    self._add_reward(
                        reward_components,
                        "bounce_location_reward",
                        self.reward_weights["bounce_xy"] * bounce_score_value,
                    )
                elif self.table_bounce_count == 2:
                    self.second_table_bounce_xy = bounce_xy
                    self.second_table_bounce_cup_dist = float(
                        np.linalg.norm(self.second_table_bounce_xy - self.cup_xy)
                    )
                    second_bounce_score = 1.0 - min(
                        self.second_table_bounce_cup_dist / DISTANCE_REWARD_SCALE,
                        1.0,
                    )
                    # second_bounce_cup_reward weight is 0 — just records metric.
                    # Episode terminates immediately after via _terminal_failure
                    # (table_bounce_count > 1).
                    self._add_reward(
                        reward_components,
                        "second_table_bounce_cup_reward",
                        self.reward_weights["second_bounce_cup"] * second_bounce_score,
                    )
                else:
                    # Bounce 3+ would never fire here normally because of the
                    # bounce > 1 termination, but kept as backstop in case
                    # multiple bounces register in a single substep.
                    self._add_reward(
                        reward_components, "extra_table_bounce_penalty", -5.0
                    )

        if self.ball_released and (any_contact["ball_floor"] or any_contact["ball_robot"]):
            self.invalid_bounce_count += int(any_contact["ball_floor"]) + int(any_contact["ball_robot"])
            self._add_reward(reward_components, "invalid_ball_contact_penalty", -25.0)
        if (
            self.ball_released
            and any_contact["ball_cup"]
            and self.table_bounce_count == 0
        ):
            self.invalid_bounce_count += 1
            self._add_reward(reward_components, "pre_bounce_cup_contact_penalty", -10.0)

    def _add_dense_reward(
        self,
        reward_components: dict[str, float],
        action: NDArray[np.float32] | None,
    ) -> None:
        self._add_reward(reward_components, "time_penalty", -0.01)
        ball_pos = self._ball_pos()
        # 2D distance for cup-XY-footprint checks (success / cup_entry).
        cup_dist_xy = float(np.linalg.norm(ball_pos[:2] - self.cup_xy))
        self.best_final_dist = min(self.best_final_dist, cup_dist_xy)

        # 3D distance to the cup-mouth target (cup XY at z = CUP_HEIGHT - 0.02,
        # i.e. just below the rim where the ball center would be when entering
        # the cup). Pure 2D shaping rewards "ball at cup XY at any z" the same,
        # which lets the policy converge on a "bounce twice next to the cup"
        # local optimum (ball ends up at z=0 next to cup → max 2D shaping
        # reward, no cup_entry). The 3D version pulls the ball *up* toward the
        # cup mouth, breaking that plateau.
        cup_target = np.array(
            [
                self.cup_xy[0],
                self.cup_xy[1],
                self.pedestal_height + CUP_HEIGHT - 0.02,
            ],
            dtype=np.float64,
        )
        cup_target_dist = float(np.linalg.norm(ball_pos - cup_target))

        if self.table_bounce_count >= 1 and not self._has_invalid_ball_contact():
            previous_best = self.closest_post_bounce_cup_dist
            self.closest_post_bounce_cup_dist = min(
                self.closest_post_bounce_cup_dist, cup_target_dist
            )
            if self.closest_post_bounce_cup_dist < previous_best:
                previous_score = 0.0
                if np.isfinite(previous_best):
                    previous_score = float(np.exp(-previous_best / CUP_DIST_REWARD_SCALE))
                cup_dist_score = float(np.exp(-self.closest_post_bounce_cup_dist / CUP_DIST_REWARD_SCALE))
                self._add_reward(
                    reward_components,
                    "post_bounce_cup_distance_reward",
                    self.reward_weights["cup_dist"] * max(cup_dist_score - previous_score, 0.0),
                )

            inside_cup_xy = cup_dist_xy < CUP_RADIUS
            inside_cup_z = (
                self.pedestal_height + 0.015 < ball_pos[2] < self.pedestal_height + CUP_HEIGHT
            )
            if inside_cup_xy and inside_cup_z and not self.ball_entered_cup:
                self.ball_entered_cup = True
                self._add_reward(reward_components, "cup_entry_bonus", self.reward_weights["cup_entry"])

    def _capture_passive_step(self, reward_components: dict[str, float]) -> None:
        if self.render_mode != "rgb_array":
            return
        frame = self.render()
        if frame is not None:
            self.passive_render_frames.append(frame.copy())
        self.passive_info_rows.append(
            {
                "step_count": self.step_count,
                "reward_so_far": float(sum(reward_components.values())),
                "success": self._success(),
                "bounce_count": self.table_bounce_count,
                "table_bounce_count": self.table_bounce_count,
                "invalid_bounce_count": self.invalid_bounce_count,
                "cup_dist": float(np.linalg.norm(self._ball_pos()[:2] - self.cup_xy)),
                "closest_post_bounce_cup_dist": self.closest_post_bounce_cup_dist,
                "second_table_bounce_cup_dist": self.second_table_bounce_cup_dist,
                "ball_released": self.ball_released,
                "ball_entered_cup": self.ball_entered_cup,
                "settled_in_cup": self.settled_in_cup,
                "robot_table_contact": self.robot_table_contact,
                "robot_cup_contact": self.robot_cup_contact,
                "ball_contacted_floor": self.ball_contacted_floor,
                "ball_contacted_robot": self.ball_contacted_robot,
                "ball_touched_water": self.ball_touched_water,
                "max_joint_vel": self.max_joint_vel,
                "max_joint_acc": self.max_joint_acc,
                "max_joint_jerk": self.max_joint_jerk,
                "motion_limit_violated": self.motion_limit_violated,
            }
        )

    def _classify_contacts(self) -> dict[str, bool]:
        contacts = {
            "ball_table": False,
            "ball_cup": False,
            "ball_water": False,
            "ball_floor": False,
            "ball_robot": False,
            "robot_table": False,
            "robot_cup": False,
        }
        for contact_idx in range(self.data.ncon):
            contact = self.data.contact[contact_idx]
            geom1 = int(contact.geom1)
            geom2 = int(contact.geom2)
            body1 = int(self.model.geom_bodyid[geom1])
            body2 = int(self.model.geom_bodyid[geom2])
            geoms = {geom1, geom2}
            bodies = {body1, body2}

            ball_in_contact = self.ball_geom_id in geoms
            if ball_in_contact:
                other_geom = geom2 if geom1 == self.ball_geom_id else geom1
                other_body = body2 if geom1 == self.ball_geom_id else body1
                contacts["ball_table"] = contacts["ball_table"] or other_geom == self.table_geom_id
                contacts["ball_floor"] = contacts["ball_floor"] or other_geom == self.floor_geom_id
                contacts["ball_water"] = contacts["ball_water"] or other_geom == self.water_geom_id
                contacts["ball_cup"] = contacts["ball_cup"] or other_body == self.cup_body_id
                contacts["ball_robot"] = contacts["ball_robot"] or other_body in self.robot_body_ids

            robot_in_contact = bool(bodies & self.robot_body_ids)
            contacts["robot_table"] = contacts["robot_table"] or (
                robot_in_contact and self.table_geom_id in geoms
            )
            contacts["robot_cup"] = contacts["robot_cup"] or (
                robot_in_contact and self.cup_body_id in bodies
            )
        return contacts

    def _has_invalid_ball_contact(self) -> bool:
        return self.ball_contacted_floor or self.ball_contacted_robot

    def _terminal_failure(self) -> bool:
        return (
            self.robot_table_contact
            or self.robot_cup_contact
            # Strict single-bounce rule (xinyi's original). Reverted to this
            # in v16 after multiple soft-constraint experiments (bounce > 4
            # in v9-v14, bounce > 2 + penalty in v15, bounce > 3 graduated
            # in v17) all converged on local optima the policy preferred
            # over single-bounce-into-cup. Strict termination forces the
            # policy to find the clean trajectory or fail.
            or self.table_bounce_count > 1
            or self._has_invalid_ball_contact()
            or self.motion_limit_violated
        )

    def _add_reward(
        self,
        reward_components: dict[str, float],
        name: str,
        value: float,
    ) -> None:
        if value == 0.0:
            return
        reward_components[name] = reward_components.get(name, 0.0) + float(value)

if __name__ == "__main__":
    from gymnasium.utils.env_checker import check_env

    check_env(RageCageEnv(), skip_render_check=True)
    print("RageCageEnv passed Gymnasium env checker")
