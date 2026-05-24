#!/usr/bin/env python3
"""rage_cage_thrower — ROS2 node that runs the trained PPO throw policy.

Layer 2: subscribes to joint feedback + cup pose, drives a 50Hz state
machine (HOMING → SETTLE_HOME → THROWING → SETTLE_RELEASE → IDLE) via
`ThrowController`, streams arm setpoints to `control/move_mit`, and
publishes one-shot gripper events to `control/joint_states`.

Talks to `agx_arm_ros` (https://github.com/agilexrobotics/agx_arm_ros,
ros2 branch) launched via:

  ros2 launch agx_arm_ctrl start_single_agx_arm.launch.py \\
      arm_type:=piper effector_type:=agx_gripper \\
      auto_enable:=true control_enabled:=true

driver topics:
  PUBLISHES: feedback/joint_states  (sensor_msgs/JointState, 7 elements:
                                     joint1..joint6 + gripper)
             feedback/arm_status    (agx_arm_msgs/AgxArmStatus — ctrl_mode,
                                     teach_status, motion_status, err_status)
  SUBSCRIBES: control/move_mit      (agx_arm_msgs/MoveMITMsg, 6 joints —
                                     MIT-mode with per-joint kp/kd matching
                                     the sim training distribution; what
                                     we publish every tick)
              control/joint_states  (sensor_msgs/JointState — gripper-only
                                     one-shot events; routes name='gripper'
                                     to the gripper actuator and ignores
                                     non-arm-joint names)

services we call (typed, not Bool topics like piper_ros used):
  /enable_agx_arm   std_srvs/SetBool — power motors on/off
  /exit_teach_mode  std_srvs/Empty   — flip ctrl_mode 2 → 1 (teach → CAN)
  /control_enable   std_srvs/SetBool — open the /control/* gate

Runs inside the `rage_cage_thrower` container; not importable on the
Mac (rclpy is container-only).

See `docs/plans/2026-05-22-ros2-inference-node-design.md` (note: that
doc still describes the old piper_ros interface; rewrite pending).
"""

from __future__ import annotations

import csv
import random
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np

# Ensure /ws is on sys.path so `real.controller` resolves when this file
# is run directly (`python3 /ws/real/rage_cage_thrower.py`).
_WS = Path(__file__).resolve().parents[1]
if str(_WS) not in sys.path:
    sys.path.insert(0, str(_WS))

import rclpy  # noqa: E402
from geometry_msgs.msg import PoseStamped  # noqa: E402
from rcl_interfaces.msg import ParameterDescriptor, ParameterType  # noqa: E402
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup  # noqa: E402
from rclpy.executors import MultiThreadedExecutor  # noqa: E402
from rclpy.node import Node  # noqa: E402
from rclpy.qos import QoSDurabilityPolicy, QoSProfile  # noqa: E402
from sensor_msgs.msg import JointState  # noqa: E402
from std_msgs.msg import String  # noqa: E402
from std_srvs.srv import Empty, SetBool, Trigger  # noqa: E402

from agx_arm_msgs.msg import MoveMITMsg  # noqa: E402

from real.controller import (  # noqa: E402
    ARM_JOINTS,
    CONTROL_DT,
    GRIPPER_HOLD_M,
    GRIPPER_OPEN_M,
    SIM_CUP_HEIGHT,
    ThrowController,
    ThrowState,
)

MODEL_DIR_DEFAULT = "/ws/models/random_stack_cup_thrower_no_ball_obs_v1"

# agx_arm_ros driver topics. The driver namespaces feedback/* (publisher)
# and control/* (subscriber). Three arm-streaming paths are available:
#   "move_js"  — driver calls agx_arm.move_js(joints), MIT-mode at the CAN
#                level with the SDK's internal default gains. Streaming
#                setpoints, no smoothing. Good when the trained model
#                already matches the SDK's defaults.
#   "move_mit" — driver calls agx_arm.move_mit(per-joint kp/kd/...). Lets
#                us tune ARM_KP/ARM_KD to a specific training distribution.
#   "move_j"   — driver calls agx_arm.move_j(joints), a PLANNED trajectory
#                (smoothed/interpolated to target). Wrong semantics for
#                50Hz streaming — each tick re-queues a new trajectory so
#                the arm stops-and-starts. Included for experimentation
#                only; expect it to fail catastrophically during THROWING.
# control/joint_states routes through fast_mode (false=move_j, true=move_js)
# but we don't use it for arm commands; only as the gripper-events channel.
ARM_FEEDBACK_TOPIC = "feedback/joint_states"
ARM_TOPIC_MOVE_JS = "control/move_js"
ARM_TOPIC_MOVE_MIT = "control/move_mit"
ARM_TOPIC_MOVE_J = "control/move_j"
GRIPPER_COMMAND_TOPIC = "control/joint_states"
CUP_POSE_TOPIC = "/cup_pose"

# Default arm command mode. Override at launch with
# `--ros-args -p arm_command_mode:=move_mit`. NOT live-reloadable: the
# publisher type is bound to the message type, so changes require a
# node restart.
ARM_COMMAND_MODE_DEFAULT = "move_js"

# Per-joint MIT-mode gains. Empirically: sim values (sim/mjcf/agilex_piper/
# piper.xml:262-268, kp=80/80/80/40/10/10) caused aggressive vibration on
# the real arm. kp=8 (sim/2.5) was still too high. Settled on kp in the
# 1-3 range — units almost certainly don't transfer 1:1 from MuJoCo.
# Ratio across joints preserved from sim: joint1-3 high, joint4 mid,
# joint5-6 low. kd left at sim values pending more tuning.
# Joint order: joint1..joint6.
ARM_KP = (10.0, 10.0, 10.0, 10.0, 10.0, 10.0)
ARM_KD = (0.5, 0.5, 0.5, 0.5, 0.5, 0.5)
# Velocity reference and torque feedforward sent every tick. v_des is a
# small forward bias; torque adds a constant feedforward term.
ARM_V_DES = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
ARM_TORQUE_FF = (0.1, 0.1, 0.1, 0.1, 0.1, 0.1)

# Default gripper closing force (Newtons). agx_arm_ctrl reads this from
# msg.effort[i]; matches the launch-time `gripper_default_effort` (1.0).
GRIPPER_DEFAULT_FORCE_N = 1.0

# Per-throw CSV logs land here. Bind-mounted from <repo>/throw_logs on
# the host so they're visible without docker cp. Created on first throw.
LOG_DIR = Path("/ws/throw_logs")

# How many internal 50Hz ticks to skip between actual publishes to
# control/move_mit. 1 = publish every tick (50Hz, normal). N = publish
# every N-th tick (50/N Hz). Live-tunable via the `arm_publish_decimation`
# ROS param. Decimation only affects the arm command stream — gripper
# events are already on-change-only and not throttled.
#
# Alternatively, set `arm_publish_hz` directly (e.g., 10.0) and we'll
# compute the matching decimation. Hz takes priority when > 0.
ARM_PUBLISH_DECIMATION_DEFAULT = 1
ARM_PUBLISH_HZ_DEFAULT = 0.0  # 0 = use decimation param instead
INTERNAL_TICK_HZ = 50  # matches CONTROL_DT = 0.02 in controller.py
REPLAY_HZ_DEFAULT = 50.0  # 50 = play recorded trajectory at native rate

# Default trajectory row index at which to fire the gripper release in
# REPLAY. 33 (in a 45-row throw recording) was empirically the best
# release timing on the real arm; the default -1 (= last row) was
# always too late.
REPLAY_RELEASE_ROW_DEFAULT = 33

# Replay lookup mode: where to find the recordings catalogue, and how
# far (Euclidean, meters in cup top-of-cup XYZ space) we'll allow a
# match to be from the requested cup pose before refusing the throw.
RECORDINGS_MANIFEST_PATH = Path("/ws/recordings/manifest.csv")
REPLAY_MAX_DISTANCE_M_DEFAULT = 0.30
REPLAY_MODE_MANUAL = "manual"
REPLAY_MODE_LOOKUP = "lookup"
REPLAY_MODE_DEFAULT = REPLAY_MODE_MANUAL
REPLAY_FREEZE_JOINTS_DEFAULT: list[int] = []  # 1-indexed; empty = freeze nothing

# Dance-mode defaults. /dance_throw chains three controller sub-cycles:
# (A) HOMING→SETTLE_HOME→DANCE — the joint1 sine sweep that lands at a
#     randomly-picked target angle from `dance_target_angles_deg`.
# (B) HOMING→SETTLE_HOME→REPLAY→SETTLE_RELEASE — existing manual replay
#     of `replay_path`, with joint1_override set to the dance target so
#     the recorded throw is retargeted to the picked angle.
# (C) HOMING→SETTLE_HOME — return to HOME_QPOS (auto-home after release).
# Duration is derived from `DANCE_NUM_SWEEPS * dance_sweep_period_s` —
# one "sweep" = one full sine cycle. The angle pool and sweep period are
# ROS-tunable so cup layouts and rhythm can be swapped without code edits.
DANCE_NUM_SWEEPS = 3
DANCE_SWEEP_AMPLITUDE_DEG = 15.0
DANCE_TARGET_ANGLES_DEG_DEFAULT: list[float] = [-6.0, 0.0, 8.0, 15.0]
DANCE_SWEEP_PERIOD_S_DEFAULT = 2.0


# ---------- recordings-manifest helpers (module-level, no rclpy) ----------


def load_recordings_manifest(
    manifest_path: Path,
) -> list[tuple[float, float, float, Path]]:
    """Parse the recordings manifest into [(cup_x, cup_y, top_z, abs_path)].

    The manifest has pedestal_height (cup base above table); we convert to
    top-of-cup z by adding SIM_CUP_HEIGHT so distances are computed in the
    same space as /cup_pose values.
    """
    out: list[tuple[float, float, float, Path]] = []
    recordings_dir = manifest_path.parent
    with manifest_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            cup_x = float(row["cup_x"])
            cup_y = float(row["cup_y"])
            pedestal_h = float(row["pedestal_height"])
            top_z = pedestal_h + SIM_CUP_HEIGHT
            traj_path = recordings_dir / row["filename"]
            out.append((cup_x, cup_y, top_z, traj_path))
    if not out:
        raise ValueError(f"empty manifest at {manifest_path}")
    return out


def find_closest_recording(
    cup_xyz: tuple[float, float, float] | np.ndarray,
    manifest: list[tuple[float, float, float, Path]],
) -> tuple[Path, float]:
    """Return (path_to_closest_recording, euclidean_distance_meters).

    cup_xyz is top-of-cup XYZ in the arm-base frame, matching /cup_pose.
    """
    target = np.asarray(cup_xyz, dtype=np.float64)
    if target.shape != (3,):
        raise ValueError(f"cup_xyz must be a 3-vector, got shape {target.shape}")
    best_path: Path | None = None
    best_dist = float("inf")
    for cup_x, cup_y, top_z, path in manifest:
        d = float(
            np.linalg.norm(target - np.array([cup_x, cup_y, top_z]))
        )
        if d < best_dist:
            best_dist = d
            best_path = path
    assert best_path is not None  # manifest is non-empty per load_recordings_manifest
    return best_path, best_dist

# Reject a /throw_trigger if joint feedback is older than this (also
# used to abort mid-cycle if the stream drops).
JOINT_STATE_STALE_S = 0.1


class RageCageThrower(Node):
    def __init__(self) -> None:
        super().__init__("rage_cage_thrower")

        self.declare_parameter("model_dir", MODEL_DIR_DEFAULT)
        # Arm command mode: "move_js" (SDK default gains, simpler, default)
        # or "move_mit" (user-specified per-joint kp/kd/v_des/torque).
        # Bound at init — see ARM_COMMAND_MODE_DEFAULT.
        self.declare_parameter("arm_command_mode", ARM_COMMAND_MODE_DEFAULT)
        # Path to a recording CSV for /replay_trajectory. Relative paths
        # resolve from /ws. Set via launch arg or `ros2 param set` before
        # calling the service.
        self.declare_parameter("replay_path", "")
        # Replay frame rate. 50 = play at native recorded rate (45 ticks
        # in 0.9s). Lower values stretch wall-clock time by holding each
        # trajectory row for round(50/replay_hz) internal 50Hz ticks; all
        # rows are still visited. NOTE: slow-motion replay won't actually
        # throw the ball — the recorded velocity profile only releases
        # correctly at the native rate. Useful for visual debugging only.
        # Live-tunable via `ros2 param set /rage_cage_thrower replay_hz N`.
        self.declare_parameter("replay_hz", REPLAY_HZ_DEFAULT)
        # Replay row index at which to open the gripper (0-based). -1 =
        # open on the last row (matches throw-style behavior). Default 33
        # is the empirically-best release row on the real arm for the
        # 45-row recordings. The arm continues following the rest of the
        # recorded trajectory after release so the follow-through plays
        # out.
        self.declare_parameter(
            "replay_release_row", REPLAY_RELEASE_ROW_DEFAULT,
        )
        # Replay mode: "manual" (use replay_path) or "lookup" (pick the
        # closest recording to /cup_pose from /ws/recordings/manifest.csv).
        self.declare_parameter("replay_mode", REPLAY_MODE_DEFAULT)
        # Maximum allowed Euclidean distance (meters, in cup top-of-cup
        # XYZ space) between requested cup pose and the nearest recording.
        # If the closest recording exceeds this, lookup mode refuses the
        # throw instead of replaying a bad-fit trajectory.
        self.declare_parameter(
            "replay_max_distance_m", REPLAY_MAX_DISTANCE_M_DEFAULT,
        )
        # 1-indexed joint numbers (1..6) to freeze during replay — those
        # joints stay at their pre-replay pose throughout HOMING + REPLAY +
        # SETTLE_RELEASE. Empty default = all 6 joints follow the recording.
        # Example: `ros2 param set /rage_cage_thrower replay_freeze_joints "[1]"`
        # freezes the base rotation, leaving joints 2..6 to follow the throw.
        # The explicit INTEGER_ARRAY descriptor is load-bearing: an empty
        # default list `[]` would otherwise be inferred as BYTE_ARRAY by
        # rclpy and reject integer sets via `ros2 param set`.
        self.declare_parameter(
            "replay_freeze_joints",
            REPLAY_FREEZE_JOINTS_DEFAULT,
            ParameterDescriptor(type=ParameterType.PARAMETER_INTEGER_ARRAY),
        )
        # Absolute joint1 target during replay, in DEGREES. Default NaN =
        # follow the recorded joint1 column normally. Set to e.g. 30.0 to
        # pin joint1 at +30° throughout HOMING/REPLAY/SETTLE_RELEASE —
        # useful for retargeting a known-good trajectory's base rotation.
        # Wins over `replay_freeze_joints=[1]` when both are set.
        self.declare_parameter(
            "replay_joint1_override_deg", float("nan"),
        )
        # Publish-rate decimation: 1 = publish every 50Hz tick, N = every
        # N-th tick (effective 50/N Hz). Live-tunable via
        # `ros2 param set /rage_cage_thrower arm_publish_decimation N`.
        self.declare_parameter(
            "arm_publish_decimation", ARM_PUBLISH_DECIMATION_DEFAULT,
        )
        # Alternative: specify publish frequency directly in Hz. > 0 takes
        # priority over arm_publish_decimation; rounds to nearest integer
        # decimation (so actual rate may differ from request for non-divisors
        # of 50). Live-tunable.
        self.declare_parameter(
            "arm_publish_hz", ARM_PUBLISH_HZ_DEFAULT,
        )
        # Dance-mode params. The angle pool is the set of base-rotation
        # targets the dance picks from uniformly at random; matches the
        # angles where cups are placed. Explicit DOUBLE_ARRAY descriptor
        # is load-bearing for the same reason as replay_freeze_joints —
        # an empty default would otherwise be misinferred as BYTE_ARRAY.
        self.declare_parameter(
            "dance_target_angles_deg",
            DANCE_TARGET_ANGLES_DEG_DEFAULT,
            ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE_ARRAY),
        )
        # Seconds per full sine cycle during the dance. 2.0 gives ~3-5
        # full sweeps in a 7-10s dance, which reads as "deliberate" to
        # an audience. Live-tunable.
        self.declare_parameter(
            "dance_sweep_period_s", DANCE_SWEEP_PERIOD_S_DEFAULT,
        )

        model_dir = Path(self.get_parameter("model_dir").value)
        self.get_logger().info(f"loading policy from {model_dir}")
        self.controller = ThrowController(model_dir=model_dir)

        # Cached inputs (written by callbacks, read by timer + service).
        self._latest_q: np.ndarray | None = None
        self._latest_qdot: np.ndarray | None = None
        self._latest_q_stamp = None
        self._latest_cup_pose: np.ndarray | None = None  # [x, y, z]
        self._cycle_done = threading.Event()
        # Track the last gripper opening we published so we don't spam
        # control/joint_states at 50Hz with redundant gripper commands.
        # In a normal cycle the gripper publishes ~3 times: HOLD on cycle
        # start, OPEN on release tick, and once on settle.
        self._last_gripper_position: float | None = None

        # CSV log buffer. Populated only between _init_throw_log() (called
        # at /throw_trigger) and _flush_throw_log() (next IDLE tick).
        # Outside a logged throw cycle, _log_meta is None and the timer
        # does no I/O beyond a single None-check.
        self._log_buffer: list[list] = []
        self._log_meta: dict | None = None
        self._log_cycle_start_time: float | None = None

        # Counter for arm-publish decimation. Increments every tick that
        # _publish_joint_command runs; only ticks where (count % decim == 0)
        # actually publish to control/move_mit.
        self._arm_publish_tick = 0
        self._last_arm_publish_was_sent = False

        # /cup_pose is latched so a late-starting node still sees the
        # last published value from the perception node.
        latched_qos = QoSProfile(
            depth=1, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
        )

        # Callback groups: the service handler blocks for up to 15s
        # waiting on the cycle. If it shared a callback group with the
        # timer or subscribers, the executor's MutuallyExclusive default
        # would block them too — the timer would never fire and the
        # cycle would never advance. Splitting service into its own
        # group lets it block while io_cbg keeps timer + subs running.
        self._io_cbg = MutuallyExclusiveCallbackGroup()
        self._service_cbg = MutuallyExclusiveCallbackGroup()

        self.create_subscription(
            JointState, ARM_FEEDBACK_TOPIC, self._on_joint_states, 10,
            callback_group=self._io_cbg,
        )
        self.create_subscription(
            PoseStamped, CUP_POSE_TOPIC, self._on_cup_pose, latched_qos,
            callback_group=self._io_cbg,
        )

        # Arm publisher type/topic depends on the command mode. move_js
        # and move_j both use JointState (same message format), only the
        # topic differs and downstream the SDK picks streaming-MIT vs
        # planned-trajectory. move_mit uses MoveMITMsg with explicit gains.
        self._arm_command_mode = str(
            self.get_parameter("arm_command_mode").value
        ).lower()
        if self._arm_command_mode == "move_js":
            self.arm_cmd_pub = self.create_publisher(
                JointState, ARM_TOPIC_MOVE_JS, 10,
            )
        elif self._arm_command_mode == "move_j":
            self.arm_cmd_pub = self.create_publisher(
                JointState, ARM_TOPIC_MOVE_J, 10,
            )
        elif self._arm_command_mode == "move_mit":
            self.arm_cmd_pub = self.create_publisher(
                MoveMITMsg, ARM_TOPIC_MOVE_MIT, 10,
            )
        else:
            raise ValueError(
                f"unknown arm_command_mode {self._arm_command_mode!r}; "
                "expected 'move_js', 'move_j', or 'move_mit'"
            )
        self.gripper_cmd_pub = self.create_publisher(
            JointState, GRIPPER_COMMAND_TOPIC, 10,
        )
        self.state_pub = self.create_publisher(String, "/throw_state", 10)

        self.create_service(
            Trigger, "/throw_trigger", self._on_throw_trigger,
            callback_group=self._service_cbg,
        )
        self.create_service(
            Trigger, "/home_arm", self._on_home_arm,
            callback_group=self._service_cbg,
        )
        self.create_service(
            Trigger, "/open_gripper", self._on_open_gripper,
            callback_group=self._service_cbg,
        )
        self.create_service(
            Trigger, "/close_gripper", self._on_close_gripper,
            callback_group=self._service_cbg,
        )
        self.create_service(
            Trigger, "/enable_arm", self._on_enable_arm,
            callback_group=self._service_cbg,
        )
        self.create_service(
            Trigger, "/disable_arm", self._on_disable_arm,
            callback_group=self._service_cbg,
        )
        self.create_service(
            Trigger, "/replay_trajectory", self._on_replay_trajectory,
            callback_group=self._service_cbg,
        )
        self.create_service(
            Trigger, "/dance_throw", self._on_dance_throw,
            callback_group=self._service_cbg,
        )

        # Service clients for driver-side control. The agx_arm_ros driver
        # boots in teach mode (ctrl_mode=2); /enable_agx_arm powers the
        # motors and /exit_teach_mode flips ctrl_mode 2 → 1 (CAN command).
        # /control_enable opens the /control/* gate (no-op if launched
        # with control_enabled:=true). All three live on _io_cbg so their
        # response futures resolve while _on_enable_arm polls them from
        # the service group.
        self._enable_client = self.create_client(
            SetBool, "enable_agx_arm", callback_group=self._io_cbg,
        )
        self._exit_teach_client = self.create_client(
            Empty, "exit_teach_mode", callback_group=self._io_cbg,
        )
        self._control_gate_client = self.create_client(
            SetBool, "control_enable", callback_group=self._io_cbg,
        )

        self.timer = self.create_timer(
            CONTROL_DT, self._on_timer, callback_group=self._io_cbg,
        )
        self.get_logger().info(
            f"rage_cage_thrower ready (arm_command_mode={self._arm_command_mode})"
        )

    # ---------- subscriber callbacks ----------

    def _on_joint_states(self, msg: JointState) -> None:
        try:
            idx = [msg.name.index(j) for j in ARM_JOINTS]
        except ValueError as e:
            self.get_logger().warn(
                f"missing arm joint in {ARM_FEEDBACK_TOPIC}: {e}",
                throttle_duration_sec=2.0,
            )
            return
        self._latest_q = np.array(msg.position, dtype=np.float32)[idx]
        # Driver publishes velocity for arm joints (indices 0-5) but not
        # gripper; safe to index by the same idx because gripper is always
        # the 7th name. If velocity is empty (some drivers omit it),
        # fall back to zeros — joint_vel doesn't affect the policy much
        # for a single-tick read.
        if len(msg.velocity) >= 6:
            self._latest_qdot = np.array(msg.velocity, dtype=np.float32)[idx]
        else:
            self._latest_qdot = np.zeros(6, dtype=np.float32)
        self._latest_q_stamp = self.get_clock().now()

    def _on_cup_pose(self, msg: PoseStamped) -> None:
        # TODO(perception): verify frame_id is the arm-base frame; transform
        # via tf2 if not. For v0 we trust the perception node.
        self._latest_cup_pose = np.array(
            [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z],
            dtype=np.float32,
        )

    # ---------- service handlers ----------

    def _on_throw_trigger(self, request, response: Trigger.Response) -> Trigger.Response:
        err = self._preflight_throw()
        if err is not None:
            self.get_logger().warn(f"throw rejected: {err}")
            response.success = False
            response.message = err
            return response

        assert self._latest_q is not None
        self.get_logger().info("throw trigger accepted; starting cycle")
        self._cycle_done.clear()
        self._arm_publish_tick = 0
        self._init_throw_log()
        self.controller.start_cycle(current_joint_pos=self._latest_q)

        # Block here until the timer thread sets _cycle_done.
        # 15s is generous: 2s home + 0.3s settle + 0.9s throw + 1s settle = ~4.2s.
        if not self._cycle_done.wait(timeout=15.0):
            self.controller.state = ThrowState.IDLE  # crude abort
            response.success = False
            response.message = "cycle timeout (>15s)"
            return response

        response.success = True
        response.message = "throw cycle complete"
        return response

    def _on_home_arm(self, request, response: Trigger.Response) -> Trigger.Response:
        err = self._preflight_motion()
        if err is not None:
            self.get_logger().warn(f"home rejected: {err}")
            response.success = False
            response.message = err
            return response

        assert self._latest_q is not None
        self.get_logger().info("home_arm trigger accepted")
        self._cycle_done.clear()
        self._arm_publish_tick = 0
        self.controller.start_home_only(current_joint_pos=self._latest_q)

        # HOMING (≤2s, plus auto-stretch if far) + SETTLE_HOME (0.3s).
        # 10s timeout covers the worst case from a fully-extended start.
        if not self._cycle_done.wait(timeout=10.0):
            self.controller.state = ThrowState.IDLE
            response.success = False
            response.message = "home cycle timeout (>10s)"
            return response

        response.success = True
        response.message = "home reached"
        return response

    def _on_open_gripper(self, request, response: Trigger.Response) -> Trigger.Response:
        return self._gripper_command(GRIPPER_OPEN_M, "opened", response)

    def _on_close_gripper(self, request, response: Trigger.Response) -> Trigger.Response:
        return self._gripper_command(GRIPPER_HOLD_M, "closed", response)

    def _gripper_command(
        self, target: float, label: str, response: Trigger.Response,
    ) -> Trigger.Response:
        err = self._preflight_motion()
        if err is not None:
            response.success = False
            response.message = err
            return response

        assert self._latest_q is not None
        # Hold the arm at its current pose; only the gripper changes.
        self._publish_joint_command(self._latest_q.copy(), target)
        # Driver-side gripper actuator takes ~0.5s to reach commanded pos.
        time.sleep(0.5)
        response.success = True
        response.message = f"gripper {label}"
        return response

    def _on_enable_arm(self, request, response: Trigger.Response) -> Trigger.Response:
        if self.controller.state != ThrowState.IDLE:
            response.success = False
            response.message = "throw or home in progress"
            return response

        # Three calls because the driver doesn't auto-exit teach on enable
        # and gates /control/* behind a separate service. Short-circuit on
        # the first failure.
        err = (
            self._call_set_bool(self._enable_client, "enable_agx_arm", True)
            or self._call_empty(self._exit_teach_client, "exit_teach_mode")
            or self._call_set_bool(self._control_gate_client, "control_enable", True)
        )
        if err is not None:
            self.get_logger().warn(f"enable failed: {err}")
            response.success = False
            response.message = err
            return response

        response.success = True
        response.message = "arm enabled, teach exited, control gate open"
        return response

    def _on_disable_arm(self, request, response: Trigger.Response) -> Trigger.Response:
        if self.controller.state != ThrowState.IDLE:
            response.success = False
            response.message = "throw or home in progress"
            return response

        err = self._call_set_bool(self._enable_client, "enable_agx_arm", False)
        if err is not None:
            self.get_logger().warn(f"disable failed: {err}")
            response.success = False
            response.message = err
            return response

        response.success = True
        response.message = "arm disabled"
        return response

    def _on_replay_trajectory(
        self, request, response: Trigger.Response,
    ) -> Trigger.Response:
        """Open-loop replay of a recorded sim trajectory.

        Two modes (controlled by the `replay_mode` ROS param):
          - "manual": load the CSV at `replay_path` directly.
          - "lookup": read `/cup_pose`, find the closest recording in
                     /ws/recordings/manifest.csv by Euclidean distance,
                     and replay that. Refuses if no recording is within
                     `replay_max_distance_m` of the requested cup.

        Either way: HOMING → SETTLE_HOME → REPLAY → SETTLE_RELEASE. No
        policy in the loop — the recorded targets stream straight to the
        driver every tick.
        """
        mode = str(self.get_parameter("replay_mode").value).lower()
        if mode == REPLAY_MODE_LOOKUP:
            err = self._preflight_throw()  # also requires cup_pose
            if err is not None:
                self.get_logger().warn(f"replay (lookup) rejected: {err}")
                response.success = False
                response.message = err
                return response
            assert self._latest_cup_pose is not None
            try:
                manifest = load_recordings_manifest(RECORDINGS_MANIFEST_PATH)
            except Exception as e:
                response.success = False
                response.message = f"failed to load manifest: {e}"
                return response
            path, distance = find_closest_recording(
                self._latest_cup_pose, manifest,
            )
            max_dist = float(self.get_parameter("replay_max_distance_m").value)
            if distance > max_dist:
                msg = (
                    f"no recording within {max_dist:.3f}m of cup at "
                    f"{tuple(self._latest_cup_pose.tolist())}; closest is "
                    f"{path.name} at {distance:.3f}m"
                )
                self.get_logger().warn(msg)
                response.success = False
                response.message = msg
                return response
            self.get_logger().info(
                f"lookup matched {path.name} at distance {distance:.3f}m"
            )
        else:
            err = self._preflight_motion()  # manual: joint feedback only
            if err is not None:
                self.get_logger().warn(f"replay rejected: {err}")
                response.success = False
                response.message = err
                return response
            path_str = str(self.get_parameter("replay_path").value)
            if not path_str:
                response.success = False
                response.message = (
                    "replay_path is empty; set via "
                    "`ros2 param set /rage_cage_thrower replay_path <path>`"
                )
                return response
            path = Path(path_str)
            if not path.is_absolute():
                # Relative paths interpreted under /ws (where bind mounts live).
                path = Path("/ws") / path
            if not path.exists():
                response.success = False
                response.message = f"trajectory file not found: {path}"
                return response
            distance = float("nan")  # not applicable in manual mode

        try:
            trajectory = self._load_trajectory_csv(path)
        except Exception as e:
            response.success = False
            response.message = f"failed to load trajectory: {e}"
            return response

        assert self._latest_q is not None
        replay_hz = float(self.get_parameter("replay_hz").value)
        if replay_hz <= 0:
            response.success = False
            response.message = f"replay_hz must be > 0, got {replay_hz}"
            return response
        hold_ticks = max(round(INTERNAL_TICK_HZ / replay_hz), 1)
        release_row = int(self.get_parameter("replay_release_row").value)
        # Resolve the effective release row for logging (controller does the
        # same clamp internally).
        effective_release_row = release_row
        if effective_release_row < 0 or effective_release_row >= len(trajectory):
            effective_release_row = len(trajectory) - 1
        self.get_logger().info(
            f"replay trigger accepted; {len(trajectory)} ticks from {path} "
            f"(replay_hz={replay_hz:.2f}, hold_ticks={hold_ticks}, "
            f"release_row={effective_release_row})"
        )
        self._cycle_done.clear()
        self._arm_publish_tick = 0
        self._init_throw_log()
        self._log_meta["replay_path"] = str(path)
        self._log_meta["replay_mode"] = mode
        self._log_meta["replay_match_distance_m"] = distance
        self._log_meta["replay_hz"] = replay_hz
        self._log_meta["replay_hold_ticks"] = hold_ticks
        self._log_meta["replay_release_row"] = effective_release_row
        freeze_joints = list(
            self.get_parameter("replay_freeze_joints").value
        )
        self._log_meta["replay_freeze_joints"] = freeze_joints
        # Convert deg → rad for the controller (NaN stays NaN).
        override_deg = float(
            self.get_parameter("replay_joint1_override_deg").value
        )
        override_rad = float(np.deg2rad(override_deg))
        self._log_meta["replay_joint1_override_deg"] = override_deg
        self.controller.start_replay(
            current_joint_pos=self._latest_q,
            trajectory=trajectory,
            hold_ticks=hold_ticks,
            release_row=release_row,
            freeze_joints=freeze_joints,
            joint1_override=override_rad,
        )

        # HOMING (≤2s) + SETTLE_HOME (0.3s) + REPLAY (N * hold * 20ms) +
        # SETTLE_RELEASE (1s). For a 45-tick traj at replay_hz=10, REPLAY
        # alone is 4.5s — scale timeout by hold_ticks so slow-motion
        # doesn't false-abort.
        cycle_timeout = 5.0 + len(trajectory) * hold_ticks * 0.02 + 5.0
        if not self._cycle_done.wait(timeout=cycle_timeout):
            self.controller.state = ThrowState.IDLE
            response.success = False
            response.message = f"replay timeout (>{cycle_timeout:.0f}s)"
            return response

        response.success = True
        response.message = f"replay complete ({len(trajectory)} ticks)"
        return response

    def _on_dance_throw(
        self, request, response: Trigger.Response,
    ) -> Trigger.Response:
        """Demo throw: sweep joint1, land at a random target angle, replay.

        Chains three controller cycles in sequence:
          A) start_dance(target_rad, duration_s, period_s, amplitude_rad)
             — HOMING → SETTLE_HOME → DANCE → IDLE
          B) start_replay(traj, ..., joint1_override=target_rad)
             — HOMING → SETTLE_HOME → REPLAY → SETTLE_RELEASE → IDLE
          C) start_home_only(current_q)
             — HOMING → SETTLE_HOME → IDLE

        Each sub-cycle writes its own throw log (dance_phase meta field
        identifies which one). A failure in any sub-cycle aborts the rest.
        """
        err = self._preflight_motion()  # joint feedback only; no cup_pose
        if err is not None:
            self.get_logger().warn(f"dance_throw rejected: {err}")
            response.success = False
            response.message = err
            return response

        # Validate replay_path up front so a bad file fails before motion.
        path_str = str(self.get_parameter("replay_path").value)
        if not path_str:
            response.success = False
            response.message = (
                "replay_path is empty; set via "
                "`ros2 param set /rage_cage_thrower replay_path <path>`"
            )
            return response
        path = Path(path_str)
        if not path.is_absolute():
            path = Path("/ws") / path
        if not path.exists():
            response.success = False
            response.message = f"trajectory file not found: {path}"
            return response
        try:
            trajectory = self._load_trajectory_csv(path)
        except Exception as e:
            response.success = False
            response.message = f"failed to load trajectory: {e}"
            return response

        # Validate dance params.
        angles_deg = list(self.get_parameter("dance_target_angles_deg").value)
        if not angles_deg:
            response.success = False
            response.message = "dance_target_angles_deg is empty"
            return response
        period_s = float(self.get_parameter("dance_sweep_period_s").value)
        if period_s <= 0:
            response.success = False
            response.message = f"dance_sweep_period_s must be > 0, got {period_s}"
            return response

        # Random pick — target angle only. Duration is deterministic:
        # DANCE_NUM_SWEEPS full sine cycles at the current period.
        target_deg = float(random.choice(angles_deg))
        duration_s = float(DANCE_NUM_SWEEPS * period_s)
        target_rad = float(np.deg2rad(target_deg))
        amplitude_rad = float(np.deg2rad(DANCE_SWEEP_AMPLITUDE_DEG))

        # Existing replay params control phase B's pacing + release timing.
        replay_hz = float(self.get_parameter("replay_hz").value)
        if replay_hz <= 0:
            response.success = False
            response.message = f"replay_hz must be > 0, got {replay_hz}"
            return response
        hold_ticks = max(round(INTERNAL_TICK_HZ / replay_hz), 1)
        release_row = int(self.get_parameter("replay_release_row").value)
        effective_release_row = release_row
        if effective_release_row < 0 or effective_release_row >= len(trajectory):
            effective_release_row = len(trajectory) - 1

        # Log the plan before any motion so judges/operators see the picks.
        self.get_logger().info(
            f"dance_throw triggered: target={target_deg:+.1f}° "
            f"({DANCE_NUM_SWEEPS} sweeps × {period_s:.2f}s = {duration_s:.2f}s)"
        )
        self.get_logger().info(
            f"  trajectory: {path} ({len(trajectory)} ticks, "
            f"release_row={effective_release_row}, replay_hz={replay_hz:.2f})"
        )
        self.get_logger().info(
            "  plan: HOMING→SETTLE_HOME→DANCE → "
            "HOMING→SETTLE_HOME→REPLAY→SETTLE_RELEASE → "
            "HOMING→SETTLE_HOME"
        )

        cycle_start_wall = time.time()
        dance_meta_extras = {
            "dance_target_deg": target_deg,
            "dance_duration_s": duration_s,
            "dance_period_s": period_s,
            "dance_amplitude_deg": DANCE_SWEEP_AMPLITUDE_DEG,
        }

        # ---- Sub-cycle A: dance ----
        assert self._latest_q is not None
        self.get_logger().info(
            f"phase 1/3: dance (sweeping joint1 ±{DANCE_SWEEP_AMPLITUDE_DEG:.0f}° "
            f"for {duration_s:.2f}s, landing at {target_deg:+.1f}°)"
        )
        self._cycle_done.clear()
        self._arm_publish_tick = 0
        self._init_throw_log()
        self._log_meta["dance_phase"] = "dance"
        self._log_meta.update(dance_meta_extras)
        try:
            self.controller.start_dance(
                current_joint_pos=self._latest_q,
                target_rad=target_rad,
                duration_s=duration_s,
                sweep_period_s=period_s,
                amplitude_rad=amplitude_rad,
            )
        except ValueError as e:
            response.success = False
            response.message = f"start_dance rejected: {e}"
            return response
        # HOMING (≤2s) + SETTLE_HOME (0.3s) + DANCE (duration_s) + ~3s slack.
        dance_timeout = 5.0 + duration_s
        if not self._cycle_done.wait(timeout=dance_timeout):
            self.controller.state = ThrowState.IDLE
            self.get_logger().warn(
                f"phase 1/3 timeout (>{dance_timeout:.0f}s) — aborting dance_throw"
            )
            response.success = False
            response.message = f"dance phase timeout (>{dance_timeout:.0f}s)"
            return response
        self.get_logger().info("phase 1/3 complete")

        # ---- Sub-cycle B: replay ----
        assert self._latest_q is not None
        self.get_logger().info(
            f"phase 2/3: replay (joint1 pinned at {target_deg:+.1f}°)"
        )
        self._cycle_done.clear()
        self._arm_publish_tick = 0
        self._init_throw_log()
        self._log_meta["dance_phase"] = "replay"
        self._log_meta.update(dance_meta_extras)
        self._log_meta["replay_path"] = str(path)
        self._log_meta["replay_hz"] = replay_hz
        self._log_meta["replay_hold_ticks"] = hold_ticks
        self._log_meta["replay_release_row"] = effective_release_row
        self.controller.start_replay(
            current_joint_pos=self._latest_q,
            trajectory=trajectory,
            hold_ticks=hold_ticks,
            release_row=release_row,
            freeze_joints=[],
            joint1_override=target_rad,
        )
        # HOMING (≤2s) + SETTLE_HOME (0.3s) + REPLAY (N * hold * 20ms) +
        # SETTLE_RELEASE (1s) + slack.
        replay_timeout = 5.0 + len(trajectory) * hold_ticks * 0.02 + 5.0
        if not self._cycle_done.wait(timeout=replay_timeout):
            self.controller.state = ThrowState.IDLE
            self.get_logger().warn(
                f"phase 2/3 timeout (>{replay_timeout:.0f}s) — aborting dance_throw"
            )
            response.success = False
            response.message = f"replay phase timeout (>{replay_timeout:.0f}s)"
            return response
        self.get_logger().info("phase 2/3 complete")

        # ---- Sub-cycle C: return home ----
        assert self._latest_q is not None
        self.get_logger().info("phase 3/3: return home")
        self._cycle_done.clear()
        self._arm_publish_tick = 0
        self._init_throw_log()
        self._log_meta["dance_phase"] = "return_home"
        self._log_meta.update(dance_meta_extras)
        self.controller.start_home_only(current_joint_pos=self._latest_q)
        home_timeout = 10.0
        if not self._cycle_done.wait(timeout=home_timeout):
            self.controller.state = ThrowState.IDLE
            self.get_logger().warn(
                f"phase 3/3 timeout (>{home_timeout:.0f}s) — aborting dance_throw"
            )
            response.success = False
            response.message = f"return-home phase timeout (>{home_timeout:.0f}s)"
            return response
        self.get_logger().info("phase 3/3 complete")

        elapsed = time.time() - cycle_start_wall
        msg = (
            f"dance_throw complete (target={target_deg:+.1f}°, "
            f"duration={duration_s:.2f}s, total elapsed={elapsed:.1f}s)"
        )
        self.get_logger().info(msg)
        response.success = True
        response.message = msg
        return response

    def _load_trajectory_csv(self, path: Path) -> np.ndarray:
        """Extract q1_target..q6_target columns from a recording CSV.

        The recording format also has q*_actual, gripper_cmd, ball_released
        etc. but we only need the per-tick joint targets — the gripper
        release is handled by the controller's SETTLE_RELEASE transition,
        and ball_released is informational.
        """
        with path.open() as f:
            reader = csv.DictReader(f)
            rows: list[list[float]] = []
            for row in reader:
                try:
                    rows.append(
                        [float(row[f"q{i}_target"]) for i in range(1, 7)]
                    )
                except KeyError as e:
                    raise ValueError(f"missing column {e} in {path}")
                except ValueError as e:
                    raise ValueError(f"non-numeric value in {path}: {e}")
        if not rows:
            raise ValueError(f"empty trajectory in {path}")
        return np.array(rows, dtype=np.float32)

    # ---------- 50Hz timer ----------

    def _on_timer(self) -> None:
        if self.controller.state == ThrowState.IDLE:
            # Flush any pending throw log (cycle finished cleanly OR was
            # aborted via state=IDLE assignment from a service or earlier
            # timer tick). Idempotent if no log is pending.
            if self._log_meta is not None:
                self._flush_throw_log()
            return

        # Joint feedback is required in every cycle state.
        if (
            self._latest_q is None
            or self._latest_qdot is None
            or self._latest_q_stamp is None
        ):
            self.get_logger().error("joint feedback missing mid-cycle — aborting")
            self.controller.state = ThrowState.IDLE
            self._cycle_done.set()
            return
        age_s = (self.get_clock().now() - self._latest_q_stamp).nanoseconds * 1e-9
        if age_s > JOINT_STATE_STALE_S:
            self.get_logger().error(
                f"joint feedback stale by {age_s:.3f}s mid-cycle — aborting",
            )
            self.controller.state = ThrowState.IDLE
            self._cycle_done.set()
            return

        # cup_pose is only required during THROWING (the policy uses it then).
        # HOMING / SETTLE_HOME / SETTLE_RELEASE don't read it; pass zeros so
        # controller.tick still has a valid array.
        if self.controller.state == ThrowState.THROWING and self._latest_cup_pose is None:
            self.get_logger().error("cup_pose missing during throw — aborting")
            self.controller.state = ThrowState.IDLE
            self._cycle_done.set()
            return
        cup_top = (
            self._latest_cup_pose
            if self._latest_cup_pose is not None
            else np.zeros(3, dtype=np.float32)
        )

        result = self.controller.tick(
            joint_pos=self._latest_q,
            joint_vel=self._latest_qdot,
            cup_top_xyz=cup_top,
        )

        if result.arm_target is not None:
            self._publish_joint_command(result.arm_target, result.gripper_position)
        self.state_pub.publish(String(data=result.state.name))

        if self._log_meta is not None:
            self._append_log_row(result)

        if result.done:
            self._cycle_done.set()

    def _current_publish_decimation(self) -> int:
        """Resolve the active arm-publish decimation, hz overrides decim."""
        hz = float(self.get_parameter("arm_publish_hz").value)
        if hz > 0:
            return max(round(INTERNAL_TICK_HZ / hz), 1)
        return max(int(self.get_parameter("arm_publish_decimation").value), 1)

    def _publish_joint_command(
        self, arm_target: np.ndarray, gripper_position: float,
    ) -> None:
        """Publish arm + (optionally) gripper setpoints to the driver.

        Arm: publishes a 6-element MoveMITMsg to control/move_mit unless
        decimation skips this tick. The internal state machine and policy
        still run at 50Hz; decimation only thins what reaches the driver,
        so the MIT controller holds the last commanded target between
        publishes.
        Gripper: only republishes when gripper_position changed since the
        last call, to control/joint_states. The driver's
        _joint_states_callback routes name='gripper' to the gripper
        actuator and ignores entries that don't match arm-joint names,
        so this won't disturb the arm's MIT-mode streaming.
        """
        decim = self._current_publish_decimation()
        published = self._arm_publish_tick % decim == 0
        if published:
            if self._arm_command_mode in ("move_js", "move_j"):
                arm_msg = JointState()
                arm_msg.header.stamp = self.get_clock().now().to_msg()
                arm_msg.name = list(ARM_JOINTS)
                arm_msg.position = arm_target.tolist()
            else:  # move_mit
                arm_msg = MoveMITMsg()
                # joint_index is 1-indexed in the SDK (joint1=1 .. joint6=6).
                # Driver validator: "Joint index should be [1, 2, 3, 4, 5, 6]".
                arm_msg.joint_index = list(range(1, 7))
                arm_msg.p_des = arm_target.tolist()
                arm_msg.v_des = list(ARM_V_DES)
                arm_msg.kp = list(ARM_KP)
                arm_msg.kd = list(ARM_KD)
                arm_msg.torque = list(ARM_TORQUE_FF)
            self.arm_cmd_pub.publish(arm_msg)
        self._arm_publish_tick += 1
        # Stash for the logger so it can mark which ticks actually went out.
        self._last_arm_publish_was_sent = published

        if gripper_position != self._last_gripper_position:
            grip_msg = JointState()
            grip_msg.header.stamp = self.get_clock().now().to_msg()
            grip_msg.name = ["gripper"]
            grip_msg.position = [float(gripper_position)]
            grip_msg.effort = [GRIPPER_DEFAULT_FORCE_N]
            self.gripper_cmd_pub.publish(grip_msg)
            self._last_gripper_position = gripper_position

    # ---------- throw logging ----------

    def _init_throw_log(self) -> None:
        """Snapshot config and clear the per-tick buffer. Called when
        /throw_trigger is accepted, just before start_cycle. Subsequent
        timer ticks fill _log_buffer; the first IDLE tick after the
        cycle ends (success or abort) flushes to disk.
        """
        cup = (
            self._latest_cup_pose.tolist()
            if self._latest_cup_pose is not None else None
        )
        self._log_meta = {
            "timestamp_iso": datetime.now().isoformat(timespec="seconds"),
            "cup_xyz": cup,
            "model_dir": str(self.controller._model_dir),
            "arm_command_mode": self._arm_command_mode,
            "arm_kp": list(ARM_KP),
            "arm_kd": list(ARM_KD),
            "arm_v_des": list(ARM_V_DES),
            "arm_torque_ff": list(ARM_TORQUE_FF),
            "arm_publish_decimation": int(
                self.get_parameter("arm_publish_decimation").value
            ),
            "arm_publish_hz": float(
                self.get_parameter("arm_publish_hz").value
            ),
            "arm_publish_decimation_effective": self._current_publish_decimation(),
        }
        self._log_buffer = []
        self._log_cycle_start_time = time.time()

    def _append_log_row(self, result) -> None:
        """Buffer one tick of state. NaN fills where data isn't present
        (e.g., raw `action` is only set during THROWING)."""
        if self._log_cycle_start_time is None:
            return
        nan6 = [float("nan")] * 6
        jp = self._latest_q.tolist() if self._latest_q is not None else nan6
        jv = self._latest_qdot.tolist() if self._latest_qdot is not None else nan6
        act = result.action.tolist() if result.action is not None else nan6
        tgt = result.arm_target.tolist() if result.arm_target is not None else nan6
        row = [
            self.controller._tick_idx,
            time.time() - self._log_cycle_start_time,
            result.state.name,
            *jp, *jv, *act, *tgt,
            result.gripper_position,
            int(self._last_arm_publish_was_sent),
        ]
        self._log_buffer.append(row)

    def _flush_throw_log(self) -> None:
        """Write _log_buffer to throw_logs/throw_<ts>.csv and reset state."""
        if self._log_meta is None:
            return
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        ts_compact = datetime.fromisoformat(
            self._log_meta["timestamp_iso"]
        ).strftime("%Y%m%d-%H%M%S")
        filename = LOG_DIR / f"throw_{ts_compact}.csv"
        with filename.open("w", newline="") as f:
            f.write(f"# timestamp: {self._log_meta['timestamp_iso']}\n")
            f.write(f"# cup_xyz: {self._log_meta['cup_xyz']}\n")
            f.write(f"# model_dir: {self._log_meta['model_dir']}\n")
            f.write(f"# arm_command_mode: {self._log_meta['arm_command_mode']}\n")
            f.write(f"# arm_kp: {self._log_meta['arm_kp']}\n")
            f.write(f"# arm_kd: {self._log_meta['arm_kd']}\n")
            f.write(f"# arm_v_des: {self._log_meta['arm_v_des']}\n")
            f.write(f"# arm_torque_ff: {self._log_meta['arm_torque_ff']}\n")
            f.write(f"# arm_publish_decimation: {self._log_meta['arm_publish_decimation']}\n")
            f.write(f"# arm_publish_hz: {self._log_meta['arm_publish_hz']}\n")
            f.write(f"# arm_publish_decimation_effective: {self._log_meta['arm_publish_decimation_effective']}\n")
            # Any extra meta keys (added by handlers like /replay_trajectory
            # or /dance_throw) get written as additional `# key: value` lines.
            base_keys = {
                "timestamp_iso", "cup_xyz", "model_dir", "arm_command_mode",
                "arm_kp", "arm_kd", "arm_v_des", "arm_torque_ff",
                "arm_publish_decimation", "arm_publish_hz",
                "arm_publish_decimation_effective",
            }
            for k, v in self._log_meta.items():
                if k not in base_keys:
                    f.write(f"# {k}: {v}\n")
            cols = ["tick", "t_rel", "state"]
            cols += [f"jp{i}" for i in range(1, 7)]
            cols += [f"jv{i}" for i in range(1, 7)]
            cols += [f"act{i}" for i in range(1, 7)]
            cols += [f"tgt{i}" for i in range(1, 7)]
            cols += ["gripper", "pub"]
            writer = csv.writer(f)
            writer.writerow(cols)
            writer.writerows(self._log_buffer)
        self.get_logger().info(
            f"throw log -> {filename} ({len(self._log_buffer)} ticks)"
        )
        self._log_meta = None
        self._log_buffer = []
        self._log_cycle_start_time = None

    # ---------- service-call helpers ----------

    def _call_set_bool(self, client, name: str, data: bool) -> str | None:
        """Call a SetBool service. Returns None on success, error str on failure.

        Polls future.done() with a 5s deadline (mirrors repl.py:_call). The
        client's response future is dispatched on _io_cbg, so polling here
        from the service group doesn't deadlock.
        """
        if not client.wait_for_service(timeout_sec=2.0):
            return f"service /{name} unavailable"
        req = SetBool.Request()
        req.data = data
        future = client.call_async(req)
        deadline = time.time() + 5.0
        while not future.done():
            if time.time() > deadline:
                return f"service /{name} timed out"
            time.sleep(0.05)
        result = future.result()
        if not result.success:
            return f"service /{name} failed: {result.message}"
        return None

    def _call_empty(self, client, name: str) -> str | None:
        """Call an Empty service. Returns None on success, error str on failure.

        Empty has no success field — any non-exception completion is success.
        """
        if not client.wait_for_service(timeout_sec=2.0):
            return f"service /{name} unavailable"
        future = client.call_async(Empty.Request())
        deadline = time.time() + 5.0
        while not future.done():
            if time.time() > deadline:
                return f"service /{name} timed out"
            time.sleep(0.05)
        return None

    # ---------- pre-flight ----------

    def _preflight_throw(self) -> str | None:
        """Pre-flight for /throw_trigger — needs joint feedback AND cup_pose."""
        return (
            self._check_idle()
            or self._check_joint_feedback()
            or self._check_cup_pose()
        )

    def _preflight_motion(self) -> str | None:
        """Pre-flight for /home_arm + gripper services — joint feedback only."""
        return self._check_idle() or self._check_joint_feedback()

    def _check_idle(self) -> str | None:
        if self.controller.state != ThrowState.IDLE:
            return "throw or home in progress"
        return None

    def _check_joint_feedback(self) -> str | None:
        if self._latest_q is None or self._latest_q_stamp is None:
            return f"no {ARM_FEEDBACK_TOPIC} received yet"
        age_s = (self.get_clock().now() - self._latest_q_stamp).nanoseconds * 1e-9
        if age_s > JOINT_STATE_STALE_S:
            return f"{ARM_FEEDBACK_TOPIC} stale by {age_s:.3f}s"
        return None

    def _check_cup_pose(self) -> str | None:
        if self._latest_cup_pose is None:
            return f"no {CUP_POSE_TOPIC} received yet"
        return None


def main() -> None:
    rclpy.init()
    node = RageCageThrower()
    # MultiThreadedExecutor is required so the blocking service callback
    # doesn't starve the 50Hz timer.
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        # Jazzy installs a SIGINT handler in rclpy.init() that shuts down
        # the context for us. Just swallow the KeyboardInterrupt so we
        # don't print a "real" traceback on a clean Ctrl-C.
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        # Guard the shutdown — if the signal handler beat us to it,
        # calling shutdown again raises RCLError ("already called").
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
