"""Shared utilities for Piper MIT-mode system identification.

This file gives sysid_step.py and sysid_ramp.py a single place to:
- own the rclpy node lifecycle and CAN safety on shutdown,
- subscribe to /feedback/joint_states and /feedback/arm_status,
- publish to /control/move_mit and /control/move_j,
- enforce hard runtime safety limits (qdot, q excursion, err_status),
- write timestamped CSV samples,
- prompt the operator interactively before each motion.

Joint indexing convention throughout: **1-based** (joint1..joint6), matching
the Piper MIT-mode CAN protocol. The arm has no joint 0.
"""

from __future__ import annotations

import csv
import os
import signal
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from sensor_msgs.msg import JointState
from agx_arm_msgs.msg import MoveMITMsg, AgxArmStatus


ARM_JOINT_NAMES = [f"joint{i}" for i in range(1, 7)]
NUM_ARM_JOINTS = 6

# SDK validator limits (from pyAgxArm piper driver). Mirrored here so we can
# clamp before sending and avoid noisy warn-spam.
SDK_KP_RANGE = (0.0, 500.0)
SDK_KD_RANGE = (-5.0, 5.0)
SDK_VDES_RANGE = (-45.0, 45.0)
SDK_PDES_RANGE = (-12.5, 12.5)
# Per-joint t_ff limits: J1-3 are larger (high-torque shoulder/elbow motors)
SDK_TFF_RANGE_LARGE = (-32.0, 32.0)  # joints 1-3
SDK_TFF_RANGE_SMALL = (-8.0, 8.0)    # joints 4-6


def tff_range(joint_index_1based: int) -> tuple[float, float]:
    return SDK_TFF_RANGE_LARGE if joint_index_1based in (1, 2, 3) else SDK_TFF_RANGE_SMALL


@dataclass
class SafetyLimits:
    """Hard runtime guards. Tripping any of these aborts the experiment."""
    tau_max: float = 3.0           # |t_ff| ceiling for any single sample (N·m)
    qdot_max: float = 1.0          # |qdot| ceiling on the test joint (rad/s)
    q_excursion_max: float = 0.3   # |q - q_start| ceiling on the test joint (rad)
    err_status_abort: bool = True  # abort on nonzero arm_status.err_status


@dataclass
class HoldGains:
    """PD used for non-test joints during an experiment."""
    kp: float = 20.0
    kd: float = 1.0


@dataclass
class JointSample:
    t_ns: int
    q: list[float]      # length 6 (joint1..joint6), radians
    qd: list[float]     # rad/s
    tau: list[float]    # effort feedback, N·m (driver-reported, see arm_feedback_high_spd 1e-3 scale)


class SysidNode(Node):
    """Common rclpy node for both sysid scripts.

    Holds a live snapshot of joint state, exposes a thread-safe publisher
    for MoveMITMsg, watches arm_status for fault codes, and ensures the arm
    is left in a safe held state on shutdown.
    """

    def __init__(self, node_name: str = "sysid_node", simulate: bool = False) -> None:
        super().__init__(node_name)
        self.simulate = simulate

        sensor_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
        )
        cmd_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
        )

        self._latest_lock = threading.Lock()
        self._latest_joint_state: Optional[JointSample] = None
        self._latest_arm_status: Optional[AgxArmStatus] = None

        self._mit_pub = self.create_publisher(MoveMITMsg, "/control/move_mit", cmd_qos)
        self._mj_pub = self.create_publisher(JointState, "/control/move_j", cmd_qos)

        self.create_subscription(
            JointState, "/feedback/joint_states", self._on_joint_state, sensor_qos
        )
        self.create_subscription(
            AgxArmStatus, "/feedback/arm_status", self._on_arm_status, sensor_qos
        )

        # Recorded at start of each experiment so abort logic and shutdown
        # cleanup know what "safe hold" looks like.
        self._safe_hold_pose: Optional[list[float]] = None
        self._shutdown_called = False

    # ---------- subscriptions ----------

    def _on_joint_state(self, msg: JointState) -> None:
        name_to_idx = {n: i for i, n in enumerate(msg.name)}
        q = [0.0] * NUM_ARM_JOINTS
        qd = [0.0] * NUM_ARM_JOINTS
        tau = [0.0] * NUM_ARM_JOINTS
        for j, name in enumerate(ARM_JOINT_NAMES):
            i = name_to_idx.get(name)
            if i is None:
                return  # incomplete frame; ignore
            q[j] = msg.position[i] if i < len(msg.position) else 0.0
            qd[j] = msg.velocity[i] if i < len(msg.velocity) else 0.0
            tau[j] = msg.effort[i] if i < len(msg.effort) else 0.0
        t_ns = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec
        with self._latest_lock:
            self._latest_joint_state = JointSample(t_ns=t_ns, q=q, qd=qd, tau=tau)

    def _on_arm_status(self, msg: AgxArmStatus) -> None:
        with self._latest_lock:
            self._latest_arm_status = msg

    def latest_state(self) -> Optional[JointSample]:
        with self._latest_lock:
            return self._latest_joint_state

    def latest_arm_status(self) -> Optional[AgxArmStatus]:
        with self._latest_lock:
            return self._latest_arm_status

    def wait_for_state(self, timeout: float = 5.0) -> JointSample:
        """Block until at least one full joint_states frame has been received."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            s = self.latest_state()
            if s is not None:
                return s
            rclpy.spin_once(self, timeout_sec=0.05)
        raise RuntimeError("Timed out waiting for /feedback/joint_states")

    # ---------- publishers ----------

    def publish_mit(
        self,
        joint_index: Sequence[int],
        p_des: Sequence[float],
        v_des: Sequence[float],
        kp: Sequence[float],
        kd: Sequence[float],
        t_ff: Sequence[float],
    ) -> None:
        """Publish a single MoveMITMsg. All arrays must be the same length.
        No-op in simulate mode."""
        if self.simulate:
            return
        msg = MoveMITMsg()
        msg.joint_index = [int(j) for j in joint_index]
        msg.p_des = [float(v) for v in p_des]
        msg.v_des = [float(v) for v in v_des]
        msg.kp = [float(v) for v in kp]
        msg.kd = [float(v) for v in kd]
        msg.torque = [float(v) for v in t_ff]
        self._mit_pub.publish(msg)

    def publish_hold_all(self, hold_pose: Sequence[float], hold: HoldGains) -> None:
        """One-shot move_mit telling every joint to hold the given pose with the
        firmware's PD. The motor will keep tracking this without further
        publishes until a new move_mit overrides it."""
        n = NUM_ARM_JOINTS
        self.publish_mit(
            joint_index=list(range(1, n + 1)),
            p_des=list(hold_pose),
            v_des=[0.0] * n,
            kp=[hold.kp] * n,
            kd=[hold.kd] * n,
            t_ff=[0.0] * n,
        )

    def move_j(self, target: Sequence[float]) -> None:
        """Send a /control/move_j position command (firmware-planned motion).
        Uses the firmware's trapezoidal planner — fast even at low speed_percent.
        Prefer move_to_mit for sysid setup. No-op in simulate mode."""
        if self.simulate:
            return
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = list(ARM_JOINT_NAMES)
        msg.position = [float(v) for v in target]
        self._mj_pub.publish(msg)

    def move_to_mit(
        self,
        target: Sequence[float],
        v_des: float = 0.01,
        kp: float = 10.0,
        kd: float = 0.5,
        rate_hz: float = 50.0,
        settle_s: float = 0.5,
        max_duration_s: float = 600.0,
    ) -> bool:
        """Software-interpolated MIT-mode position move at `v_des` rad/s.

        Each joint travels at speed `v_des` in the direction of its delta;
        the per-joint trajectory ends when that joint reaches its target.
        The full move ends when the longest-delta joint finishes plus
        `settle_s` for the firmware PD to converge.

        At v_des=0.01 rad/s, a 1 rad reposition takes ~100 seconds. Set
        max_duration_s low to refuse impractically long moves.

        Returns True if the move ran to completion, False if simulate mode
        skipped it or the planned duration exceeded max_duration_s.
        """
        if self.simulate:
            self.get_logger().info(
                f"[simulate] move_to_mit target={[f'{v:+.3f}' for v in target]} "
                f"v_des={v_des} (skipped)"
            )
            return False

        s = self.latest_state()
        if s is None:
            self.get_logger().error("move_to_mit: no joint_states yet")
            return False
        start = list(s.q)
        target = list(target)
        deltas = [target[j] - start[j] for j in range(NUM_ARM_JOINTS)]
        max_abs_delta = max(abs(d) for d in deltas) if deltas else 0.0
        if max_abs_delta < 1e-4:
            # Already at the target; just publish a hold and return.
            self.publish_mit(
                joint_index=list(range(1, NUM_ARM_JOINTS + 1)),
                p_des=target, v_des=[0.0] * NUM_ARM_JOINTS,
                kp=[kp] * NUM_ARM_JOINTS, kd=[kd] * NUM_ARM_JOINTS,
                t_ff=[0.0] * NUM_ARM_JOINTS,
            )
            return True

        duration_s = max_abs_delta / v_des
        if duration_s > max_duration_s:
            self.get_logger().error(
                f"move_to_mit: planned {duration_s:.0f}s exceeds max_duration_s "
                f"({max_duration_s:.0f}s). Largest delta={max_abs_delta:.3f}rad. "
                f"Either move arm manually closer first, increase v_des, or "
                f"raise max_duration_s."
            )
            return False

        self.get_logger().info(
            f"move_to_mit: v_des={v_des} rad/s, longest delta={max_abs_delta:.3f}rad, "
            f"planned duration ~{duration_s:.1f}s"
        )

        # Per-joint signed velocity feedforward.
        signed_vdes = [
            (v_des if d > 0 else -v_des) if abs(d) > 1e-4 else 0.0
            for d in deltas
        ]
        # Per-joint completion fraction lookup: once a joint is done, its
        # signed_vdes becomes 0 and its p_des stays at target.
        per_joint_done_at = [
            (abs(d) / v_des) if abs(d) > 1e-4 else 0.0 for d in deltas
        ]
        n = NUM_ARM_JOINTS
        dt = 1.0 / rate_hz
        t0 = time.monotonic()
        deadline = t0 + duration_s + settle_s + 0.5
        while time.monotonic() < deadline:
            elapsed = time.monotonic() - t0
            p_des = [0.0] * n
            v_des_now = [0.0] * n
            for j in range(n):
                if elapsed >= per_joint_done_at[j] or abs(deltas[j]) < 1e-4:
                    p_des[j] = target[j]
                    v_des_now[j] = 0.0
                else:
                    p_des[j] = start[j] + signed_vdes[j] * elapsed
                    v_des_now[j] = signed_vdes[j]
            self.publish_mit(
                joint_index=list(range(1, n + 1)),
                p_des=p_des, v_des=v_des_now,
                kp=[kp] * n, kd=[kd] * n, t_ff=[0.0] * n,
            )
            rclpy.spin_once(self, timeout_sec=0.0)
            # Allow early-exit once we're inside settle window and tracking
            # error is small.
            if elapsed > duration_s:
                cur = self.latest_state()
                if cur and max(abs(cur.q[j] - target[j]) for j in range(n)) < 0.005:
                    return True
            time.sleep(dt)
        return True

    def wait_until_settled(
        self, target: Sequence[float], tol: float = 0.015, timeout: float = 8.0
    ) -> bool:
        """Block until max |q - target| < tol over all 6 joints, or timeout."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)
            s = self.latest_state()
            if s is None:
                continue
            if all(abs(s.q[j] - target[j]) < tol for j in range(NUM_ARM_JOINTS)):
                # require it to stay there for ~150 ms to filter ringing
                stable_until = time.monotonic() + 0.15
                while time.monotonic() < stable_until:
                    rclpy.spin_once(self, timeout_sec=0.02)
                    s2 = self.latest_state()
                    if not all(abs(s2.q[j] - target[j]) < tol for j in range(NUM_ARM_JOINTS)):
                        break
                else:
                    return True
        return False

    # ---------- safety ----------

    def safety_check(
        self,
        sample: JointSample,
        test_joint_1based: int,
        q_start: float,
        limits: SafetyLimits,
    ) -> Optional[str]:
        """Return None if safe, or a string describing the violation."""
        j = test_joint_1based - 1
        if abs(sample.qd[j]) > limits.qdot_max:
            return f"joint{test_joint_1based} qdot={sample.qd[j]:+.3f} exceeds {limits.qdot_max}"
        if abs(sample.q[j] - q_start) > limits.q_excursion_max:
            return (
                f"joint{test_joint_1based} q={sample.q[j]:+.3f} excursion "
                f"|q - q_start|={abs(sample.q[j] - q_start):.3f} exceeds {limits.q_excursion_max}"
            )
        if limits.err_status_abort:
            st = self.latest_arm_status()
            if st is not None and st.err_status != 0:
                return f"arm err_status={st.err_status}"
        return None

    def emergency_hold(self) -> None:
        """Send a one-shot move_mit holding every joint at current position
        with conservative PD. Called on abort and on shutdown."""
        s = self.latest_state()
        if s is None:
            return
        self.publish_hold_all(s.q, HoldGains(kp=20.0, kd=1.5))

    # ---------- shutdown ----------

    def install_signal_handlers(self) -> None:
        def _h(signum, _frame):
            if self._shutdown_called:
                return
            self._shutdown_called = True
            self.get_logger().warn(f"signal {signum} -> emergency hold")
            self.emergency_hold()
            # Allow the publisher to flush before exit. rclpy's executor will
            # handle the SIGINT and break us out of spin() naturally.
            time.sleep(0.1)
            sys.exit(130 if signum == signal.SIGINT else 1)

        signal.signal(signal.SIGINT, _h)
        signal.signal(signal.SIGTERM, _h)


# ----------------------------- CSV logging -----------------------------

class CsvLogger:
    """Append-only CSV writer with a fixed schema for joint-state samples."""

    HEADER = (
        ["t_ns", "phase", "tau_cmd"]
        + [f"q{i}" for i in range(1, 7)]
        + [f"qd{i}" for i in range(1, 7)]
        + [f"tau{i}" for i in range(1, 7)]
    )

    def __init__(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self._f = open(path, "w", newline="")
        self._w = csv.writer(self._f)
        self._w.writerow(self.HEADER)

    def write(self, sample: JointSample, phase: str = "", tau_cmd: float = 0.0) -> None:
        self._w.writerow([sample.t_ns, phase, tau_cmd] + sample.q + sample.qd + sample.tau)

    def close(self) -> None:
        self._f.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


# ----------------------------- interactive prompt -----------------------------

def prompt_continue(message: str, auto: bool = False) -> bool:
    """Print a message and wait for the operator to press Enter. Returns False
    if the operator types 'q' or 'skip' to skip this step."""
    if auto:
        print(message, flush=True)
        return True
    sys.stdout.write(f"\n{message}\n  [Enter] continue, [q] abort run, [s] skip this joint: ")
    sys.stdout.flush()
    try:
        line = input().strip().lower()
    except EOFError:
        return False
    if line in ("q", "quit", "abort"):
        raise KeyboardInterrupt("operator abort")
    if line in ("s", "skip"):
        return False
    return True


def countdown(seconds: int, prefix: str = "starting in") -> None:
    for i in range(seconds, 0, -1):
        sys.stdout.write(f"\r{prefix} {i}s ...   ")
        sys.stdout.flush()
        time.sleep(1.0)
    sys.stdout.write("\r" + " " * 60 + "\r")
    sys.stdout.flush()


def countdown_keepalive(
    seconds: float,
    prefix: str,
    keepalive_fn: Optional[Callable[[], None]] = None,
    rate_hz: float = 20.0,
) -> None:
    """Like countdown(), but invokes keepalive_fn at rate_hz so MIT-mode
    publishes can keep firing during the wait. Many CAN motor controllers
    revert to internal hold mode if MIT messages stop for >1s; passing a
    keepalive that re-publishes the last command keeps the loop warm."""
    import math as _math
    dt = 1.0 / rate_hz
    end = time.monotonic() + seconds
    last_print_sec = -1
    while time.monotonic() < end:
        remaining = end - time.monotonic()
        sec = int(_math.ceil(remaining))
        if sec != last_print_sec:
            sys.stdout.write(f"\r{prefix} {sec}s ...   ")
            sys.stdout.flush()
            last_print_sec = sec
        if keepalive_fn is not None:
            keepalive_fn()
        time.sleep(dt)
    sys.stdout.write("\r" + " " * 60 + "\r")
    sys.stdout.flush()


# ----------------------------- ROS bootstrap -----------------------------

def init_and_make_node(node_name: str, simulate: bool = False) -> SysidNode:
    rclpy.init()
    node = SysidNode(node_name, simulate=simulate)
    node.install_signal_handlers()
    # Wait for first feedback so downstream calls don't race the subscription.
    node.wait_for_state(timeout=5.0)
    node.get_logger().info(
        f"connected; first joint_states received{' (SIMULATE)' if simulate else ''}"
    )
    return node


def shutdown_node(node: SysidNode) -> None:
    try:
        node.emergency_hold()
        time.sleep(0.1)
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
