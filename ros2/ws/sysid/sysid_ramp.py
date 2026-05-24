#!/usr/bin/env python3
"""Piper MIT-mode torque-ramp sysid.

Three open-loop probes per joint, all with the test joint commanded as
kp=0, kd=kd_min (tiny damping to keep numerical-derivative noise from
amplifying through the firmware loop). Other joints are held with stiff PD
via a one-shot move_mit at experiment start.

  Phase A — static friction probe
    Slow ramp t_ff: 0 -> +tau_max over `--ramp-duration-s`. Record τ at
    which |qdot| first exceeds `vel_breakaway`. Repeat for -tau_max. Yields
    tau_static_pos and tau_static_neg (Coulomb floor; viscous coupling is
    small at these slew rates).

  Phase B — inertia step probe
    Apply a square pulse of τ_step = 1.5 * max(|tau_static_pos|, |tau_static_neg|)
    for `pulse_ms` ms. Fit qdd via finite difference on the velocity trace.
    I_eff = τ_step / qdd  (after Coulomb subtraction).

  Phase C — viscous damping probe
    Hold τ_const for up to 1 s or until |qdot| stops increasing (terminal
    velocity). b_eff = (τ_const - τ_static) / qdot_terminal.

Outputs (under --output-dir / sysid_<timestamp>/):
  - raw/joint{J}_{phaseA|phaseB|phaseC}_{sign}.csv
  - fit.yaml                                 per-joint fitted parameters
  - mujoco_xml_snippet.xml                   ready-to-paste <joint>/<actuator>
  - plots/joint{J}_phase{A|B|C}.png          (if --plot)

The SDK silently divides t_ff by 4 for joints 1-3 before encoding it to CAN.
We don't compensate; the fitted alpha will simply reflect whatever scaling
the firmware actually applies, and the MuJoCo snippet uses that alpha to
convert commanded-to-delivered.

Safety: same SafetyLimits as sysid_step, plus a hard ceiling on torque
ramp magnitude and an abort if |qdot| > qdot_max during ramp.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from datetime import datetime

import numpy as np
import rclpy
import yaml
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sysid_common import (  # noqa: E402
    ARM_JOINT_NAMES,
    NUM_ARM_JOINTS,
    CsvLogger,
    HoldGains,
    JointSample,
    SafetyLimits,
    SysidNode,
    countdown,
    countdown_keepalive,
    init_and_make_node,
    prompt_continue,
    shutdown_node,
    tff_range,
)

POSES_PATH_DEFAULT = os.path.join(os.path.dirname(__file__), "joint_isolation_poses.yaml")


# ----------------------------- argparse -----------------------------

def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Piper MIT-mode torque sysid")
    p.add_argument(
        "--joints", type=int, nargs="+", default=[4, 6, 5, 1, 3, 2],
        choices=[1, 2, 3, 4, 5, 6],
        help="1-based joint indices. Default order is low-load-first.",
    )
    p.add_argument(
        "--poses", type=str, default=POSES_PATH_DEFAULT,
        help="path to joint_isolation_poses.yaml",
    )
    p.add_argument(
        "--tau-max", type=float, default=1.0,
        help="absolute ceiling on commanded |t_ff| (N·m). Default 1.0 (very conservative; "
             "raise for stickier joints if breakaway not detected).",
    )
    p.add_argument(
        "--ramp-duration-s", type=float, default=8.0,
        help="Phase A duration of each unidirectional torque ramp (s).",
    )
    p.add_argument(
        "--pulse-ms", type=float, default=200.0,
        help="Phase B trapezoid flat-hold width (ms). Total pulse adds 80ms ramp-up "
             "and 80ms ramp-down to soften onset/offset.",
    )
    p.add_argument(
        "--pulse-ramp-ms", type=float, default=80.0,
        help="Phase B trapezoid ramp-up/ramp-down duration (ms each).",
    )
    p.add_argument(
        "--damping-s", type=float, default=1.0,
        help="Phase C duration of constant-torque hold (s).",
    )
    p.add_argument(
        "--vel-breakaway", type=float, default=0.05,
        help="|qdot| (rad/s) at which we consider static friction overcome. "
             "Must hold for --breakaway-consec samples in a row before firing.",
    )
    p.add_argument(
        "--breakaway-consec", type=int, default=3,
        help="Number of consecutive samples above vel_breakaway required to "
             "declare breakaway. Filters single-sample noise spikes.",
    )
    p.add_argument(
        "--kd-min", type=float, default=0.1,
        help="tiny kd on test joint to suppress derivative noise without "
             "introducing meaningful position feedback.",
    )
    p.add_argument("--hold-kp", type=float, default=20.0)
    p.add_argument("--hold-kd", type=float, default=1.5)
    p.add_argument(
        "--reposition-vel", type=float, default=0.01,
        help="MIT-interpolated reposition speed (rad/s). Default 0.01 (~0.57 deg/s). "
             "At v=0.01 a 2 rad move takes 200s; raise to 0.05-0.1 for tractable times.",
    )
    p.add_argument(
        "--reposition-kp", type=float, default=15.0,
        help="kp used by the MIT reposition.",
    )
    p.add_argument(
        "--reposition-kd", type=float, default=1.0,
        help="kd used by the MIT reposition.",
    )
    p.add_argument("--qdot-max", type=float, default=0.5,
                   help="abort if test joint exceeds this |qdot| (rad/s). Default 0.5.")
    p.add_argument("--q-excursion-max", type=float, default=0.1,
                   help="abort if test joint travels beyond this from pose start (rad). "
                        "Default 0.1 rad (~6 deg).")
    p.add_argument("--output-dir", type=str, default="/ws/sysid/runs")
    p.add_argument("--plot", action="store_true")
    p.add_argument("--auto", action="store_true",
                   help="skip per-joint prompts; still does safety countdowns.")
    p.add_argument(
        "--simulate", action="store_true",
        help="dry-run: never publish /control/move_mit; useful to exercise the flow.",
    )
    return p.parse_args(argv)


# ----------------------------- low-level helpers -----------------------------

def publish_test_joint_torque(
    node: SysidNode,
    joint_1based: int,
    hold_pose: list[float],
    hold: HoldGains,
    test_kd: float,
    tau_cmd: float,
    simulate: bool,
) -> None:
    """Send a full 6-joint move_mit where the test joint runs kp=0, kd=test_kd,
    t_ff=tau_cmd, and the others are held with (hold.kp, hold.kd)."""
    n = NUM_ARM_JOINTS
    j = joint_1based - 1
    kp = [hold.kp] * n
    kd = [hold.kd] * n
    p_des = list(hold_pose)
    v_des = [0.0] * n
    t_ff = [0.0] * n
    kp[j] = 0.0
    kd[j] = test_kd
    t_ff[j] = tau_cmd
    if simulate:
        return
    node.publish_mit(
        joint_index=list(range(1, n + 1)),
        p_des=p_des, v_des=v_des, kp=kp, kd=kd, t_ff=t_ff,
    )


def load_poses(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def measure_gravity_baseline(
    node: SysidNode,
    joint_1based: int,
    hold_pose: list[float],
    hold: HoldGains,
    settle_s: float = 1.0,
    sample_s: float = 0.5,
    simulate: bool = False,
) -> float:
    """Hold the arm with stiff PD (kp=hold.kp, kd=hold.kd, t_ff=0) and read the
    firmware-reported effort on the test joint. At steady state the motor's
    PD output balances gravity, so this effort *is* the gravity torque
    (plus any static-friction component the joint happens to be sitting on).

    Used as a feedforward baseline for kp=0 sysid phases so the test joint
    doesn't fall when its position-holding gain is removed.
    """
    if simulate:
        node.get_logger().info("[simulate] gravity baseline = 0.0 (skipped)")
        return 0.0

    # Establish the stiff hold and let the joint settle.
    node.publish_hold_all(hold_pose, hold)
    settle_end = time.monotonic() + settle_s
    while time.monotonic() < settle_end:
        node.publish_hold_all(hold_pose, hold)
        rclpy.spin_once(node, timeout_sec=0.0)
        time.sleep(0.02)

    # Average effort over sample_s.
    j = joint_1based - 1
    efforts: list[float] = []
    sample_end = time.monotonic() + sample_s
    while time.monotonic() < sample_end:
        node.publish_hold_all(hold_pose, hold)
        rclpy.spin_once(node, timeout_sec=0.0)
        s = node.latest_state()
        if s is not None:
            efforts.append(s.tau[j])
        time.sleep(0.02)
    if not efforts:
        node.get_logger().warn("gravity baseline: no effort samples collected")
        return 0.0
    baseline = float(np.median(efforts))
    node.get_logger().info(
        f"j{joint_1based} gravity baseline (median effort over {len(efforts)} samples): "
        f"{baseline:+.3f} N·m"
    )
    return baseline


# ----------------------------- Phase A — static friction -----------------------------

def phase_a_static_friction(
    node: SysidNode,
    joint_1based: int,
    hold_pose: list[float],
    hold: HoldGains,
    args: argparse.Namespace,
    csv_path: str,
    sign: float,
    tau_baseline: float = 0.0,
) -> tuple[float, list, list, list, list]:
    """Slow torque ramp to find the Coulomb breakaway torque in one direction."""
    tau_low, tau_high = tff_range(joint_1based)
    tau_max = sign * min(args.tau_max, abs(tau_high if sign > 0 else tau_low))
    duration = args.ramp_duration_s
    rate_hz = 100.0
    dt = 1.0 / rate_hz
    t_list, tau_cmd_list, q_list, qd_list = [], [], [], []

    # Snapshot baseline.
    s0 = node.latest_state()
    q_start = s0.q[joint_1based - 1]
    limits = SafetyLimits(
        tau_max=args.tau_max, qdot_max=args.qdot_max,
        q_excursion_max=args.q_excursion_max,
    )
    breakaway_tau = float("nan")
    consec_above = 0
    first_crossing_tau: float | None = None  # tau at the *first* sample that crossed

    with CsvLogger(csv_path) as csvw:
        t0 = time.monotonic()
        baseline_t_ns = None
        while True:
            now = time.monotonic()
            elapsed = now - t0
            if elapsed > duration:
                break
            # Linear ramp 0 -> tau_max
            tau = tau_max * (elapsed / duration)
            publish_test_joint_torque(
                node, joint_1based, hold_pose, hold,
                args.kd_min, tau_baseline + tau, args.simulate,
            )
            rclpy.spin_once(node, timeout_sec=0.0)
            s = node.latest_state()
            if s is None:
                time.sleep(dt)
                continue
            if baseline_t_ns is None:
                baseline_t_ns = s.t_ns
            t_rel = (s.t_ns - baseline_t_ns) * 1e-9
            t_list.append(t_rel)
            tau_cmd_list.append(tau)
            q_list.append(s.q[joint_1based - 1])
            qd_list.append(s.qd[joint_1based - 1])
            csvw.write(s, phase=f"A_{'+' if sign > 0 else '-'}", tau_cmd=tau)
            # Detect breakaway: need `consec` samples in a row above threshold,
            # all with qdot sign matching commanded direction. Reset counter
            # if any sample falls back below threshold. This filters out
            # single-sample noise spikes (e.g. encoder quantization).
            if math.isnan(breakaway_tau):
                if sign * s.qd[joint_1based - 1] > args.vel_breakaway:
                    if consec_above == 0:
                        first_crossing_tau = tau
                    consec_above += 1
                else:
                    consec_above = 0
                    first_crossing_tau = None
                if consec_above >= args.breakaway_consec:
                    # The torque that actually overcame friction was the one
                    # commanded at the first crossing, not the third.
                    breakaway_tau = (
                        first_crossing_tau if first_crossing_tau is not None else tau
                    )
                    node.get_logger().info(
                        f"j{joint_1based} breakaway at tau={tau:+.3f} "
                        f"qdot={s.qd[joint_1based - 1]:+.3f}"
                    )
                    # Stop ramping further once we've found it. Drop torque
                    # smoothly to zero over ~150 ms so the joint doesn't snap.
                    decay_until = time.monotonic() + 0.15
                    while time.monotonic() < decay_until:
                        frac = max(
                            0.0, 1.0 - (time.monotonic() - (decay_until - 0.15)) / 0.15
                        )
                        publish_test_joint_torque(
                            node, joint_1based, hold_pose, hold,
                            args.kd_min, tau_baseline + tau * frac, args.simulate,
                        )
                        rclpy.spin_once(node, timeout_sec=0.005)
                    publish_test_joint_torque(
                        node, joint_1based, hold_pose, hold,
                        args.kd_min, tau_baseline, args.simulate,
                    )
                    break
            err = node.safety_check(s, joint_1based, q_start, limits)
            if err is not None:
                node.get_logger().error(f"PHASE A SAFETY ABORT: {err}")
                # Hold at current pose with stiff PD — gravity baseline is
                # measured at the original pose and stops being valid once
                # the joint has drifted.
                node.emergency_hold()
                break
            time.sleep(dt)
    # Drop experimental component; keep gravity baseline so the joint stays put.
    publish_test_joint_torque(
        node, joint_1based, hold_pose, hold, args.kd_min, tau_baseline, args.simulate,
    )

    # Post-breakaway settle: low-inertia joints (esp. wrists) become weakly-
    # damped pendulums under kp=0; the breakaway impulse sends them swinging
    # and the next phase trips qdot_max before any test torque is applied.
    # Engage stiff PD at a FIXED setpoint (the position at settle start) so
    # kp pulls the joint back, not chases it.
    if not args.simulate:
        s_at_settle = node.latest_state()
        if s_at_settle is not None:
            settle_pose = list(s_at_settle.q)
            stiff_hold = HoldGains(kp=20.0, kd=1.5)
            settle_deadline = time.monotonic() + 1.5
            while time.monotonic() < settle_deadline:
                node.publish_hold_all(settle_pose, stiff_hold)
                rclpy.spin_once(node, timeout_sec=0.02)
                s = node.latest_state()
                if s is not None and abs(s.qd[joint_1based - 1]) < 0.02:
                    break
                time.sleep(0.02)
        # Re-arm gravity baseline for the next phase.
        publish_test_joint_torque(
            node, joint_1based, hold_pose, hold, args.kd_min, tau_baseline, args.simulate,
        )
    return breakaway_tau, t_list, tau_cmd_list, q_list, qd_list


# ----------------------------- Phase B — inertia step -----------------------------

def phase_b_inertia(
    node: SysidNode,
    joint_1based: int,
    hold_pose: list[float],
    hold: HoldGains,
    args: argparse.Namespace,
    csv_path: str,
    tau_static_mag: float,
    tau_baseline: float = 0.0,
) -> tuple[dict, list, list, list, list]:
    """Square pulse for `pulse_ms`. Fit qdd from velocity slope."""
    tau_low, tau_high = tff_range(joint_1based)
    tau_step = 1.5 * tau_static_mag
    tau_step = max(min(tau_step, args.tau_max, tau_high), -args.tau_max, tau_low)
    flat_s = args.pulse_ms / 1000.0
    ramp_s = args.pulse_ramp_ms / 1000.0
    rate_hz = 200.0
    dt = 1.0 / rate_hz
    t_list, tau_cmd_list, q_list, qd_list = [], [], [], []

    s0 = node.latest_state()
    q_start = s0.q[joint_1based - 1]
    limits = SafetyLimits(
        tau_max=args.tau_max, qdot_max=args.qdot_max,
        q_excursion_max=args.q_excursion_max,
    )

    def _trapezoid_tau(t_in_pulse: float) -> float:
        """Trapezoidal command: 0 -> tau_step over ramp_s, hold for flat_s,
        tau_step -> 0 over ramp_s. Beyond the pulse window, returns 0."""
        if t_in_pulse < 0:
            return 0.0
        if t_in_pulse < ramp_s:
            return tau_step * (t_in_pulse / ramp_s)
        if t_in_pulse < ramp_s + flat_s:
            return tau_step
        if t_in_pulse < 2 * ramp_s + flat_s:
            return tau_step * (1.0 - (t_in_pulse - ramp_s - flat_s) / ramp_s)
        return 0.0

    pulse_duration = 2 * ramp_s + flat_s
    pre_s = 0.05   # baseline quiet period before pulse
    post_s = 0.50  # post-pulse decay window (used to fit b_eff and re-fit tau_static)
    total_duration = pre_s + pulse_duration + post_s

    with CsvLogger(csv_path) as csvw:
        t0 = time.monotonic()
        baseline_t_ns = None
        pulse_start_t: Optional[float] = None
        while time.monotonic() - t0 < total_duration:
            now = time.monotonic()
            elapsed = now - t0
            if elapsed < pre_s:
                seg_tau = 0.0
            else:
                if pulse_start_t is None:
                    pulse_start_t = elapsed
                seg_tau = _trapezoid_tau(elapsed - pulse_start_t)
            publish_test_joint_torque(
                node, joint_1based, hold_pose, hold,
                args.kd_min, tau_baseline + seg_tau, args.simulate,
            )
            rclpy.spin_once(node, timeout_sec=0.0)
            s = node.latest_state()
            if s is None:
                time.sleep(dt)
                continue
            if baseline_t_ns is None:
                baseline_t_ns = s.t_ns
            t_rel = (s.t_ns - baseline_t_ns) * 1e-9
            t_list.append(t_rel)
            # Record only the experimental component (seg_tau) so the analysis
            # sees the on-axis torque, not the cancellation of gravity.
            tau_cmd_list.append(seg_tau)
            q_list.append(s.q[joint_1based - 1])
            qd_list.append(s.qd[joint_1based - 1])
            csvw.write(s, phase="B", tau_cmd=seg_tau)
            err = node.safety_check(s, joint_1based, q_start, limits)
            if err is not None:
                node.get_logger().error(f"PHASE B SAFETY ABORT: {err}")
                node.emergency_hold()
                return _fit_inertia(
                    t_list, qd_list, tau_cmd_list, tau_step, tau_static_mag,
                ), t_list, tau_cmd_list, q_list, qd_list
            time.sleep(dt)
    return (
        _fit_inertia(t_list, qd_list, tau_cmd_list, tau_step, tau_static_mag),
        t_list, tau_cmd_list, q_list, qd_list,
    )


def _fit_damping_from_decay(
    t: list, qd: list, tau_cmd: list, I_eff: float,
) -> dict:
    """Fit b_eff and a refined tau_static from Phase B's post-pulse coasting.

    With commanded tau == 0 and joint coasting in one direction,
    `qd_dot = -(b/I) * qd - sign(qd) * tau_static / I`.

    Rearranged: `-I * qd_dot = b * qd + tau_static`, linear in (b, tau_static)
    if we restrict to samples with consistent sign(qd).
    """
    nan_result = {
        "b_eff_decay": float("nan"),
        "tau_static_decay": float("nan"),
        "n_samples_fit": 0,
    }
    if math.isnan(I_eff) or abs(I_eff) < 1e-6 or len(t) < 10:
        return nan_result

    t_arr = np.asarray(t)
    qd_arr = np.asarray(qd)
    tau_arr = np.asarray(tau_cmd)

    # Identify the post-pulse region: after the trapezoid ramp-down has
    # finished AND commanded tau is approximately zero AND the joint is still
    # moving (so it's coasting under friction+damping, not held by Coulomb).
    pulse_off = np.abs(tau_arr) < 1e-3
    if not pulse_off.any():
        return nan_result
    # First contiguous run of pulse-off after the pulse fired.
    pulse_on_max = int(np.argmax(np.abs(tau_arr)))
    decay_mask = pulse_off.copy()
    decay_mask[:pulse_on_max] = False  # only after the pulse peak
    decay_idx = np.where(decay_mask)[0]
    if decay_idx.size < 8:
        return nan_result

    t_dec = t_arr[decay_idx]
    qd_dec = qd_arr[decay_idx]
    # Sign of motion: take the dominant sign in the decay region.
    sign_dec = 1.0 if np.mean(qd_dec) >= 0 else -1.0
    same_sign = (np.sign(qd_dec) == sign_dec) & (np.abs(qd_dec) > 0.01)
    t_dec = t_dec[same_sign]
    qd_dec = qd_dec[same_sign]
    if t_dec.size < 6:
        return nan_result

    # Compute qd_dot via central difference. Trim endpoints.
    qd_dot = np.gradient(qd_dec, t_dec)
    # Trim a few endpoint samples where gradient is noisy.
    qd_dec_c = qd_dec[2:-2]
    qd_dot_c = qd_dot[2:-2]
    if qd_dec_c.size < 4:
        return nan_result

    # Solve y = b*x1 + tau_static*x2 where y = -I*qd_dot, x1 = qd, x2 = sign(qd)
    y = -I_eff * qd_dot_c
    X = np.column_stack([qd_dec_c, sign_dec * np.ones_like(qd_dec_c)])
    try:
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    except np.linalg.LinAlgError:
        return nan_result
    b_eff, tau_static = float(coef[0]), float(coef[1])
    return {
        "b_eff_decay": b_eff,
        "tau_static_decay": tau_static,
        "n_samples_fit": int(qd_dec_c.size),
    }


def _fit_inertia(
    t: list, qd: list, tau_cmd: list, tau_step: float, tau_static_mag: float,
) -> dict:
    """Fit `qdd` in the trapezoid's flat-top region (where commanded tau is
    constant at tau_step). Then I_eff = (tau_step - sign(qdd)*tau_static) / qdd.

    The flat-top is the longest run of consecutive samples where
    |tau_cmd - tau_step| < 5% of tau_step. We linear-regress qdot vs t over
    that window."""
    if not t:
        return {"I_eff": float("nan"), "qdd": float("nan"), "tau_eff": float("nan")}
    t_arr = np.asarray(t)
    qd_arr = np.asarray(qd)
    tau_arr = np.asarray(tau_cmd)
    if abs(tau_step) < 1e-6:
        return {"I_eff": float("nan"), "qdd": float("nan"), "tau_eff": float("nan")}

    # Find indices where commanded tau is at the flat-top (within 5%).
    flat_mask = np.abs(tau_arr - tau_step) < 0.05 * abs(tau_step)
    flat_idx = np.where(flat_mask)[0]
    if flat_idx.size < 6:
        return {"I_eff": float("nan"), "qdd": float("nan"), "tau_eff": float("nan")}

    # Use only the longest contiguous run of flat-top samples.
    diffs = np.diff(flat_idx)
    breaks = np.where(diffs > 1)[0]
    if breaks.size:
        # Pick the longest run.
        segments = np.split(flat_idx, breaks + 1)
        run = max(segments, key=len)
    else:
        run = flat_idx
    if run.size < 6:
        return {"I_eff": float("nan"), "qdd": float("nan"), "tau_eff": float("nan")}

    tt = t_arr[run]
    qq = qd_arr[run]
    # Linear fit qdot(t) = qdd*t + b
    A = np.vstack([tt - tt[0], np.ones_like(tt)]).T
    coef, *_ = np.linalg.lstsq(A, qq, rcond=None)
    qdd = float(coef[0])
    tau_eff = float(tau_step) - math.copysign(tau_static_mag, qdd)
    I_eff = tau_eff / qdd if abs(qdd) > 1e-3 else float("nan")
    return {"I_eff": I_eff, "qdd": qdd, "tau_eff": tau_eff}


# ----------------------------- Phase C — viscous damping -----------------------------

def phase_c_damping(
    node: SysidNode,
    joint_1based: int,
    hold_pose: list[float],
    hold: HoldGains,
    args: argparse.Namespace,
    csv_path: str,
    tau_static_mag: float,
    tau_baseline: float = 0.0,
) -> tuple[dict, list, list, list, list]:
    """Hold constant torque; record terminal velocity once qdd ~= 0.

    Starts with a 100 ms smooth ramp from 0 to tau_const so the firmware
    doesn't see a step input (which appears to be filtered/watchdogged on
    Piper firmware). Keeps publishing at 200 Hz the whole time."""
    tau_low, tau_high = tff_range(joint_1based)
    tau_const = 1.2 * tau_static_mag
    tau_const = max(min(tau_const, args.tau_max, tau_high), -args.tau_max, tau_low)
    rate_hz = 200.0
    dt = 1.0 / rate_hz
    ramp_in_s = 0.10  # smooth ramp to avoid step-input filtering
    t_list, tau_cmd_list, q_list, qd_list = [], [], [], []

    s0 = node.latest_state()
    q_start = s0.q[joint_1based - 1]
    limits = SafetyLimits(
        tau_max=args.tau_max, qdot_max=args.qdot_max,
        q_excursion_max=args.q_excursion_max,
    )

    with CsvLogger(csv_path) as csvw:
        t0 = time.monotonic()
        baseline_t_ns = None
        end = time.monotonic() + args.damping_s + ramp_in_s
        while time.monotonic() < end:
            elapsed = time.monotonic() - t0
            if elapsed < ramp_in_s:
                tau_now = tau_const * (elapsed / ramp_in_s)
            else:
                tau_now = tau_const
            publish_test_joint_torque(
                node, joint_1based, hold_pose, hold,
                args.kd_min, tau_baseline + tau_now, args.simulate,
            )
            rclpy.spin_once(node, timeout_sec=0.0)
            s = node.latest_state()
            if s is None:
                time.sleep(dt)
                continue
            if baseline_t_ns is None:
                baseline_t_ns = s.t_ns
            t_rel = (s.t_ns - baseline_t_ns) * 1e-9
            t_list.append(t_rel)
            # Log experimental component only (excluding gravity baseline).
            tau_cmd_list.append(tau_now)
            q_list.append(s.q[joint_1based - 1])
            qd_list.append(s.qd[joint_1based - 1])
            csvw.write(s, phase="C", tau_cmd=tau_now)
            err = node.safety_check(s, joint_1based, q_start, limits)
            if err is not None:
                node.get_logger().error(f"PHASE C SAFETY ABORT: {err}")
                node.emergency_hold()
                break
            time.sleep(dt)
    publish_test_joint_torque(
        node, joint_1based, hold_pose, hold, args.kd_min, tau_baseline, args.simulate,
    )
    return (
        _fit_damping(t_list, qd_list, tau_const, tau_static_mag),
        t_list, tau_cmd_list, q_list, qd_list,
    )


def _fit_damping(
    t: list, qd: list, tau_const: float, tau_static_mag: float,
) -> dict:
    if len(t) < 20:
        return {"b_eff": float("nan"), "qdot_terminal": float("nan")}
    qd_arr = np.asarray(qd)
    # Treat the last 20% of samples as steady-state.
    n_tail = max(5, int(0.2 * len(qd_arr)))
    qdot_terminal = float(np.mean(qd_arr[-n_tail:]))
    if abs(qdot_terminal) < 1e-3:
        return {"b_eff": float("nan"), "qdot_terminal": qdot_terminal}
    tau_eff = tau_const - math.copysign(tau_static_mag, qdot_terminal)
    b_eff = tau_eff / qdot_terminal
    return {"b_eff": b_eff, "qdot_terminal": qdot_terminal}


# ----------------------------- plotting -----------------------------

def save_phase_plot(
    path: str,
    title: str,
    t: list, q: list, qd: list, tau_cmd: list,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    if not t:
        return
    t_arr = np.asarray(t)
    fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
    axes[0].plot(t_arr, np.degrees(q), color="C0")
    axes[0].set_ylabel("q (deg)")
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title(title)

    axes[1].plot(t_arr, qd, color="C1")
    axes[1].set_ylabel("qdot (rad/s)")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(t_arr, tau_cmd, color="C2")
    axes[2].set_ylabel("τ_cmd (N·m)")
    axes[2].set_xlabel("time (s)")
    axes[2].grid(True, alpha=0.3)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ----------------------------- MuJoCo snippet emitter -----------------------------

def mujoco_snippet(fits: dict) -> str:
    """Emit <joint>/<actuator> XML stubs from the fitted I_eff, b_eff,
    tau_static, alpha. The user can paste these into their Piper MJCF."""
    lines = ["<!-- generated by sysid_ramp.py -->\n<mujoco>\n  <worldbody>"]
    for j in sorted(fits):
        f = fits[j]
        I = f.get("I_eff", float("nan"))
        b = f.get("b_eff", float("nan"))
        tau_s = max(abs(f.get("tau_static_pos", 0.0)),
                    abs(f.get("tau_static_neg", 0.0)))
        if math.isnan(I) or math.isnan(b):
            lines.append(f"    <!-- joint{j}: insufficient fit, skipping -->")
            continue
        # Approximate joint armature = I_eff (treat as link rotational inertia
        # added at the joint). damping = b_eff. frictionloss = tau_static.
        lines.append(
            f'    <joint name="joint{j}" armature="{I:.5f}" damping="{b:.5f}" '
            f'frictionloss="{tau_s:.4f}"/>'
        )
    lines.append("  </worldbody>\n  <actuator>")
    for j in sorted(fits):
        # Use a `motor` actuator for direct-torque sim; gear=1 so commanded
        # torque maps 1:1 to applied. Scale via the policy if needed.
        lines.append(f'    <motor name="m_joint{j}" joint="joint{j}" gear="1"/>')
    lines.append("  </actuator>\n</mujoco>")
    return "\n".join(lines) + "\n"


# ----------------------------- main loop -----------------------------

def main(argv=None) -> int:
    args = parse_args(argv)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = os.path.join(args.output_dir, f"sysid_{stamp}")
    raw_dir = os.path.join(out_root, "raw")
    plots_dir = os.path.join(out_root, "plots")
    os.makedirs(raw_dir, exist_ok=True)
    if args.plot:
        os.makedirs(plots_dir, exist_ok=True)

    print(f"\n==== sysid_ramp: writing to {out_root} ====", flush=True)
    print(f"joints={args.joints} tau_max={args.tau_max}N·m "
          f"qdot_max={args.qdot_max}rad/s q_exc_max={args.q_excursion_max}rad",
          flush=True)
    if args.simulate:
        print("*** SIMULATE MODE: no /control/move_mit publishes ***", flush=True)

    poses = load_poses(args.poses)
    node = init_and_make_node("sysid_ramp", simulate=args.simulate)
    hold = HoldGains(kp=args.hold_kp, kd=args.hold_kd)

    fits: dict[int, dict] = {}

    try:
        for j in args.joints:
            pose_entry = poses.get(f"joint{j}")
            if pose_entry is None:
                node.get_logger().warn(f"no isolation pose for joint{j}, skipping")
                continue
            hold_pose = list(pose_entry["hold"])

            cont = prompt_continue(
                f"--- joint{j} isolation pose: {hold_pose}\n"
                f"    {pose_entry.get('notes', '')}\n"
                f"    arm will reposition with move_j and then begin torque tests.",
                auto=args.auto,
            )
            if not cont:
                continue

            # Slow MIT-interpolated reposition to isolation pose.
            node.get_logger().info(
                f"j{j}: repositioning to isolation pose at v_des={args.reposition_vel}rad/s"
            )
            node.move_to_mit(
                hold_pose,
                v_des=args.reposition_vel,
                kp=args.reposition_kp,
                kd=args.reposition_kd,
            )

            # Measure the gravity-compensation baseline: hold with stiff PD
            # and read the firmware-reported motor torque. This becomes the
            # t_ff baseline for kp=0 phases so the test joint doesn't fall
            # when its position-holding gain is removed.
            tau_baseline = measure_gravity_baseline(
                node, j, hold_pose, hold,
                settle_s=1.0, sample_s=0.5,
                simulate=args.simulate,
            )
            # Establish stiff hold on all joints before we start touching the
            # MIT loop on the test joint.
            if not args.simulate:
                node.publish_hold_all(hold_pose, hold)
            time.sleep(0.3)

            # --- Phase A: static friction ±
            countdown(3, prefix=f"j{j} Phase A (static friction); starting in")
            tau_pos, t_a_p, tau_a_p, q_a_p, qd_a_p = phase_a_static_friction(
                node, j, hold_pose, hold, args,
                os.path.join(raw_dir, f"joint{j}_phaseA_pos.csv"),
                sign=+1.0, tau_baseline=tau_baseline,
            )
            time.sleep(0.5)
            tau_neg, t_a_n, tau_a_n, q_a_n, qd_a_n = phase_a_static_friction(
                node, j, hold_pose, hold, args,
                os.path.join(raw_dir, f"joint{j}_phaseA_neg.csv"),
                sign=-1.0, tau_baseline=tau_baseline,
            )
            tau_static_mag = max(
                abs(tau_pos) if not math.isnan(tau_pos) else 0.0,
                abs(tau_neg) if not math.isnan(tau_neg) else 0.0,
            )
            print(
                f"  j{j} static friction: pos={tau_pos:+.3f}  neg={tau_neg:+.3f}  "
                f"|τ_s|={tau_static_mag:.3f}", flush=True,
            )

            if tau_static_mag <= 0 or math.isnan(tau_static_mag):
                node.get_logger().warn(
                    f"j{j}: did not find breakaway within tau_max={args.tau_max}. "
                    f"Skipping phases B and C."
                )
                fits[j] = {
                    "tau_static_pos": tau_pos, "tau_static_neg": tau_neg,
                    "I_eff": float("nan"), "b_eff": float("nan"),
                    "alpha_torque_scale": float("nan"),
                }
                continue

            # --- Phase B: inertia step
            # Keep-alive between Phase A and B: maintain kp=0, kd=kd_min,
            # t_ff=tau_baseline so the firmware sees a continuous MIT stream
            # AND the test joint stays gravity-balanced during the countdown.
            keepalive = lambda: publish_test_joint_torque(
                node, j, hold_pose, hold, args.kd_min, tau_baseline, args.simulate,
            )
            countdown_keepalive(
                3, f"j{j} Phase B (inertia step); starting in", keepalive,
            )
            inertia_fit, t_b, tau_b, q_b, qd_b = phase_b_inertia(
                node, j, hold_pose, hold, args,
                os.path.join(raw_dir, f"joint{j}_phaseB.csv"),
                tau_static_mag=tau_static_mag,
                tau_baseline=tau_baseline,
            )
            print(
                f"  j{j} inertia: I_eff={inertia_fit['I_eff']:.4f} kg·m²  "
                f"qdd={inertia_fit['qdd']:+.3f}", flush=True,
            )

            # Fit b_eff (and refined tau_static) from Phase B's post-pulse decay.
            # This often gives a usable b_eff even when Phase C times out.
            decay_fit = _fit_damping_from_decay(
                t_b, qd_b, tau_b, inertia_fit["I_eff"],
            )
            print(
                f"  j{j} decay fit: b_eff={decay_fit['b_eff_decay']:.4f}  "
                f"tau_static_decay={decay_fit['tau_static_decay']:+.3f}  "
                f"n={decay_fit['n_samples_fit']}", flush=True,
            )

            # --- Phase C: viscous damping (validation against decay fit)
            countdown_keepalive(
                3, f"j{j} Phase C (viscous damping); starting in", keepalive,
            )
            damping_fit, t_c, tau_c, q_c, qd_c = phase_c_damping(
                node, j, hold_pose, hold, args,
                os.path.join(raw_dir, f"joint{j}_phaseC.csv"),
                tau_static_mag=tau_static_mag,
                tau_baseline=tau_baseline,
            )
            print(
                f"  j{j} damping: b_eff={damping_fit['b_eff']:.4f} N·m·s/rad  "
                f"qdot_term={damping_fit['qdot_terminal']:+.3f}", flush=True,
            )

            # Prefer the decay fit's b_eff if Phase C didn't yield one. The
            # decay fit uses Phase B data already collected and works even
            # when Phase C hits excursion/qdot limits.
            b_eff_use = damping_fit["b_eff"]
            if math.isnan(b_eff_use):
                b_eff_use = decay_fit["b_eff_decay"]

            # alpha = ratio between commanded torque and torque inferred from
            # the dynamic model (I*qdd + b*qd + tau_static) during the steady
            # phase of Phase C. alpha < 1 means firmware delivers less than
            # we commanded (e.g., the 0.25 scaling on J1-3).
            tau_inferred = (
                damping_fit["b_eff"] * damping_fit["qdot_terminal"]
                + math.copysign(tau_static_mag, damping_fit["qdot_terminal"])
            ) if not math.isnan(damping_fit["b_eff"]) else float("nan")
            tau_commanded = 1.2 * tau_static_mag
            alpha = (
                tau_inferred / tau_commanded
                if not math.isnan(tau_inferred) and abs(tau_commanded) > 1e-6
                else float("nan")
            )

            fits[j] = {
                "tau_static_pos": float(tau_pos),
                "tau_static_neg": float(tau_neg),
                "tau_gravity_baseline": float(tau_baseline),
                "I_eff": float(inertia_fit["I_eff"]),
                "qdd_at_step": float(inertia_fit["qdd"]),
                "b_eff": float(b_eff_use),
                "b_eff_phase_c": float(damping_fit["b_eff"]),
                "b_eff_decay_fit": float(decay_fit["b_eff_decay"]),
                "tau_static_decay_fit": float(decay_fit["tau_static_decay"]),
                "decay_fit_n_samples": int(decay_fit["n_samples_fit"]),
                "qdot_terminal": float(damping_fit["qdot_terminal"]),
                "alpha_torque_scale": float(alpha),
            }
            print(f"  j{j} alpha = {alpha:.3f}\n", flush=True)

            if args.plot:
                save_phase_plot(
                    os.path.join(plots_dir, f"joint{j}_phaseA_pos.png"),
                    f"j{j} Phase A (τ ramp +)", t_a_p, q_a_p, qd_a_p, tau_a_p,
                )
                save_phase_plot(
                    os.path.join(plots_dir, f"joint{j}_phaseA_neg.png"),
                    f"j{j} Phase A (τ ramp −)", t_a_n, q_a_n, qd_a_n, tau_a_n,
                )
                save_phase_plot(
                    os.path.join(plots_dir, f"joint{j}_phaseB.png"),
                    f"j{j} Phase B (inertia step)", t_b, q_b, qd_b, tau_b,
                )
                save_phase_plot(
                    os.path.join(plots_dir, f"joint{j}_phaseC.png"),
                    f"j{j} Phase C (viscous)", t_c, q_c, qd_c, tau_c,
                )

        node.get_logger().info("returning to home (zeros) via slow MIT")
        node.move_to_mit(
            [0.0] * NUM_ARM_JOINTS,
            v_des=args.reposition_vel,
            kp=args.reposition_kp,
            kd=args.reposition_kd,
        )
    except KeyboardInterrupt:
        print("\nKeyboardInterrupt; emergency hold.", flush=True)
        node.emergency_hold()
    finally:
        with open(os.path.join(out_root, "fit.yaml"), "w") as f:
            yaml.safe_dump({"args": vars(args), "fits": fits}, f, sort_keys=False)
        with open(os.path.join(out_root, "mujoco_xml_snippet.xml"), "w") as f:
            f.write(mujoco_snippet(fits))
        print(f"\nfit -> {os.path.join(out_root, 'fit.yaml')}", flush=True)
        print(f"mjcf -> {os.path.join(out_root, 'mujoco_xml_snippet.xml')}", flush=True)
        shutdown_node(node)
    return 0


if __name__ == "__main__":
    sys.exit(main())
