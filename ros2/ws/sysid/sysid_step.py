#!/usr/bin/env python3
"""Piper MIT-mode step-response sysid.

For each requested joint and each requested step amplitude, this script:

  1. Repositions the arm to a known starting pose with /control/move_j.
  2. Sends a single /control/move_mit step command with a chosen (kp, kd).
  3. Logs /feedback/joint_states at full rate (~200 Hz) for `--log-duration`.
  4. Computes rise time, overshoot, settling time, peak velocity.

The point isn't absolute accuracy of the metrics — it's matching the *same*
closed-loop behavior in MuJoCo. Use the per-amplitude curves to tune MuJoCo's
position-actuator kp/kv until its step response matches what's recorded here.

Outputs (under --output-dir / step_<timestamp>/):
  - raw/joint{J}_amp{deg}_kp{Kp}_kd{Kd}.csv    one CSV per shot
  - summary.yaml                                fitted metrics
  - plots/joint{J}_amp{deg}_kp{Kp}_kd{Kd}.png   per-shot plot (if --plot)

Safety: the script's emergency_hold runs on SIGINT / SIGTERM and at the end
of every shot. It also aborts any shot whose live joint state breaches the
SafetyLimits in sysid_common (qdot, |q - q_start|, err_status).
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from datetime import datetime
from typing import Iterable

import numpy as np
import rclpy
import yaml

POSES_PATH_DEFAULT = os.path.join(os.path.dirname(__file__), "joint_isolation_poses.yaml")

# Allow `python3 /ws/sysid/sysid_step.py` without installing as a package.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sysid_common import (  # noqa: E402
    ARM_JOINT_NAMES,
    NUM_ARM_JOINTS,
    CsvLogger,
    HoldGains,
    SDK_KD_RANGE,
    SDK_KP_RANGE,
    SafetyLimits,
    SysidNode,
    countdown,
    init_and_make_node,
    prompt_continue,
    shutdown_node,
    tff_range,
)


# ----------------------------- argparse -----------------------------

def parse_gain_pair(s: str) -> tuple[float, float]:
    """`'10,0.5'` -> (10.0, 0.5)."""
    try:
        kp_s, kd_s = s.split(",")
        kp = float(kp_s)
        kd = float(kd_s)
    except Exception as e:
        raise argparse.ArgumentTypeError(f"expected kp,kd got {s!r}") from e
    if not (SDK_KP_RANGE[0] <= kp <= SDK_KP_RANGE[1]):
        raise argparse.ArgumentTypeError(f"kp {kp} out of SDK range {SDK_KP_RANGE}")
    if not (SDK_KD_RANGE[0] <= kd <= SDK_KD_RANGE[1]):
        raise argparse.ArgumentTypeError(f"kd {kd} out of SDK range {SDK_KD_RANGE}")
    return kp, kd


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Piper MIT-mode step-response sysid")
    p.add_argument(
        "--joints", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6],
        choices=[1, 2, 3, 4, 5, 6],
        help="1-based joint indices to test (default: all six).",
    )
    p.add_argument(
        "--amplitudes-deg", type=float, nargs="+", default=[5.0, 10.0, 20.0],
        help="Step amplitudes in degrees (centered around start pose).",
    )
    p.add_argument(
        "--gains", type=parse_gain_pair, nargs="+", default=[(10.0, 0.5)],
        help="(kp,kd) pairs to sweep. Repeat the flag for multiple values, e.g. "
             "--gains 10,0.5 20,1.0 30,1.5",
    )
    p.add_argument(
        "--log-duration", type=float, default=3.0,
        help="Seconds to log after the step command (default 3.0).",
    )
    p.add_argument(
        "--hold-kp", type=float, default=20.0, help="kp for non-test joints (default 20).",
    )
    p.add_argument(
        "--hold-kd", type=float, default=1.0, help="kd for non-test joints (default 1.0).",
    )
    p.add_argument(
        "--reposition-vel", type=float, default=0.01,
        help="MIT-interpolated reposition speed (rad/s). Default 0.01 (~0.57 deg/s). "
             "At v=0.01 a 1 rad move takes 100s; raise to 0.05 for tractable times.",
    )
    p.add_argument(
        "--reposition-kp", type=float, default=15.0,
        help="kp used by the MIT reposition.",
    )
    p.add_argument(
        "--reposition-kd", type=float, default=1.0,
        help="kd used by the MIT reposition.",
    )
    p.add_argument(
        "--poses", type=str, default=POSES_PATH_DEFAULT,
        help="joint_isolation_poses.yaml path. Each joint's base pose for "
             "step tests is read from this yaml (same poses sysid_ramp uses). "
             "WITHOUT this, the default base pose [0,0,0,0,0,0] puts J2 and J3 "
             "at their joint limits — unsafe.",
    )
    p.add_argument(
        "--qdot-max", type=float, default=5.0,
        help="abort if |qdot| on test joint exceeds (rad/s). Permissive default: "
             "an underdamped PD step (low kd) easily produces 1-3 rad/s peaks even "
             "on small amplitudes; q_excursion_max is the real safety guard for steps.",
    )
    p.add_argument(
        "--q-excursion-max", type=float, default=0.5,
        help="abort if test joint travels beyond this from start (rad).",
    )
    p.add_argument(
        "--output-dir", type=str, default="/ws/sysid/runs",
        help="parent directory for results (default /ws/sysid/runs).",
    )
    p.add_argument("--plot", action="store_true", help="save PNG plots per shot.")
    p.add_argument(
        "--auto", action="store_true",
        help="skip per-joint interactive prompts (still waits for safety countdown).",
    )
    p.add_argument(
        "--simulate", action="store_true",
        help="dry-run: never publish /control/move_mit; useful to exercise the flow.",
    )
    return p.parse_args(argv)


# ----------------------------- per-shot core -----------------------------

def run_one_shot(
    node: SysidNode,
    joint_1based: int,
    amplitude_rad: float,
    kp: float,
    kd: float,
    hold: HoldGains,
    limits: SafetyLimits,
    log_path: str,
    log_duration: float,
    simulate: bool,
    reposition_vel: float,
    reposition_kp: float,
    reposition_kd: float,
    base_pose: list[float],
) -> tuple[list[float], list[float], list[float], list[float], int]:
    """Run a single step-response shot. Returns time/q/qd/tau arrays and the
    sample index of the step command, which the metric fitter uses as t=0.
    """
    # Reposition: take the per-joint base pose from the yaml (Pose A for
    # shoulders/elbow, Pose B for wrists — same poses sysid_ramp uses, all
    # validated to be safely inside joint limits), then bias the test joint
    # by -A/2 for the pre-step pose and +A/2 for the post-step pose.
    start_pose = list(base_pose)
    start_pose[joint_1based - 1] = base_pose[joint_1based - 1] - amplitude_rad / 2.0
    target_pose = list(start_pose)
    target_pose[joint_1based - 1] = base_pose[joint_1based - 1] + amplitude_rad / 2.0

    node.get_logger().info(
        f"j{joint_1based} amp={math.degrees(amplitude_rad):.1f}deg kp={kp} kd={kd}: "
        f"pre-positioning (slow MIT)"
    )
    node.move_to_mit(
        start_pose,
        v_des=reposition_vel,
        kp=reposition_kp,
        kd=reposition_kd,
    )

    # Snapshot start state for safety reference.
    s0 = node.latest_state()
    q_start = s0.q[joint_1based - 1]

    # Hold every joint at start with the requested PD via a single move_mit.
    n = NUM_ARM_JOINTS
    hold_kp = [hold.kp] * n
    hold_kd = [hold.kd] * n
    hold_p = list(start_pose)
    if not simulate:
        node.publish_mit(
            joint_index=list(range(1, n + 1)),
            p_des=hold_p,
            v_des=[0.0] * n,
            kp=hold_kp,
            kd=hold_kd,
            t_ff=[0.0] * n,
        )
    time.sleep(0.4)  # let the hold establish

    # Build the step command: only the test joint changes; others are
    # included so the publish snaps the whole MIT state coherently.
    step_kp = list(hold_kp)
    step_kd = list(hold_kd)
    step_p = list(start_pose)
    step_kp[joint_1based - 1] = kp
    step_kd[joint_1based - 1] = kd
    step_p[joint_1based - 1] = +amplitude_rad / 2.0

    # Begin logging *before* the step so we capture the pre-command baseline.
    pre_log = 0.2  # seconds before
    log_until = time.monotonic() + pre_log + log_duration

    t_list: list[float] = []
    q_list: list[float] = []
    qd_list: list[float] = []
    tau_list: list[float] = []
    step_sent_at: float | None = None
    step_idx = -1

    with CsvLogger(log_path) as csvw:
        t0 = time.monotonic()
        baseline_t_ns = None
        while time.monotonic() < log_until:
            rclpy.spin_once(node, timeout_sec=0.005)
            s = node.latest_state()
            if s is None:
                continue
            if baseline_t_ns is None:
                baseline_t_ns = s.t_ns
            # Issue the step exactly once, after pre-log seconds.
            if step_sent_at is None and (time.monotonic() - t0) >= pre_log:
                if not simulate:
                    node.publish_mit(
                        joint_index=list(range(1, n + 1)),
                        p_des=step_p,
                        v_des=[0.0] * n,
                        kp=step_kp,
                        kd=step_kd,
                        t_ff=[0.0] * n,
                    )
                step_sent_at = time.monotonic()
                step_idx = len(t_list)
                node.get_logger().info(f"j{joint_1based} step sent")
            # Hard safety guard.
            err = node.safety_check(s, joint_1based, q_start, limits)
            if err is not None:
                node.get_logger().error(f"SAFETY ABORT: {err}")
                node.emergency_hold()
                break
            t_list.append((s.t_ns - baseline_t_ns) * 1e-9)
            q_list.append(s.q[joint_1based - 1])
            qd_list.append(s.qd[joint_1based - 1])
            tau_list.append(s.tau[joint_1based - 1])
            csvw.write(s, phase="step", tau_cmd=0.0)

    # Always re-assert hold after a shot.
    if not simulate:
        node.publish_hold_all(start_pose, hold)
    return t_list, q_list, qd_list, tau_list, step_idx


# ----------------------------- metrics -----------------------------

def fit_step_metrics(
    t: list[float], q: list[float], qd: list[float],
    step_idx: int, amplitude_rad: float,
) -> dict:
    """Compute rise time, overshoot, settling time, peak velocity. Time axis
    is reset to t=0 at the step command. Returns NaN-friendly results."""
    if step_idx < 0 or step_idx >= len(t):
        return {
            "rise_time_s": float("nan"),
            "overshoot": float("nan"),
            "settling_time_s": float("nan"),
            "peak_velocity_rad_s": float("nan"),
            "steady_state_error_rad": float("nan"),
        }
    t_arr = np.asarray(t[step_idx:]) - t[step_idx]
    q_arr = np.asarray(q[step_idx:])
    qd_arr = np.asarray(qd[step_idx:])
    q0 = q[step_idx]
    target_delta = amplitude_rad  # step goes from -A/2 to +A/2, total delta = A
    q_after = q_arr - q0

    # Rise time: 10% -> 90% of step amplitude.
    t10 = t90 = float("nan")
    sign = 1.0 if target_delta >= 0 else -1.0
    target = sign * target_delta
    for i, v in enumerate(q_after):
        if math.isnan(t10) and sign * v >= 0.1 * target:
            t10 = float(t_arr[i])
        if sign * v >= 0.9 * target:
            t90 = float(t_arr[i])
            break
    rise_time = (t90 - t10) if not math.isnan(t10) and not math.isnan(t90) else float("nan")

    # Overshoot: max excursion past final / amplitude.
    q_final = float(q_after[-1])
    if abs(q_final) > 1e-6:
        peak = float(sign * np.max(sign * q_after))
        overshoot = max(0.0, (peak - target) / target) if target != 0 else float("nan")
    else:
        overshoot = float("nan")

    # Settling time: last time |q - target| > 0.05*target.
    band = 0.05 * abs(target)
    settling = float("nan")
    for i in range(len(q_after) - 1, -1, -1):
        if abs(q_after[i] - target) > band:
            if i + 1 < len(t_arr):
                settling = float(t_arr[i + 1])
            break
    if math.isnan(settling) and len(t_arr):
        settling = 0.0  # already inside band on first sample

    peak_velocity = float(np.max(np.abs(qd_arr))) if qd_arr.size else float("nan")
    sse = float(q_after[-1] - target) if q_after.size else float("nan")

    return {
        "rise_time_s": rise_time,
        "overshoot": overshoot,
        "settling_time_s": settling,
        "peak_velocity_rad_s": peak_velocity,
        "steady_state_error_rad": sse,
    }


# ----------------------------- plotting -----------------------------

def save_plot(
    path: str, t: list[float], q: list[float], qd: list[float],
    step_idx: int, amplitude_rad: float, joint_1based: int, kp: float, kd: float,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"matplotlib unavailable; skipping {path}", flush=True)
        return

    t_arr = np.asarray(t)
    q_arr = np.asarray(q)
    qd_arr = np.asarray(qd)
    t_step = t[step_idx] if 0 <= step_idx < len(t) else 0.0
    q_target = (q[step_idx] + amplitude_rad) if step_idx >= 0 else 0.0

    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    axes[0].plot(t_arr, np.degrees(q_arr), label="q (deg)")
    axes[0].axvline(t_step, color="k", lw=0.8, ls="--")
    if step_idx >= 0:
        axes[0].axhline(math.degrees(q_target), color="r", lw=0.8, ls=":", label="target")
    axes[0].set_ylabel("position (deg)")
    axes[0].set_title(
        f"joint{joint_1based}  amp={math.degrees(amplitude_rad):.1f}°  kp={kp} kd={kd}"
    )
    axes[0].legend(loc="best")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t_arr, qd_arr, label="qdot (rad/s)", color="C1")
    axes[1].axvline(t_step, color="k", lw=0.8, ls="--")
    axes[1].set_ylabel("velocity (rad/s)")
    axes[1].set_xlabel("time (s)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="best")

    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ----------------------------- main loop -----------------------------

def main(argv=None) -> int:
    args = parse_args(argv)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = os.path.join(args.output_dir, f"step_{stamp}")
    raw_dir = os.path.join(out_root, "raw")
    plots_dir = os.path.join(out_root, "plots")
    os.makedirs(raw_dir, exist_ok=True)
    if args.plot:
        os.makedirs(plots_dir, exist_ok=True)

    print(f"\n==== sysid_step: writing to {out_root} ====", flush=True)
    print(f"joints={args.joints} amps_deg={args.amplitudes_deg} gains={args.gains}")
    if args.simulate:
        print("*** SIMULATE MODE: no /control/move_mit publishes ***", flush=True)

    node = init_and_make_node("sysid_step", simulate=args.simulate)
    hold = HoldGains(kp=args.hold_kp, kd=args.hold_kd)
    limits = SafetyLimits(
        qdot_max=args.qdot_max, q_excursion_max=args.q_excursion_max,
    )

    # Load per-joint base poses from yaml.
    with open(args.poses) as f:
        poses_yaml = yaml.safe_load(f)

    summary: dict = {"args": vars(args), "results": {}}

    try:
        for j in args.joints:
            entry = poses_yaml.get(f"joint{j}")
            if entry is None:
                print(f"  -- no base pose for joint{j} in yaml; skipping", flush=True)
                continue
            base_pose = list(entry["hold"])
            cont = prompt_continue(
                f"--- ready to test joint{j} ({len(args.amplitudes_deg)} amplitudes × "
                f"{len(args.gains)} gain pairs).\n"
                f"    base pose: {[f'{v:+.3f}' for v in base_pose]}\n"
                f"    {entry.get('notes', '')}\n"
                f"    e-stop ready?",
                auto=args.auto,
            )
            if not cont:
                continue
            for amp_deg in args.amplitudes_deg:
                amp_rad = math.radians(amp_deg)
                for kp, kd in args.gains:
                    countdown(3, prefix=f"j{j} amp={amp_deg}° kp={kp} kd={kd}; starting in")
                    stem = f"joint{j}_amp{int(round(amp_deg))}_kp{kp}_kd{kd}"
                    csv_path = os.path.join(raw_dir, stem + ".csv")
                    t, q, qd, tau, idx = run_one_shot(
                        node=node, joint_1based=j, amplitude_rad=amp_rad,
                        kp=kp, kd=kd, hold=hold, limits=limits,
                        log_path=csv_path, log_duration=args.log_duration,
                        simulate=args.simulate,
                        reposition_vel=args.reposition_vel,
                        reposition_kp=args.reposition_kp,
                        reposition_kd=args.reposition_kd,
                        base_pose=base_pose,
                    )
                    metrics = fit_step_metrics(t, q, qd, idx, amp_rad)
                    key = f"joint{j}_amp{int(round(amp_deg))}_kp{kp}_kd{kd}"
                    summary["results"][key] = metrics
                    print(
                        f"  -> {key}: t_r={metrics['rise_time_s']:.3f}s "
                        f"OS={metrics['overshoot']:.2%} t_s={metrics['settling_time_s']:.3f}s "
                        f"qd_peak={metrics['peak_velocity_rad_s']:.2f} "
                        f"sse={metrics['steady_state_error_rad']:.4f}rad",
                        flush=True,
                    )
                    if args.plot:
                        save_plot(
                            os.path.join(plots_dir, stem + ".png"),
                            t, q, qd, idx, amp_rad, j, kp, kd,
                        )
        # Return to the last joint's base pose at the end (not [0]s which is
        # at joint limits). The user can manually move from there if needed.
    except KeyboardInterrupt:
        print("\nKeyboardInterrupt; emergency hold.", flush=True)
        node.emergency_hold()
    finally:
        with open(os.path.join(out_root, "summary.yaml"), "w") as f:
            yaml.safe_dump(summary, f, sort_keys=False)
        print(f"\nsummary -> {os.path.join(out_root, 'summary.yaml')}", flush=True)
        shutdown_node(node)
    return 0


if __name__ == "__main__":
    sys.exit(main())
