#!/usr/bin/env python3
"""MuJoCo mirror of sysid_step.py — runs the same step-input protocol in
simulation and emits the same summary.yaml metric format, so the two outputs
can be directly compared.

This file is intentionally **standalone**: copy it to whatever environment
has `mujoco` and `numpy` installed. It does NOT import anything from
sysid_common; it duplicates the metric fitter to stay self-contained.

What it simulates (identical to sysid_step.run_one_shot on the real arm):

  1. qpos = base_pose, with the test joint biased to -A/2 from its base.
  2. Pre-step hold for `pre_hold_s` seconds with all joints under PD:
       (hold_kp, hold_kd, p_des = start_pose, t_ff = 0)
  3. Step: at t=pre_hold_s, switch the test joint's gains to (kp_test, kd_test)
     and its target to base + A/2. Other joints keep their hold gains.
  4. Log qpos/qvel/qfrc_applied for `log_duration_s` seconds at the MJCF's
     simulation rate (defaults to mjcf timestep).

Control law applied via `data.qfrc_applied` (one component per joint dof):

     τ = kp * (p_des - q) - kd * qd + t_ff

This bypasses the model's own actuators entirely so we don't have to match
the actuator type. It's the simulator equivalent of /control/move_mit.

CLI:
    python3 mujoco_step_runner.py --mjcf path/to/piper.xml \\
        --joints 1 2 3 --amplitudes-deg 5 10 --gains 10,0.5 --plot
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from datetime import datetime
from typing import Optional

import numpy as np
import yaml


# ----------------------------- argparse -----------------------------

def parse_gain_pair(s: str) -> tuple[float, float]:
    try:
        kp_s, kd_s = s.split(",")
        return float(kp_s), float(kd_s)
    except Exception as e:
        raise argparse.ArgumentTypeError(f"expected kp,kd got {s!r}") from e


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--mjcf", type=str, required=True,
                   help="path to the Piper MJCF (XML) file")
    p.add_argument("--joints", type=int, nargs="+", default=[1, 2, 3],
                   choices=[1, 2, 3, 4, 5, 6],
                   help="1-based joint indices to test")
    p.add_argument("--amplitudes-deg", type=float, nargs="+", default=[5.0, 10.0],
                   help="step amplitudes in degrees")
    p.add_argument("--gains", type=parse_gain_pair, nargs="+", default=[(10.0, 0.5)],
                   help="(kp,kd) test-joint gain pairs. Repeat the flag for sweeps.")
    p.add_argument("--hold-kp", type=float, default=20.0,
                   help="kp on non-test joints (matches sysid_step default)")
    p.add_argument("--hold-kd", type=float, default=1.0,
                   help="kd on non-test joints (matches sysid_step default)")
    p.add_argument("--pre-hold-s", type=float, default=0.4,
                   help="seconds of pre-step hold before the step is applied")
    p.add_argument("--log-duration-s", type=float, default=3.0,
                   help="seconds to simulate and log after the step command")
    p.add_argument("--poses", type=str,
                   default=os.path.join(os.path.dirname(__file__), "joint_isolation_poses.yaml"),
                   help="joint_isolation_poses.yaml (same one sysid_ramp uses)")
    p.add_argument("--joint-names", type=str, nargs="+",
                   default=[f"joint{i}" for i in range(1, 7)],
                   help="MJCF joint names in joint1..joint6 order. Override if your "
                        "MJCF names them differently.")
    p.add_argument("--output-dir", type=str, default="./mujoco_runs",
                   help="parent directory for results")
    p.add_argument("--plot", action="store_true", help="save PNG plots per shot")
    return p.parse_args(argv)


# ----------------------------- simulation -----------------------------

def run_one_shot(
    mjcf_path: str,
    joint_names: list[str],
    joint_1based: int,
    base_pose: list[float],
    amplitude_rad: float,
    kp_test: float,
    kd_test: float,
    hold_kp: float,
    hold_kd: float,
    pre_hold_s: float,
    log_duration_s: float,
):
    import mujoco

    model = mujoco.MjModel.from_xml_path(mjcf_path)
    data = mujoco.MjData(model)
    dt = float(model.opt.timestep)

    # Map joint name -> (qpos_addr, dof_addr).
    qpos_adr = []
    dof_adr = []
    for name in joint_names:
        try:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        except Exception:
            jid = -1
        if jid < 0:
            raise RuntimeError(f"joint {name!r} not found in {mjcf_path}")
        qpos_adr.append(int(model.jnt_qposadr[jid]))
        dof_adr.append(int(model.jnt_dofadr[jid]))

    j = joint_1based - 1
    # Start: base pose with test joint biased to -A/2.
    start_pose = list(base_pose)
    start_pose[j] = base_pose[j] - amplitude_rad / 2.0
    target_p_test = base_pose[j] + amplitude_rad / 2.0

    # Apply start pose to qpos.
    for i, adr in enumerate(qpos_adr):
        data.qpos[adr] = start_pose[i]
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    # Per-joint PD setpoints during the run.
    p_des = list(start_pose)
    kps = [hold_kp] * 6
    kds = [hold_kd] * 6

    # Discretize into pre-hold + log windows.
    n_pre = max(1, int(round(pre_hold_s / dt)))
    n_log = max(1, int(round(log_duration_s / dt)))
    step_idx = n_pre  # the step command lands at this index

    t_list: list[float] = []
    q_list: list[float] = []
    qd_list: list[float] = []
    tau_list: list[float] = []

    step_applied = False
    for k in range(n_pre + n_log):
        # Switch to test gains at step_idx.
        if not step_applied and k >= step_idx:
            kps[j] = kp_test
            kds[j] = kd_test
            p_des[j] = target_p_test
            step_applied = True

        # MIT-style PD law: τ = kp*(p_des - q) - kd*qd. No t_ff (zero).
        for i, (qadr, dadr) in enumerate(zip(qpos_adr, dof_adr)):
            q = float(data.qpos[qadr])
            qd = float(data.qvel[dadr])
            tau = kps[i] * (p_des[i] - q) - kds[i] * qd
            data.qfrc_applied[dadr] = tau

        mujoco.mj_step(model, data)

        # Log AFTER the step so the very first sample reflects the applied torque.
        t_list.append(k * dt)
        q_list.append(float(data.qpos[qpos_adr[j]]))
        qd_list.append(float(data.qvel[dof_adr[j]]))
        tau_list.append(float(data.qfrc_applied[dof_adr[j]]))

    return t_list, q_list, qd_list, tau_list, step_idx


# ----------------------------- metric fitting -----------------------------
# (Identical algorithm to sysid_step._fit_step_metrics, duplicated here so
# this script has no project-local imports.)

def fit_step_metrics(
    t: list[float], q: list[float], qd: list[float],
    step_idx: int, amplitude_rad: float,
) -> dict:
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
    target_delta = amplitude_rad
    q_after = q_arr - q0

    sign = 1.0 if target_delta >= 0 else -1.0
    target = sign * target_delta

    t10 = t90 = float("nan")
    for i, v in enumerate(q_after):
        if math.isnan(t10) and sign * v >= 0.1 * target:
            t10 = float(t_arr[i])
        if sign * v >= 0.9 * target:
            t90 = float(t_arr[i])
            break
    rise_time = (t90 - t10) if not math.isnan(t10) and not math.isnan(t90) else float("nan")

    q_final = float(q_after[-1])
    if abs(q_final) > 1e-6:
        peak = float(sign * np.max(sign * q_after))
        overshoot = max(0.0, (peak - target) / target) if target != 0 else float("nan")
    else:
        overshoot = float("nan")

    band = 0.05 * abs(target)
    settling = float("nan")
    for i in range(len(q_after) - 1, -1, -1):
        if abs(q_after[i] - target) > band:
            if i + 1 < len(t_arr):
                settling = float(t_arr[i + 1])
            break
    if math.isnan(settling) and len(t_arr):
        settling = 0.0

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
        f"[MuJoCo] joint{joint_1based}  amp={math.degrees(amplitude_rad):.1f}°  "
        f"kp={kp} kd={kd}"
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


# ----------------------------- CSV dumping -----------------------------

def write_csv(path: str, t: list, q: list, qd: list, tau: list, step_idx: int) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_s", "step_applied", "q_rad", "qd_rad_s", "tau_applied_Nm"])
        for k, (ti, qi, qdi, taui) in enumerate(zip(t, q, qd, tau)):
            w.writerow([f"{ti:.6f}", int(k >= step_idx), f"{qi:.6f}", f"{qdi:.6f}", f"{taui:.6f}"])


# ----------------------------- main loop -----------------------------

def main(argv=None) -> int:
    args = parse_args(argv)

    try:
        import mujoco  # noqa: F401
    except ImportError as e:
        print(f"\nERROR: mujoco package not importable in this Python env: {e}", file=sys.stderr)
        print("Install with: pip install mujoco", file=sys.stderr)
        return 2

    if not os.path.exists(args.mjcf):
        print(f"ERROR: MJCF not found: {args.mjcf}", file=sys.stderr)
        return 2

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = os.path.join(args.output_dir, f"sim_step_{stamp}")
    raw_dir = os.path.join(out_root, "raw")
    plots_dir = os.path.join(out_root, "plots")
    os.makedirs(raw_dir, exist_ok=True)
    if args.plot:
        os.makedirs(plots_dir, exist_ok=True)

    with open(args.poses) as f:
        poses_yaml = yaml.safe_load(f)

    print(f"==== mujoco_step_runner: writing to {out_root} ====", flush=True)
    print(f"mjcf={args.mjcf} joints={args.joints} amps_deg={args.amplitudes_deg} "
          f"gains={args.gains}", flush=True)

    summary: dict = {"args": vars(args), "results": {}}

    for j in args.joints:
        entry = poses_yaml.get(f"joint{j}")
        if entry is None:
            print(f"  -- no base pose for joint{j} in yaml; skipping", flush=True)
            continue
        base_pose = list(entry["hold"])
        for amp_deg in args.amplitudes_deg:
            amp_rad = math.radians(amp_deg)
            for kp, kd in args.gains:
                stem = f"joint{j}_amp{int(round(amp_deg))}_kp{kp}_kd{kd}"
                t, q, qd, tau, idx = run_one_shot(
                    mjcf_path=args.mjcf,
                    joint_names=args.joint_names,
                    joint_1based=j,
                    base_pose=base_pose,
                    amplitude_rad=amp_rad,
                    kp_test=kp, kd_test=kd,
                    hold_kp=args.hold_kp, hold_kd=args.hold_kd,
                    pre_hold_s=args.pre_hold_s,
                    log_duration_s=args.log_duration_s,
                )
                metrics = fit_step_metrics(t, q, qd, idx, amp_rad)
                summary["results"][stem] = metrics
                print(
                    f"  -> {stem}: t_r={metrics['rise_time_s']:.3f}s "
                    f"OS={metrics['overshoot']:.2%} t_s={metrics['settling_time_s']:.3f}s "
                    f"qd_peak={metrics['peak_velocity_rad_s']:.2f} "
                    f"sse={metrics['steady_state_error_rad']:.4f}rad",
                    flush=True,
                )
                write_csv(os.path.join(raw_dir, stem + ".csv"), t, q, qd, tau, idx)
                if args.plot:
                    save_plot(
                        os.path.join(plots_dir, stem + ".png"),
                        t, q, qd, idx, amp_rad, j, kp, kd,
                    )

    with open(os.path.join(out_root, "summary.yaml"), "w") as f:
        yaml.safe_dump(summary, f, sort_keys=False)
    print(f"\nsummary -> {os.path.join(out_root, 'summary.yaml')}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
