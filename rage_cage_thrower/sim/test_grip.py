"""Visually verify the ball weld holds across arm motion and releases cleanly.

Run via:  mjpython sim/test_grip.py     (macOS — launch_passive needs main-thread Cocoa)

Sequence: coils joint2 (shoulder pitch) backward for 2 seconds with the ball welded
to the gripper — sinusoid is centered well behind the home angle so the gripper
never swings forward into the cup. Then deactivates the weld and commands the
gripper open; the ball should fall free and bounce on the table.
"""
import time

import mujoco
import mujoco.viewer
import numpy as np

SCENE = "sim/mjcf/rage_cage.xml"

model = mujoco.MjModel.from_xml_path(SCENE)
data = mujoco.MjData(model)

key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "rage_home")
grip_eq_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, "ball_grip")
joint2_act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "joint2")
gripper_act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "gripper")

def init_episode():
    """Reset to keyframe, re-arm the weld, clear release state. Returns wallclock start time."""
    mujoco.mj_resetDataKeyframe(model, data, key_id)
    data.eq_active[grip_eq_id] = 1
    return time.time(), False


t0, released = init_episode()

with mujoco.viewer.launch_passive(model, data) as v:
    prev_sim_time = data.time
    while v.is_running():
        # Viewer reset (Backspace) calls mj_resetData which zeros qpos/qvel/data.time
        # but doesn't reload the keyframe or restore eq_active. Detect via time going
        # backward and re-run our full init.
        if data.time < prev_sim_time:
            t0, released = init_episode()
            print("episode reset")

        t = time.time() - t0
        # Center at 1.0 rad (~30° behind home), ±0.4 rad swing → range 0.6..1.4,
        # entirely on the "coil back" side of home. Gripper traces an arc up and
        # behind the base, never toward the cup.
        data.ctrl[joint2_act_id] = 1.0 + 0.4 * np.sin(2 * np.pi * 0.5 * t)
        if t > 2.0 and not released:
            data.eq_active[grip_eq_id] = 0
            data.ctrl[gripper_act_id] = 0.035
            released = True
            print("released")
        mujoco.mj_step(model, data)
        prev_sim_time = data.time
        v.sync()
        time.sleep(model.opt.timestep)
