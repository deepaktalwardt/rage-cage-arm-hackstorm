"""Verify MuJoCo loads the PiPER scene, steps, and renders offscreen on M1.

Primer gotcha 7.2 flagged Mac offscreen rendering as a day-0 risk — this is the check.

Pass --print-link6-pose to print link6 world pose and the gripper finger-pad world
positions at the home keyframe. Used to compute the ball_grip weld relpose.
"""

import argparse
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

SCENE = Path(__file__).parent / "mjcf" / "rage_cage.xml"
OUT_DIR = Path(__file__).parent / "_smoke_out"
WIDTH, HEIGHT = 640, 480


def print_link6_pose(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    """Print link6 world pose and the world positions of the two finger-pad outer geoms.

    The midpoint of the two outer-pad world positions is where the welded ball should sit.
    Inverting link6's world transform on that midpoint gives the weld relpose in link6 frame.
    """
    mujoco.mj_kinematics(model, data)

    link6_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "link6")
    link6_pos = data.xpos[link6_id].copy()
    link6_quat = data.xquat[link6_id].copy()
    link6_mat = data.xmat[link6_id].reshape(3, 3).copy()
    print(f"link6 world pos:  {np.array2string(link6_pos, precision=5)}")
    print(f"link6 world quat: {np.array2string(link6_quat, precision=5)}")

    # The outer finger pads are the 2nd collision box on each of link7/link8 (the larger
    # 0.015 x 0.015 x 0.0025 ones at finger-local (0, ±0.045, -0.0025)). They have no name,
    # so locate them by body. Each finger body has: 1 mesh visual + 2 collision boxes;
    # we want the last (outer) collision box on each.
    pad_positions = []
    for body_name in ("link7", "link8"):
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        # Geoms attached to this body: pick the last one (outer pad, finger-y = ±0.045).
        body_geom_ids = [g for g in range(model.ngeom) if model.geom_bodyid[g] == body_id]
        outer_pad_geom = body_geom_ids[-1]
        pad_world = data.geom_xpos[outer_pad_geom].copy()
        print(f"  {body_name} outer-pad geom (id={outer_pad_geom}) world pos: "
              f"{np.array2string(pad_world, precision=5)}")
        pad_positions.append(pad_world)

    midpoint_world = (pad_positions[0] + pad_positions[1]) / 2
    print(f"pad midpoint world (target ball pos): {np.array2string(midpoint_world, precision=5)}")

    # Express midpoint in link6 local frame: relpose = R_link6^T @ (midpoint - link6_pos)
    relpose_local = link6_mat.T @ (midpoint_world - link6_pos)
    print(f"ball_grip weld relpose (link6 frame): {np.array2string(relpose_local, precision=5)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--print-link6-pose", action="store_true",
                        help="Print link6 world pose and pad geom positions for weld setup.")
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path(str(SCENE))
    data = mujoco.MjData(model)

    print(f"model: nq={model.nq} nv={model.nv} nu={model.nu} (actuators)")
    print(f"timestep: {model.opt.timestep}s   integrator: {model.opt.integrator}")

    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "rage_home")
    mujoco.mj_resetDataKeyframe(model, data, key_id)
    print(f"reset to keyframe 'rage_home' (id={key_id})")
    print(f"qpos: {np.array2string(data.qpos, precision=3)}")
    print(f"ctrl: {np.array2string(data.ctrl, precision=3)}")

    if args.print_link6_pose:
        print_link6_pose(model, data)
        return

    for _ in range(100):
        mujoco.mj_step(model, data)
    print(f"after 100 steps qpos: {np.array2string(data.qpos, precision=3)}")

    with mujoco.Renderer(model, height=HEIGHT, width=WIDTH) as renderer:
        renderer.update_scene(data)
        frame = renderer.render()

    print(f"render: shape={frame.shape} dtype={frame.dtype} "
          f"min={frame.min()} max={frame.max()} mean={frame.mean():.1f}")

    assert frame.shape == (HEIGHT, WIDTH, 3), "unexpected render shape"
    assert frame.dtype == np.uint8, "unexpected render dtype"
    assert frame.max() > 0, "render returned all-black frame"

    OUT_DIR.mkdir(exist_ok=True)
    out_path = OUT_DIR / "home_pose.png"
    Image.fromarray(frame).save(out_path)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
