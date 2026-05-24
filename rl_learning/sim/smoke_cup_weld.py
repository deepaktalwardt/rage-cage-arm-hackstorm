"""Diagnose whether the cup_world weld snaps the cup back to (1.10, 0).

Resets the env with a +10cm override, takes physics steps without any
policy action, and prints the cup's world-frame XY at each step. If
the weld is pulling, we'll see the cup drift back toward (1.10, 0).

Run via:  uv run python -m sim.smoke_cup_weld
"""

from __future__ import annotations

import mujoco
import numpy as np

from sim.env import RageCageEnv


def main() -> None:
    env = RageCageEnv(randomize_cup=False)

    cup_world_eq_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_EQUALITY, "cup_world")
    print(f"cup_world_eq_id={cup_world_eq_id}")
    print(f"eq_data[cup_world] (initial) = {env.model.eq_data[cup_world_eq_id]}")
    print(f"eq_active[cup_world] = {env.data.eq_active[cup_world_eq_id]}")

    target = np.array([1.20, 0.10], dtype=np.float32)
    env.set_next_cup(target)
    env.reset(seed=0)
    print(f"\nafter reset: self.cup_xy={env.cup_xy.tolist()}")
    print(f"cup body xpos (post-reset): {env.data.xpos[env.cup_body_id].tolist()}")

    print("\nstepping physics with no actuation, no ball release:")
    for step in range(0, 80, 8):
        for _ in range(8):
            mujoco.mj_step(env.model, env.data)
        cup_xpos = env.data.xpos[env.cup_body_id]
        drift = float(np.linalg.norm(cup_xpos[:2] - target))
        print(
            f"  step={step + 8:3d} cup_xpos=({cup_xpos[0]:.4f}, {cup_xpos[1]:.4f}, {cup_xpos[2]:.4f}) "
            f"drift_from_target={drift:.4f}"
        )

    final_drift = float(np.linalg.norm(env.data.xpos[env.cup_body_id][:2] - target))
    if final_drift > 0.01:
        raise SystemExit(f"FAIL: cup drifted {final_drift:.4f}m from target — weld still pulling back")
    print("smoke_cup_weld OK — cup stays at target")


if __name__ == "__main__":
    main()
