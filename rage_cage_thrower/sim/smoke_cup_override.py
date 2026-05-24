"""Smoke-test the per-reset cup-position override and the dynamic cup-range setter.

Verifies the two new env hooks needed by the multi-position training run:

- ``set_next_cup(xy)`` consumes a one-shot override on the next ``reset()``,
  then clears itself so subsequent resets fall back to the configured
  randomization. Used by the ``grid``-mode evaluator to land cup at a fixed
  set of positions.
- ``set_cup_range(x_range, y_range)`` mutates the per-episode randomization
  ranges so the curriculum can grow the workspace as training proceeds.

Run via:  python -m sim.smoke_cup_override
"""

from __future__ import annotations

import numpy as np

from sim.env import NOMINAL_CUP_XY, RageCageEnv


def main() -> None:
    env = RageCageEnv(randomize_cup=True)
    env.reset(seed=0)

    # Override consumes the next reset.
    target = np.array([1.18, 0.07], dtype=np.float32)
    env.set_next_cup(target)
    env.reset()
    assert np.allclose(env.cup_xy, target), f"override not applied: {env.cup_xy} != {target}"
    print(f"override_applied cup_xy={env.cup_xy.tolist()}")

    # Subsequent reset falls back to randomization. Sample several to confirm.
    follow_xys = []
    for i in range(4):
        env.reset(seed=10 + i)
        follow_xys.append(env.cup_xy.copy())
    follow = np.stack(follow_xys)
    assert not np.allclose(follow, target), "override leaked past one reset"
    print(f"override_cleared follow_cup_xys={follow.tolist()}")

    # Range setter shrinks/grows the randomization box.
    env.set_cup_range((1.05, 1.15), (-0.05, 0.05))
    samples = []
    for i in range(20):
        env.reset(seed=100 + i)
        samples.append(env.cup_xy.copy())
    sample_arr = np.stack(samples)
    in_x = (sample_arr[:, 0] >= 1.05 - 1e-6) & (sample_arr[:, 0] <= 1.15 + 1e-6)
    in_y = (sample_arr[:, 1] >= -0.05 - 1e-6) & (sample_arr[:, 1] <= 0.05 + 1e-6)
    assert in_x.all() and in_y.all(), f"out-of-range samples: {sample_arr[~(in_x & in_y)]}"
    print(
        f"range_applied n={len(samples)} "
        f"x=[{sample_arr[:,0].min():.4f}, {sample_arr[:,0].max():.4f}] "
        f"y=[{sample_arr[:,1].min():.4f}, {sample_arr[:,1].max():.4f}]"
    )

    # Setting back to default (or a wider range) should reflect immediately.
    env.set_cup_range((1.00, 1.20), (-0.10, 0.10))
    wide = []
    for i in range(40):
        env.reset(seed=200 + i)
        wide.append(env.cup_xy.copy())
    wide_arr = np.stack(wide)
    assert wide_arr[:, 0].max() > 1.15, "wide range x-max didn't extend past prior 1.15"
    assert abs(wide_arr[:, 1]).max() > 0.05, "wide range y-extent didn't grow past prior 0.05"
    print(
        f"wide_range_applied "
        f"x=[{wide_arr[:,0].min():.4f}, {wide_arr[:,0].max():.4f}] "
        f"y=[{wide_arr[:,1].min():.4f}, {wide_arr[:,1].max():.4f}]"
    )

    # Sanity check: nominal-cup baseline is unaffected.
    nominal_env = RageCageEnv(randomize_cup=False)
    nominal_env.reset(seed=0)
    assert np.allclose(nominal_env.cup_xy, NOMINAL_CUP_XY), "nominal default broken"
    print(f"nominal_unchanged cup_xy={nominal_env.cup_xy.tolist()}")

    print("smoke_cup_override OK")


if __name__ == "__main__":
    main()
