"""Smoke-test the randomization-stage curriculum logic.

Verifies:

- ``RANDOMIZATION_STAGES`` covers R0-R3 with monotonically widening
  cup ranges centered on NOMINAL_CUP_XY[0], reaching ±10cm at R3.
- ``next_rand_stage`` promotes only when the per-stage success_rate
  threshold (0.5 / 0.4 / 0.3) is met, and returns None at R3 (the
  terminal stage).
- ``apply_rand_stage_to_env`` mutates the env's cup_range to the
  configured stage values, so the next reset() samples from the new
  box.

Run via:  uv run python -m sim.smoke_rand_curriculum
"""

from __future__ import annotations

import numpy as np

from sim.env import NOMINAL_CUP_XY, RageCageEnv
from sim.train_rl import (
    RANDOMIZATION_STAGES,
    apply_rand_stage_to_env,
    next_rand_stage,
)


def main() -> None:
    # R0..R3 span the expected widening box, all centered on the nominal
    # cup x. Final R3 stage matches the ±10cm operational envelope.
    expected_half_widths = {0: 0.02, 1: 0.05, 2: 0.08, 3: 0.10}
    for stage, cfg in RANDOMIZATION_STAGES.items():
        x_lo, x_hi = cfg["x_range"]
        y_lo, y_hi = cfg["y_range"]
        h = expected_half_widths[stage]
        assert abs((x_hi - x_lo) / 2 - h) < 1e-6, f"R{stage} x half-width != {h}"
        assert abs((y_hi - y_lo) / 2 - h) < 1e-6, f"R{stage} y half-width != {h}"
        assert abs((x_lo + x_hi) / 2 - NOMINAL_CUP_XY[0]) < 1e-6, f"R{stage} x not centered"
        assert abs((y_lo + y_hi) / 2) < 1e-6, f"R{stage} y not centered"
        print(f"R{stage}_ok x={cfg['x_range']} y={cfg['y_range']} promote_at={cfg['promote_at']}")

    # Promotion rules: thresholds 0.5 / 0.4 / 0.3, terminal R3.
    assert next_rand_stage(0, {"success_rate": 0.49}) is None
    assert next_rand_stage(0, {"success_rate": 0.50}) == 1
    assert next_rand_stage(1, {"success_rate": 0.39}) is None
    assert next_rand_stage(1, {"success_rate": 0.40}) == 2
    assert next_rand_stage(2, {"success_rate": 0.29}) is None
    assert next_rand_stage(2, {"success_rate": 0.30}) == 3
    assert next_rand_stage(3, {"success_rate": 1.00}) is None
    print("promotion_thresholds_ok")

    # Applying a stage to a live env updates the per-instance range.
    env = RageCageEnv(randomize_cup=True)
    apply_rand_stage_to_env(env, 3)
    samples = []
    for i in range(40):
        env.reset(seed=500 + i)
        samples.append(env.cup_xy.copy())
    arr = np.stack(samples)
    assert arr[:, 0].max() > float(NOMINAL_CUP_XY[0]) + 0.05, "R3 didn't widen x past NOMINAL+5cm"
    assert abs(arr[:, 1]).max() > 0.05, "R3 didn't widen y past 0.05"
    print(f"apply_rand_stage_R3_ok x=[{arr[:,0].min():.3f},{arr[:,0].max():.3f}] y=[{arr[:,1].min():.3f},{arr[:,1].max():.3f}]")

    apply_rand_stage_to_env(env, 0)
    samples = []
    for i in range(40):
        env.reset(seed=600 + i)
        samples.append(env.cup_xy.copy())
    arr = np.stack(samples)
    r0_x_lo, r0_x_hi = float(NOMINAL_CUP_XY[0]) - 0.02, float(NOMINAL_CUP_XY[0]) + 0.02
    assert (arr[:, 0] >= r0_x_lo - 1e-6).all() and (arr[:, 0] <= r0_x_hi + 1e-6).all()
    assert (arr[:, 1] >= -0.02 - 1e-6).all() and (arr[:, 1] <= 0.02 + 1e-6).all()
    print(f"apply_rand_stage_R0_ok x=[{arr[:,0].min():.3f},{arr[:,0].max():.3f}] y=[{arr[:,1].min():.3f},{arr[:,1].max():.3f}]")

    print("smoke_rand_curriculum OK")


if __name__ == "__main__":
    main()
