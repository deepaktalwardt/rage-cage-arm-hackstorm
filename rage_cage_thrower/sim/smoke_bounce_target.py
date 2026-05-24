"""Smoke-test the cup-aware elliptical bounce-target geometry.

Verifies the new bounce-reward geometry (env.bounce_score and env._bounce_target):

- ``_bounce_target(cup_xy) = BOUNCE_FRACTION * cup_xy`` — sits on the
  line from arm base (world origin) to cup, NOT a fixed offset behind it.
  This rotates correctly when the cup is off-axis.
- ``bounce_score(target, cup_xy) == 1.0`` — bounce exactly at target.
- ``bounce_score(point at SIGMA_LONG along throw_dir, cup_xy) == 0.0`` —
  on the long-axis ellipse boundary.
- ``bounce_score(point at SIGMA_PERP perpendicular to throw_dir, cup_xy) == 0.0`` —
  on the perp-axis ellipse boundary.
- For an off-axis cup (e.g. (0.85, +0.10)), the score function correctly
  rotates the ellipse: the perpendicular-error direction is no longer
  world y, but the world-frame normal to the throw axis.

Run via:  uv run python -m sim.smoke_bounce_target
"""

from __future__ import annotations

import numpy as np

from sim.env import (
    BOUNCE_FRACTION,
    BOUNCE_SIGMA_LONG,
    BOUNCE_SIGMA_PERP,
    NOMINAL_CUP_XY,
    RageCageEnv,
    bounce_score,
)


def main() -> None:
    env = RageCageEnv(randomize_cup=False)

    # Target sits at α * cup_xy. At nominal cup (0.85, 0) and α=0.7, target ≈ (0.595, 0).
    nominal_cup = np.asarray(NOMINAL_CUP_XY, dtype=np.float64)
    env.cup_xy = NOMINAL_CUP_XY.astype(np.float32).copy()
    target = env._bounce_target()
    expected = BOUNCE_FRACTION * nominal_cup
    assert np.allclose(target, expected, atol=1e-6), f"nominal target {target} != {expected}"
    print(f"nominal_target_ok cup={nominal_cup.tolist()} target={target.tolist()}")

    # bounce AT target → score 1.
    score_at_target = bounce_score(target, nominal_cup)
    assert abs(score_at_target - 1.0) < 1e-6, f"score at target should be 1.0, got {score_at_target}"
    print(f"score_at_target_ok score={score_at_target:.4f}")

    # SIGMA_LONG along throw direction → score 0 (boundary).
    throw_dir = nominal_cup / np.linalg.norm(nominal_cup)
    along_pt = target + BOUNCE_SIGMA_LONG * throw_dir
    score_along = bounce_score(along_pt, nominal_cup)
    assert abs(score_along) < 1e-6, f"score on long-axis boundary should be 0, got {score_along}"
    print(f"long_axis_boundary_ok score={score_along:.4f}")

    # SIGMA_PERP perpendicular → score 0.
    perp_dir = np.array([-throw_dir[1], throw_dir[0]])
    perp_pt = target + BOUNCE_SIGMA_PERP * perp_dir
    score_perp = bounce_score(perp_pt, nominal_cup)
    assert abs(score_perp) < 1e-6, f"score on perp-axis boundary should be 0, got {score_perp}"
    print(f"perp_axis_boundary_ok score={score_perp:.4f}")

    # Outside the ellipse → score clipped to 0.
    far = target + np.array([1.0, 1.0])
    score_far = bounce_score(far, nominal_cup)
    assert score_far == 0.0, f"score outside ellipse should be 0 (clipped), got {score_far}"
    print(f"outside_ellipse_clipped_ok score={score_far:.4f}")

    # Off-axis cup: the ellipse must rotate with the throw axis. For cup
    # at (0.85, 0.30), the throw direction is roughly 19° off from world
    # x. A bounce displaced PERPENDICULAR to the throw axis (in world
    # frame) by SIGMA_PERP should score 0. A displacement of the same
    # magnitude purely in world y (which is NOT perpendicular to the
    # throw axis here) should score above 0 because part of it falls
    # along the more-tolerant long axis.
    cup_off = np.array([0.85, 0.30])
    target_off = BOUNCE_FRACTION * cup_off
    throw_off = cup_off / np.linalg.norm(cup_off)
    perp_off = np.array([-throw_off[1], throw_off[0]])
    pt_truly_perp = target_off + BOUNCE_SIGMA_PERP * perp_off
    score_truly_perp = bounce_score(pt_truly_perp, cup_off)
    assert abs(score_truly_perp) < 1e-6, f"truly-perp boundary should score 0, got {score_truly_perp}"
    pt_world_y = target_off + np.array([0.0, BOUNCE_SIGMA_PERP])
    score_world_y = bounce_score(pt_world_y, cup_off)
    assert score_world_y > 0.05, f"world-y displacement should score >0 for off-axis cup, got {score_world_y}"
    print(f"off_axis_rotation_ok truly_perp_score={score_truly_perp:.4f} world_y_score={score_world_y:.4f}")

    # ±10cm sweep: target tracks cup naturally.
    for cup in [(0.75, 0.0), (0.95, 0.10), (0.95, -0.10), (0.85, -0.10)]:
        env.cup_xy = np.array(cup, dtype=np.float32)
        t = env._bounce_target()
        expected_t = BOUNCE_FRACTION * np.asarray(cup, dtype=np.float64)
        assert np.allclose(t, expected_t, atol=1e-6), f"cup={cup}: target {t} != {expected_t}"
    print(f"sweep_ok all 4 corner-ish cups")

    print("smoke_bounce_target OK")


if __name__ == "__main__":
    main()
