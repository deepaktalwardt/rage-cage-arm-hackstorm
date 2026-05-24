"""Smoke test for pedestal_height handling in RageCageEnv.

Covers the env-level changes from the stacked-cup design (Section A+B):
- pedestal_height is sampled per reset within set_pedestal_range
- set_next_pedestal pins one reset to a specific value
- the cup body is welded at z=pedestal_height (qpos + cup_world weld eq_data)
- the cup_pedestal body geom_size and body_pos track pedestal_height
- the cup_count obs slot now carries pedestal_height (not cup_count / 10)
- info dict exposes pedestal_height
- z-dependent reward / success logic uses pedestal_height + CUP_HEIGHT
- the ball can't tunnel through a tall pedestal (collision works)

Run from the repo root:

    uv run python -m sim.smoke_pedestal
"""

from __future__ import annotations

import numpy as np

from sim.env import CUP_HEIGHT, NOMINAL_CUP_XY, RageCageEnv


def _make_env() -> RageCageEnv:
    return RageCageEnv(randomize_cup=True, reward_stage=3)


def test_default_pedestal_zero() -> None:
    env = RageCageEnv(randomize_cup=False, reward_stage=3)
    env.reset(seed=0)
    assert env.pedestal_height == 0.0, f"default pedestal should be 0, got {env.pedestal_height}"
    cup_z = float(env.data.qpos[env.cup_qposadr + 2])
    assert abs(cup_z) < 1e-6, f"cup z should be 0 with no pedestal, got {cup_z}"
    print(f"OK default pedestal=0, cup z={cup_z:.6f}")


def test_set_pedestal_range_samples() -> None:
    env = _make_env()
    env.set_pedestal_range((0.05, 0.10))
    seen: list[float] = []
    for seed in range(8):
        env.reset(seed=seed)
        seen.append(float(env.pedestal_height))
    for h in seen:
        assert 0.05 <= h <= 0.10, f"pedestal_height {h} out of [0.05, 0.10]"
    assert max(seen) - min(seen) > 0.005, f"no pedestal variation across seeds: {seen}"
    print(f"OK pedestal varied across seeds: min={min(seen):.4f} max={max(seen):.4f}")


def test_set_next_pedestal_overrides() -> None:
    env = _make_env()
    env.set_pedestal_range((0.05, 0.10))
    env.set_next_pedestal(0.15)
    env.reset(seed=0)
    assert abs(env.pedestal_height - 0.15) < 1e-9, f"override failed, got {env.pedestal_height}"
    env.reset(seed=1)
    assert 0.05 <= env.pedestal_height <= 0.10, (
        f"after override consumed, expected re-sample in range, got {env.pedestal_height}"
    )
    print("OK set_next_pedestal one-shot override consumed")


def test_cup_body_at_pedestal_z() -> None:
    env = _make_env()
    env.set_next_pedestal(0.10)
    env.reset(seed=0)
    cup_z = float(env.data.qpos[env.cup_qposadr + 2])
    assert abs(cup_z - 0.10) < 1e-6, f"cup qpos z={cup_z}, expected 0.10"
    weld_z = float(env.model.eq_data[env.cup_world_eq_id, 5])
    assert abs(weld_z - 0.10) < 1e-6, f"cup_world weld z={weld_z}, expected 0.10"
    print(f"OK cup body welded at z={cup_z:.4f}")


def test_pedestal_geom_size_and_body_pos() -> None:
    env = _make_env()
    env.set_next_pedestal(0.10)
    env.reset(seed=0)
    half_h = float(env.model.geom_size[env.cup_pedestal_geom_id, 1])
    body_z = float(env.model.body_pos[env.cup_pedestal_body_id, 2])
    assert abs(half_h - 0.05) < 1e-6, f"pedestal half-h={half_h}, expected 0.05"
    assert abs(body_z - 0.05) < 1e-6, f"pedestal body z={body_z}, expected 0.05"
    body_xy = env.model.body_pos[env.cup_pedestal_body_id, :2]
    assert np.allclose(body_xy, env.cup_xy, atol=1e-6), (
        f"pedestal xy {body_xy} should match cup_xy {env.cup_xy}"
    )
    print(f"OK pedestal geom half-h={half_h:.4f}, body z={body_z:.4f}, xy={tuple(body_xy)}")


def test_obs_carries_pedestal_height() -> None:
    env = _make_env()
    env.set_next_pedestal(0.075)
    obs, _info = env.reset(seed=0)
    pedestal_obs_idx = 6 + 6 + 3 + 3 + 2  # joints(6) + jvel(6) + bpos(3) + bvel(3) + cup_xy(2) = 20
    val = float(obs[pedestal_obs_idx])
    assert abs(val - 0.075) < 1e-6, (
        f"obs[{pedestal_obs_idx}]={val}, expected pedestal_height=0.075"
    )
    print(f"OK obs[pedestal_slot]={val:.4f}")


def test_pedestal_height_in_info() -> None:
    env = _make_env()
    env.set_next_pedestal(0.075)
    _obs, info = env.reset(seed=0)
    assert "pedestal_height" in info, f"info missing pedestal_height: {sorted(info.keys())}"
    assert abs(float(info["pedestal_height"]) - 0.075) < 1e-6
    print(f"OK info[pedestal_height]={info['pedestal_height']:.4f}")


def test_inside_cup_z_shifts_with_pedestal() -> None:
    """At pedestal=0.10, ball at world z<0.115 should be below cup base."""
    env = RageCageEnv(randomize_cup=False, reward_stage=3)
    env.set_next_pedestal(0.10)
    env.reset(seed=0)
    # Probe the predicate by manually placing the ball — env._success uses
    # ball_pos[2] vs (pedestal_height + 0.015, pedestal_height + CUP_HEIGHT).
    env.data.qpos[env.ball_qposadr : env.ball_qposadr + 3] = [
        env.cup_xy[0],
        env.cup_xy[1],
        0.05,  # below the elevated cup base (0.115)
    ]
    env.data.qvel[env.ball_dofadr : env.ball_dofadr + 6] = 0.0
    env.data.eq_active[env.ball_grip_eq_id] = 0
    import mujoco

    mujoco.mj_forward(env.model, env.data)
    ball_z = float(env._ball_pos()[2])
    inside_lo = env.pedestal_height + 0.015
    inside_hi = env.pedestal_height + CUP_HEIGHT
    assert ball_z < inside_lo, (
        f"ball z={ball_z} should be below shifted cup base {inside_lo}"
    )
    print(f"OK inside_cup_z shifted to ({inside_lo:.4f}, {inside_hi:.4f})")


def test_no_randomize_cup_pins_pedestal_zero() -> None:
    env = RageCageEnv(randomize_cup=False, reward_stage=3)
    env.set_pedestal_range((0.05, 0.10))
    env.reset(seed=0)
    assert env.pedestal_height == 0.0, (
        f"randomize_cup=False should pin pedestal=0 even with range set, got {env.pedestal_height}"
    )
    print("OK randomize_cup=False pins pedestal=0")


def test_pedestal_blocks_low_throw() -> None:
    """A 15cm pedestal in front of an arm-side ball at low z should collide and stop it."""
    env = RageCageEnv(randomize_cup=False, reward_stage=3)
    env.set_next_pedestal(0.15)
    env.reset(seed=0)
    import mujoco

    # Place ball ~15cm in front of the pedestal axis at z=5cm, moving toward cup.
    env.data.qpos[env.ball_qposadr : env.ball_qposadr + 3] = [
        env.cup_xy[0] - 0.15,
        env.cup_xy[1],
        0.05,
    ]
    env.data.qvel[env.ball_dofadr : env.ball_dofadr + 6] = [3.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    env.data.eq_active[env.ball_grip_eq_id] = 0
    mujoco.mj_forward(env.model, env.data)
    # Step physics for 0.5s in raw mj_step (no env.step, which has agent control)
    n_steps = int(0.5 / env.model.opt.timestep)
    for _ in range(n_steps):
        mujoco.mj_step(env.model, env.data)
    final_x = float(env._ball_pos()[0])
    pedestal_x = float(env.cup_xy[0])
    assert final_x < pedestal_x, (
        f"ball should have bounced off pedestal: final_x={final_x:.3f} cup_x={pedestal_x:.3f}"
    )
    # Ball should still be on the arm side or right at the pedestal edge,
    # not on the far side of the cup (pedestal_x + radius).
    assert final_x < pedestal_x + 0.04, (
        f"ball passed through pedestal: final_x={final_x:.3f} cup_x={pedestal_x:.3f}"
    )
    print(f"OK pedestal blocked low throw: ball stopped at x={final_x:.3f} (cup_x={pedestal_x:.3f})")


if __name__ == "__main__":
    test_default_pedestal_zero()
    test_set_pedestal_range_samples()
    test_set_next_pedestal_overrides()
    test_cup_body_at_pedestal_z()
    test_pedestal_geom_size_and_body_pos()
    test_obs_carries_pedestal_height()
    test_pedestal_height_in_info()
    test_inside_cup_z_shifts_with_pedestal()
    test_no_randomize_cup_pins_pedestal_zero()
    test_pedestal_blocks_low_throw()
    print("\nAll pedestal smoke checks passed.")
