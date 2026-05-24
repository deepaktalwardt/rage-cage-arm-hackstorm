"""Smoke-test fixed release at the configured ``release_step`` with 6-D action.

Verifies:

- ``action_space.shape == (6,)`` — joint deltas only; no release dim.
- Obs slot at index -1 is ``release_countdown`` in [0, 1], decreasing
  linearly from 1.0 at reset to 0.0 at the release step.
- Release fires automatically when ``step_count == release_step`` (default
  60); the env then runs the passive-flight loop and terminates.

Run via:  uv run python -m sim.smoke_fixed_release
"""

from __future__ import annotations

import numpy as np

from sim.env import RageCageEnv


def main() -> None:
    env = RageCageEnv(randomize_cup=False, reward_stage=3)
    release_step = env.release_step
    print(f"configured_release_step={release_step}")

    assert env.action_space.shape == (6,), f"expected 6-d action, got {env.action_space.shape}"
    print(f"action_space_ok shape={env.action_space.shape}")

    obs, info = env.reset(seed=0)
    countdown = float(obs[-1])
    assert 0.99 <= countdown <= 1.0, f"reset countdown should be ~1.0, got {countdown}"
    assert info["ball_released"] is False
    print(f"reset_ok release_countdown={countdown}")

    arm_zero = np.zeros(6, dtype=np.float32)

    # Step until just before release (release_step - 1 steps). Verify
    # countdown decreases linearly and ball stays welded.
    for i in range(1, release_step):
        obs, reward, terminated, truncated, info = env.step(arm_zero)
        countdown = float(obs[-1])
        expected = max(release_step - i, 0) / release_step
        assert abs(countdown - expected) < 1e-5, f"step {i}: countdown {countdown} != expected {expected}"
        assert info["ball_released"] is False, f"release fired prematurely at step {i}"
        assert not (terminated or truncated), f"unexpected termination at step {i}"
    print(f"countdown_decreases_ok {release_step - 1} steps, final pre-release countdown={countdown:.4f}")

    # Step `release_step`: release should fire automatically.
    obs, reward, terminated, truncated, info = env.step(arm_zero)
    assert info["ball_released"] is True, f"release didn't fire at step {release_step}"
    assert terminated or truncated, "release didn't run passive flight to termination"
    print(f"release_at_step_{release_step}_ok info_step_count={info['step_count']} terminated={terminated} truncated={truncated} reward={reward:.2f}")

    # After release, calling step again raises (env enforces single release).
    try:
        env.step(arm_zero)
    except RuntimeError as exc:
        print(f"post_release_step_blocks_ok: {exc}")
    else:
        raise AssertionError("env allowed step() after release without raising")

    print("smoke_fixed_release OK")


if __name__ == "__main__":
    main()
