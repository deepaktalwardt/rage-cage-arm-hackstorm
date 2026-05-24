"""Verify that `StubEnv` is shaped correctly for `VecNormalize.load`.

We need a Gym env with matching observation_space + action_space so the
controller can recover obs-normalization stats from the training-time
`vecnormalize.pkl` without dragging MuJoCo + `sim/` onto the container.
"""

from __future__ import annotations

from pathlib import Path

from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from real.stub_env import StubEnv

MODEL_DIR = (
    Path(__file__).resolve().parents[2]
    / "models"
    / "random_stack_cup_thrower_no_ball_obs_v1"
)


def test_vecnormalize_loads_with_stub_env() -> None:
    vecnorm_pkl = MODEL_DIR / "vecnormalize.pkl"
    assert vecnorm_pkl.exists(), f"missing test fixture: {vecnorm_pkl}"

    dummy = DummyVecEnv([lambda: StubEnv()])
    vn = VecNormalize.load(str(vecnorm_pkl), dummy)

    # 16-d obs: joint_pos(6), joint_vel(6), cup_xy(2), pedestal(1), countdown(1).
    assert vn.obs_rms.mean.shape == (16,)
    assert vn.obs_rms.var.shape == (16,)


def test_vecnormalize_loads_with_22dim_stub_env() -> None:
    """The legacy v1 model uses 22-d obs (with ball_pos+ball_vel slots)."""
    v1_dir = MODEL_DIR.parent / "random_stack_cup_thrower_v1"
    vecnorm_pkl = v1_dir / "vecnormalize.pkl"
    assert vecnorm_pkl.exists(), f"missing test fixture: {vecnorm_pkl}"

    dummy = DummyVecEnv([lambda: StubEnv(obs_dim=22)])
    vn = VecNormalize.load(str(vecnorm_pkl), dummy)

    assert vn.obs_rms.mean.shape == (22,)
    assert vn.obs_rms.var.shape == (22,)
