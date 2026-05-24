"""Minimal `gym.Env` exposing the obs and 6-d action spaces used by the
throw policies.

Used only so `VecNormalize.load(...)` has something to bind to when the
inference node recovers training-time obs-normalization stats — avoids
dragging MuJoCo + `sim/` into the deployment container.

`obs_dim` parameterizes for the two policy variants we ship:
  - 16: random_stack_cup_thrower_no_ball_obs_v1 (default)
  - 22: random_stack_cup_thrower_v1 (legacy with ball_pos/ball_vel slots)

Bounds are intentionally infinite: only shape matters for the
normalization-stat load path; the policy itself never sees this env.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class StubEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, obs_dim: int = 16) -> None:
        super().__init__()
        self._obs_dim = obs_dim
        self.observation_space = spaces.Box(
            low=np.full(obs_dim, -np.inf, dtype=np.float32),
            high=np.full(obs_dim, np.inf, dtype=np.float32),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(6,), dtype=np.float32,
        )

    def reset(self, *, seed=None, options=None):  # noqa: D401, ANN001
        super().reset(seed=seed)
        return np.zeros(self._obs_dim, dtype=np.float32), {}

    def step(self, action):  # noqa: ANN001
        return np.zeros(self._obs_dim, dtype=np.float32), 0.0, True, False, {}
