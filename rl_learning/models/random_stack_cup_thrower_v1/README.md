# random_stack_cup_thrower_v1

PPO policy for a 6-DOF Piper arm to throw a ping-pong ball into a Solo
cup placed anywhere within ±10cm of nominal `(0.85, 0)` *and* elevated
by 0–15cm of pedestal — a 20cm × 20cm × 15cm operational envelope. The
pedestal height simulates a stack of 1–9 nested cups in real rage cage
(Solo cups nest at ~1.8cm per cup). The policy reads cup_xy and
`pedestal_height` from the observation and conditions its throw on both.

This is the stacked-cup successor to `random_pos_cup_thrower_v1`, which
handled cup_xy randomization but only at table height (`z=0`).

## Files

- `policy.zip` — the SB3 PPO model (MlpPolicy, medium MLP 256×256).
- `vecnormalize.pkl` — running observation/reward normalization stats.
  **Required at inference** — without it the model receives unnormalized
  observations and produces wrong actions.

## Performance at this checkpoint

Final 25M-step snapshot of the v36 from-scratch run. Eval seed=0.

| metric | value |
|---|---|
| `success_rate` (8 fixed-cup eval episodes at NOMINAL, pedestal=0) | 1.000 |
| `range_success_rate` (64 random episodes uniform on R3 × Z3) | 0.828 |
| `grid3d_success_rate` (3×3×3 grid across ±10cm × {0, 7.5, 15}cm) | 0.741 |
| per-layer z=0cm | 7/9 |
| per-layer z=7.5cm | 8/9 |
| per-layer z=15cm | 5/9 |
| `valid_bounce_rate` | 1.000 |
| `exact_one_bounce_rate` | 1.000 |
| `bounce_target_rate` | 1.000 |
| `median_closest_cup_dist` (grid3d) | 0.019 m |
| `mean_reward` (fixed-cup) | 141.7 |

The remaining ~25% misses across the workspace are rim grazes at the
hardest corners (cup_xy at ±10cm with pedestal at 15cm), same failure
mode as v34's `(0.85, +0.10)` — `closest_cup_dist` ≤ 7cm. Not a
learning shortfall; a geometry ceiling of the current bounce-throw
style at the workspace boundary.

## Training config (v36)

25M timesteps trained **from scratch** (not warm-started from v34).
The warm-start path collapsed: v34's policy had near-zero weights on
the repurposed cup_count obs slot, and surgical resetting the
normalization stats let gradient flow there too fast for the policy to
adapt without first losing v34's specific 1-bounce arc. From-scratch
with pedestal randomized from t=0 lets pedestal-conditioning develop
alongside the throw, not after.

Key knobs (see `sim/env.py`, `sim/train_rl.py`, and
`docs/rl_experiment_log.md` for full context):

- 16 parallel envs (SubprocVecEnv), n_steps=2048, batch_size=512.
- LR=2e-4 with linear schedule, gamma=0.99, gae_lambda=0.95, ent_coef=0.01.
- `release_step=45` (fixed; ball releases at control step 45).
- `action_delta=0.06`, `RESET_NOISE_STD=0.005`.
- 6-D action space (joint deltas only).
- Strict `bounce > 1` termination (single-bounce required for success).
- 4-stage reward curriculum (1=bounce, 2=+bounce_xy, 3=+cup_dist+entry+success).
  Auto-promoted 1→2→3 in ~230K steps.
- R3 from t=0 (full ±10cm cup_xy randomization, no R-stage curriculum).
- Z3 from t=0 (pedestal sampled uniformly in [0, 0.15m] each reset, no
  Z-stage curriculum). v36 doesn't use the Z curriculum because the
  warm-start runs proved the curriculum was firing on mechanical
  headroom rather than learned conditioning.
- Bounce-target geometry unchanged from v34: `target = 0.7 * cup_xy`
  along arm-base→cup line, elliptical reward tolerance
  `σ_long=0.30, σ_perp=0.08` in throw-frame coordinates.
- 3D post-bounce cup_dist reward uses target z = `pedestal_height +
  CUP_HEIGHT - 0.02`, which is the implicit "throw to the right
  height" signal.

Reward / mean-reward trajectory:
- 230K: stages 1→2→3 done, learning the bounce target.
- 7M: `median_closest_cup_dist` reaches ~5cm but `fixed_success` still 0
  (mostly 2-bounce roll-to-cup at this point).
- 8.2M: first sustained 50% fixed_success — clean 1-bounce arcs emerge.
- 13M onwards: sustained convergence; fixed=1.0, grid3d oscillating
  ~0.6–0.8 across evals.

## Caveat: matching `release_step` at inference

The policy was trained against `release_step=45`. Loading it into an env
with a different release step will produce off-target throws (the policy
peaks gripper velocity at step 45; releasing earlier or later misses the
cup mouth).

The current `sim/env.py` default is `release_step=45`, so loading via
the standard helpers works out of the box.

## Loading

The simplest way to watch the policy is the live viewer script:

```bash
mjpython -m sim.play_policy --model models/random_stack_cup_thrower_v1 \
    --rand-stage 3 --rand-stage-z 3
mjpython -m sim.play_policy --model models/random_stack_cup_thrower_v1 \
    --cup-grid --pedestal-grid
mjpython -m sim.play_policy --model models/random_stack_cup_thrower_v1 \
    --no-randomize-cup --pedestal 0.15
mjpython -m sim.play_policy --model models/random_stack_cup_thrower_v1 \
    --cup-xy 0.95,0.10 --pedestal 0.10
```

(macOS requires `mjpython`, not plain `python` — see the script docstring.)

To use the policy programmatically:

```python
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from sim.env import RageCageEnv

env = DummyVecEnv([lambda: RageCageEnv(randomize_cup=True, reward_stage=3)])
env = VecNormalize.load("models/random_stack_cup_thrower_v1/vecnormalize.pkl", env)
env.training = False
env.norm_reward = False
# Optional: set pedestal range; defaults to (0, 0) without this call.
env.env_method("set_pedestal_range", (0.0, 0.15))

model = PPO.load("models/random_stack_cup_thrower_v1/policy.zip", env=env)

obs = env.reset()
done = [False]
while not done[0]:
    action, _ = model.predict(obs, deterministic=True)
    obs, _reward, done, _info = env.step(action)
```

To evaluate across the full 27-cell workspace:

```bash
uv run python -m sim.eval_grid \
    --run-dir models/random_stack_cup_thrower_v1 \
    --reward-stage 3 \
    --also-fixed
```

(eval_grid currently writes 9-cell z=0 grid only via the legacy
heatmap; use `evaluate_policy_grid3d` from `sim.train_rl` for the full
27-cell breakdown — see the snippet in `docs/rl_experiment_log.md`.)
