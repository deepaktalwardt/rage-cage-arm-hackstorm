# random_pos_cup_thrower_v1

PPO policy for a 6-DOF Piper arm to throw a ping-pong ball into a Solo
cup placed anywhere within ±10cm of the nominal position `(0.85, 0)` —
a 20cm × 20cm operational envelope. The policy reads `cup_xy` from the
observation and conditions its throw on the target. Trained in the
rage_cage MuJoCo sim with curriculum-driven cup randomization.

This is the multi-position successor to `single_bounce_cup_thrower_v1`,
which only worked at a single fixed cup pose.

## Files

- `policy.zip` — the SB3 PPO model (MlpPolicy, medium MLP 256×256).
- `vecnormalize.pkl` — running observation/reward normalization stats.
  **Required at inference** — without it the model receives unnormalized
  observations and produces wrong actions.

## Performance at this checkpoint

Captured by the per-R-stage best-snapshot saver during the v34 run. Eval
numbers below use 100 random R3 episodes plus the canonical 3×3 grid
across the full ±10cm workspace.

| metric | value |
|---|---|
| `success_rate` (8 fixed-cup eval episodes at NOMINAL) | 1.000 |
| `range_success_rate` (100 random R3 episodes, uniform ±10cm) | 0.870 |
| `grid_success_rate` (3×3 grid across ±10cm) | 8/9 = 0.889 |
| `valid_bounce_rate` | 1.000 |
| `bounce_target_rate` | 1.000 |
| `median_closest_cup_dist` | 0.015 m |
| `mean_reward` (fixed-cup) | 140.9 |

The single failing grid cell is `(0.85, +0.10)` — closest_cup_dist
0.035m, just outside the 0.047m cup radius. The same "rim graze" failure
mode appears at ~13% of random R3 positions — a real ceiling of the
current bounce-throw geometry, not a learning shortfall.

Across uniformly-random R3 cups: the policy throws with a single bounce,
threads the cup mouth most of the time, and lands the ball within ~3cm
of cup center on average.

## Training config (v34)

20M timesteps, ~14M to sustained convergence. Key knobs (see `sim/env.py`,
`sim/train_rl.py`, and `docs/rl_experiment_log.md` for full context):

- 16 parallel envs (SubprocVecEnv), n_steps=2048, batch_size=512.
- LR=2e-4 with linear schedule, gamma=0.99, gae_lambda=0.95, ent_coef=0.01.
- `release_step=45` (fixed; ball releases at control step 45).
- `action_delta=0.06`, `RESET_NOISE_STD=0.005`.
- 6-D action space (joint deltas only).
- Strict `bounce > 1` termination (single-bounce required for success).
- 4-stage reward curriculum (1=bounce, 2=+bounce_xy, 3=+cup_dist+entry+success).
- 4-stage randomization curriculum within reward stage 3:
  R0=±2cm → R1=±5cm → R2=±8cm → R3=±10cm.
- Bounce-target geometry: `target = 0.7 * cup_xy` along arm-base→cup
  line, elliptical reward tolerance `σ_long=0.30`, `σ_perp=0.08` in
  throw-frame coordinates (replaces the v22-era hardcoded
  `(-0.32, 0)` offset).

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
mjpython -m sim.play_policy --model models/random_pos_cup_thrower_v1
mjpython -m sim.play_policy --model models/random_pos_cup_thrower_v1 --cup-grid --episodes 9
mjpython -m sim.play_policy --model models/random_pos_cup_thrower_v1 --rand-stage 3
mjpython -m sim.play_policy --model models/random_pos_cup_thrower_v1 --cup-xy 0.95,0.10
```

(macOS requires `mjpython`, not plain `python` — see the script docstring.)

To use the policy programmatically:

```python
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from sim.env import RageCageEnv

env = DummyVecEnv([lambda: RageCageEnv(randomize_cup=True, reward_stage=3)])
env = VecNormalize.load("models/random_pos_cup_thrower_v1/vecnormalize.pkl", env)
env.training = False
env.norm_reward = False

model = PPO.load("models/random_pos_cup_thrower_v1/policy.zip", env=env)

obs = env.reset()
done = [False]
while not done[0]:
    action, _ = model.predict(obs, deterministic=True)
    obs, _reward, done, _info = env.step(action)
```

To evaluate across the full workspace:

```bash
uv run python -m sim.eval_grid \
    --model models/random_pos_cup_thrower_v1/policy.zip \
    --vecnormalize models/random_pos_cup_thrower_v1/vecnormalize.pkl \
    --reward-stage 3 \
    --out sim/_rl_eval/grid.csv \
    --also-fixed
```
