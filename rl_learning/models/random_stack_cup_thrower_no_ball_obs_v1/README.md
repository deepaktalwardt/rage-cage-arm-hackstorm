# random_stack_cup_thrower_no_ball_obs_v1

PPO policy for the same task as `random_stack_cup_thrower_v1` — throw a
ping-pong ball into a Solo cup placed within ±10cm of nominal `(0.85, 0)`
*and* elevated by 0–15cm of pedestal — but trained on a reduced
**16-dim observation** that excludes ball position and velocity.

This is the sim2real-aligned successor to `random_stack_cup_thrower_v1`.
Ball state cannot be reliably estimated on the real Piper at sub-throw
timescales, so a policy that depends on it would not transfer. The
removed slots were largely redundant with joint state during the welded
windup and unused after release, but the policy still has to discover
that — hence a from-scratch retrain rather than a fine-tune.

## Observation change

Old (22-dim, v1):

```
joint_pos(6), joint_vel(6), ball_pos(3), ball_vel(3), cup_xy(2),
pedestal_height(1), release_countdown(1)
```

New (16-dim, this model):

```
joint_pos(6), joint_vel(6), cup_xy(2), pedestal_height(1),
release_countdown(1)
```

Everything else — action space, reward shape, MJCF, release timing,
randomization box — is identical to v1.

## Files

- `policy.zip` — the SB3 PPO model (MlpPolicy, medium MLP 256×256).
- `vecnormalize.pkl` — running observation/reward normalization stats.
  **Required at inference** — without it the model receives unnormalized
  observations and produces wrong actions.

## Performance at this checkpoint

Final 25M-step snapshot. Eval seed=0.

| metric | value |
|---|---|
| `range_success_rate` (300 random episodes uniform on R3 × Z3) | 0.940 |
| `exact_one_bounce_rate` (range eval) | 0.940 |
| `valid_bounce_rate` (range eval) | 1.000 |
| `median_closest_cup_dist` (range eval) | 0.013 m |
| `grid3d_success_rate` (3×3×3 grid across ±10cm × {0, 7.5, 15}cm) | 0.963 |
| per-layer z=0cm | 9/9 |
| per-layer z=7.5cm | 9/9 |
| per-layer z=15cm | 8/9 |
| `median_closest_cup_dist` (grid3d) | 0.014 m |
| `success_rate` (8 fixed-cup eval episodes at NOMINAL) | 0.750 |
| `mean_reward` (grid3d) | 135.7 |

The single failing grid3d cell is at the `(cup_xy=corner, pedestal=15cm)`
geometric ceiling — same failure mode as v1.

The fixed-cup `0.750` is below v1's `1.000`. Late-training evals
(curriculum.csv) showed the policy oscillating between 0.750 and 1.000
on the 8-episode fixed eval over the final ~2M steps. The 300-episode
random eval, the 27-cell grid eval, and `median_closest_cup_dist` all
came out ahead of v1 — the model is generalizing better, not regressing
on hard cases. The fixed-eval dip looks like late-training variance on
a small (8-episode) sample, not a real regression.

### Comparison to `random_stack_cup_thrower_v1`

| metric | v1 (22-dim obs) | this (16-dim obs) | delta |
|---|---:|---:|---:|
| range_success (300 ep) | 0.807 | 0.940 | +0.133 |
| grid3d_success (27 cells) | 0.741 | 0.963 | +0.222 |
| z=0 layer | 7/9 | 9/9 | +2 |
| z=7.5cm layer | 8/9 | 9/9 | +1 |
| z=15cm layer | 5/9 | 8/9 | +3 |
| median_closest (grid3d) | 0.019 m | 0.014 m | tighter |
| fixed (8 ep nominal) | 1.000 | 0.750 | -0.250 |

Removing ball state did not cost performance on the broader workspace
distribution — it improved on every randomized metric. Plausible
explanation: the dropped 6 obs slots were redundant during the welded
windup (ball_pos = gripper_pos) and uninformative after release, so the
policy had no useful signal there to begin with. Forcing it to operate
on joint state alone may have served as a mild regularizer that pushed
the policy toward more robust trajectories.

## Training config

25M timesteps, **from scratch**, identical to v1's hyperparams except
the env's observation space:

- 16 parallel envs (SubprocVecEnv), n_steps=2048, batch_size=512.
- LR=2e-4 with linear schedule, gamma=0.99, gae_lambda=0.95, ent_coef=0.01.
- `release_step=45` (fixed; ball releases at control step 45).
- `action_delta=0.06`, `RESET_NOISE_STD=0.005`.
- 6-D action space (joint deltas only).
- Strict `bounce > 1` termination.
- 4-stage reward curriculum (1=bounce, 2=+bounce_xy, 3=+cup_dist+entry+success).
  Auto-promoted 1→2→3.
- R3 from t=0 (full ±10cm cup_xy randomization, no R-stage curriculum).
- Z3 from t=0 (pedestal sampled uniformly in [0, 0.15m] each reset).
- MlpPolicy medium 256×256, tanh.

Curriculum milestones (curriculum.csv):

| event | timesteps |
|---|---|
| Stage 1 → 2 | 0.13M |
| Stage 2 → 3 | ~0.23M |
| First non-zero range_success | ~3.0M |
| Sustained range_success ≥ 0.5 | ~12M |
| Sustained range_success ≥ 1.0 (16-ep eval) | ~17M |
| Final eval | 25M |

## Caveat: matching `release_step` at inference

The policy was trained against `release_step=45`. Loading it into an env
with a different release step will produce off-target throws (the policy
peaks gripper velocity at step 45; releasing earlier or later misses the
cup mouth). The current `sim/env.py` default is `release_step=45`, so
loading via the standard helpers works out of the box.

## Caveat: env obs space must match

This policy's input layer is 16-dim. It will fail to load against the
22-dim env from before this branch. Conversely, v1's 22-dim policy will
fail to load against the current env. Old playable models live on the
parent commit if needed.

## Loading

Live MuJoCo viewer:

```bash
mjpython -m sim.play_policy --model models/random_stack_cup_thrower_no_ball_obs_v1 \
    --rand-stage 3 --rand-stage-z 3
mjpython -m sim.play_policy --model models/random_stack_cup_thrower_no_ball_obs_v1 \
    --cup-grid --pedestal-grid
mjpython -m sim.play_policy --model models/random_stack_cup_thrower_no_ball_obs_v1 \
    --no-randomize-cup --pedestal 0.15
mjpython -m sim.play_policy --model models/random_stack_cup_thrower_no_ball_obs_v1 \
    --cup-xy 0.95,0.10 --pedestal 0.10
```

(macOS requires `mjpython`, not plain `python` — see the script docstring.)

To use the policy programmatically:

```python
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from sim.env import RageCageEnv

env = DummyVecEnv([lambda: RageCageEnv(randomize_cup=True, reward_stage=3)])
env = VecNormalize.load("models/random_stack_cup_thrower_no_ball_obs_v1/vecnormalize.pkl", env)
env.training = False
env.norm_reward = False
env.env_method("set_pedestal_range", (0.0, 0.15))

model = PPO.load("models/random_stack_cup_thrower_no_ball_obs_v1/policy.zip", env=env)

obs = env.reset()
done = [False]
while not done[0]:
    action, _ = model.predict(obs, deterministic=True)
    obs, _reward, done, _info = env.step(action)
```

To evaluate across the full 27-cell workspace:

```bash
uv run python -m sim.eval_grid \
    --run-dir models/random_stack_cup_thrower_no_ball_obs_v1 \
    --reward-stage 3 \
    --also-fixed
```
