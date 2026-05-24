# single_bounce_cup_thrower_v1

PPO policy for a 6-DOF Piper arm to throw a ping-pong ball into a Solo
cup with one table bounce (beer-pong shot). Trained in the rage_cage
MuJoCo sim against a randomized cup position.

## Files

- `policy.zip` — the SB3 PPO model (MlpPolicy, medium MLP 256×256).
- `vecnormalize.pkl` — running observation/reward normalization stats.
  **Required at inference** — without it the model receives unnormalized
  observations and produces wrong actions.

## Performance at this checkpoint

Captured by the best-snapshot callback at 3.0M training steps in the v22
run, the eval where `success_rate` peaked:

| metric | value |
|---|---|
| success_rate (8 fixed-cup eval episodes) | 0.75 |
| mean_reward | 9.18 |
| closest_cup_dist | 0.015 m (ball through cup-mouth center) |
| valid_bounce_rate | 0.75 |
| bounce_target_rate | 0.75 |

Stage 3 of the curriculum, just past auto-promotion from stage 2.

## Training config

Key knobs that produced this checkpoint (see `sim/env.py`,
`sim/train_rl.py`, and `docs/rl_experiment_log.md` for full context):

- 10M-timestep run target; this snapshot was saved at 3.0M when peak was
  hit. Training continued past this but never beat 0.75 — see the v22
  entry in the experiment log for the stage-3 collapse pattern observed
  after ~5M.
- 16 parallel envs (SubprocVecEnv), n_steps=2048, batch_size=512.
- LR=2e-4 with linear schedule, gamma=0.99, gae_lambda=0.95, ent_coef=0.01.
- `action_delta=0.06`, `RESET_NOISE_STD=0.005`, `release_step=35`.
- Strict `bounce > 1` termination (single-bounce required for success).
- Post-bounce cup-distance reward: `exp(-dist / 0.04)` (the v21 reward
  change that unlocked cup entries — see experiment log).
- 4-stage auto-curriculum, with stage 3 → 4 promotion disabled (stage 4
  weights destabilized the policy in v21).

## Loading

The simplest way to watch the policy is the live viewer script:

```bash
mjpython -m sim.play_policy --model models/single_bounce_cup_thrower_v1
```

(macOS requires `mjpython`, not plain `python` — see the script docstring.)

To use the policy programmatically:

```python
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from sim.env import RageCageEnv

env = DummyVecEnv([lambda: RageCageEnv(randomize_cup=False, reward_stage=3)])
env = VecNormalize.load("models/single_bounce_cup_thrower_v1/vecnormalize.pkl", env)
env.training = False
env.norm_reward = False

model = PPO.load("models/single_bounce_cup_thrower_v1/policy.zip", env=env)

obs = env.reset()
done = [False]
while not done[0]:
    action, _ = model.predict(obs, deterministic=True)
    obs, _reward, done, _info = env.step(action)
```

Set `randomize_cup=True` and a different `reward_stage` if you want to
evaluate generalization or under stage-4 reward shaping.
