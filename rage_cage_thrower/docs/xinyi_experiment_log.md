# Xinyi Experiment Log

Personal notes for PPO training runs and machine-specific profiling. These
numbers are useful context for Xinyi's laptop, but collaborators should run
`sim.profile_train` on their own machines before choosing final training
settings.

## Training Experiments

### Post-Grip-Fix 60cm Cup Hparam Study

Context:

- `main` merged a grip/release fix using a `ball_grip` weld.
- Cup curriculum target moved to about 60cm from the held ball.
- Distance shaping currently uses `DISTANCE_REWARD_SCALE = 2.0`, so misses on
  the 60cm task still get a directional reward signal.
- Each trial was capped under 10 minutes wall time.

Trial 2, default MLP, 500k:

Change from previous trial:

- This was the post-merge baseline for the new setup: grip-fix weld release,
  60cm cup target, and `DISTANCE_REWARD_SCALE = 2.0`.
- Used the conservative PPO settings we had been using before changing the
  model: `default` MLP, `LR=3e-4`, linear schedule, `N_ENVS=2`,
  `N_STEPS=1024`.

```bash
/usr/bin/time -l uv run python -m sim.train_rl --timesteps 500000 --n-envs 2 --n-steps 1024 --batch-size 64 --lr 0.0003 --lr-schedule linear --net-arch default --activation tanh --log-interval 20 --seed 0 --out sim/_rl_out/exp02_gripfix_500k_default_2env1024_linear --train-rollout-viz-dir sim/_rl_out/exp02_gripfix_500k_default_2env1024_linear_train_rollouts --train-rollout-viz-every 250000 --train-rollout-viz-steps 150 --train-rollout-viz-fixed-cup
```

Result:

- Completed in 4m41s, peak RSS about 1.39GB.
- Training snapshot at 252k: `bounce_count=1`, `cup_dist=1.945m`.
- Eval: `reward=1.72`, `success=False`, `bounce_count=1`, `cup_dist=1.446m`.

Conclusion:

Default MLP can learn to produce one bounce, but it misses far from the cup.

Trial 3, medium MLP, 500k:

Change from Trial 2:

- Changed only `NET_ARCH`: `default` -> `medium`.
- Kept `LR=3e-4`, linear schedule, `N_ENVS=2`, `N_STEPS=1024`, and 500k
  timesteps fixed.
- Reason: Trial 2 learned one bounce but missed far, so the next question was
  whether a larger policy/value MLP had enough capacity to aim the throw.

```bash
/usr/bin/time -l uv run python -m sim.train_rl --timesteps 500000 --n-envs 2 --n-steps 1024 --batch-size 64 --lr 0.0003 --lr-schedule linear --net-arch medium --activation tanh --log-interval 20 --seed 0 --out sim/_rl_out/exp03_gripfix_500k_medium_2env1024_linear --train-rollout-viz-dir sim/_rl_out/exp03_gripfix_500k_medium_2env1024_linear_train_rollouts --train-rollout-viz-every 250000 --train-rollout-viz-steps 150 --train-rollout-viz-fixed-cup
```

Result:

- Completed in 9m08s, peak RSS about 1.25GB.
- Mean reward climbed from `-13.4` to about `43`.
- Early updates were aggressive: KL around `0.02-0.027`, clip fraction around
  `0.25-0.29`.
- Training snapshot at 252k: `reward=22.27`, `bounce_count=1`,
  `cup_dist=1.064m`.
- Eval: `reward=46.63`, `success=False`, `bounce_count=1`, `cup_dist=0.474m`.

Conclusion:

Medium MLP is much better than default MLP, but `3e-4` linear still misses by
about 47cm.

Trial 4, medium MLP, lower LR:

Change from Trial 3:

- Changed only `LR`: `3e-4` -> `1e-4`.
- Kept `medium` MLP, linear schedule, `N_ENVS=2`, `N_STEPS=1024`, and 500k
  timesteps fixed.
- Reason: Trial 3 improved geometry but early PPO updates were aggressive
  (`KL ~= 0.02-0.027`, clip fraction around `0.25-0.29`), so this tested
  whether smaller updates would improve stability and final aim.

```bash
/usr/bin/time -l uv run python -m sim.train_rl --timesteps 500000 --n-envs 2 --n-steps 1024 --batch-size 64 --lr 0.0001 --lr-schedule linear --net-arch medium --activation tanh --log-interval 20 --seed 0 --out sim/_rl_out/exp04_gripfix_500k_medium_2env1024_linear1e4 --train-rollout-viz-dir sim/_rl_out/exp04_gripfix_500k_medium_2env1024_linear1e4_train_rollouts --train-rollout-viz-every 250000 --train-rollout-viz-steps 150 --train-rollout-viz-fixed-cup
```

Result:

- Completed in 5m26s, peak RSS about 1.41GB.
- PPO metrics were stable, but behavior was too timid.
- Training snapshot at 252k: `reward=-0.80`, `bounce_count=1`,
  `cup_dist=3.613m`.
- Eval: `reward=17.48`, `success=False`, `bounce_count=1`, `cup_dist=2.879m`.

Conclusion:

`1e-4` linear is too slow for the 500k budget.

Trial 5, medium MLP, constant `3e-4`:

Change from Trial 4:

- Changed `LR`: `1e-4` -> `3e-4`.
- Changed `LR_SCHEDULE`: `linear` -> `constant`.
- Kept `medium` MLP, `N_ENVS=2`, `N_STEPS=1024`, and 500k timesteps fixed.
- Reason: Trial 4 was stable but too timid. This tested the opposite extreme:
  keep large updates throughout training and see if it can push the ball closer
  before LR decay removes learning pressure.

```bash
/usr/bin/time -l uv run python -m sim.train_rl --timesteps 500000 --n-envs 2 --n-steps 1024 --batch-size 64 --lr 0.0003 --lr-schedule constant --net-arch medium --activation tanh --log-interval 20 --seed 0 --out sim/_rl_out/exp05_gripfix_500k_medium_2env1024_constant3e4 --train-rollout-viz-dir sim/_rl_out/exp05_gripfix_500k_medium_2env1024_constant3e4_train_rollouts --train-rollout-viz-every 250000 --train-rollout-viz-steps 150 --train-rollout-viz-fixed-cup
```

Result:

- Completed in 5m28s, peak RSS about 1.58GB.
- Highest reward early, but update metrics became unhealthy: final KL about
  `0.185`, clip fraction about `0.644`, policy std collapsed toward `0.377`.
- Training snapshot at 252k: `reward=61.69`, `bounce_count=1`,
  `cup_dist=1.471m`.
- Eval: `reward=86.80`, `success=False`, `bounce_count=1`, `cup_dist=0.730m`.

Conclusion:

Constant `3e-4` optimizes reward aggressively but is less stable and worse on
cup distance than `3e-4` linear.

Trial 6, medium MLP, `2e-4` linear, 1024-step rollouts:

Change from Trial 5:

- Changed `LR`: `3e-4` -> `2e-4`.
- Changed `LR_SCHEDULE`: `constant` -> `linear`.
- Kept `medium` MLP, `N_ENVS=2`, `N_STEPS=1024`, and 500k timesteps fixed.
- Reason: Trial 5 got high reward but unstable PPO metrics and worse cup
  distance than Trial 3. This tested a middle-ground LR with decay: less
  aggressive than constant `3e-4`, less timid than Trial 4's `1e-4`.

```bash
/usr/bin/time -l uv run python -m sim.train_rl --timesteps 500000 --n-envs 2 --n-steps 1024 --batch-size 64 --lr 0.0002 --lr-schedule linear --net-arch medium --activation tanh --log-interval 20 --seed 0 --out sim/_rl_out/exp06_gripfix_500k_medium_2env1024_linear2e4 --train-rollout-viz-dir sim/_rl_out/exp06_gripfix_500k_medium_2env1024_linear2e4_train_rollouts --train-rollout-viz-every 250000 --train-rollout-viz-steps 150 --train-rollout-viz-fixed-cup
```

Result:

- Completed in 5m21s, peak RSS about 1.63GB.
- More stable than constant `3e-4`: final KL about `0.0038`, clip fraction
  about `0.012`.
- Training snapshot at 252k: `reward=90.27`, `bounce_count=1`,
  `cup_dist=1.137m`.
- Eval: `reward=108.07`, `success=False`, `bounce_count=1`, `cup_dist=0.264m`.

Conclusion:

Best 1024-step rollout config. It balances reward and stability well.

Trial 7, medium MLP, `2e-4` linear, 2048-step rollouts:

Change from Trial 6:

- Changed only `N_STEPS`: `1024` -> `2048`.
- Kept `medium` MLP, `LR=2e-4`, linear schedule, `N_ENVS=2`, batch size 64,
  and 500k timesteps fixed.
- Reason: Trial 6 was the best 1024-step config. This tested whether larger
  on-policy rollouts per PPO update would improve stability/geometry without
  exceeding the 10-minute trial budget.

```bash
/usr/bin/time -l uv run python -m sim.train_rl --timesteps 500000 --n-envs 2 --n-steps 2048 --batch-size 64 --lr 0.0002 --lr-schedule linear --net-arch medium --activation tanh --log-interval 10 --seed 0 --out sim/_rl_out/exp07_gripfix_500k_medium_2env2048_linear2e4 --train-rollout-viz-dir sim/_rl_out/exp07_gripfix_500k_medium_2env2048_linear2e4_train_rollouts --train-rollout-viz-every 250000 --train-rollout-viz-steps 150 --train-rollout-viz-fixed-cup
```

Result:

- Completed in 5m07s, peak RSS about 1.60GB.
- PPO metrics stayed controlled: final KL about `0.0056`, clip fraction about
  `0.034`.
- Training snapshot at 254k: `reward=100.12`, `bounce_count=1`,
  `cup_dist=0.371m`.
- Eval: `reward=114.05`, `success=False`, `bounce_count=1`, `cup_dist=0.173m`.

Conclusion:

Best configuration from this study. Larger rollouts improved the final geometry
without extra wall-time cost.

Recommended next run:

```bash
FULL_RUN=1 TIMESTEPS=5000000 N_ENVS=2 N_STEPS=2048 BATCH_SIZE=64 LR=0.0002 LR_SCHEDULE=linear NET_ARCH=medium TRAIN_ROLLOUT_VIZ=1 TRAIN_ROLLOUT_VIZ_EVERY=250000 TRAIN_ROLLOUT_VIZ_STEPS=150 EPISODES=1 OUT=sim/_rl_out/ppo_5m_gripfix_medium_2env2048_lr2e4_linear ./scripts/train_ppo.sh
```

Rationale:

- Keep `N_ENVS=2` for 16GB laptop memory safety.
- Use `N_STEPS=2048`; it was no slower in this study and improved cup distance.
- Use `NET_ARCH=medium`; default MLP missed much farther.
- Use `LR=2e-4 LR_SCHEDULE=linear`; this was more stable than constant `3e-4`
  and less timid than `1e-4`.
- Keep training rollout snapshots on, because mid-run behavior can be better
  than the final model and should be inspected.

### Initial Baseline: 500k Timesteps, 2 Envs, 1024-Step Rollouts

Command:

```bash
uv run python -m sim.train_rl --timesteps 500000 --n-envs 2 --n-steps 1024 --batch-size 64 --lr 0.0003 --lr-schedule linear --log-interval 20 --seed 0 --out sim/_rl_out/exp01_500k_2env1024_linear3e4
```

Reasoning:

- Memory-safe 16GB laptop baseline.
- `N_ENVS=2` and `N_STEPS=1024` keep memory lower while preserving about 1.5k
  steps/sec on the profiled machine.

Result:

- Completed in 7m31s with peak RSS about 1.05GB.
- Mean episode reward rose from about 62 at 41k timesteps, peaked around 145
  near 246k timesteps, and ended around 137 as the linear LR decayed near zero.
- KL stayed around 0.001-0.016.
- Clip fraction stayed below about 0.18.
- Policy std decayed gradually.

Eval:

- `success=True`
- `bounce_count=1`
- final `cup_dist=0.005m`
- fixed-cup deterministic eval

Conclusion:

This is the current recommended baseline before trying larger networks.

### Training Rollout Visualization Smoke

Command:

```bash
uv run python -m sim.train_rl --timesteps 32 --n-envs 1 --n-steps 16 --batch-size 16 --log-interval 1 --out sim/_rl_out/viz_smoke --train-rollout-viz-dir sim/_rl_out/viz_smoke_train_rollouts --train-rollout-viz-every 16 --train-rollout-viz-steps 40 --train-rollout-viz-fixed-cup
```

Result:

- Saved `train_rollout_000000016.gif/csv`
- Saved `train_rollout_000000032.gif/csv`
- Saved `train_rollout_final_000000032.gif/csv`
- Rollout rewards were about 30 because this was only a pipeline check.

### Real Rollout-Length Visualization Smoke

Command:

```bash
uv run python -m sim.train_rl --timesteps 2048 --n-envs 1 --n-steps 1024 --batch-size 64 --lr-schedule linear --log-interval 1 --out sim/_rl_out/viz_real_rollout_smoke --train-rollout-viz-dir sim/_rl_out/viz_real_rollout_smoke_train_rollouts --train-rollout-viz-every 1024 --train-rollout-viz-steps 300 --train-rollout-viz-fixed-cup
```

Result:

- Completed in about 20s.
- Saved `train_rollout_000001024.gif/csv`
- Saved `train_rollout_000002048.gif/csv`
- Saved `train_rollout_final_000002048.gif/csv`
- Rewards were about 111-112; this was still too short to learn a solved policy.

## Profiling Notes

Measured 5M-timestep estimates on Xinyi's profiled machine:

| Config | Peak RSS | Estimate |
|---|---:|---:|
| `N_ENVS=2 N_STEPS=1024` | ~1.05 GB | ~54m 29s |
| `N_ENVS=2 N_STEPS=2048` | ~0.98 GB | ~53m 8s to 54m 37s |
| `N_ENVS=4 N_STEPS=2048` | ~1.13 GB | ~50m 14s to 51m 25s |

Use these as local reference points only. Runtime and memory can change
substantially across laptops due to CPU, memory bandwidth, thermal throttling,
background load, and OS differences.

## Next Trials

- Run the recommended 5M-timestep constant-`1e-4` baseline above.
- Add checkpoint saving if we want to preserve the best mid-run policy, not just
  the final policy.
- Consider a curriculum that starts closer than 60cm, then increases distance
  once fixed-cup eval reaches the cup reliably.
