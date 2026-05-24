# random_stack_cup_thrower_latency_robust_v1

**Sim2real-targeted PPO policy.** Trained against the same soft-gain MuJoCo
controller as `random_stack_cup_thrower_soft_gains_candidate`, with three
additional changes to the observation/action pipeline aimed at the
command-vs-joint-state latency that was driving jitter on the real PiPER.

Same task as the prior `random_stack_*` models: throw a ping-pong ball
into a Solo cup placed within ±10cm of nominal `(0.85, 0)` and elevated
by 0–15cm of pedestal.

## Why this run exists

On the real arm the `soft_gains_candidate` policy oscillates: it issues
a command, sees the joint state lag a few control ticks behind, "corrects"
too aggressively, and the loop chatters. Standard closed-loop-policy-meets-
unmodeled-feedback-delay failure mode.

Three coordinated changes target it:

1. **Drop `joint_vel` from obs.** Host-side finite-difference velocity on
   CAN-encoder reads is the dominant noise source on the real obs vector.
   The policy can't usefully learn against it; better to let the policy
   estimate velocity from the joint_pos history itself if it needs to.
2. **Add history of `joint_pos` and previous actions.** 4 frames each.
   Gives the policy enough state to do its own filtering / velocity
   estimation, and lets it see what commands are still in flight through
   the latency queue.
3. **Model the action pipeline.** Per-episode-sampled action latency in
   `{2, 3, 4}` ticks, plus a first-order low-pass filter (α=0.5) modeling
   finite motor-controller bandwidth. The policy trains under the same
   delay/smoothing it will face on the robot.

If the hypothesis holds, this policy should retain reactive behavior
without the jitter. If sim convergence regresses too far or the real arm
gets worse, fall back to the open-loop step-conditioned design from
`docs/plans/2026-05-23-openloop-step-conditioned-design.md`.

## Env / obs / action / reward / randomization

Same MJCF (soft gains, joint 1–3 `kp=6.5/kv=0.4`, joint 4–6 `kp=2.5/kv=0.2`)
as `soft_gains_candidate`. Same auto curriculum 1→3, same R3+Z3
randomization, same medium MLP, same 25M steps. From scratch (obs shape
incompatible with all prior checkpoints).

```
obs (52-dim):  cup_xy(2), pedestal_height(1), release_countdown(1),
               joint_pos_history(4*6 = 24)    # current + 3 previous, σ=0.001 rad noise
               action_history(4*6 = 24)        # last 4 raw policy actions

action (6-dim): joint deltas, scaled by action_delta=0.06
               then -> latency queue (length sampled per-episode from {2,3,4})
               then -> low-pass filter (α=0.5)
               then -> integrated onto arm joint target

reward: 4-stage curriculum; auto-promoted 1→2→3 (stage 4 disabled)
randomization: cup_xy ±10cm (R3), pedestal 0–15cm (Z3)
release_step=45, RESET_NOISE_STD=0.005
```

### Diff vs `random_stack_cup_thrower_soft_gains_candidate`

| field | soft_gains_candidate | latency_robust_v1 |
|---|---|---|
| obs dim | 16 | **52** |
| `joint_vel` in obs | yes (6 dims) | **dropped** |
| `joint_pos` history | 1 frame (current) | **4 frames** |
| action history in obs | none | **4 frames** |
| `obs_joint_pos_noise_std` | 0.0 | **0.001 rad** |
| `action_latency_range` | (0, 0) | **(2, 4)** per-episode sampled |
| `action_filter_alpha` | 1.0 (no filter) | **0.5** |

Everything else (MJCF, reward, randomization, network, hyperparams, seed,
release_step, action_delta) is held constant.

## Training

- Command: `FULL_RUN=1 TIMESTEPS=25000000 N_ENVS=16 N_STEPS=2048 BATCH_SIZE=512 LR=0.0002 LR_SCHEDULE=linear NET_ARCH=medium ACTIVATION=tanh ENT_COEF=0.01 REWARD_STAGE=1 CURRICULUM=auto CURRICULUM_EVAL_EVERY=100000 CURRICULUM_EVAL_EPISODES=8 RAND_STAGE=3 Z_STAGE=3 RESET_NOISE_STD=0.005 ACTION_LATENCY_RANGE="2,4" ACTION_FILTER_ALPHA=0.5 OBS_JOINT_POS_NOISE_STD=0.001 JOINT_POS_HISTORY_LEN=4 ACTION_HISTORY_LEN=4 SEED=0 OUT=sim/_rl_out/latency_robust_v1_25m ./scripts/train_ppo.sh`
- Wall clock: ~1h on M-series Mac, n_envs=16 SubprocVecEnv.
- Curriculum promotions: **stage 1 → 2 at 131K**, **stage 2 → 3 at 229K**
  (identical to `soft_gains_v1`; the env-side modeling did not slow early
  learning).
- First non-zero `range_success_rate` at ~8.9M (vs `soft_gains_v1`'s
  ~1.8M — this is the cost of the harder learning problem).

## Performance at this snapshot

Auto-curriculum evals over the final 22–25M window (31 evals, 8-ep fixed
and 16-ep R3×Z3, seed=0). Numbers are representative of the steady-state
band, not a single eval row:

| metric | typical band | mean (final 22-25M) |
|---|---|---:|
| `success_rate` (8 ep fixed nominal) | 0.625–1.000 | 0.798 |
| `valid_bounce_rate` | 1.000 | 1.000 |
| `exact_one_bounce_rate` | 0.625–1.000 | matches success |
| `bounce_target_rate` | 1.000 | 1.000 |
| `median_closest_cup_dist` | 0.010–0.020 m | 0.016 m |
| `range_success_rate` (16 ep, ±10cm × 0–15cm) | 0.44–0.94 | 0.679 |
| `grid_success_rate` (3×3 cup_xy grid) | 0.33–0.89 | 0.638 |
| `grid3d_success_rate` (3×3×3 cup_xy × pedestal) | 0.37–0.74 | 0.556 |
| `invalid_contact_rate` | 0.000 | 0.000 |

## Comparison to `random_stack_cup_thrower_soft_gains_candidate` (plateau)

| metric | soft_gains_candidate (plateau) | latency_robust_v1 (final 22-25M mean) |
|---|---:|---:|
| range_success | 0.94–1.00 | 0.679 |
| grid_success | 0.78–1.00 | 0.638 |
| grid3d_success | 0.78–0.96 | 0.556 |
| median_closest (range) | 0.007–0.010 m | 0.016 m |
| fixed-cup success (8 ep, sustained) | 1.000 | 0.798 |

**Sim performance is materially lower** (~20–30 pp on range_success,
~25 pp on grid3d). That's the cost of the harder env modeling and the
reduced obs fidelity. The whole point is whether this trade buys a
working policy on the real arm — that's the test that matters.

Caveat: these are 8/16-episode auto-curriculum evals, not a 300-episode
benchmark. Use them as a directional signal; final apples-to-apples
comparison would require running `eval_grid` against the `models/`
artifact.

## Files

- `policy.zip` — SB3 PPO model, MlpPolicy medium 256×256 tanh, 52-dim input.
- `vecnormalize.pkl` — observation/reward normalization stats. Required
  at inference.
- `training.json` — env config persistence so playback/eval tools
  reconstruct the matching env automatically via
  `env_kwargs_from_training_json`. Without this file the loaders default
  to a 10-dim env and obs-shape-mismatch on `PPO.load`.

## Caveats

- **52-dim obs is incompatible with every prior checkpoint.** v1, v2,
  `random_pos_*`, `single_bounce_*`, `no_ball_obs_v1` (16-dim), and
  `soft_gains_candidate` (16-dim) all fail to load against the current
  env. To play those, check out their parent commits.
- **MJCF must match at inference.** This policy expects the soft-gain
  `piper.xml`. The soft-gain MJCF has been the default since the
  `soft_gains_candidate` commit; loading against the older stiff-gain
  MJCF will produce different joint dynamics than during training.
- **Real-arm validation pending.** Sim numbers regressed against
  `soft_gains_candidate`. The expected outcome is: less jitter, possibly
  similar end-to-end success on the real arm despite lower sim metrics.
  Treat the on-device result as ground truth.
- **Latency assumption is 2–4 ticks (40–80 ms at 50 Hz).** If the actual
  cmd→state delay on hardware is materially outside this range, the
  policy may still misbehave. If you re-tune, the rollback ladder is:
  drop `ACTION_FILTER_ALPHA` first → narrow `ACTION_LATENCY_RANGE` →
  drop `OBS_JOINT_POS_NOISE_STD` → shrink `ACTION_HISTORY_LEN`. See
  the design doc.
- **`joint_vel` is gone for good in this obs schema.** If the real-arm
  comparison shows the policy needed feedback velocity, retraining with
  a denoised velocity signal (e.g. host-side low-pass on finite-diff)
  is a separate experiment, not a tweak to this model.

## Design + implementation docs

- `docs/plans/2026-05-24-latency-robust-history-conditioned-design.md`
  — full design rationale, scope decisions, rollback ladder.
- `docs/plans/2026-05-24-latency-robust-implementation.md` — task-by-task
  implementation plan (env changes, plumbing, tests).

## Loading

Live MuJoCo viewer:

```bash
mjpython -m sim.play_policy --model models/random_stack_cup_thrower_latency_robust_v1 \
    --rand-stage 3 --rand-stage-z 3
mjpython -m sim.play_policy --model models/random_stack_cup_thrower_latency_robust_v1 \
    --cup-grid --pedestal-grid
mjpython -m sim.play_policy --model models/random_stack_cup_thrower_latency_robust_v1 \
    --no-randomize-cup --pedestal 0.15
```

`play_policy.py` reads `training.json` next to the model and reconstructs
the env with `action_latency_range`, `action_filter_alpha`,
`obs_joint_pos_noise_std`, `joint_pos_history_len`, `action_history_len`
automatically.

Programmatic:

```python
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from sim.env import RageCageEnv, env_kwargs_from_training_json

env_kwargs = env_kwargs_from_training_json(
    "models/random_stack_cup_thrower_latency_robust_v1/training.json"
)
env = DummyVecEnv([lambda: RageCageEnv(randomize_cup=True, reward_stage=3, **env_kwargs)])
env = VecNormalize.load(
    "models/random_stack_cup_thrower_latency_robust_v1/vecnormalize.pkl", env
)
env.training = False
env.norm_reward = False
env.env_method("set_pedestal_range", (0.0, 0.15))

model = PPO.load(
    "models/random_stack_cup_thrower_latency_robust_v1/policy.zip", env=env
)

obs = env.reset()
done = [False]
while not done[0]:
    action, _ = model.predict(obs, deterministic=True)
    obs, _reward, done, _info = env.step(action)
```

27-cell workspace eval:

```bash
uv run python -m sim.eval_grid \
    --run-dir models/random_stack_cup_thrower_latency_robust_v1 \
    --reward-stage 3 \
    --also-fixed
```

300-episode randomized headline eval:

```bash
uv run python -m sim.eval_rl \
    --model models/random_stack_cup_thrower_latency_robust_v1/policy.zip \
    --vecnormalize models/random_stack_cup_thrower_latency_robust_v1/vecnormalize.pkl \
    --episodes 300 \
    --reward-stage 3 \
    --out-dir sim/_rl_eval/latency_robust_v1_300ep
```
