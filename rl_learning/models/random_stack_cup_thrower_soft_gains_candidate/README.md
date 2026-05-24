# random_stack_cup_thrower_soft_gains_candidate

**Candidate snapshot, not a final model.** Pulled mid-training from an
in-progress run for early on-device validation. Will be superseded by a
final 25M-step model from the same run.

PPO policy for the same task as `random_stack_cup_thrower_no_ball_obs_v1`
— throw a ping-pong ball into a Solo cup placed within ±10cm of nominal
`(0.85, 0)` *and* elevated by 0–15cm of pedestal — but trained against a
**softened MuJoCo controller** intended to match the real PiPER's
effective dynamics under MIT kp=10, kd=0.5.

## Why this run exists

The v37 / `no_ball_obs_v1` policy was trained against MuJoCo gains
(joints 1–3 `kp=80, kv=5`; joint 4 `kp=40, kv=5`; joints 5–6 `kp=10,
kv=1.5`) that are physically out of reach for the real PiPER given CAN
latency, encoder noise, and motor bandwidth. On the real arm the policy
issued position commands the controller couldn't track at the assumed
rate, blowing up the sim2real gap.

This run lowers MuJoCo's actuator gains to the real arm's effective
operating regime, so the policy learns trajectories the physical
controller can actually execute. The fastest sim2real fix isn't tuning
the real arm up to match MuJoCo — it's tuning MuJoCo down to match the
real arm.

## MJCF gain change

`sim/mjcf/agilex_piper/piper.xml` actuators:

| joint | old kp | old kv | new kp | new kv |
|---|---:|---:|---:|---:|
| 1 | 80 | 5.0 | 6.5 | 0.4 |
| 2 | 80 | 5.0 | 6.5 | 0.4 |
| 3 | 80 | 5.0 | 6.5 | 0.4 |
| 4 | 40 | 5.0 | 2.5 | 0.2 |
| 5 | 10 | 1.5 | 2.5 | 0.2 |
| 6 | 10 | 1.5 | 2.5 | 0.2 |

Gripper actuator is unchanged. Friction (`frictionloss=0.3`) is
unchanged — known potential issue (deadband ≈ 2.6° at the new joint 1–3
gains) but not retuned for this run.

## Observation / action / reward / randomization

Identical to `random_stack_cup_thrower_no_ball_obs_v1`:

```
obs (16-dim): joint_pos(6), joint_vel(6), cup_xy(2), pedestal_height(1),
              release_countdown(1)
action (6-dim): joint deltas, scaled by action_delta=0.06
reward: 4-stage curriculum; auto-promoted 1→2→3 (stage 4 disabled)
randomization: cup_xy ±10cm, pedestal 0–15cm
release_step=45, RESET_NOISE_STD=0.005
```

No action filter or latency was applied (`action_filter_alpha=1.0`,
`action_latency_steps=0`). The smooth_v1 design knobs landed in the
codebase before this run, but were left at their no-op defaults to
isolate the effect of the gain change.

## Snapshot provenance

- Source: `sim/_rl_out/soft_gains_v1_25m/best_Z3.zip` (CurriculumCallback's
  rolling best at Z-stage 3, the full 0–15cm pedestal randomization).
- Snapshot time: 2026-05-23 19:55 PDT (refreshed at training stop).
  Underlying `best_Z3.zip` was last updated by the callback at 19:37,
  corresponding to ~11–15M training timesteps.
- Training was stopped early at ~16M of the 25M-step budget after
  observing a clear plateau: fixed-cup success pinned at 1.000 across
  50+ consecutive evals, `range_success_rate` 0.94–1.00,
  `grid3d_success_rate` oscillating 0.78–0.96 with no upward trend, and
  the callback's `best_Z3.zip` not refreshing for the final ~5M
  timesteps. Continuing past 16M was unlikely to yield meaningful
  improvement on this MJCF.

## Performance at this snapshot

Auto-curriculum evals during the plateau region (8-episode fixed
deterministic + 16-episode range, both seed=0). Numbers below are
representative of the steady-state band rather than a single eval row;
exact values fluctuate per eval due to small-sample noise:

| metric | typical value |
|---|---|
| `success_rate` (8 ep fixed nominal cup) | 1.000 (sustained) |
| `valid_bounce_rate` | 1.000 |
| `exact_one_bounce_rate` | 1.000 |
| `bounce_target_rate` | 1.000 |
| `median_closest_cup_dist` | 0.007–0.010 m |
| `range_success_rate` (16 ep, ±10cm × 0–15cm) | 0.94–1.00 |
| `grid_success_rate` (3×3 cup_xy grid) | 0.78–1.00 |
| `grid3d_success_rate` (3×3×3 cup_xy × pedestal) | 0.78–0.96 |
| `mean_reward` | 135–140 |
| `invalid_contact_rate` | 0.000 |

Stage 1→2 promoted at 131K, stage 2→3 at 229K (mirroring v37 timing).
First cup-entries at 1.8M (vs v37's ~3.0M). `grid3d_success` reached
v37's final 25M number of 0.963 by 11.1M training steps — less than
half the budget. The soft-gain run converged measurably faster than the
stiff-gain v37 baseline despite training against an objectively harder
controller.

## Comparison to `random_stack_cup_thrower_no_ball_obs_v1` (final 25M)

| metric | v37 final 25M | this candidate (plateau) |
|---|---:|---:|
| range_success | 0.940 (300 ep) | 0.94–1.00 (16 ep) |
| grid3d_success (27 cells) | 0.963 | 0.78–0.96 (range) |
| median_closest (range) | 0.013 m | 0.007–0.010 m |
| fixed-cup success (8 ep) | 0.750 | 1.000 (sustained) |

Caveat: v37 numbers are the 300-episode final eval; this candidate
numbers are 8-episode and 16-episode auto-curriculum evals. Eval-set
sizes differ. Use this candidate as a directional signal, not a
head-to-head. Final apples-to-apples comparison would require running
`eval_grid` against the `models/` artifact.

## Files

- `policy.zip` — SB3 PPO model, MlpPolicy medium 256×256, 16-dim input.
- `vecnormalize.pkl` — observation/reward normalization stats. Required
  at inference; without it the model receives unnormalized observations
  and produces wrong actions.

## Caveats

- **Plateau snapshot, not final.** Training was stopped at ~16M of the
  25M budget after the policy plateaued. If you re-run later, more
  steps on this MJCF are unlikely to improve the policy meaningfully —
  consider changing reward shape, friction, or filter/latency knobs
  instead of just running longer.
- **MJCF must match at inference.** This policy expects the soft-gain
  `piper.xml`. Loading it against an older stiff-gain MJCF will not
  reproduce the trained dynamics — actions will produce different joint
  motion than during training.
- **Real-arm validation pending.** This is the first soft-gain candidate;
  the whole point is to test whether the gain change closes the sim2real
  gap. Treat the on-device result as the ground truth, not the sim
  metrics.
- **No filter / no latency.** The `action_filter_alpha` and
  `action_latency_steps` knobs are at no-op defaults. If the real-arm
  test still shows tracking issues, the next iteration would be to
  enable filter (α=0.6) and latency (1 step) and retrain.
- **Friction not retuned.** With softer kp the static friction deadband
  is now ~2.6° at joints 1–3. Hasn't visibly hurt training so far but
  may show up as a sim2real difference if the real arm has a smaller
  effective friction.

## Loading

Live MuJoCo viewer:

```bash
mjpython -m sim.play_policy --model models/random_stack_cup_thrower_soft_gains_candidate \
    --rand-stage 3 --rand-stage-z 3
mjpython -m sim.play_policy --model models/random_stack_cup_thrower_soft_gains_candidate \
    --cup-grid --pedestal-grid
mjpython -m sim.play_policy --model models/random_stack_cup_thrower_soft_gains_candidate \
    --no-randomize-cup --pedestal 0.15
```

Programmatic:

```python
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from sim.env import RageCageEnv

env = DummyVecEnv([lambda: RageCageEnv(randomize_cup=True, reward_stage=3)])
env = VecNormalize.load(
    "models/random_stack_cup_thrower_soft_gains_candidate/vecnormalize.pkl", env
)
env.training = False
env.norm_reward = False
env.env_method("set_pedestal_range", (0.0, 0.15))

model = PPO.load(
    "models/random_stack_cup_thrower_soft_gains_candidate/policy.zip", env=env
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
    --run-dir models/random_stack_cup_thrower_soft_gains_candidate \
    --reward-stage 3 \
    --also-fixed
```
