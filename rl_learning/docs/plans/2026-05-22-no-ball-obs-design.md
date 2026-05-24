# No-ball-obs cup thrower: design

Strip `ball_pos` (3) and `ball_vel` (3) from the policy's observation
vector and retrain `random_stack_cup_thrower_v1` from scratch on the
reduced 16-dim obs. Motivation: ball state is sim-privileged and cannot
be reliably estimated on the real Piper at sub-throw timescales, so a
policy that depends on it will not transfer.

## Goal

Random-range success rate (300-episode R3×Z3 eval) within ~0.05 of
`random_stack_cup_thrower_v1` (which hit 0.807). Materially below that
suggests the ablation cost real performance and we revisit
(architecture, longer training, observation noise instead of full
removal). Materially above is unexpected and worth understanding before
moving on.

Headline secondary metric: 3×3×3 grid3d success rate. v1 hit 0.741.

## Why this is a v36 ablation, not a new paradigm

Same env mechanics as v36 (the from-scratch run that produced v1). Same
reward shape, same release_step=45, same 6-D action space, same R3+Z3
randomization, same auto reward curriculum 1→2→3. The only change is
the obs vector. This makes results directly comparable to v1 and isolates
the cost of removing ball state.

## Findings from current code that shape the design

**F1. Ball state is welded to the gripper during windup.** The ball is
constrained to the gripper from `reset()` until `release_step=45`. Across
those 45 steps, `ball_pos` and `ball_vel` are deterministic functions of
joint state and gripper geometry — i.e. fully redundant with `joint_pos`
+ `joint_vel`. The policy could in principle compute them from existing
obs. Removing the slots forces it to do so implicitly.

**F2. Post-release ball state does not affect outcomes.** After the
ball is released at step 45, the policy continues to act on the arm but
those actions can no longer influence the throw. The reward function
reads ball state from `data` directly (not from the obs vector), so the
removal does not affect reward shaping. Net: post-release, the missing
slots carried information the policy had no use for.

**F3. Reward function reads ball state directly.** `_get_info`,
`_compute_reward`, `_check_success` all use `self._ball_pos()` /
`self._ball_vel()` which read `data.qpos` / `data.qvel` — not the obs
vector. Stripping the obs slots leaves rewards untouched.

**F4. Existing checkpoints are 22-dim.** v1, v2, and `random_pos_*` all
have policy first-layer + vecnormalize stats sized to 22 obs dims.
Loading any of them against a 16-dim env will fail at `PPO.load`. We
accept this — old models stay playable on the parent commit if needed.

## Scope decisions

**S1. Strip ball_pos + ball_vel only.** `cup_xy` and `pedestal_height`
remain in the obs. On the real robot, `cup_xy` comes from perception or
fiducials, and `pedestal_height` from a known stack count. Both are
out-of-policy abstractions but observable, so they stay.

**S2. Keep `pedestal_height` encoding as-is.** No rename to `cup_rim_z`
or `cup_count`. The slot stays a continuous height in [0, 0.15m]. On
real hardware, the operator counts cups, computes height as
`(N-1) × 1.8cm`, and injects it. The `CUP_HEIGHT` offset and any other
geometry conversions are a runtime concern, not a training-time concern.

**S3. From-scratch, no warm-start.** 22 → 16 obs change makes weight
transfer impractical. Surgical first-layer reinit was considered and
rejected — the surviving deep layers' representations encode an obs
distribution the new env will never produce.

**S4. Reward stage 3 only for run #1.** Skip stage-4 cup-entry-depth
fine-tune. We want a clean v1-comparable baseline first; stage-4 is a
second run if the baseline holds up, mirroring the v1 → v2 pattern.

## Approach

### A. Env change (sim/env.py)

Three edits, all surgical:

1. `_get_obs` (line 649): drop `self._ball_pos()` and `self._ball_vel()`
   from the concat. New layout: `joint_pos(6) + joint_vel(6) + cup_xy(2)
   + pedestal_height(1) + release_countdown(1) = 16 dims`.
2. `obs_low` (line 320) and `obs_high` (line 331): drop the corresponding
   `np.array([-1.0, -1.0, -0.5])` / `np.full(3, -10.0)` (and mirrored
   high bounds). New `observation_space.shape = (16,)`.
3. Class docstring (line 180): update obs description.

No MJCF changes. No reward changes. No curriculum changes. No
`set_cup_range` / `set_pedestal_range` / `set_next_cup` /
`set_next_pedestal` changes.

### B. Training config

Mirror v36 exactly except for obs dim. Env vars for `scripts/train_ppo.sh`:

```bash
FULL_RUN=1
TIMESTEPS=25000000
N_ENVS=16
N_STEPS=2048
BATCH_SIZE=512
LR=0.0002
LR_SCHEDULE=linear
NET_ARCH=medium
REWARD_STAGE=1
CURRICULUM=auto
CURRICULUM_EVAL_EVERY=100000
CURRICULUM_EVAL_EPISODES=8
RAND_STAGE=3
Z_STAGE=3
ENT_COEF=0.01
SEED=0
OUT=sim/_rl_out/no_ball_obs_v1_25m
```

Auto curriculum promotes reward stages 1→2→3 (~230K steps in v36; expect
similar). `RAND_STAGE=3` and `Z_STAGE=3` pin the randomization breadth at
maximum from t=0 (no R-stage or Z-stage curriculum), matching v36.

### C. Naming

- Run dir during training: `sim/_rl_out/no_ball_obs_v1_25m/`
- Final model dir: `models/random_stack_cup_thrower_no_ball_obs_v1/`

### D. Evaluation

Same suite as v1/v2 for apples-to-apples:

1. 8-episode fixed-cup eval at NOMINAL `(0.85, 0)`, pedestal=0.
2. 3×3×3 grid eval (cup_xy ∈ {-10cm, 0, +10cm}², pedestal ∈ {0, 7.5cm,
   15cm}). Wired into training via `evaluate_policy_grid3d`.
3. 300-episode random R3×Z3 eval at end-of-training via
   `sim/eval_grid.py`.

Headline comparison vs v1: random-range success rate.

## Risks and contingencies

**R1. Convergence stalls or hits a lower ceiling.** If after 25M steps
random-range success is materially below v1, the leading hypothesis is
that joint-state-only obs makes timing the throw harder than expected.
Contingency: extend to 30-40M steps; if still stalled, try a recurrent
policy (LSTM) so the network can integrate joint trajectory history.

**R2. Stage promotion fails or thrashes.** v36's auto-curriculum
promoted cleanly. If the new run thrashes between stages, fall back to
manual stage 3 with a longer warmup at stage 1.

**R3. Eval at training boundaries shows no learning past stage 1.**
Indicates the bounce reward signal is harder to find without ball state.
Unlikely (joint state suffices for arm trajectory), but if it happens,
inspect via `watch_rollouts` to confirm the gripper trajectory is
varying and the issue is reward attribution, not exploration.

## Out of scope for this run

- Stage-4 cup-entry-depth fine-tune (separate follow-up if v1 baseline
  holds up).
- Multi-seed sweep (single seed=0 first; replicate later if needed).
- Recurrent policy (only if MLP stalls).
- Removing `cup_xy` or `pedestal_height` from the obs.
- Adding observation noise to `cup_xy` / `pedestal_height` to simulate
  perception uncertainty.
- MJCF changes to model literal stacked cups.
- Real-robot transfer experiments.

## Deliverables

- Env code change (3 edits in `sim/env.py`).
- Implementation plan (separate doc via `superpowers:writing-plans`).
- Training run output in `sim/_rl_out/no_ball_obs_v1_25m/`.
- Promoted model in `models/random_stack_cup_thrower_no_ball_obs_v1/`
  with README following the v1 template.
- Experiment log entry in `docs/rl_experiment_log.md` summarizing
  config, results, and v1 comparison.
