# Cup-position randomization: design

Extend the v22 single-bounce thrower (fixed cup at (1.10, 0)) to handle a
cup placed anywhere within ±10cm of nominal in the xy-plane. The policy
should learn from trial and error during training; at inference the user
places the cup, the cup_xy is fed in via the obs, and the same policy
throws to wherever it is.

## Goal

R3 grid-eval `success_rate` ≥ 0.30 across all 9 cells of a
{-10cm, 0, +10cm} × {-10cm, 0, +10cm} grid, with no cell at 0, while
not regressing v22's fixed-cup `success_rate` by more than 0.15.

## Why this is goal-conditioned RL, not a different paradigm

The obs already includes `cup_xy` (env.py:122) and the reward is
already cup-relative (env.py:480, :524, :819). Per-episode randomization
+ target-in-obs + target-relative reward is the standard formulation for
"learn a function of the target." The 256×256 MLP has plenty of capacity
for a 2-D conditioning input. The only reason v22 didn't generalize is
that the training distribution was 4cm × 4cm — comparable to reset noise
— so the policy never had reason to use `cup_xy`. Widening the
distribution forces the policy to use it.

## Findings from current code that shape the design

**F1. Curriculum eval was hardcoded to fixed cup.**
`evaluate_policy_metrics` is called with `fixed_cup=True` at
`train_rl.py:480`. Auto-promotion, the curriculum CSV, and the
best-snapshot saver all read it. v22's claimed `success_rate=0.75` was
measured at (1.10, 0); the policy's behavior across the workspace was
never observed during training.

**F2. The bounce-XY reward uses a fixed target.**
`BOUNCE_TARGET_XY = (0.78, 0.0)` (env.py:40), used for the stage-2/3
`bounce_xy` reward at env.py:767-769. With cup at e.g. (1.20, 0.10),
the geometrically correct first-bounce point is offset toward the
cup; rewarding (0.78, 0) actively fights cup randomization.

**F3. The narrow `exp(-d/0.04)` post-bounce reward stays.**
This shape is what unlocked v22 (see `rl_experiment_log.md` v21 entry).
Without it the policy hits the cup rim and never closes. The bootstrap
problem at corner cups is not solved by widening the reward — it is
solved by a curriculum on the randomization range itself, so that at
every point in training the policy is asked to generalize a little
past what it already knows.

**F4. `cup_count` stays in obs.**
Currently always 1; reserved for the future cup-stacking task.

## Approach

### A. Bounce target becomes cup-relative

Replace `BOUNCE_TARGET_XY` constant with a per-step computation:

```python
bounce_target = self.cup_xy + np.array([-0.32, 0.0])
```

The 32cm pre-cup offset matches the existing 1.10 → 0.78 setup. Y
tracks cup_y directly; this is geometrically slightly off (true
throw-plane bounce is at ~half cup_y) but well inside the 45cm
`BOUNCE_TARGET_SCALE` tolerance.

### B. Randomization curriculum

Sub-stages within reward stage 3 (reward weights unchanged):

| sub-stage | cup x-range | cup y-range | promote when |
|---|---|---|---|
| R0 | ±2cm | ±2cm | success_rate ≥ 0.5 on R0 range eval |
| R1 | ±5cm | ±5cm | success_rate ≥ 0.4 on R1 range eval |
| R2 | ±8cm | ±8cm | success_rate ≥ 0.3 on R2 range eval |
| R3 | ±10cm | ±10cm | train to convergence |

R0 doubles as a warm-start verification that the bounce-target change
hasn't broken anything. Promotion threshold drops with range because
the harder task should not be required to match easier-task numbers
before moving on.

Implementation: extend `CurriculumCallback` with a parallel
randomization stage. Add `set_cup_range(x_range, y_range)` env method.
Auto-promotion triggers off `range`-mode eval `success_rate`.

### C. Eval and viz infrastructure

**C1. `evaluate_policy_metrics` takes a `cup_eval_mode`:**
- `fixed` — current behavior. Sanity-check vs. v22 baseline number.
- `range` — N=16 episodes uniformly sampled from the *current*
  R-stage range. Drives auto-promotion.
- `grid` — 3×3 fixed grid spanning the full ±10cm workspace, regardless
  of current sub-stage. Drives best-snapshot selection and progress
  monitoring. 9 episodes per eval.

**C2. Per-cell metrics CSV.** `<run>_grid.csv` with columns
`timesteps,cup_x,cup_y,success,closest_cup_dist,valid_bounce`.
Heatmap over (cup_x, cup_y) at end of training (or live).

**C3. Multi-cup viz rollouts.** `TrainingRolloutVizCallback` renders a
3×3 composite GIF — one rollout per grid cell, side by side, each cell
labelled `cup=(x.xxx, y.xxx)`.

**C4. Probe.** Before any retraining, run the new `grid` eval against
v22 unchanged. Confirms F1 empirically and gives a starting-point
heatmap to compare against.

### D. Warm-start training run

- Load `models/single_bounce_cup_thrower_v1/policy.zip` and
  `vecnormalize.pkl`.
- Start at `reward_stage=3`, randomization stage R0.
- LR = 1e-4, linear schedule (half v22's 2e-4 — fine-tune, not
  from-scratch).
- ent_coef = 0.015 (v22 used 0.01; small bump for cup-conditioned
  exploration).
- action_delta=0.06, RESET_NOISE_STD=0.005, release_step=35,
  net=medium, 16 envs, n_steps=2048, batch_size=512 — all unchanged
  from v22.
- Total budget: 15M timesteps (~55 min at 4500 fps × 16 envs). Expected
  convergence at 8-12M.
- Best snapshot saver upgraded to keep a per-R-stage best
  (`<out>_best_R0.zip`, etc.) — same mitigation we wished we'd had
  for v21's stage-4 collapse.
- VecNormalize obs-norm stats: load v22's, accept online drift through
  R-stage promotions. If turbulence dominates, fall back to surgical
  reset of cup_xy entries of `obs_rms` at promotion.

## Implementation order

1. Land C1-C3 (eval/viz infrastructure). Verify by running the new
   `grid` eval against v22 — this is the C4 probe.
2. Land A (cup-relative bounce target) and B (randomization curriculum).
   Sanity-check by running `grid` eval on v22 with the env changes —
   should be identical to step 1 since v22 is at fixed cup and
   bounce-target only affects training reward.
3. Run training (D). Watch the live grid heatmap and per-R-stage
   success_rate.
4. Save best snapshot to `models/multi_pos_cup_thrower_v1/` with
   README and `vecnormalize.pkl`.

## Risks and kill criteria

**Risks (in order of probability):**
- R-stage promotion turbulence larger than expected. Mitigation:
  surgical reset of cup_xy `obs_rms` at promotion.
- Bounce-target geometry off — y-component shift is geometrically too
  much. Symptom would be y-asymmetric grid heatmap. Mitigation:
  switch to half-cup-y formula.
- VecNormalize cup_xy stats stale on warm load. Self-corrects in
  <100K steps; same surgical-reset fallback.
- 256×256 hits capacity. Unlikely. Fallback: `large` (512×512) or
  `deep` (256×256×128). Don't pre-emptively switch.

**Kill criteria (abort and rethink, not just retrain):**
- R0 doesn't reach 0.5 success_rate within 2M steps — warm-start is
  broken, probably bounce-target.
- R1 doesn't recover from promotion turbulence within 3M additional
  steps — implies policy isn't using `cup_xy` at all.
- Grid heatmap shows persistent isolated dead spots that don't fade
  with training — discontinuous policy behavior; check env asymmetry
  before retraining.

## Out of scope

- Cup stacking (`cup_count > 1`). The obs slot is reserved.
- Larger-than-±10cm randomization. Once we have ±10cm working, the
  same machinery extends.
- Real-arm transfer eval. Separate work item once sim policy lands
  cup entries reliably across the workspace.

---

## Postscript: what actually shipped

This plan was written before v23. By the time training succeeded
(v34), the design had drifted in several ways. Recording for the
record; the full play-by-play is in `docs/rl_experiment_log.md` under
"v23–v34: multi-position cup randomization".

**Five env bugs landed before training would converge:**

1. The `cup_world` weld snapped the cup back to (1.10, 0) regardless
   of `qpos` overrides — meaning v22's "randomization" was a no-op.
   Fixed by rewriting `model.eq_data` per reset.
2. Pre-release ball-finger contacts terminated episodes silently
   (no -25 penalty signal pre-release). Fixed by gating
   `ball_contacted_robot` / `ball_contacted_floor` accumulation on
   `self.ball_released`.
3. Reward-stage promotion (1→2→3) was guarded out when
   `rand_stage_ref` was set. Fixed by always running `_maybe_promote`.
4. The original additive bounce target `cup_xy + (-0.32, 0)` was
   geometrically wrong for off-axis cups. Replaced with
   `α · cup_xy` along base→cup line, with elliptical reward
   tolerance in throw-frame coords.
5. Render-loop "cup snap-back" viz artifact (post-step `env.render()`
   captured the auto-reset state). Fixed by skipping the post-step
   render on `done` and capturing a terminal-state frame inside
   `env.step()`.

**Design changes vs the original plan:**

- **Cup nominal moved** (1.10, 0) → (0.85, 0) for easier throw geometry.
- **`release_step` is now a fixed env parameter, not action-controlled.**
  We tried adding a 7th action dim for release timing (v25/v29/v30/v31);
  the credit-assignment problem was too hard, convergence stalled at
  ~25% peak success. Reverted to fixed release. v33 used 60, v34 used
  45 (faster convergence, same endpoint quality).
- **Bounce target redesigned.** Replaced with the fractional /
  elliptical formulation described above.
- The R-stage curriculum (R0→R1→R2→R3 with 0.5/0.4/0.3 thresholds)
  is implemented and active, but in v34 promotions fired in tight
  succession (8.3M → 9.3M) once the policy started entering cups —
  the curriculum smoothes the transition but isn't doing heavy
  lifting once the bugs were fixed.

**Final result:** v34 trained from scratch in 20M steps. Sustained
convergence at 14.2M. Final eval: 100% fixed cup, 87% on uniform
random R3 ±10cm, 8/9 on the 3×3 grid. Snapshot checked in at
`models/random_pos_cup_thrower_v1/`.

**Definition-of-done check vs the original plan:**

| design goal | result |
|---|---|
| R3 grid_success_rate ≥ 0.30 across all 9 cells, no cell at 0 | 8/9 cells succeed; 1 cell at 0 (rim graze, closest=0.035m) |
| Don't regress fixed-cup success_rate by > 0.15 | Improved (1.0 vs v22's 0.75) |
| Smooth heatmap, no isolated dead spots | Smooth — failing cell varies run-to-run, suggests geometry ceiling not learning shortfall |
| Best snapshot in `models/multi_pos_cup_thrower_v1/` | Shipped as `models/random_pos_cup_thrower_v1/` |
