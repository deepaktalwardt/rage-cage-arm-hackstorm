# RL Training Experiment Log

Beer-pong throw RL on rage_cage. Branch: `deepak/rl-merge-v2`. PPO via SB3,
xinyi's training scaffold + deepak's physics fixes (COR, thick table, water
cylinder, contact-exclude, per-substep bounce detection).

This document covers ~21 training-config experiments. v8–v18 plateaued at
`closest_cup_dist ≈ 0.07m` and zero cup-entries — we attributed this to a
physics ceiling and reverted to a v16 baseline. **v21 broke through that
plateau and produced the first cup successes** (success_rate up to 0.75
at 3M steps). The single change that unlocked it was reshaping the
post-bounce cup-distance reward from a linear ramp to an exponential, which
flipped the reward gradient from "saturating at 6cm" to "78% of reward
gated on the last 6cm."

## TL;DR

- **v21 is the first working policy.** First non-zero `success_rate` ever
  observed at 2.9M steps; 0.75 success rate at 3.0M. `closest_cup_dist`
  reached `0.015m` (ball going dead-center through the cup mouth).
- **What unlocked it:** the post-bounce `cup_dist` reward was changed from
  `1 - dist/0.5` (linear) to `exp(-dist / 0.04)`. Old form gave score=0.88
  at dist=0.06 — the policy got 88% of the available reward by just getting
  to the rim and stopping, with only 12% left to motivate threading the
  cup mouth. Exponential gives score=0.22 at dist=0.06 and 1.0 at dist=0,
  so 78% of the reward lives in the last 6cm.
- **The "physics ceiling" framing in earlier TL;DRs was wrong.** The policy
  could already reach the cup; it just had no gradient pulling it the last
  6cm because of saturating linear reward shape.
- **Diagnostic mistake to avoid:** earlier sections of this log inferred
  policy behavior from aggregate eval metrics (`closest_post_bounce_cup_dist`,
  `valid_bounce_rate`, `invalid_contact_rate`). Those collapse very different
  trajectory shapes into the same number. Always trace the per-step rollout
  CSV before claiming "the policy is doing X."

## What's solid (the foundation)

- **xinyi's training scaffold**: SB3 PPO + medium MLP (256×256) + linear LR
  2e-4 + 16 envs SubprocVecEnv + 4-stage auto-curriculum + tensorboard +
  rollout viz. Works well at ~4500 fps.
- **deepak's physics fixes** in MJCF + env: thick (10cm) table, water
  cylinder for in-cup detection, contact-exclude for cup↔world solver
  stability, COR correction in substep loop (table=0.88, cup=0.70, water=0).
- **Per-substep bounce detection** (replacing xinyi's per-control-step
  check). Caught a real bug — xinyi's logic missed ~75% of bounces because
  COR-corrected fast bounces last <1 control step.
- **Water-touch as success criterion** (in addition to xinyi's "settled in
  cup volume").
- **Cup walls 3mm + COR_CUP=0.5** (vs original 1.5mm + 0.7). More realistic
  for plastic cup, less rim-deflection.
- **release_step=35** (vs xinyi's 25). More windup time helps but isn't
  enough on its own.
- **`DISTANCE_REWARD_SCALE=0.5`** (vs original 2.0). Sharper gradient near
  cup mouth.
- **3D shaping**: `post_bounce_cup_distance_reward` now uses 3D distance to
  `(cup_xy, z=CUP_HEIGHT-0.02=0.10)` instead of pure 2D xy. Pulls ball
  *upward* toward cup mouth instead of just toward cup XY-at-any-z.
- **bounce-counter fix** (the per-substep change above): the policy can no
  longer get credit from skip-bounce trajectories that the buggy counter
  read as "single bounce."

## Best result: v21

`action_delta=0.06`, `RESET_NOISE_STD=0.005`, `bounce > 1` termination,
exp-shaped post-bounce cup-distance reward (`exp(-dist / 0.04)`), all the
foundation listed above.

Curriculum progression:

| timesteps | stage | success | valid_bounce | exact_one_bounce | closest_cup_dist |
|---|---|---|---|---|---|
| 131K | 1→2 | 0.00 | 0.875 | 0.000 | — (no bounce yet) |
| 1.4M | 2 | 0.00 | 0.625 | 0.000 | 0.068 |
| 1.8M | 2 | 0.00 | 0.000 | 0.000 | **0.051** |
| 2.9M | 2 | **0.50** | 0.500 | 0.500 | — |
| 3.0M | 2→3 | **0.75** | 0.750 | 0.750 | **0.015** |
| 3.3M | 3 | 0.375 | 0.625 | 0.375 | 0.032 |
| 3.5M | 3 | 0.250 | 0.875 | 0.250 | 0.055 |

Stage 2 plateau (1.4M–2.5M) was where the policy was bouncing on the table,
arcing into the cup XY footprint at `cup_dist ≈ 0.03–0.05`, and getting
deflected by the cup wall. The exp-reward gradient eventually pushed it past
the wall-grazing zone and into clean cup entries via the water-touch path.

Stage 3 turbulence (regressions to 0 success at some checkpoints) is the
policy adjusting to the new reward weights — `cup_dist` jumps from 0.1 to
10, `cup_entry` from 0 to 30, `success` from 0 to 100. closest_cup_dist
stays in the 0.03–0.06 range during the regressions, so the throw isn't
being unlearned.

### Earlier best: v14 (historical reference)

Before v21, the closest thing to a working policy was v14 — `bounce > 2`
termination + `extra_table_bounce_penalty=-5`. Achieved `closest_3d=0.045m`
and intermittent `cup_entry_bonus` firing, but never a single success.
v21 supersedes it.

## Experiment timeline

### v8 (initial stable config)

- `action_delta=0.05`, `JOINT_VEL_LIMIT=8`, `ent_coef=0.005`,
  `RESET_NOISE_STD=0.005`, `release_step=35`, `bounce > 1` termination
- Result: trains stably, plateaus at `closest_3d=0.07m`, `success_rate=0`

### v9: throw-power + exploration bump

- `action_delta: 0.05 → 0.07`, motion limits raised (8→11, 80→110,
  10000→14000), `ent_coef: 0.005 → 0.01`, `cup_entry: 50 → 100`,
  `bounce > 1 → bounce > 4`
- Result: **catastrophic instability**. mean_reward went to -54 at 131k.
  Combined variance (action_delta + ent_coef + raised limits) was too much.

### v10: tighten motion limits, keep action_delta=0.07

- Limits back to v8 values (8/80/10000)
- Result: still unstable. With action_delta=0.07, max vel_norm = 8.57 >
  limit=8.0. Motion-limit violations firing constantly during throws.

### v11: JOINT_VEL_LIMIT=9.5

- 10% headroom over the 8.57 max-possible vel.
- Result: same problem. Episodes ending pre-release. ep_len_mean=11.

### v12: revert action_delta to 0.05

- action_delta=0.05, but kept v9's `bounce > 4` and `extra_bounce_penalty=-25`
- Result: policy threw the ball **sideways off the table** (-10) to avoid
  the multi-bounce stacking penalty (-75 for 5-bounce). Local optimum from
  perverse incentives.

### v13: reduce extra_bounce_penalty

- `extra_table_bounce_penalty: -25 → -5`. bounce > 4 kept.
- Result: policy converged on **multi-bounce skip-train along the table**.
  Got reasonable reward (~+10). `bounce_target_rate=0` (skipping doesn't need
  to hit target). Closer than v12 but not single-bounce.

### v14: bounce > 2 termination — **best result**

- `bounce > 4 → bounce > 2`. Skip-trains die after 3 bounces.
- Result: **first run to trigger cup_entry**. closest_3d=0.045m, mean_reward
  +13 spikes (= one cup_entry per 8 eval episodes). Policy converges on
  2-bounce-into-cup. exact_one_bounce_rate=0 throughout.

### v15: penalize 2nd bounce

- Added flat -15 penalty when `table_bounce_count==2`.
- Result: **failed**. Mean reward dropped to -9. The policy abandoned the
  v14 working strategy without finding 1-bounce. Trajectories fragmented
  across off-table escapes and incomplete 1-bounce attempts. The penalty
  was too disruptive.

### v16: strict bounce > 1 (xinyi original)

- Reverted to `bounce > 1` termination. Removed second_bounce_penalty.
- Result: stable but plateaued. `exact_one_bounce_rate=0` throughout —
  policy STILL produces 2-bounce trajectories, they just terminate at
  bounce 2 with no cup_entry. closest_3d~0.08m. Confirmed: physics
  constraint is the bottleneck, not reward shaping.

### v17: action_delta=0.06 + reset_noise=0.01 + bounce > 3 graduated

- `action_delta: 0.05 → 0.06`, `RESET_NOISE_STD: 0.005 → 0.01`, bounce > 3,
  graduated penalties: 2nd=-5, 3rd=-15, 4th=-30+terminate
- Result: chaotic. Combined exploration variance (1.4× × 2× × 1.2× = 4.8×
  v8) too high. Most episodes terminating pre-release. Pre-release crashes.

### v18: add in-flight shaping

- All v17 + per-step reward `exp(-5×d_xy)` while ball released and
  pre-bounce. Stage weights 0.03/0.05/0.05/0.03.
- Result: same v17 chaos. The inflight reward fires post-release, but most
  episodes never reach release. Killed early.

### Reverted to v16-style baseline before v20

Before v20, `sim/env.py` was rolled back to the v16 reward structure
(`bounce > 1` termination, no graduated bounce penalty, no in-flight
shaping). `action_delta` was kept at `0.06` (carried from v17/v18) and
`RESET_NOISE_STD` at `0.01`. v19 was a separate experiment in this state
(see author's notes); v20/v21 below changed the reward shape on top of
this baseline.

### v20: exponential cup-distance reward (failed by setup)

- Changed post-bounce cup-distance score from `1 - dist / 0.5` (linear)
  to `exp(-dist / 0.04)`. Kept `RESET_NOISE_STD=0.01` from v17/v18.
- Result: stage 1 promotion was much slower than v16 (`valid_bounce_rate`
  hovered at 0.25 through 426K vs v16's 0.875 at 131K). Killed early.
  Almost certainly not the reward change — stage 1 has `cup_dist` weight=0
  so the new score has no effect there. RNG variance + the elevated reset
  noise was the most likely cause.

### v21: same exp-reward, RESET_NOISE_STD reduced to 0.005

- Same exp-shaped cup-distance reward as v20.
- `RESET_NOISE_STD: 0.01 → 0.005` (matches the value the v16-era log
  claimed was being used).
- Result: **first working policy.** Stage 1 → 2 promoted at 131K with
  `valid_bounce_rate=0.875` (identical to v16). Then divergence: `closest_cup_dist`
  fell to 0.05 by 1.4M (vs v16's plateau of 0.07 at 10M), to 0.026 by
  1.8M, and triggered the first `success_rate=0.5` at 2.9M, `0.75` at
  3.0M with `closest_cup_dist=0.015` (ball through cup mouth). Stage 2 → 3
  auto-promoted at 3M; cup-entries continued in stage 3 with `mean_reward`
  spikes to +70 and `success_rate` between 0.25–0.50 across most of 3M–5M.
- **Stage 4 destabilization.** Auto-promoted to stage 4 at 6M. Stage 4 weights
  halve `cup_dist` (10 → 5) and triple `cup_entry` (30 → 100), aiming to
  amplify the rare-event signal. Effect: the dense gradient the policy was
  riding got cut, the rare-event reward alone wasn't enough to maintain the
  trajectory, and `success_rate` collapsed to 0 for the rest of training
  (6M–10M). `closest_cup_dist` stayed in the 0.06–0.09 range — the policy
  was still throwing toward the cup but not entering. The peak policy at
  3M–5M was lost because no intermediate checkpoints were saved.
- **Followups landed in this PR after the v21 run:**
  - Best-snapshot saving in `CurriculumCallback` and periodic saving via
    SB3's `CheckpointCallback`, so future runs can recover the peak policy
    even if late-stage training destabilizes it.
  - Stage 3 → 4 auto-promotion disabled. Stage 3 is the last stage and
    training runs longer there. Stage 4's weight schedule needs a redesign
    before it should be re-enabled (likely keep `cup_dist` at 10 instead
    of halving it).

## Key learnings

1. **Reward gradient saturation was the real bottleneck, not physics.** v8–v18
   plateaued at `closest_cup_dist ≈ 0.07m` because the linear cup-distance
   score `1 - dist/0.5` only differs by 0.12 between dist=0.06 and dist=0.0
   — the marginal reward of getting closer was outweighed by the marginal
   motion-penalty cost of throwing more precisely. v21's exponential score
   inverted this: 78% of the reward lives in the last 6cm. Lesson: when a
   policy plateaus, look at the *gradient* of the reward at the plateau
   distance, not just the absolute reward magnitude there.

2. **Aggregate eval metrics hide trajectory shape.** Multiple times in this
   log we inferred the wrong failure mode from `valid_bounce_rate`,
   `closest_post_bounce_cup_dist`, or `invalid_contact_rate` and proposed
   fixes that didn't address the actual problem. The fix is to trace the
   per-step rollout CSV (and ideally extract a few GIF frames at closest
   approach) before claiming "the policy is doing X." Examples that burned
   us: "v16 is doing flat skip-bounces past the cup" (actually it was
   getting deflected by the rim), "v21 is oscillating between clean throws
   and floor crashes" (actually it was reliably entering the cup XY footprint
   and the floor contacts were rim-deflections rolling off the table edge).

3. **Action_delta increases destabilize.** Every attempt — 0.10 (v5), 0.07
   (v9-v11), 0.06 (v17-v18) — produced too-wide exploration variance, which
   manifests as pre-release arm crashes (motion_limit_violated, robot_table
   contact, ball-into-arm). v21 ran at 0.06 successfully but only with
   reset_noise pulled back to 0.005.

4. **Combined randomness multiplies.** ent_coef × reset_noise × action_delta
   variances compound. v17 was 4.8× v8 — unmanageable. If experimenting,
   bump ONE knob at a time. v20 (reset_noise=0.01 + exp-reward) is also a
   data point here — slow stage 1 progress vs v21 (reset_noise=0.005 +
   same exp-reward).

5. **Skip-bounce is a real local optimum.** With `bounce > N` termination
   for N≥3 and small extra_bounce_penalty, policy finds "skip flat along
   table" as cheaper than arcing. Extra_bounce_penalty needs to be JUST
   right — not 0 (allows skip), not -25 (forces sideways escape). -5 was
   our sweet spot in v13-v14.

6. **2D shaping creates a "near cup at z=0" plateau.** The 3D-shaping fix
   (target = (cup_xy, z=0.10)) was real progress — it differentiates "ball
   above cup" from "ball next to cup at z=0". But the differential is small
   so it can't break through alone — the exponential gradient was needed
   to actually drive the ball into the mouth.

7. **Bounce-counter bug was real and significant.** xinyi's per-control-step
   contact check missed brief bounces (1-2 substeps duration after COR
   correction). Fixed by per-substep detection. Caught the bug because v3-v4
   metrics (`bounce_target_rate=1.0`) were *too good* given the policy's
   actual behavior.

8. **fps depends on VecEnv class.** xinyi's `make_vec_env` defaults to
   `DummyVecEnv` (serial). Switching to `SubprocVecEnv` was a free 2-3×
   speedup. ~4500 fps with 16 envs.

9. **Sim-to-real principle stays.** We resisted lowering table friction
   (would help in sim, break real-arm transfer). Lesson: physics realism
   constrains what's achievable in sim and that's the point.

## Open directions to try next

With v21 producing a working policy, the remaining gaps are:

1. **Redesign stage 4.** The current weights destabilized v21 (see v21
   timeline entry). Likely fix: keep `cup_dist` at 10 (don't halve) and
   only adjust `cup_entry` / `success` upward. Until then, stage 3 → 4
   auto-promotion is disabled; training ends in stage 3.

2. **Wider cup randomization.** Currently 4cm × 4cm — policy can essentially
   memorize. 10cm × 10cm would force the policy to USE `cup_xy` in the obs
   and improve generalization. Worth doing once we have a baseline working
   policy at the narrow range.

3. **Real-arm transfer eval.** With a sim policy that lands cup entries,
   the next step is checking whether the trajectory is achievable on the
   physical Piper.

4. **Larger network.** Currently medium MLP (256×256). With a working
   gradient signal now, a `large` (512×512) or `deep` (256×256×128) may
   help by giving more capacity to specialize the entry trajectory.

## Bookkeeping

- `tb/PPO_*` are tensorboard logs from each run (gitignored).
- `rage_v*_seed*.curriculum.csv` are the curriculum eval logs (gitignored).
- `rage_v*_seed*_train_rollouts/` contain GIFs and per-step CSVs of training
  rollouts for visual inspection (gitignored).
- `rage_v*_seed*_best.{zip,vecnormalize.pkl}` are the best snapshots saved
  by `CurriculumCallback._maybe_save_best` during training (gitignored).
- `rage_v*_seed*_checkpoints/` hold periodic snapshots saved by
  `CheckpointCallback` every 500K env steps (gitignored).
- v8 produced the pre-cup-entry policy plateau. v14 produced an intermittent
  multi-bounce cup-entry. v21 was the first single-bounce cup-entry policy
  with a non-zero `success_rate`. v22 reproduced v21 with auto-saving on
  and captured the 3M-timestep peak (`success_rate=0.75`,
  `closest_cup_dist=0.015m`) — that snapshot is checked in at
  `models/single_bounce_cup_thrower_v1/`.

## v23–v34: multi-position cup randomization

This block covers the extension from "throw at fixed cup at (1.10, 0)"
to "throw at any cup within ±10cm of (0.85, 0)". The final policy
(checked in at `models/random_pos_cup_thrower_v1/`) hits **100% at
fixed cup, 87% on uniform-random R3 ±10cm, 8/9 on the canonical grid**,
trained from scratch in 14M timesteps to sustained convergence.

### TL;DR

- **The bigger story is bug-finding, not RL tuning.** Five real bugs
  blocked progress; once they were fixed, the multi-position task
  trained cleanly in <20M steps from scratch. Most of the 11 runs
  (v23–v34) were diagnosing those bugs.
- **Final design:** fixed `release_step=45` + 6-D action (joint deltas
  only) + cup-relative bounce target (`0.7 · cup_xy` along arm-base→cup
  line, elliptical reward in throw-frame coords) + R-stage curriculum
  (R0→R1→R2→R3, ±2/5/8/10cm) layered inside reward stage 3.
- **What v22 actually learned:** v22's "75% success" was at fixed cup
  only because the `cup_world` weld pinned the cup to (1.10, 0)
  regardless of the randomization sample. v22 never saw cup variation
  during training; it memorized one throw. The C4 probe (grid eval
  after fixing the weld) confirmed v22 generalizes 0/9 across ±10cm.

### Bugs found and fixed

These were each load-bearing — every time we found one, training
immediately got better. They're listed in roughly the order discovered.

1. **`cup_world` weld snapped cup back to nominal regardless of qpos
   override.** The MJCF `<weld>` between world and cup had a hard-coded
   `relpose="1.10 0 0 ..."`; writing to `data.qpos[cup_qpos]` at reset
   only briefly placed the cup elsewhere — within ~10 solver steps the
   compliant weld pulled it back. v22 trained against ±2cm
   randomization but the cup actually never moved. Fix: rewrite
   `model.eq_data[cup_world_eq_id, 3:6]` at every reset so the weld's
   anchor matches the sampled cup_xy.
2. **Pre-release ball-robot contact terminated episodes silently.**
   The ball is welded to the gripper with a compliant solver; small
   solver jitter lets it briefly clip a finger geom (link7/link8). The
   env's `_terminal_failure()` counted ball-robot contacts always, but
   the `-25 invalid_ball_contact_penalty` only fired post-release —
   so episodes died with no signal beyond the time penalty. PPO had no
   gradient to learn "don't trigger this". Fix: gate the
   `ball_contacted_robot` / `ball_contacted_floor` flags on
   `self.ball_released`. Immediate effect: valid_bounce_rate jumped
   from 0.6–0.7 to 1.0 at the first eval.
3. **Reward-stage promotion (1→2→3) was guarded out when the
   randomization curriculum was active.** A leftover `if
   self.rand_stage_ref is None:` from earlier scaffolding meant fresh
   runs sat in reward stage 1 forever (no bounce_xy / cup_dist
   reward). Fix: always run `_maybe_promote(fixed_metrics)`; rand-stage
   promotion is independent and gated by reward stage 3.
4. **Bounce-target offset was hand-coded for fixed cup.** The original
   rule was `bounce_target = cup_xy + (-0.32, 0)`, geometrically
   correct only when the cup is on-axis. For an off-axis cup at
   (0.95, +0.10), the rule placed the target on a line parallel to
   world-x rather than on the actual base→cup throw axis. Replaced
   with `target = α · cup_xy` (α=0.7) along the throw axis, plus an
   elliptical reward tolerance with `σ_long=0.30`, `σ_perp=0.08` in
   throw-frame coords. The asymmetric tolerance is physically
   motivated: perpendicular bounce error translates ~1:1 into y-miss
   at the cup, longitudinal error is partially absorbed by ball
   bounce-out behavior.
5. **Render-loop "snap-back" viz artifact.** GIFs of training rollouts
   appeared to show the cup teleporting at the end. Cause: the
   render loop captured `env.render()` after `env.step()` returned
   `done=True`, which is *after* SB3's VecEnv auto-reset — so the
   final frame showed the next episode's reset state. Fix: skip the
   post-step render when `done=True`, and capture a terminal-state
   frame inside `env.step()` before returning. Pure cosmetic but
   misled the user (and me) into chasing physics bugs that weren't
   there.

### Action-space release: tried, abandoned

We initially tried making release time a 7th action dimension —
`action[6] > 0` triggers release, gated to a [30, 100] window. Hypothesis
was that adapting release timing per cup position would help. Result
(v25, v29, v30, v31): policy did learn to release within the window,
but convergence was 3-5× slower than fixed release. The credit-assignment
problem of "was this miss because of the trajectory or the release
timing?" dominated. v31 reached 25% peak success at 5M steps then
plateaued. Reverted to fixed release (v33 onward).

### Final design choices and their effects

- **Cup nominal moved (1.10, 0) → (0.85, 0).** Closer to robot base;
  shorter throw is mechanically easier and fits the arm's effective
  reach. Workspace becomes ±10cm of (0.85, 0) — i.e., x ∈ [0.75, 0.95],
  y ∈ [-0.10, 0.10].
- **`release_step` tuned 35 → 60 → 45.** v22 used 35 (tuned for cup at
  1.10). v33 used 60 (longer windup, broader workspace). v34 tightened
  to 45, which converged faster (stage 2→3 promotion in 229K vs 819K
  for v33) without hurting endpoint quality. The throw is shorter at
  (0.85, 0) so less windup is needed.
- **R-stage curriculum.** R0 ±2cm → R1 ±5cm → R2 ±8cm → R3 ±10cm,
  promoted on `range_success ≥ 0.5 / 0.4 / 0.3`. Auto-promotion
  happened in tight succession at v34 (R0→R1→R2→R3 between 8.3M and
  9.3M); the curriculum smoothes the broadening of the cup
  distribution but isn't a hard requirement once the bug fixes
  landed.
- **Bounce-target ellipse parameters.** α=0.7, σ_long=0.30,
  σ_perp=0.08. α=0.7 reproduces (0.77, 0) at the v22-era cup,
  matching the working prior. σ_perp=0.08 is tighter than σ_long
  because perpendicular bounce error directly causes y-miss at the
  cup with no compensating physics, while longitudinal error is
  partially absorbed by ball bounce-out variability.

### Result curves (v34, 20M steps)

| milestone | timesteps |
|---|---|
| Stage 1 → 2 | 0.13M |
| Stage 2 → 3 | 0.23M |
| First fixed-cup success | 5.6M |
| R0 → R1 | 8.3M |
| R1 → R2 → R3 | 8.8M / 8.9M |
| First fixed_succ ≥ 0.875 | 8.2M |
| First grid_succ ≥ 0.8 | 12.9M |
| Sustained convergence (3-eval streak fixed≥0.875 AND grid≥0.7) | 14.2M |

Last 5M of training was flat at fixed=1.0, grid≈0.89, range≈0.94.
Running longer didn't move the ceiling. The single failing grid cell
shifts run-to-run (v33 fails at (0.75, -0.10), v34 fails at (0.85,
+0.10)) — looks like a real "rim graze" geometry limit rather than a
learning shortfall.

### Caveats / things to watch out for

- **Models are tied to the `release_step` they were trained with.**
  v22 (release=35), v33 (release=60), v34 (release=45) are all
  *incompatible at inference* if you load them into an env with a
  different `release_step` — the policy peaks gripper velocity at
  the trained step, releasing earlier or later misses. The
  shipped `random_pos_cup_thrower_v1` is the v34 R3-best, so the
  current `sim/env.py` default (`release_step=45`) matches. If you
  retrain with a different release step, save it in metadata or
  pass it explicitly to play_policy / eval_grid.
- **Eval mode mismatch.** `evaluate_policy_metrics` has three modes:
  `fixed` (cup at NOMINAL), `range` (uniform sample within current
  R-stage box, gates auto-promotion), and `grid` (the canonical 3×3
  ±10cm grid, drives best-snapshot selection). The grid is only 9
  points; the random uniform success rate is the more honest
  "operational" metric. Don't promote/select on grid_success alone if
  you suspect overfitting to the cells.
- **VecNormalize stats need to match the model.** All evals copy the
  obs_rms saved in `vecnormalize.pkl`. If you tweak the env's obs
  space (add/remove dimensions, change bounds), retrain — the
  pretrained obs_rms won't generalize.

## v35–v36: stacked-cup pedestal extension

This block covers the extension from "throw at any cup within ±10cm of
(0.85, 0) at table height" to "throw at any cup within that same
±10cm window *and* with the cup elevated 0–15cm on a pedestal" — i.e.,
simulating a real-rage-cage stack of 1–9 nested cups. The final policy
(`models/random_stack_cup_thrower_v1/`) hits **100% at fixed
nominal cup, 83% on uniform-random R3 × Z3 sampling, 0.74 on the
canonical 3×3×3 grid**, trained from scratch in 25M timesteps.

### TL;DR

- **The warm-start strategy failed cleanly.** v34 had near-zero
  weights on the cup_count obs slot (always 1 during v34 training).
  Repurposing that slot to carry `pedestal_height` and surgical-resetting
  the obs_rms stats let gradient flow through, but the policy ignored
  the new input initially and gradients destroyed v34's specific
  1-bounce arc faster than slot-20 weights could grow. The policy
  drifted into a 2-bounce roll-to-cup attractor and never recovered.
- **From-scratch with R3 × Z3 from t=0 worked.** Same hyperparams as
  v34 plus pedestal randomization in the env, 25M steps, hit same
  caliber result as v34 extended over the new z axis.
- **Reward shape unchanged.** The 3D `cup_dist` reward in stage 3 has
  target z = `pedestal_height + CUP_HEIGHT - 0.02`, which is the
  implicit "throw to the right height" gradient signal. Stages 1 and 2
  are pedestal-blind by design (just bounce mechanics + bounce target).
- **Z curriculum is implemented but unused in the shipped run.** The
  warm-start attempts used Z0→Z3 promotion; the curriculum fired on
  *mechanical headroom* (v34's natural 37cm-apex throw clears 0-10cm
  pedestals without learning) rather than learned pedestal-conditioning,
  so it gave a misleading "good progress" signal right up until Z3
  forced actual learning and the policy collapsed. From-scratch +
  full Z3 from t=0 sidesteps the trap.

### Env changes from v34

1. **Pedestal body in MJCF.** `cup_pedestal` cylinder (radius 4cm,
   variable half-height) welded under cup with collision. The crucial
   gotcha: `model.geom_size` is mutated at reset, but the broadphase
   uses `model.geom_aabb` and `model.geom_rbound` which are *cached at
   compile time*. Without updating those alongside `geom_size`, the
   ball passes through the pedestal — broadphase culls the contact
   pair. `env.py:reset()` writes all three.
2. **Cup welded at z=pedestal_height.** `cup_world` weld eq_data and
   cup qpos z both updated to `pedestal_height` per reset.
3. **Obs slot repurposed.** The cup_count slot (env.py:594, was always
   `1/10.0`) now carries `pedestal_height` in meters. Obs bounds widened
   from `[0.1, 1.0]` to `[0.0, 0.20]` on that slot. Dim unchanged so
   warm-loading v34 didn't need an input-layer expansion (it just needed
   the surgical reset that turned out not to be enough).
4. **Z-dependent reward sites shifted by pedestal_height.** Three sites
   in env.py: `inside_cup_z` in `_success` (line 649), `inside_cup_z`
   in `_update_reward` (line 977), and the 3D `cup_target` z (line 956).
5. **`cup_water` rides along for free.** The water cylinder is a child
   geom of the cup body (`pos="0 0 0.018"` body-local), so welding the
   cup body up to pedestal_height moves the water with it. Matches the
   "only top cup has water" real-game rule automatically.

### What failed (v35 series, warm-start)

Three runs, three variants of the warm-start strategy:

1. **v35_probe (2M, Z curriculum, warm-start):** Z0 → Z1 at 130K
   (v34 distribution match), Z1 → Z2 at 230K, Z2 → Z3 at 430K. All
   three Z promotions fired on mechanical headroom — `range_success`
   stayed at 0.8 because v34's high-arc throw still hit the lower
   pedestals. By 1M steps in Z3, `median_2nd_bounce_cup_dist` dropped
   from `10.0` (no second bounce) to `0.13m` — the policy abandoned
   v34's 1-bounce arc for a 2-bounce roll-to-cup mode. `fixed_success`
   crashed from 1.0 to 0.0 and stayed there.
2. **v35_stacked_warm_seed2 (20M, Z curriculum, warm-start):** Same
   trajectory as the 2M probe. Different seed, same wall. Killed at
   1.7M after pattern confirmed.
3. **v35_stacked_direct_z3_seed3 (20M, direct Z3, warm-start):**
   Hypothesized that the curriculum was the problem — maybe direct
   Z3 from t=0 would force pedestal-conditioning before v34's arc
   collapsed. Same outcome by 1M steps. Diagnosis: not a curriculum
   problem, a warm-start problem. Killed at 6.6M.

The 2-bounce roll-to-cup mode is a real local optimum: the ball lands
within 4-5cm of the cup *via* a second table bounce. Gets partial
cup_dist reward (3D distance is small once ball is on table near cup
base) but never triggers `cup_entry` (z criterion fails). Strict
`bounce_count == 1` success rule means it's permanently stuck at 0%.

### What worked (v36, from scratch)

`runs/v36_stacked_scratch_seed1`. 25M timesteps, ~50min wallclock.

- LR=2e-4 linear, ent_coef=0.01, net=medium, n_envs=16, n_steps=2048,
  batch_size=512 — identical to v34's hyperparams.
- `--reward-stage 1 --rand-stage 3 --z-stage 3` — start at reward
  stage 1 (auto-promote), full R3 cup_xy from t=0, full Z3 pedestal
  (0-15cm) from t=0.
- No warm-start, no surgical reset.

Result curves:

| milestone | timesteps |
|---|---|
| Stage 1 → 2 | 0.13M |
| Stage 2 → 3 | 0.23M |
| `median_closest` drops below 5cm | 7M |
| First sustained fixed_success ≥ 0.5 | 8.2M |
| First grid3d ≥ 0.3 | 8.0M |
| `median_2nd_bounce` flips back to `~10` (clean 1-bounce arcs) | 8.2-9.1M |
| Sustained fixed=1.0 + grid3d ≥ 0.5 | ~13M |
| Final eval | 25M |

Final eval (deterministic seed=0):

| metric | value |
|---|---|
| fixed (8 ep nominal) | 1.000 |
| range R3 × Z3 (64 ep random) | 0.828 |
| grid3d (27 cells) | 0.741 |
| z=0 layer (9 cells) | 7/9 |
| z=7.5cm layer (9 cells) | 8/9 |
| z=15cm layer (9 cells) | 5/9 |
| median_closest_cup_dist (grid3d) | 1.9 cm |

The 7 failing cells are rim grazes at the workspace corners (cup at
the far edge of ±10cm xy combined with the tallest pedestal), same
failure mode as v34's single failing grid cell. Closest distance on
those cells is 4-7cm — the ball lands right next to cup, not far away.

### Tooling additions for v35/v36

- **Run subfolder layout.** Replaces v22-v34's scattered ~14 files per
  run at top of `runs/`. All artifacts now under `runs/<name>/`:
  `policy.zip`, `vecnormalize.pkl`, `best_R*.zip`, `best_Z*.zip`,
  `curriculum.csv`, `grid.csv`, `checkpoints/`, `train_rollouts/`,
  `watch_rollouts/`, `tb/`, `training.json`.
- **`evaluate_policy_grid3d`** (train_rl.py) — 3×3×3 eval (cup_xy 3-grid
  × pedestal {0, 7.5, 15cm}). Drives best-snapshot selection when Z
  curriculum is enabled.
- **`--z-stage` + `Z_RANDOMIZATION_STAGES`** in train_rl.py — Z0/Z1/Z2/Z3
  curriculum stages parallel to R0/R1/R2/R3. Unused in shipped run but
  available.
- **`--surgical-reset-obs-rms-slots`** in train_rl.py — for warm-starts
  where an obs slot's semantics has changed. Used in v35; doesn't make
  warm-start succeed but does unbreak it cosmetically.
- **`watch_rollouts --pedestals`** flag — comma-separated list of
  pedestal heights to render at each cup_xy cell. Default
  `0.0,0.02,0.05,0.10,0.15`. Without this you only see the policy at
  pedestal=0 and miss the whole point of the trained distribution.
- **`play_policy --pedestal / --pedestal-grid / --rand-stage-z`** —
  mirrors the cup-xy flags for pedestal viewing.
- **`probe_apex.py`** — quick measurement of max ball z-height a saved
  policy produces. Used as the v35 pre-training reachability check
  (confirmed v34 reaches ~37cm apex, well above the 27cm needed for
  pedestal=15cm).

### Caveats / things to watch out for (additions to v34's list)

- **Warm-start with semantic-changed obs slot is fragile.** The
  v35 attempts confirmed it. If you change what an obs dim *means*,
  retrain from scratch unless you've got a very compelling reason
  and budget to babysit the run.
- **Runtime mutation of `model.geom_size` requires updating
  `model.geom_aabb` and `model.geom_rbound`.** MuJoCo's broadphase
  uses the cached AABB/rbound, not the live geom_size. Silent
  ball-through-pedestal bug if you forget.
- **`cup_dist` reward in stage 3 is the only pedestal-aware signal.**
  Stages 1-2 are pedestal-blind. If a future change disables stage 3
  or moves the cup_target z computation, the policy loses its
  height-conditioning signal.
- **Z curriculum eval is misleading on a warm-start.** v34's natural
  37cm-apex throw clears 0-10cm pedestals without learning, so Z0→Z2
  range eval looks great even though the policy isn't using
  pedestal_height. The promotion signal only becomes honest at Z3.

