# Stacked-cup thrower: design

Extend the v34 multi-position cup thrower (cup randomized within ±10cm
of (0.85, 0)) to handle a cup elevated by a continuous pedestal height
∈ [0, 15cm] — equivalent to stacks of 1-9 cups in real rage cage. The
policy reads `pedestal_height` via the existing `cup_count` obs slot
and conditions its throw arc on it. Single training run, warm-started
from v34.

## Goal

Z3 `grid3d_success_rate` ≥ 0.30 across all 27 cells of the 3×3×3 grid
(xy {-10, 0, +10} cm × pedestal {0, 7.5, 15} cm), with no cell at 0,
while not regressing v34's pedestal=0 fixed cup or 3×3 xy grid numbers
by more than 0.15.

## Why this is a v34 extension, not a new paradigm

Same machinery as v34's xy randomization. Add a continuous pedestal
height to per-episode randomization; expose it through the (already
in-obs) `cup_count` slot; let the existing 3D post-bounce reward
`exp(-d_to_cup_3d / 0.04)` pull the throw apex up to higher cup_z
without formula change. Z curriculum (Z0=0 → Z3=0-15cm) parallel to
the proven R-stage design forces the policy to use the new conditioning
input.

## Findings from current code that shape the design

**F1. `cup_count` is in obs but always 1.** env.py:594 packs
`cup_count / 10` into the obs vector; env.py:434 always sets it to 1.
The slot was reserved for cup-stacking per the v23 design (F4).
Repurposing it as `pedestal_height` keeps obs dim unchanged and lets
v34 warm-load directly.

**F2. `cup_water` is a child geom of the cup body.** rage_cage.xml:82
defines `cup_water` at body-local pos `(0, 0, 0.018)` with friction
`0.6 0.5 0.05`. When the `cup_world` weld moves the cup body up, the
water cylinder rides along automatically — matches the "top cup has
water" rule for free.

**F3. `inside_cup_z` and the 3D mouth target bake `CUP_HEIGHT` as if
cup base were at table.** env.py:649, 977, 956. All three sites shift
by `pedestal_height`. Mechanical search-replace.

**F4. v34's reward shape stays.** The narrow `exp(-d/0.04)` post-bounce
term and 3D distance-to-cup-center already account for cup_z naturally
— apex pulled up as cup_top_z grows.

## Approach

### A. Pedestal body

Add `cup_pedestal` to MJCF: cylinder, cup-radius (0.045), welded to
world with collision. Per-reset mutate
`model.geom_size[pedestal_id, 2] = pedestal_height/2` and
`model.eq_data[pedestal_eq_id, 5] = pedestal_height/2`. Cup
`cup_world` weld z updated from 0 → `pedestal_height` so the cup sits
on top.

Collision is the point — balls hitting the stack side become a real
failure mode the policy must learn to clear, matching real-game
physics. Without collision the policy could exploit "fly under"
trajectories that don't transfer to real cups.

`inside_cup_z` (env.py:649, 977) and the 3D mouth target (env.py:956)
shift by `pedestal_height`.

### B. Obs change

Repurpose the `cup_count` slot to hold `pedestal_height` directly.
Avoids changing input dim → v34 warm-load just works. Surgical reset
of that slot's `obs_rms` before training kicks off — old stats had
var≈0 (cup_count always =1), would clip `pedestal_height` insanely.

### C. Z curriculum

Sub-stages within R3 (R-axis stays pinned, since v34 mastered xy):

| stage | pedestal range | promote when |
|---|---|---|
| Z0 | 0 only | range eval success ≥ 0.5 |
| Z1 | 0-5cm | range eval success ≥ 0.4 |
| Z2 | 0-10cm | range eval success ≥ 0.3 |
| Z3 | 0-15cm | train to convergence |

Z0 doubles as warm-start sanity (pedestal=0 should match v34 exactly).
Promotion thresholds mirror R-stage's. Range eval = 16 episodes
uniform in `(R3 xy box × current Z range)`.

Implementation: extend `CurriculumCallback` with `zrand_stage_ref`
parallel to `rand_stage_ref`. R-axis stays fixed at R3. Per-Z best
snapshot saver (`<run>/best_Z0.zip` ... `best_Z3.zip`) — same playbook
as v34's per-R pattern.

New env method `set_pedestal_range(z_range)` analogous to
`set_cup_range`. Per-reset uniform sample within current Z-stage range.

### D. Eval and viz infrastructure

**D1.** `evaluate_policy_metrics` gains a `cup_eval_mode='grid3d'` —
3×3×3 grid (xy {-10, 0, +10}cm × pedestal {0, 7.5, 15}cm). 27 episodes.
Drives best-snapshot selection at full range. Existing modes (`fixed`,
`range`, `grid`) preserved.

**D2.** `<run>/grid3d.csv` per-cell metrics with columns
`timesteps, cup_x, cup_y, pedestal_height, success, closest_cup_dist,
valid_bounce`. End-of-training heatmap: 3 stacked xy heatmaps, one per
z layer.

**D3.** `TrainingRolloutVizCallback` extended to render 3 composite
3×3 GIFs — one per z layer, labelled `pedestal=0cm` / `pedestal=7.5cm`
/ `pedestal=15cm`.

**D4. Probe.** Before retraining, run `grid3d` eval against v34
unchanged. Expected: pedestal=0 layer matches v34's 8/9; pedestal>0
layers ~0/9 (policy doesn't know to throw higher). Confirms the new
eval infrastructure works and gives a baseline heatmap.

### E. Warm-start training run

- Load v34: `models/random_pos_cup_thrower_v1/policy.zip` and
  `vecnormalize.pkl`.
- Surgical reset of `cup_count` slot in `obs_rms` before kickoff.
- Start at reward_stage=3, R-stage=3, Z-stage=0.
- LR=1e-4 linear (half v34's 2e-4 — fine-tune, not from-scratch).
- ent_coef=0.012 (small bump from v34's 0.01 for new z exploration).
- action_delta=0.06, RESET_NOISE_STD=0.005, release_step=45,
  net=medium, 16 envs, n_steps=2048, batch_size=512 — all unchanged
  from v34.
- Total budget: 15M timesteps. Per-stage rough estimate: Z0 ~1M
  (sanity), Z1 ~3M, Z2 ~3M, Z3 ~5M.

### F. Run artifact layout

All v35 artifacts go under `runs/<run_name>/`:

```
runs/v35_stacked_warm/
  policy.zip                  # final snapshot
  vecnormalize.pkl
  best_Z0.zip                 # per-Z best snapshots
  best_Z0.vecnormalize.pkl
  best_Z1.zip / .vecnormalize.pkl
  best_Z2.zip / .vecnormalize.pkl
  best_Z3.zip / .vecnormalize.pkl
  curriculum.csv
  grid3d.csv
  checkpoints/
    checkpoint_001000000_steps.zip
    ...
  train_rollouts/             # GIFs from TrainingRolloutVizCallback
  watch_rollouts/             # GIFs from sim.watch_rollouts
```

Replaces the scattered ~14-entries-per-run layout of v22-v34. Code
changes:

- `sim/train_rl.py` — `--out` becomes a directory (was a path stem).
  All artifacts written with fixed names inside.
- `sim/watch_rollouts.py` — collapse `--checkpoints` + `--out` into a
  single `--run-dir`. Internally derives `<dir>/checkpoints/` (read)
  and `<dir>/watch_rollouts/` (write).
- `sim/eval_grid.py` — accept `--run-dir` as a shortcut that pulls
  policy/vecnormalize from inside and writes CSV to
  `<run-dir>/grid3d.csv`. Keep explicit flags for ad-hoc evals against
  shipped models.
- `scripts/train_ppo.sh` — pass dir-style `--out`.
- `sim/play_policy.py` — already handles dir layout
  (`_resolve_model_paths` at play_policy.py:42-43). No change.

Existing v22/v34-style files still loadable via the old paths. No
renaming of past runs. New layout applies v35 onward.

## Implementation order

1. **Pre-training smoke (probe).** Land sections A+B with smokes; run
   v34 unchanged at pedestal=15cm. Confirm: ball can physically reach
   the cup mouth at high pedestal (max apex > 25cm) at any R3 cell. If
   not, drop upper bound to 0-12cm before going further. Run grid3d
   eval — confirms infra works, gives baseline heatmap.
2. **Land section F** (run subfolder layout) — affects all downstream
   tooling, so flush early.
3. **Land sections C+D** (Z curriculum + grid3d eval + viz).
4. **Run training (E).** Watch the live grid3d heatmap and per-Z-stage
   success_rate.
5. **Ship snapshot** to `models/<final_name>/` with README and
   vecnormalize.pkl. Candidate name: `random_stack_cup_thrower_v1`.

## Risks and kill criteria

**Risks (in order of probability):**
- Physical ceiling at pedestal=15cm. `action_delta=0.06` +
  `release_step=45` bound max ball velocity. Surfaces in step-1 probe.
- VecNormalize `cup_count` stats stale on warm load. Surgical reset is
  the mitigation; same trick worked for cup_xy stats during R-stage
  promotions.
- Z-stage promotion turbulence (analogous to R-stage). Same surgical
  reset fallback if needed.
- Warm-start brittleness — policy stuck in v34's z=0 throw pattern,
  unable to learn the higher-arc variant. Falls into kill criteria
  below.

**Kill criteria (abort and rethink, not just retrain):**
- Step-1 probe shows max ball apex < 25cm at any R3 cell — pedestal=15cm
  unreachable; reduce upper bound or revisit release_step / action_delta.
- Z0 doesn't reach 0.5 success_rate within 1M warm-start steps — warm
  load broken (probably `cup_count` slot semantics).
- Z1 doesn't recover from promotion within 3M additional steps — policy
  isn't using `pedestal_height` at all.
- Grid3d heatmap shows persistent dead spots in high-z layers that
  don't fade with training — physical ceiling hit, not learning
  shortfall.

## Out of scope

- Real cup pyramid layouts (multiple cups arranged spatially).
- Pedestal heights > 15cm (would need release_step / action_delta
  tuning).
- Real-arm transfer eval. Separate work item once sim policy lands
  cup entries reliably across the (xy, z) workspace.

---

## Postscript: what actually shipped

The plan called for a warm-start from v34 with a Z curriculum. By
the time the run converged (v36), the design had drifted in a few
ways. Full play-by-play is in `docs/rl_experiment_log.md` under
"v35–v36: stacked-cup pedestal extension".

**Warm-start failed across three variants.** v35_probe (Z curriculum),
v35_warm (Z curriculum, different seed), and v35_direct_z3 (no
curriculum, full Z3 from t=0) all collapsed within 1M warm-start
steps. v34's policy has near-zero weights on the cup_count obs slot
(slot was always 1 during v34 training); surgical-resetting obs_rms
let pedestal_height flow as input, but the policy couldn't develop
slot-20 conditioning fast enough to offset gradient destruction of
v34's specific 1-bounce arc. The policy drifted into a 2-bounce
roll-to-cup attractor that lands close (~5cm) but never triggers
cup_entry — `fixed_success` crashed from 1.0 to 0.0.

**From-scratch worked.** v36, same v34 hyperparams plus
`--rand-stage 3 --z-stage 3` from t=0, 25M timesteps. Reward-stage
curriculum 1→2→3 auto-promoted in ~230K. Long bounce-target
refinement phase to 7M. Breakthrough to 1-bounce arcs at 8M.
Sustained convergence ~13M.

**Z curriculum is implemented but unused in v36.** The warm-start
attempts proved the Z curriculum fires on *mechanical headroom*
(v34's natural 37cm-apex throw clears 0-10cm pedestals without
learning) rather than learned conditioning. The curriculum code
stays in for future use.

**Env gotcha that bit us.** Runtime mutation of `model.geom_size`
silently doesn't update the collision broadphase — you also have to
write `model.geom_aabb` and `model.geom_rbound`. Without those, the
ball passes through a pedestal regardless of how tall geom_size says
it is. Cost a couple hours of debugging in Section A.

**Final result:** v36 from-scratch in 25M steps. Sustained
convergence at ~13M. Final eval: 100% fixed cup, 83% on uniform
random R3×Z3, 0.74 on the 3×3×3 grid (7/9 at z=0, 8/9 at z=7.5cm,
5/9 at z=15cm). Snapshot checked in at
`models/random_stack_cup_thrower_v1/`.

**Definition-of-done check vs the original plan:**

| design goal | result |
|---|---|
| Z3 grid3d_success_rate ≥ 0.30 across all 27 cells, no cell at 0 | aggregate 0.74; 7/27 cells at 0 (rim grazes, closest ≤ 7cm) |
| Don't regress v34's fixed-cup or pedestal=0 grid by > 0.15 | fixed 1.0 (matches v34); z=0 grid 7/9 vs v34's 8/9 (within 0.15) |
| Best snapshot in `models/<final_name>/` with README | Shipped as `models/random_stack_cup_thrower_v1/` with README |
| Warm-start from v34 with Z curriculum | Did not work; switched to from-scratch |

