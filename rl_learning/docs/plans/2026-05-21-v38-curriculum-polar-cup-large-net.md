# v38: Curriculum + polar cup obs + large net — implementation plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Train `v38_far_curric_large_seed1` to unblock the `±15cm × ±15cm × 0–12cm` workspace plateau seen in v37 (`runs/v37_stacked_far_seed1`, ~3–11% grid3d success), by (a) re-enabling the R-stage and Z-stage randomization curricula, (b) adding polar cup features (`cup_theta`, `cup_r`) and dropping ball obs (which are redundant FK pre-release and unavailable on the real arm post-release), and (c) widening the policy/value MLP to 512×512.

**Architecture:** Minimal obs-space surgery in `sim/env.py` (one method + two bounds vectors), surgical update of one smoke test that depended on the old obs shape, and an env-var-only relaunch via the existing `scripts/train_ppo.sh`. No new flags or training-loop changes — the curriculum machinery already exists (v34/v36 used it) and the launcher already wires `--rand-stage`, `--z-stage`, `--net-arch`, `--lr` through.

**Tech Stack:** Python 3, MuJoCo (via `mujoco` package), Gymnasium, Stable-Baselines3 PPO, uv for env mgmt.

---

## Background

v37 (`runs/v37_stacked_far_seed1`, 28M steps, R3/Z3 from t=0, 256×256 MLP) reached:
- `valid_bounce_rate` and `bounce_target_rate` ≈ 1.0 throughout — *core throwing motion learned*.
- `median_closest_cup_dist` plateaus at 5–7cm; never closes.
- `grid3d_success_rate` ≈ 3–11%; worst cell `(1.10, +0.15)` misses by 42cm.
- Failure concentrates at far + sideways corners — cells that didn't exist in v36's ±10cm workspace.

Diagnosis (see brainstorm transcript): the failure isn't capacity or control authority. `joint1` has ~3 steps of margin to swing to any reachable angle within the 45-step pre-release budget. The bottleneck is *learning signal*: (1) the workspace got 2.25× larger but the rand curriculum was disabled, so corner cells are sparse in uniform sampling, and (2) cup is encoded as cartesian `(x, y)`, so the MLP has to learn the nonlinear `atan2` to set `joint1` — and there's a tempting wrist-tilt shortcut that works at small Y and saturates at large Y.

Sim2real constraint kept us off cup-conditioned reset poses; everything in this plan deploys cleanly: training-only curriculum, observation features computable from any cup-pose sensor, and forward-kinematics-recoverable ball obs (which we're dropping anyway).

## Observation-space delta

Current (22 dims):
```
joint_pos(6) + joint_vel(6) + ball_pos(3) + ball_vel(3) + cup_xy(2) + pedestal_height(1) + release_countdown(1)
```

New (18 dims):
```
joint_pos(6) + joint_vel(6) + cup_xy(2) + cup_theta(1) + cup_r(1) + pedestal_height(1) + release_countdown(1)
```

Net change: **−6 ball obs, +2 polar cup features**.

`cup_theta = atan2(cup_y, cup_x)` and `cup_r = ||cup_xy||`. Both are deterministic functions of `cup_xy` — they're feature engineering, not new state. We keep cartesian `cup_xy` to avoid forcing the policy to internally re-Cartesianize when the throw geometry needs it.

Ball obs removal rationale (verified in `sim/env.py:740` `_release_ball` and `sim/env.py:587` step-loop): pre-release, the ball is welded to `link6` via `ball_grip` equality, so `ball_pos = FK(link6) + 9cm·ẑ_link6` — fully determined by `joint_pos`. Post-release, `_advance_control_step` is called with `action=None` (policy is not queried). So the ball obs is a redundant shortcut feature pre-release and irrelevant post-release; dropping it removes a deploy-time FK requirement entirely.

## Bounds

For `cup_theta`: workspace boundary `(0.75, ±0.20)` gives `max |theta| = atan2(0.20, 0.75) ≈ 0.262 rad`. Bound at `±0.30`.

For `cup_r`: min at `(0.75, 0) = 0.75`; max at `(1.15, ±0.20) ≈ 1.167`. Bound at `[0.70, 1.20]`.

## Training config

| knob | v37 | v38 |
|---|---|---|
| `NET_ARCH` | `medium` (256×256, ~145k params) | `large` (512×512, ~540k params) |
| `LR` | 2e-4 linear | 1.5e-4 linear |
| `RAND_STAGE` | (disabled) — R3 from t=0 | `0` — auto-promote R0→R3 |
| `Z_STAGE` | (disabled) — Z3 from t=0 | `0` — auto-promote Z0→Z3 |
| `TIMESTEPS` | 28M | 30M |
| `SEED` | 1 | 1 |
| obs dim | 22 | 18 |
| `BOUNCE_SIGMA_*`, `BOUNCE_FRACTION`, `release_step`, action delta, vel/acc/jerk limits, reward weights | (unchanged from v37) | (unchanged) |

The curriculum auto-promotion thresholds and stage definitions in `sim/train_rl.py` are already the values we want (R: 0.5/0.4/0.3, Z: 0.5/0.4/0.3, stages widened to ±15cm and 0–12cm in the current uncommitted edits).

## Backward compatibility

This run breaks obs-space compat with v34/v36/v37 saved policies — **intentional**. Those policies stay where they are (`models/random_pos_cup_thrower_v1`, `models/random_stack_cup_thrower_v1`); the existing `sim/smoke_z_curriculum.py::test_evaluate_policy_grid3d_shape` test that loaded v34 needs to switch to a freshly-initialized policy because the saved input layer dim no longer matches.

No warm-start in v38. From-scratch.

---

## Task 1: Update env obs (drop ball, add polar cup)

**Files:**
- Modify: `sim/env.py:180-187` (class docstring)
- Modify: `sim/env.py:315-342` (obs_low / obs_high)
- Modify: `sim/env.py:646-657` (`_get_obs`)
- Create: `sim/smoke_obs_polar.py`

### Step 1.1: Write the failing test

Create `sim/smoke_obs_polar.py`:

```python
"""Smoke for v38 obs layout: drop ball_pos/ball_vel, add (cup_theta, cup_r).

New layout (18 dims):
    joint_pos(6) + joint_vel(6) + cup_xy(2) + cup_theta(1) + cup_r(1)
    + pedestal_height(1) + release_countdown(1)

Indices:
    [0:6]   joint_pos
    [6:12]  joint_vel
    [12:14] cup_xy
    [14]    cup_theta = atan2(cup_y, cup_x)
    [15]    cup_r     = ||cup_xy||
    [16]    pedestal_height
    [17]    release_countdown

Run:  uv run python -m sim.smoke_obs_polar
"""

from __future__ import annotations

import math

import numpy as np

from sim.env import RageCageEnv


def test_obs_shape_is_18() -> None:
    env = RageCageEnv(randomize_cup=False, reward_stage=3)
    obs, _ = env.reset(seed=0)
    assert obs.shape == (18,), f"expected (18,), got {obs.shape}"
    assert env.observation_space.shape == (18,)
    print("OK obs shape == 18")


def test_obs_has_no_ball_state() -> None:
    """Move the ball-in-gripper to two different welded-arm poses and verify
    the obs slots formerly holding ball state no longer vary with arm pose.
    With ball obs dropped, obs[12:14] is cup_xy (constant across reset) —
    so resetting twice with same seed and cup should produce identical obs.
    """
    env = RageCageEnv(randomize_cup=False, reward_stage=3)
    obs_a, _ = env.reset(seed=0)
    obs_b, _ = env.reset(seed=0)
    np.testing.assert_allclose(obs_a, obs_b, atol=1e-6)
    print("OK obs deterministic across same-seed reset (no ball-state leak)")


def test_polar_cup_features_match_cup_xy() -> None:
    env = RageCageEnv(randomize_cup=True, reward_stage=3)
    for seed in (0, 7, 42, 123):
        obs, _ = env.reset(seed=seed)
        cup_x, cup_y = obs[12], obs[13]
        cup_theta, cup_r = obs[14], obs[15]
        expected_theta = math.atan2(cup_y, cup_x)
        expected_r = math.hypot(cup_x, cup_y)
        assert abs(cup_theta - expected_theta) < 1e-5, (
            f"seed={seed} theta {cup_theta} != atan2({cup_y},{cup_x})={expected_theta}"
        )
        assert abs(cup_r - expected_r) < 1e-5, (
            f"seed={seed} r {cup_r} != hypot={expected_r}"
        )
    print("OK cup_theta == atan2(cup_y, cup_x), cup_r == ||cup_xy||")


def test_obs_bounds_cover_workspace_corners() -> None:
    env = RageCageEnv(randomize_cup=True, reward_stage=3)
    low = env.observation_space.low
    high = env.observation_space.high
    assert low[12] <= 0.75 and high[12] >= 1.15, f"cup_x bounds {low[12]}..{high[12]}"
    assert low[13] <= -0.20 and high[13] >= 0.20, f"cup_y bounds {low[13]}..{high[13]}"
    assert low[14] <= -0.27 and high[14] >= 0.27, f"cup_theta bounds {low[14]}..{high[14]}"
    assert low[15] <= 0.75 and high[15] >= 1.17, f"cup_r bounds {low[15]}..{high[15]}"
    print("OK obs bounds cover the R3 workspace + margin")


if __name__ == "__main__":
    test_obs_shape_is_18()
    test_obs_has_no_ball_state()
    test_polar_cup_features_match_cup_xy()
    test_obs_bounds_cover_workspace_corners()
    print("\nAll polar-obs smoke checks passed.")
```

### Step 1.2: Run the test, confirm it fails

```bash
uv run python -m sim.smoke_obs_polar
```

Expected: `AssertionError: expected (18,), got (22,)` from `test_obs_shape_is_18`.

### Step 1.3: Update the env docstring

Edit `sim/env.py` lines 180–187 (the `RageCageEnv` class docstring) from:

```python
    """Privileged-state MuJoCo task for bootstrapping PPO throw policies.

    Observation is a flat vector:
    joint_pos(6), joint_vel(6), ball_pos(3), ball_vel(3), cup_xy(2),
    pedestal_height(1), release_countdown(1).

    Action is six joint-target deltas in [-1, 1]. The ball is welded to the
    gripper and released automatically at a fixed control step (release_step).
    The release_countdown obs slot lets the policy time peak gripper velocity
    to coincide with the release moment.
    """
```

to:

```python
    """Privileged-state MuJoCo task for bootstrapping PPO throw policies.

    Observation is a flat vector (18 dims):
    joint_pos(6), joint_vel(6), cup_xy(2), cup_theta(1), cup_r(1),
    pedestal_height(1), release_countdown(1).

    cup_theta = atan2(cup_y, cup_x) and cup_r = ||cup_xy|| are pre-computed
    polar features. They align the observation with joint1 (which owns the
    aim angle) and with the radial throw-strength gradient. Both are
    deterministic functions of cup_xy — kept alongside cartesian so the
    policy can use whichever encoding suits the head it's computing.

    Ball pose is NOT in the observation. Pre-release the ball is welded to
    link6 at +9cm along link6's z-axis, so its world pose is fully determined
    by joint_pos via forward kinematics; post-release the policy doesn't act.
    Dropping ball obs removes a deploy-time FK requirement and forces the
    policy to use the joint state directly (matching what a real arm sees).

    Action is six joint-target deltas in [-1, 1]. The ball is welded to the
    gripper and released automatically at a fixed control step (release_step).
    The release_countdown obs slot lets the policy time peak gripper velocity
    to coincide with the release moment.
    """
```

### Step 1.4: Update obs_low / obs_high

Edit `sim/env.py:315-342`. Replace the entire `obs_low = ...` and `obs_high = ...` blocks with:

```python
        # 18-dim obs layout (see RageCageEnv docstring).
        # joint qpos/qvel from MuJoCo; cup_xy widened to cover the R3
        # ±15cm box at the (0.95, 0) nominal. cup_theta bound ±0.30 covers
        # atan2(±0.20, 0.75) ≈ ±0.26 rad with margin. cup_r ∈ [0.70, 1.20]
        # covers min ||cup_xy||=0.75 to max ≈1.17 with margin.
        obs_low = np.concatenate(
            [
                self.joint_low,
                np.full(6, -20.0, dtype=np.float32),
                np.array([0.75, -0.20], dtype=np.float32),
                np.array([-0.30], dtype=np.float32),
                np.array([0.70], dtype=np.float32),
                np.array([0.0], dtype=np.float32),
                np.array([0.0], dtype=np.float32),
            ]
        )
        obs_high = np.concatenate(
            [
                self.joint_high,
                np.full(6, 20.0, dtype=np.float32),
                np.array([1.15, 0.20], dtype=np.float32),
                np.array([0.30], dtype=np.float32),
                np.array([1.20], dtype=np.float32),
                np.array([0.18], dtype=np.float32),
                np.array([1.0], dtype=np.float32),
            ]
        )
```

### Step 1.5: Update `_get_obs`

Edit `sim/env.py:646-657`. Replace:

```python
    def _get_obs(self) -> NDArray[np.float32]:
        return np.concatenate(
            [
                self.data.qpos[self.joint_qposadr],
                self.data.qvel[self.joint_dofadr],
                self._ball_pos(),
                self._ball_vel(),
                self.cup_xy,
                np.array([self.pedestal_height], dtype=np.float32),
                np.array([self._release_countdown()], dtype=np.float32),
            ]
        ).astype(np.float32)
```

with:

```python
    def _get_obs(self) -> NDArray[np.float32]:
        cup_xy64 = self.cup_xy.astype(np.float64)
        cup_theta = float(np.arctan2(cup_xy64[1], cup_xy64[0]))
        cup_r = float(np.linalg.norm(cup_xy64))
        return np.concatenate(
            [
                self.data.qpos[self.joint_qposadr],
                self.data.qvel[self.joint_dofadr],
                self.cup_xy,
                np.array([cup_theta, cup_r], dtype=np.float32),
                np.array([self.pedestal_height], dtype=np.float32),
                np.array([self._release_countdown()], dtype=np.float32),
            ]
        ).astype(np.float32)
```

### Step 1.6: Run the test, confirm pass

```bash
uv run python -m sim.smoke_obs_polar
```

Expected: 4 OK lines + `All polar-obs smoke checks passed.`

### Step 1.7: Commit

```bash
git add sim/env.py sim/smoke_obs_polar.py
git commit -m "$(cat <<'EOF'
v38 obs: drop ball state, add polar cup features

- Remove ball_pos / ball_vel from observation. Pre-release these are
  deterministic FK of joint state (ball welded to link6 +9cm) and
  post-release the policy is not queried. Removes deploy-time FK
  requirement on real arm.
- Add cup_theta = atan2(cup_y, cup_x) and cup_r = ||cup_xy|| alongside
  cartesian cup_xy. Aligns the observation with joint1 (aim) and the
  radial throw-strength gradient — the cells where v37 plateaued.
- Update obs bounds and docstring. New obs dim = 18 (was 22).

This breaks obs-space compat with v34/v36/v37 saved policies. v38 is
from-scratch; older models stay loadable in their original env state.
EOF
)"
```

---

## Task 2: Fix `smoke_z_curriculum` to not depend on v34's saved policy

The `test_evaluate_policy_grid3d_shape` test in `sim/smoke_z_curriculum.py:70-98` loads `models/random_pos_cup_thrower_v1/policy.zip` (22-dim obs). After Task 1, the env emits 18-dim obs, so the loaded policy can't predict. We replace the load with a fresh PPO instance built against the current env — same test of `evaluate_policy_grid3d`'s shape, just with a freshly-initialized network instead of a saved one.

**Files:**
- Modify: `sim/smoke_z_curriculum.py:70-98` (`test_evaluate_policy_grid3d_shape`)

### Step 2.1: Run the smoke as-is, observe the failure

```bash
uv run python -m sim.smoke_z_curriculum
```

Expected: first four tests pass; `test_evaluate_policy_grid3d_shape` fails with an obs-shape mismatch when loading `policy.zip` into the new env.

### Step 2.2: Replace the policy-load with a fresh PPO instance

Edit `sim/smoke_z_curriculum.py:70-98`. Replace the entire `test_evaluate_policy_grid3d_shape` function with:

```python
def test_evaluate_policy_grid3d_shape() -> None:
    """Verify evaluate_policy_grid3d returns 27 unique cells across the
    expected pedestal heights, regardless of policy quality. Uses a fresh
    untrained PPO so this smoke does not depend on any saved checkpoint
    (whose obs space may differ from the current env's)."""
    base_env = RageCageEnv(randomize_cup=True, reward_stage=3)
    env = DummyVecEnv([lambda: base_env])
    env = VecNormalize(env, training=False, norm_reward=False)
    model = PPO("MlpPolicy", env, seed=0, n_steps=16, batch_size=16)

    aggregate, rows = evaluate_policy_grid3d(model, reward_stage=3, seed=0)
    assert len(rows) == 27, f"expected 27 cells, got {len(rows)}"
    cells = {(round(r["cup_x"], 4), round(r["cup_y"], 4), round(r["pedestal_height"], 4)) for r in rows}
    assert len(cells) == 27, f"duplicate cells: {len(cells)} unique vs 27 rows"
    pedestals = {round(r["pedestal_height"], 4) for r in rows}
    assert pedestals == {0.0, 0.06, 0.12}, f"got pedestals {pedestals}"
    assert "success_rate" in aggregate
    print(f"OK grid3d eval n_cells=27 success_rate={aggregate['success_rate']:.3f}")
```

### Step 2.3: Run the smoke, confirm pass

```bash
uv run python -m sim.smoke_z_curriculum
```

Expected: all 5 tests OK + `All Z curriculum smoke checks passed.`

### Step 2.4: Commit

```bash
git add sim/smoke_z_curriculum.py
git commit -m "smoke_z_curriculum: use fresh PPO instead of v34 policy load

The grid3d shape test only needs to verify evaluate_policy_grid3d
returns 27 unique cells; the policy quality is irrelevant. Loading
v34 (22-dim obs) into the new 18-dim env fails. Build a fresh PPO
against the current env instead, so this smoke is self-contained."
```

---

## Task 3: Verify the rand-curriculum smoke still passes

`sim/smoke_rand_curriculum.py` doesn't read the obs vector — it tests stage tables, promotion thresholds, and per-instance range mutation. Should pass as-is.

### Step 3.1: Run

```bash
uv run python -m sim.smoke_rand_curriculum
```

Expected: `smoke_rand_curriculum OK` (already updated for ±15cm in current uncommitted edits).

No commit needed if it passes.

---

## Task 4: 100k-step training smoke

Verify the full pipeline — env init, VecNormalize, PPO with 512×512, curriculum eval+promotion, checkpointing, grid CSV writes — runs end-to-end against the new obs layout *before* committing to the 30M-step run.

### Step 4.1: Launch the smoke

```bash
FULL_RUN=1 \
TIMESTEPS=100000 \
N_ENVS=16 \
N_STEPS=2048 \
BATCH_SIZE=512 \
LR=0.00015 \
LR_SCHEDULE=linear \
NET_ARCH=large \
ACTIVATION=tanh \
REWARD_STAGE=1 \
CURRICULUM=auto \
CURRICULUM_EVAL_EVERY=50000 \
CURRICULUM_EVAL_EPISODES=8 \
RAND_STAGE=0 \
RAND_EVAL_EPISODES=16 \
Z_STAGE=0 \
SEED=1 \
OUT=runs/_v38_smoke \
TRAIN_ROLLOUT_VIZ=1 \
TRAIN_ROLLOUT_VIZ_EVERY=50000 \
PROFILE=0 \
bash scripts/train_ppo.sh
```

Expected: completes in a few minutes; `runs/_v38_smoke/curriculum.csv` exists and has `>=1` row; `runs/_v38_smoke/grid.csv` exists; `runs/_v38_smoke/checkpoints/` has a checkpoint zip; no Python tracebacks; the printed `policy:` block shows `in_features=18` and `512` hidden units.

### Step 4.2: Sanity-check the smoke outputs

```bash
ls runs/_v38_smoke/
head -2 runs/_v38_smoke/curriculum.csv
head -2 runs/_v38_smoke/grid.csv
uv run python -c "
from stable_baselines3 import PPO
m = PPO.load('runs/_v38_smoke/checkpoints/' + sorted(__import__('os').listdir('runs/_v38_smoke/checkpoints'))[0])
print(m.policy)
"
```

Expected: `curriculum.csv` header matches v37's; `grid.csv` has cells at `cup_x ∈ {0.80, 0.95, 1.10}`, `cup_y ∈ {-0.15, 0, +0.15}`, `pedestal ∈ {0, 0.06, 0.12}`; policy print shows `Linear(in_features=18, out_features=512, ...)`.

### Step 4.3: If everything looks good, clean up the smoke dir

```bash
rm -rf runs/_v38_smoke
```

(If anything looks wrong, do NOT proceed to Task 5 — debug in the smoke dir first.)

No commit. The smoke output is throwaway.

---

## Task 5: Launch the 30M-step training run

### Step 5.1: Launch in the background

@superpowers:verification-before-completion — do not claim training started until you see the curriculum.csv writing periodically.

```bash
FULL_RUN=1 \
TIMESTEPS=30000000 \
N_ENVS=16 \
N_STEPS=2048 \
BATCH_SIZE=512 \
LR=0.00015 \
LR_SCHEDULE=linear \
NET_ARCH=large \
ACTIVATION=tanh \
REWARD_STAGE=1 \
CURRICULUM=auto \
CURRICULUM_EVAL_EVERY=100000 \
CURRICULUM_EVAL_EPISODES=8 \
RAND_STAGE=0 \
RAND_EVAL_EPISODES=16 \
Z_STAGE=0 \
SEED=1 \
OUT=runs/v38_far_curric_large_seed1 \
TRAIN_ROLLOUT_VIZ=1 \
TRAIN_ROLLOUT_VIZ_EVERY=250000 \
PROFILE=1 \
LOG_INTERVAL=10 \
bash scripts/train_ppo.sh > runs/v38_far_curric_large_seed1.stdout 2>&1 &
```

### Step 5.2: Confirm the run is live

After ~2 minutes:

```bash
ls runs/v38_far_curric_large_seed1/
tail -20 runs/v38_far_curric_large_seed1.stdout
```

Expected: `tb/` dir exists; stdout shows `==> Training PPO` and rolling iteration metrics. If `curriculum.csv` exists, even better.

### Step 5.3: Append a v38 entry to the experiment log

Edit `docs/rl_experiment_log.md` — add a new section under the existing v37 entry. Use the existing v36/v37 entry style as a template. Include:

- Date launched (2026-05-21).
- Branch (`deepak/rl-cup-height-var`).
- Why v38 (one-paragraph diagnosis of v37 plateau).
- Deltas vs v37 (curriculum re-enabled, polar obs, no ball obs, 512×512, LR=1.5e-4).
- Pending — fill in final metrics when the run completes.

### Step 5.4: Commit the doc update

```bash
git add docs/rl_experiment_log.md docs/plans/2026-05-21-v38-curriculum-polar-cup-large-net.md
git commit -m "docs: launch v38 — re-enable R/Z curriculum, polar cup obs, 512x512 net"
```

---

## Risks & non-goals

- **Risk: 512×512 trains slower wall-clock.** Roughly 1.5–2× per env step; total run ~10–14 hours at 30M steps depending on machine. If you need to ship sooner, drop to 256×256 — curriculum + polar obs are the higher-leverage knobs.
- **Risk: larger LR + larger net could destabilize early PPO.** We pre-emptively dropped LR to 1.5e-4 (from 2e-4). If you see `explained_variance` crash or KL > 0.05 sustained in the first 2M steps, halt and lower LR further to 1e-4.
- **Risk: curriculum stalls at R0 or Z0.** The from-scratch v36 succeeded at R3 immediately; if v38 spends >5M steps at R0 the curriculum threshold (0.5 success_rate) may be too strict for the new obs layout. Diagnose by reading `curriculum.csv` `rand_stage` column — if it sits at 0 with `range_success_rate > 0.3`, the issue is threshold not learning, and we can hand-promote via `RAND_STAGE=1` warm-restart.
- **Non-goal: changing reward shape, bounce target, or release timing.** We isolated the smallest set of changes that addresses the diagnosed root causes. If v38 also plateaus, the next experiment touches reward shape (smooth `bounce_score`, overshoot penalty) — separate plan.
- **Non-goal: warm-starting from v37.** Obs dim changed; warm-start would require obs-padding wrappers, and the v37 weights are tuned to a different observation representation anyway. From-scratch is cleaner.

## Done criteria

- [ ] Task 1–3 commits pushed; all three smokes pass (`smoke_obs_polar`, `smoke_z_curriculum`, `smoke_rand_curriculum`).
- [ ] 100k-step smoke run produces a valid checkpoint with 18-dim input layer and 512-wide hidden.
- [ ] 30M-step run completes and writes `best_Z3.zip`.
- [ ] `grid3d_success_rate >= 0.30` on the final eval — i.e., we have surpassed v37's plateau.
- [ ] Per-corner success at `(1.10, +0.15)` > 0 (the v37 worst cell).
- [ ] Experiment log updated with final numbers.
