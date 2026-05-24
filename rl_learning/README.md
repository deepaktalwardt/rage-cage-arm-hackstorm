# rage-cage-arm

A 48-hour hackathon project: an AgileX PiPER 6-DOF arm that bounces a ping pong ball off a table into a stack of cups, with an adjustable "drunkenness" parameter that makes the throw wobbly.

Full project spec, design rationale, and 48-hour timeline: [`docs/rage_cage_vla_primer.md`](docs/rage_cage_vla_primer.md).

Current PPO reward/constraint design notes: [`docs/ppo_reward_design.md`](docs/ppo_reward_design.md).

## Status

The project runs in two phases (full scope, timeline, and rationale in the [primer](docs/rage_cage_vla_primer.md)):

- **Phase 1 — Simulation** (this Mac): MuJoCo scene → PPO thrower in sim → demo collection → OpenVLA LoRA fine-tune → closed-loop sim test.
- **Phase 2 — Real hardware** (Linux box): ship the fine-tuned VLA to the physical PiPER over ROS2, collect ~20 real demos, fine-tune again, run the live demo.

This repo is currently in **Phase 1**. Sub-stage progress:

- [x] MuJoCo scene: PiPER + table + 1 cup + ping pong ball (single-cup MVP; multi-cup nested stack later)
- [x] Bouncy ball physics, hollow cup with composite-primitive collision
- [x] Offscreen rendering verified on Apple Silicon
- [ ] Custom render camera for RL observations
- [x] Gym env wrapper with privileged state observations
- [ ] PPO thrower policy training
- [ ] Demo collection → OpenVLA LoRA fine-tune
- [ ] Closed-loop sim test

## Prerequisites

- [`uv`](https://docs.astral.sh/uv/) for Python env + dependency management
- Python 3.11 (managed automatically by `uv`)
- macOS (Apple Silicon tested) or Linux

## Install

```bash
git clone git@github.com:deepaktalwardt/rage-cage-arm.git
cd rage-cage-arm
uv sync
```

`uv sync` creates a `.venv/` in the repo root and installs everything pinned in `uv.lock`. To add a new dependency later: `uv add <package>`.

## Get started with Sim

### Smoke test (loads scene, steps physics, renders one offscreen frame)

```bash
uv run python sim/smoke_test.py
```

Prints model dimensions and ball trajectory, writes `sim/_smoke_out/home_pose.png`. If this passes, the sim stack is good.

### Interactive MuJoCo viewer

```bash
uv run python -m mujoco.viewer --mjcf="$(pwd)/sim/mjcf/rage_cage.xml"
```

Controls worth knowing:

| Action | How |
|---|---|
| Rotate camera | Left-click + drag |
| Pan camera | Right-click + drag |
| Zoom | Scroll |
| Focus on object | Double-click it |
| Apply force to body (e.g., shove the ball) | **Ctrl + right-click + drag** |
| Rotate selected body | **Ctrl + left-click + drag** |
| Pause / resume sim | Space |
| Reset to initial state | Backspace |
| Hide collision-only geoms (green wall slats) | Rendering tab → toggle **Group 3** |

Things to try:
1. In the **Simulation** panel, set **Key** to `1`, click **Load key** to load `rage_home`, then press space → ball falls and bounces on the table.
2. Ctrl-right-drag the ball into the cup → should drop in cleanly through the open top.
3. Drag arm joint sliders (Joints panel) to confirm the arm reaches the cup.

Note: keyframe IDs are numeric in the MuJoCo viewer. Because the included PiPER arm XML contributes key `0` named `home`, the full-scene `rage_home` key is `1`.

### Gym environment check

```bash
uv run python sim/env.py
```

Runs Gymnasium's environment checker against the privileged-state `RageCageEnv`.
The env preloads the ball at the gripper, holds it briefly, then releases it as
a free MuJoCo body with inherited gripper velocity.
The first curriculum cup target is about 60cm in front of the preloaded ball,
with a small randomized reset range around that pose.

### PPO Training

Use the workflow script for normal local runs:

```bash
./scripts/train_ppo.sh
```

This runs dependency sync, the MuJoCo smoke test, the Gymnasium env check, PPO
training, and eval GIF rendering. By default this is a minimized smoke run. It
checks the full pipeline quickly, but it is not expected to learn a throw.

For a conservative longer run on a laptop:

```bash
FULL_RUN=1 TIMESTEPS=5000000 N_ENVS=2 N_STEPS=2048 BATCH_SIZE=64 LR=0.0002 LR_SCHEDULE=linear NET_ARCH=medium EPISODES=1 OUT=sim/_rl_out/ppo_5m_gripfix_medium_2env2048_lr2e4_linear ./scripts/train_ppo.sh
```

Full mode runs a short PPO profile before training, prints expected training
time from measured rollout/update speed, then prints actual training, eval, and
workflow durations at the end. Disable the pre-training profile with `PROFILE=0`
if you want to skip that overhead. The estimate is machine-specific; each
collaborator should profile on their own laptop before starting a long run.

Useful knobs:

- `FULL_RUN=1` switches defaults from smoke settings to a larger training run.
- `TIMESTEPS` controls training experience. More timesteps usually improve
  learning but increase runtime roughly linearly.
- `N_ENVS` controls parallel MuJoCo envs. More envs collect rollout data faster
  on multi-core machines, at higher CPU/memory cost.
- `N_STEPS` controls rollout length before each PPO update. Larger values make
  updates more stable but delay each update.
- `BATCH_SIZE` controls PPO optimizer minibatches. It should usually divide
  `N_ENVS * N_STEPS`.
- `LR` controls PPO's optimizer learning rate. Default is `0.0003`.
- `LR_SCHEDULE=linear` decays LR to zero over training; default is `constant`.
- `NET_ARCH` controls the policy/value MLP size: `default`, `medium`, `large`,
  or `deep`. Start with `default`; try `medium` if reward plateaus.
- `ACTIVATION` controls MLP activation: `tanh` or `relu`. PPO defaults are tanh;
  relu can help larger networks but is a separate experiment.
- `REWARD_STAGE` selects the starting reward curriculum stage. Stage 1
  emphasizes any valid table bounce, stage 2 adds bounce-location shaping,
  stage 3 emphasizes post-bounce cup approach, and stage 4 emphasizes cup
  entry/success. The workflow starts at stage 1 by default.
- `CURRICULUM=auto` periodically runs deterministic fixed-cup eval rollouts and
  promotes stages when the policy meets the next-stage criterion. Full runs
  default to `auto`; smoke runs default to `manual`.
- `CURRICULUM_EVAL_EVERY` controls how often promotion checks run, in env
  timesteps. `CURRICULUM_EVAL_EPISODES` controls how many eval rollouts each
  check uses.
- `EPISODES` controls how many post-training eval GIFs are rendered.
- `FIXED_CUP=0` evaluates randomized cup positions instead of the nominal cup.
- `PROFILE=1` runs rollout/update timing before training. It defaults on for
  `FULL_RUN=1` and off for smoke mode.
- `PROFILE_ROLLOUTS` controls how many PPO iterations are timed for the estimate.
- `TRAIN_ROLLOUT_VIZ=1` saves visual training rollout snapshots during training.
  It defaults on for `FULL_RUN=1` and off for smoke mode.
- `TRAIN_ROLLOUT_VIZ_EVERY` controls the snapshot interval in environment
  timesteps. Snapshots are saved at PPO rollout boundaries, so they may land up
  to `N_ENVS * N_STEPS` steps after the target interval.
- `TRAIN_ROLLOUT_VIZ_DIR` receives the training rollout GIFs and reward CSVs.
- `TRAIN_ROLLOUT_VIZ_STEPS` caps policy steps in each visualized rollout.
  Post-release passive frames may extend the GIF within the env episode limit.

Direct training command, useful for one-off experiments:

```bash
uv run python -m sim.train_rl --timesteps 500000 --n-envs 2 --n-steps 1024 --batch-size 64 --lr 0.0003 --lr-schedule linear --net-arch default --out sim/_rl_out/exp_default_500k
```

Training writes:

- `OUT.zip` — saved PPO policy
- `OUT.vecnormalize.pkl` — observation/reward normalization stats
- `OUT.training.json` — training metadata, including final reward stage
- `OUT.curriculum.csv` — auto-curriculum eval metrics and promotions
- `sim/_rl_out/tb/` — TensorBoard logs

Automatic curriculum promotion rules:

- Stage 1 → 2: fixed-cup clean first-table-bounce rate reaches 75%.
- Stage 2 → 3: clean first-table-bounce rate reaches 70% and bounce-target hit
  rate reaches 60%.
- Stage 3 → 4: clean first-table-bounce rate reaches 60% and median second
  table-bounce impact distance to the cup is at most 25cm.

The final eval run uses the final reward stage recorded in `OUT.training.json`.
The curriculum CSV also records `exact_one_bounce_rate`; final success requires
exactly one clean table bounce before the ball settles in the cup. Bounces
inside the cup should be cup contacts, not table bounces. Early stages can
promote after learning the first clean table bounce even if a missed ball later
bounces again.
When auto-curriculum promotes stages and training rollout visualization is
enabled, it also writes `reward_stage_<stage>_end_<timestep>.gif/csv`
snapshots to the training rollout directory.

### Observe PPO Rollouts

There are two rollout visualizations:

- **Training rollout snapshots**: one-env deterministic rollouts rendered
  periodically while training is still running. Use these to see whether the
  current policy is improving between PPO updates.
- **Eval rollouts**: rollouts rendered after training from a saved model. Use
  these to inspect the final policy artifact and compare experiments.

Full workflow runs enable training rollout snapshots by default:

```bash
FULL_RUN=1 TIMESTEPS=5000000 N_ENVS=2 N_STEPS=2048 BATCH_SIZE=64 LR=0.0002 LR_SCHEDULE=linear NET_ARCH=medium REWARD_STAGE=1 CURRICULUM=auto TRAIN_ROLLOUT_VIZ=1 TRAIN_ROLLOUT_VIZ_EVERY=250000 TRAIN_ROLLOUT_VIZ_STEPS=150 EPISODES=1 OUT=sim/_rl_out/ppo_5m_auto_stage_medium_2env2048_lr2e4_linear ./scripts/train_ppo.sh
```

Training rollout files are written to:

```text
${OUT}_train_rollouts/train_rollout_<timestep>.gif
${OUT}_train_rollouts/train_rollout_<timestep>.csv
${OUT}_train_rollouts/reward_stage_<stage>_end_<timestep>.gif
${OUT}_train_rollouts/reward_stage_<stage>_end_<timestep>.csv
${OUT}_train_rollouts/train_rollout_final_<timestep>.gif
${OUT}_train_rollouts/train_rollout_final_<timestep>.csv
```

Each GIF overlays the step reward, cumulative reward, bounce count, cup
distance, and success flag. After release, the policy stops issuing actions,
the arm keeps the last commanded target and gradually settles, and the env
simulates passive ball physics internally; the GIF still includes those passive
flight frames. Each CSV contains policy-step rows plus `phase=passive` rows for
post-release frames, contact flags, closest post-bounce cup distance, motion
limit metrics, and reward component breakdowns. The CSV also records
`second_table_bounce_cup_dist`, the exact table impact distance to the cup if a
missed ball bounces a second time. Motion-limit penalties apply to
policy-controlled pre-release steps, not to post-release controller settling.

For a direct `sim.train_rl` experiment, pass the visualization args explicitly:

```bash
uv run python -m sim.train_rl --timesteps 500000 --n-envs 2 --n-steps 1024 --batch-size 64 --lr 0.0003 --lr-schedule linear --out sim/_rl_out/exp_default_500k --train-rollout-viz-dir sim/_rl_out/exp_default_500k_train_rollouts --train-rollout-viz-every 100000 --train-rollout-viz-steps 150 --train-rollout-viz-fixed-cup
```

The periodic snapshot interval is checked at PPO rollout boundaries. For
example, with `N_ENVS=2 N_STEPS=2048`, PPO collects `4096` transitions before
each update, so a `TRAIN_ROLLOUT_VIZ_EVERY=100000` snapshot may be saved shortly
after the exact 100k target.

To render eval rollouts from a saved model:

```bash
uv run python -m sim.eval_rl --model sim/_rl_out/exp_default_500k.zip --vecnormalize sim/_rl_out/exp_default_500k.vecnormalize.pkl --episodes 1 --fixed-cup
```

Eval GIFs are written to `sim/_rl_eval/episode_*.gif`. The eval command also
prints episode reward, success, bounce count, and final cup distance. Pass
`--out-dir <path>` to keep eval GIFs from different experiments separate.

Current success definition requires exactly one clean table bounce before the
ball settles in the cup. Bounces inside the cup should be cup contacts, not
table bounces. More than one table bounce, ball contact with the floor or robot,
or robot contact with the table/cup terminates the episode with a penalty.
Cup-distance shaping is only positive after a valid table bounce.

### PPO Experiment Logs

Keep machine-specific results, training notes, and personal profiling numbers
outside the shared workflow instructions. Xinyi's current notes live in
[`docs/xinyi_experiment_log.md`](docs/xinyi_experiment_log.md).

### PPO Profiling

```bash
uv run python -m sim.profile_train --n-envs 2 --n-steps 1024 --rollouts 3 --estimate-timesteps 5000000
```

Prints rollout collection time, network update time, measured steps/second, and
an estimated wall-clock time for the requested training length. Start with
`N_ENVS=2 N_STEPS=1024` for laptop-friendly profiling. If memory and CPU headroom
look comfortable, compare against `N_ENVS=4 N_STEPS=1024` on the same machine.

### Regenerate the cup mesh (only if you change cup geometry)

```bash
uv run python sim/cup_model.py --count 1 --rim-radius 0.047 --base-radius 0.030
```

Outputs `sim/mjcf/cups/cup.stl`. Run with `--help` for all knobs.

## Project layout

```
docs/
  rage_cage_vla_primer.md    Full project spec — read this first
sim/
  cup_model.py               Parametric cup STL generator (trimesh)
  env.py                     Gymnasium MuJoCo wrapper for SB3 experiments
  train_rl.py                PPO training entry point
  eval_rl.py                 PPO evaluation + rollout GIF renderer
  profile_train.py           PPO rollout/update timing profiler
  smoke_test.py              MuJoCo load + step + offscreen render check
  mjcf/
    rage_cage.xml            Top-level scene (PiPER + table + cup + ball)
    agilex_piper/            PiPER MJCF + meshes (from MuJoCo Menagerie)
    cups/cup.stl             Generated cup visual mesh
scripts/
  train_ppo.sh               One-command sync/check/train/eval workflow
pyproject.toml               Project + dependency pins
uv.lock                      Fully-resolved lockfile (commit this)
```

## Attribution

The AgileX PiPER MJCF in `sim/mjcf/agilex_piper/` is vendored from [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) (MIT, contributed by [Omar Rayyan](https://orayyan.com/)). The original `LICENSE` and `README.md` are preserved in that subdirectory.
