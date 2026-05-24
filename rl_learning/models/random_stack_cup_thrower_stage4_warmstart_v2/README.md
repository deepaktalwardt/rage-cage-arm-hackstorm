# random_stack_cup_thrower_stage4_warmstart_v2

PPO policy for the stacked-cup Rage Cage task, fine-tuned from
`models/random_stack_cup_thrower_v1` with reward stage 4.

This checkpoint keeps the stage-3 bounce, bounce-location, and cup-distance
weights, uses a stage-4 success bonus of 120, and adds an extra stage-4
cup-entry depth reward. Once the ball is inside the cup XY/Z volume, the policy
receives incremental reward for reaching a lower ball `z` inside the cup. The
goal is to prefer throws that enter deeper rather than graze the rim or inner
wall and bounce out.

## Files

- `policy.zip` - SB3 PPO model.
- `vecnormalize.pkl` - observation/reward normalization stats. Required for
  inference.

## Source Run

Copied from:

```text
sim/_rl_out/stack_stage4_depth_finetune_success_120_5m/
```

Training metadata:

```text
initial_reward_stage = 4
final_reward_stage = 4
curriculum = manual
timesteps = 5,013,504
```

Fine-tune config:

```bash
FULL_RUN=1
TIMESTEPS=5000000
N_ENVS=16
N_STEPS=2048
BATCH_SIZE=512
LR=0.00005
LR_SCHEDULE=linear
NET_ARCH=medium
REWARD_STAGE=4
CURRICULUM=manual
RAND_STAGE=3
Z_STAGE=3
WARM_START_POLICY=models/random_stack_cup_thrower_v1/policy.zip
WARM_START_VECNORMALIZE=models/random_stack_cup_thrower_v1/vecnormalize.pkl
ENT_COEF=0.005
```

Stage-4 reward weights at training time:

```python
{
    "bounce": 5.0,
    "bounce_xy": 2.0,
    "cup_dist": 10.0,
    "second_bounce_cup": 0.0,
    "cup_entry": 20.0,
    "success": 120.0,
}
```

## Evaluation

Compared against `models/random_stack_cup_thrower_v1` using eval seed 0.

### 3x3x3 Fixed Grid

The 3x3x3 grid tests cup XY offsets `{-10cm, 0, +10cm}` and pedestal heights
`{0, 7.5cm, 15cm}`.

| model | fixed success | grid3d success | z=0 | z=7.5cm | z=15cm | median closest |
|---|---:|---:|---:|---:|---:|---:|
| `random_stack_cup_thrower_v1` | 1.000 | 0.778 | 9/9 | 9/9 | 3/9 | 0.018m |
| `random_stack_cup_thrower_stage4_warmstart_v2` | 1.000 | 0.778 | 9/9 | 9/9 | 3/9 | 0.019m |

On this coarse fixed grid, the fine-tuned model is neutral: no regression, but
no grid-cell success improvement.

### Random R3 x Z3 Range

Random range eval samples cup XY uniformly within +-10cm of nominal `(0.85, 0)`
and pedestal height uniformly in `[0, 0.15m]`.

| model | episodes | success | exact one bounce | valid bounce | median closest |
|---|---:|---:|---:|---:|---:|
| `random_stack_cup_thrower_v1` | 300 | 0.807 | 0.807 | 1.000 | 0.018m |
| `random_stack_cup_thrower_stage4_warmstart_v2` | 300 | 0.843 | 0.843 | 0.997 | 0.017m |

The fine-tuned model shows a modest random-distribution improvement, but the
hard fixed high-pedestal grid cells remain unsolved.

## Known Failure Mode

Remaining fixed-grid failures are concentrated at high pedestal height. Visual
inspection suggests the ball often contacts the cup from the outside or near
the wall/rim and bounces back out. A useful next training step is likely
targeted hard-case sampling or a reward/diagnostic that activates before cup
entry for outside-wall or rim-graze approaches.

## Loading

Live MuJoCo viewer:

```bash
uv run mjpython -m sim.play_policy \
  --model models/random_stack_cup_thrower_stage4_warmstart_v2 \
  --reward-stage 4 \
  --rand-stage 3 \
  --rand-stage-z 3 \
  --seed 0 \
  --episodes 20 \
  --speed 0.5
```

Fixed 3x3x3 grid playback:

```bash
uv run mjpython -m sim.play_policy \
  --model models/random_stack_cup_thrower_stage4_warmstart_v2 \
  --reward-stage 4 \
  --cup-grid \
  --pedestal-grid \
  --episodes 27 \
  --speed 0.5
```

Legacy 3x3 flat-grid eval:

```bash
uv run python -m sim.eval_grid \
  --run-dir models/random_stack_cup_thrower_stage4_warmstart_v2 \
  --reward-stage 4 \
  --also-fixed
```
