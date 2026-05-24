# PPO Reward and Constraint Design

This document defines the next training-environment changes for the state-based
PPO thrower. The goal is to make the learning problem match the real task:
the arm chooses a throw motion, releases the ball, then the ball passively
bounces on the table and may land in the cup.

## Implementation Status

The current implementation follows this design in `sim.env.RageCageEnv`:

- Policy actions are consumed only before release. On the release step the env
  opens the gripper, keeps the last commanded arm target, runs passive
  post-release physics internally, accumulates the delayed reward, and returns
  `done=True` on success, invalid contact, more than one table bounce, or
  timeout.
- Training/eval rollout GIFs still show passive ball-flight frames. The
  underlying PPO rollout buffer only receives the pre-release policy actions.
- Reward stage can be fixed manually with `reward_stage` / `REWARD_STAGE`, or
  promoted automatically by the PPO training callback when `CURRICULUM=auto`.
- Training rollout CSVs log contact flags, table/invalid bounce counts, closest
  post-bounce cup distance, motion limit metrics, and reward components.
- Velocity/acceleration/jerk penalties and hard motion-limit termination apply
  only to policy-controlled pre-release steps. The passive settling phase still
  logs motion metrics and still terminates on unsafe contacts, but it does not
  punish the policy for the arm gradually decelerating after release.

## Current Problem

The current environment has three issues:

- The policy can keep moving the arm after ball release. Those actions cannot
  meaningfully affect a free ball trajectory, but PPO will still assign gradient
  credit to them.
- The reward mostly measures ball-to-cup distance each step, so the policy can
  get shaped reward without learning the required sequence: release, table
  bounce, approach cup, enter cup.
- Safety constraints are weak. The policy can learn motions that would be
  unrealistic or unsafe for the real arm, such as hitting the table/cup or
  making large acceleration/jerk changes.

## Release-Time Credit Assignment

After release, the robot arm should stop receiving new policy actions. This
should be hard-coded in the environment. It should not be interpreted as an
instantaneous physical stop, because an instant stop is itself an acceleration
and jerk event.

Important PPO detail: simply ignoring actions after release is not enough.
Stable Baselines 3 will still store those post-release actions in the rollout
buffer and update the policy from their log-probs. That creates bad credit
assignment: the network is trained on actions that the environment ignored.

Recommended design:

1. Before release, each env step consumes one policy action and moves the arm.
2. At `release_step`, deactivate `ball_grip`, open the gripper, and keep the
   last commanded arm target. Do not snap the target to the current joint pose.
   The position actuators should decelerate the arm gradually.
3. After release, run the remaining passive ball physics internally without
   requesting more policy actions.
4. Accumulate post-release rewards during that internal simulation.
5. Return the accumulated delayed reward on the release step, with `done=True`
   once the passive rollout reaches success, too many bounces, or timeout.

This preserves the long physics rollout needed to calculate reward, but avoids
training the policy on meaningless post-release actions. PPO then correctly
credits the throw actions that happened before release. Motion penalties and
hard motion-limit failures should be computed on the pre-release controlled
steps; the passive settling phase can still be logged, but should not create a
jerk-limit failure just because the arm is naturally decelerating after release.

Alternative design:

- Keep post-release steps in the Gym episode but use a custom rollout buffer or
  loss mask to remove post-release action log-probs from policy-gradient loss.

This is more invasive and not worth doing first.

## Arm Motion Constraints

The action is currently a 6D normalized joint target delta. We should keep that,
but add physical guardrails.

### Hard Constraints

These should terminate the episode with a penalty:

- Arm or gripper contacts the table.
- Arm or gripper contacts the cup.
- Joint position exceeds the model joint range.
- Joint velocity exceeds a configured velocity limit.
- Joint acceleration exceeds a configured acceleration limit.
- Joint jerk exceeds a configured jerk limit.

The table/cup contacts should be detected through MuJoCo contacts. Ball contacts
with table/cup are allowed; robot body contacts are not.

### Soft Penalties

These should be part of the reward each pre-release control step:

- Joint velocity penalty: discourages fast motions.
- Joint acceleration penalty: discourages abrupt velocity changes.
- Joint jerk penalty: discourages twitchy commands.
- Action delta penalty: discourages large target jumps.
- Optional joint-limit margin penalty: discourages operating close to joint
  limits even when technically valid.

Implementation state needed:

- Previous joint velocity.
- Previous joint acceleration.
- Previous action or previous arm target.
- A list of robot body/geom ids that count as unsafe if they contact the table
  or cup.

## Bounce Semantics

Only table bounces should count toward the task. A bounce on the floor, cup,
gripper, arm, or any other object should not count as a valid table bounce.

Recommended state:

- `table_bounce_count`
- `first_table_bounce_xy`
- `first_table_bounce_time`
- `invalid_bounce_count`
- `ball_contacted_table`
- `ball_contacted_cup`
- `ball_contacted_floor`
- `ball_contacted_robot`

Valid bounce detection:

- Ball is released.
- Ball contacts the `table` geom.
- Ball vertical velocity is downward or near-impact immediately before contact.
- It is the first transition from not touching table to touching table.

Invalid bounce detection:

- Ball contacts floor before a valid table bounce.
- Ball contacts robot after release.
- Ball contacts cup before a valid table bounce.
- More than one table bounce occurs when the task requires exactly one.

## Reward Design

The reward should reflect task stages. The policy should first learn to produce
a valid table bounce, then learn where that bounce should happen, then learn to
approach the cup after the bounce, and finally learn to land in the cup.

### Terms

Recommended reward components:

- `time_penalty`: small negative cost per pre-release control step.
- `motion_penalty`: velocity/acceleration/jerk/action penalties before release.
- `safety_penalty`: large penalty and termination for robot-table or robot-cup
  contact.
- `valid_table_bounce_bonus`: one-time bonus for the first table bounce.
- `invalid_contact_penalty`: penalty for ball contact with invalid surfaces.
- `extra_bounce_penalty`: penalty and termination for more than one table bounce.
- `bounce_location_reward`: reward based on first table bounce point relative to
  a target bounce region.
- `post_bounce_cup_approach_reward`: reward based on closest ball-to-cup
  distance after a valid table bounce.
- `cup_entry_bonus`: large bonus when the ball enters the cup after exactly one
  valid table bounce.
- `settled_in_cup_bonus`: final success bonus when the ball is inside the cup
  and slow enough.

### Table-Gated Cup Distance

Distance to the cup should only produce positive reward after the ball has made
a valid table bounce. Before that, cup distance should not help the policy.

Reason:

- Without table gating, the policy may learn to carry, drop, or fling the ball
  near the cup without solving the bounce requirement.
- The task is specifically "bounce on table, then cup." Reward should encode
  that sequence.

Recommended rule:

```text
if table_bounce_count == 0:
    cup_distance_reward = 0
else:
    cup_distance_reward = positive function of closest post-bounce cup distance
```

The reward should use closest post-bounce distance, not just final distance,
because early training may miss the cup but still pass near it.

For miss trajectories that bounce a second time on the table, the cup-approach
metric should also record the exact second table impact location:

```text
second_table_bounce_cup_dist =
    distance(second_table_bounce_xy, cup_xy)
```

That second-impact distance is a better curriculum signal than an arbitrary
closest in-flight point when the policy is learning to bounce near the cup.
Final task success requires exactly one clean table bounce before the ball
settles in the cup. Bounces inside the cup should be cup contacts, not table
bounces. More than one table bounce remains a failure condition.

## Curriculum / Weighted Reward Schedule

A staged weighted reward makes sense here. The task is sparse and sequential;
asking PPO to discover "exactly one table bounce then cup entry" from the start
is too hard.

The current training script supports an automatic curriculum. The env exposes
`set_reward_stage(stage)`, and `sim.train_rl` periodically evaluates the current
policy on deterministic fixed-cup rollouts. If the policy crosses a promotion
threshold, the callback updates every training env to the next stage. The PPO
network is not reset; training continues from the same policy with new reward
weights.

Current promotion checks:

- Stage 1 -> 2: clean first-table-bounce rate is at least 75%.
- Stage 2 -> 3: clean first-table-bounce rate is at least 70% and
  bounce-target hit rate is at least 60%.
- Stage 3 -> 4: clean first-table-bounce rate is at least 60% and median second
  table-bounce impact distance to the cup is at most 25cm.

Promotion metrics are written to `OUT.curriculum.csv`. Training metadata,
including the final reward stage, is written to `OUT.training.json`.
The CSV also tracks `exact_one_bounce_rate`. Final task success still requires
the ball to settle in the cup after exactly one clean table bounce, but early
curriculum stages should not block promotion merely because a miss later
bounces a second time.
When training rollout visualization is enabled, the auto-curriculum callback
also writes a `reward_stage_<stage>_end_<timestep>.gif/csv` snapshot immediately
before each promotion.

Recommended curriculum:

### Stage 1: Learn Valid Table Bounce

Goal:

- Release the ball.
- Hit the table exactly once.
- Avoid robot/table/cup collisions.

Weights:

- High `valid_table_bounce_bonus`.
- High safety penalties.
- Low or zero cup-distance reward.
- No cup-entry requirement.

Promotion criterion:

- Valid table bounce rate above a threshold, e.g. 70-80% over fixed-cup eval.

### Stage 2: Learn Bounce Point

Goal:

- Bounce at a useful table region between gripper and cup.

Weights:

- Keep valid-bounce bonus.
- Add `bounce_location_reward`.
- Keep cup-distance reward low.

Bounce target:

- Start with a broad target zone.
- Narrow it as the policy improves.

Promotion criterion:

- Valid bounce rate stays high.
- First bounce point is consistently within the target region.

### Stage 3: Learn Post-Bounce Cup Approach

Goal:

- After one table bounce, minimize closest distance to the cup.

Weights:

- Keep valid-bounce bonus.
- Keep bounce-location reward.
- Increase post-bounce closest-distance reward.
- Penalize extra bounces and invalid contacts strongly.

Promotion criterion:

- Median post-bounce closest cup distance below a threshold.

### Stage 4: Learn Cup Entry and Settling

Goal:

- Exactly one table bounce, then enter and settle in the cup.

Weights:

- Highest weight on cup entry and settled-in-cup success.
- Keep safety penalties.
- Reduce shaping weights enough that the sparse success objective dominates.

Promotion criterion:

- Fixed-cup success rate above target, then begin randomized cup positions.

## Proposed Reward Formula

At a high level:

```text
reward =
    time_penalty
  + motion_penalty
  + safety_penalty
  + w_bounce * valid_table_bounce_bonus
  + w_bounce_xy * bounce_location_reward
  + w_cup_dist * post_bounce_closest_cup_reward
  + w_entry * cup_entry_bonus
  + w_success * settled_success_bonus
  + invalid_contact_penalties
```

The curriculum changes the weights:

```text
Stage 1: high w_bounce, low w_cup_dist, zero/low w_entry
Stage 2: high w_bounce + w_bounce_xy
Stage 3: high w_cup_dist
Stage 4: high w_entry + w_success
```

## Success Definition

Final success should require:

- Ball released.
- Exactly one valid table bounce.
- No invalid post-release contacts.
- Ball enters the cup volume from above.
- Ball is inside cup bounds.
- Ball speed is below a settling threshold, or the ball remains in the cup for
  a short dwell window.

## Metrics to Log

Training/eval logs should include:

- `success`
- `table_bounce_count`
- `invalid_bounce_count`
- `first_table_bounce_xy`
- `first_table_bounce_dist_to_target`
- `closest_post_bounce_cup_dist`
- `final_cup_dist`
- `ball_entered_cup`
- `settled_in_cup`
- `robot_table_contact`
- `robot_cup_contact`
- `max_joint_vel`
- `max_joint_acc`
- `max_joint_jerk`
- Reward component breakdown.

These should be written into training rollout CSVs so visual rollouts can be
debugged without replaying the simulation.

## Implementation Order

1. Add contact classification helpers for table, cup, floor, ball, and robot
   bodies/geoms.
2. Add table-only bounce tracking and invalid-contact tracking.
3. Freeze the arm at release and run passive post-release physics internally so
   PPO does not train on ignored post-release actions.
4. Add motion-limit tracking for velocity, acceleration, jerk, and action deltas.
5. Replace the reward with componentized reward terms and log each component.
6. Add curriculum weight config to the environment constructor and training
   script.
7. Update rollout GIF/CSV output to include reward components and key event
   flags.

## Open Questions

- Exact velocity, acceleration, and jerk limits should be chosen from PiPER
  specs or conservative empirical values from the MuJoCo model.
- The best bounce target region may need to be learned empirically from
  successful trajectories. Start broad.
- The automatic promotion thresholds are initial guesses and should be tuned
  after a few longer runs.
