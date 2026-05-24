# Dance mode: design

A new service `/dance_throw` for the `rage_cage_thrower` node that turns a
throw into a small bit of showmanship: home the arm, sweep joint1
side-to-side for three full sine cycles, land at one of four preset base
angles chosen at random, replay the recorded throw with `joint1_override`
set to that angle, and finally return home. Four cups are placed at the
matching angles (-6°, 0°, +8°, +15° — all validated on the real arm) so
judges can guess where the ball is headed while the arm "dances".

Replay only — the trained policy is not used. We rely on the existing
recording at `replay_path` plus the `joint1_override` retargeting
already plumbed through `start_replay`.

## Scope

In:

- One new service `/dance_throw` (std_srvs/Trigger) on `RageCageThrower`.
- One new state `ThrowState.DANCE` and mode `"dance"` in
  `ThrowController` for the sweep itself.
- One new controller entry point `start_dance(...)`.
- Two new ROS params (`dance_target_angles_deg`, `dance_sweep_period_s`).
- Verbose `info`-level logging of every random pick and phase transition.

Out:

- No changes to `/throw_trigger`, `/home_arm`, `/replay_trajectory` or
  the existing replay state machine.
- No new perception (no `/cup_pose` requirement — dance uses manual
  `replay_path`, not lookup mode).
- No new trajectory format. The same recording is reused for all five
  angles via `joint1_override`.

## End-to-end flow

The service handler chains three independent controller cycles. Each
cycle ends in IDLE and flushes its own throw log, so a dance throw
produces three CSVs.

```
/dance_throw
  ├── preflight + random target pick + log plan
  │
  ├── sub-cycle A  "dance"
  │     HOMING (≤2s) → SETTLE_HOME (0.3s) → DANCE (3 × period_s) → IDLE
  │
  ├── sub-cycle B  "replay" (existing start_replay, joint1_override=target)
  │     HOMING (≤2s) → SETTLE_HOME (0.3s) → REPLAY (0.9s) → SETTLE_RELEASE (1s) → IDLE
  │
  └── sub-cycle C  "return_home" (existing start_home_only)
        HOMING (≤2s) → SETTLE_HOME (0.3s) → IDLE
```

Worst-case wall-clock at default period (2.0s → 6.0s dance): ~13s.
Per-cycle timeouts: A = 5 + duration_s, B = 10s, C = 10s.

## Controller changes (`real/controller.py`)

### New state

`ThrowState.DANCE`.

### New mode

`"dance"`. Drives `HOMING → SETTLE_HOME → DANCE → IDLE`. The
`SETTLE_HOME → DANCE` transition is gated on `_mode == "dance"`; the
existing modes are unaffected.

### New API

```python
def start_dance(
    self,
    current_joint_pos: NDArray[np.float32],
    target_rad: float,
    duration_s: float,
    sweep_period_s: float,
    amplitude_rad: float,
) -> None
```

Sets `state = HOMING`, `mode = "dance"`, `_home_target = HOME_QPOS`
(so the sweep starts from a known pose with joint1 = 0), and stashes
`_dance_target_rad`, `_dance_total_ticks = int(duration_s / CONTROL_DT)`,
`_dance_period_ticks = sweep_period_s / CONTROL_DT`, `_dance_amplitude_rad`.

### DANCE tick handler

```
t_s = tick_idx * CONTROL_DT
progress = tick_idx / total_ticks            # 0 → 1

# 60% full amplitude, 40% cos² ease-out
if progress < 0.6:
    damp = 1.0
else:
    damp = cos(0.5 * π * (progress - 0.6) / 0.4) ** 2

joint1 = damp * amplitude_rad * sin(2π * t_s / period_s)
       + (1 - damp) * target_rad

arm_target = [joint1, HOME_QPOS[1], ..., HOME_QPOS[5]]
gripper    = GRIPPER_HOLD_M
```

Transition: `tick_idx > total_ticks` → `IDLE`.

At handoff, joint1 = `target_rad` with zero velocity (cos² damping
lands flat). Joints 2-6 = HOME values. The next sub-cycle's HOMING
interps joints 2-6 from HOME to `trajectory[0]`; `joint1_override`
pins joint1 at `target_rad` for the entire replay cycle.

## Node changes (`real/rage_cage_thrower.py`)

### New constants (top of file)

```python
DANCE_NUM_SWEEPS = 3                  # full sine cycles per dance
DANCE_SWEEP_AMPLITUDE_DEG = 15.0
DANCE_TARGET_ANGLES_DEG_DEFAULT = [-6.0, 0.0, 8.0, 15.0]
DANCE_SWEEP_PERIOD_S_DEFAULT = 2.0    # → 6.0s dance by default
```

### New ROS params

| name | type | default | notes |
|---|---|---|---|
| `dance_target_angles_deg` | `DOUBLE_ARRAY` | `[-6, 0, 8, 15]` | Explicit `ParameterDescriptor(type=PARAMETER_DOUBLE_ARRAY)` per the same gotcha that bit `replay_freeze_joints`. Four angles validated on the real arm against four cup positions. |
| `dance_sweep_period_s` | `DOUBLE` | `2.0` | One full sine cycle in this many seconds. |

### New service `/dance_throw`

Handler:

1. `_preflight_motion()` — joint feedback only. Reject if controller
   isn't IDLE or feedback is missing/stale.
2. Validate `replay_path` is set, resolves to an existing file, and
   parses via `_load_trajectory_csv` (fail fast before any motion).
3. Read & validate `dance_target_angles_deg` (non-empty) and
   `dance_sweep_period_s` (`> 0`).
4. Random pick (`random.choice`, no fixed seed — we want true randomness
   per demo run):
   - `target_deg = random.choice(angles_deg)`
   - `duration_s = DANCE_NUM_SWEEPS * dance_sweep_period_s` (deterministic)
5. Log the plan (see "Logging" below).
6. Sub-cycle A — `controller.start_dance(...)`, wait on `_cycle_done`
   with 13s timeout.
7. Sub-cycle B — `controller.start_replay(traj, hold_ticks, release_row,
   freeze_joints=[], joint1_override=target_rad)` reading `replay_hz`
   and `replay_release_row` from existing params. 10s timeout.
8. Sub-cycle C — `controller.start_home_only(current_q)`. 10s timeout.
9. Return `success=True` with
   `f"dance throw complete (target={target_deg:+.1f}°, dance={duration_s:.2f}s)"`.

On any sub-cycle timeout: assign `controller.state = IDLE`, return
`success=False` with the phase that timed out.

Before each sub-cycle wait: `self._cycle_done.clear()`,
`self._arm_publish_tick = 0`, `self._init_throw_log()` (so each phase
gets its own log).

### Logging

At service accept (after random rolls, before motion):

```
[INFO] dance_throw triggered: target=+10.0° duration=8.43s period=2.00s
[INFO]   trajectory: /ws/recordings/throw_x0.20_y0.10.csv (45 ticks, release_row=33)
[INFO]   plan: HOMING→SETTLE_HOME→DANCE → HOMING→SETTLE_HOME→REPLAY→SETTLE_RELEASE → HOMING→SETTLE_HOME
```

Per sub-cycle (just before its `_cycle_done.wait`):

```
[INFO] phase 1/3: dance (sweeping joint1 ±15° for 8.43s, landing at +10.0°)
[INFO] phase 2/3: replay (joint1 pinned at +10.0°)
[INFO] phase 3/3: return home
```

Per sub-cycle completion:

```
[INFO] phase 1/3 complete
[WARN] phase 2/3 timeout (>10s) — aborting dance throw
```

At final return:

```
[INFO] dance_throw complete (target=+10.0°, total elapsed=18.4s)
```

`/throw_state` continues publishing the per-tick state name as before,
so live monitors see `DANCE` ticking during the sweep.

### Throw-log meta

Each sub-cycle log gets new meta fields so the three CSVs are linkable:

- `dance_phase` — `"dance"`, `"replay"`, or `"return_home"`
- `dance_target_deg` — the rolled angle
- `dance_duration_s` — the rolled duration
- `dance_period_s` — the active sweep period
- `dance_amplitude_deg` — the hardcoded amplitude (15°)

## Testing

Unit (Mac, no rclpy):

- `start_dance` sets state/mode correctly and stores params.
- DANCE tick at `tick_idx = 1`: joint1 ≈ 0, joints 2-6 = HOME.
- DANCE tick at `tick_idx = total_ticks`: joint1 = target_rad, damp = 0.
- DANCE tick during ease-out: joint1 between `(1-damp)*target` and
  `(1-damp)*target ± damp*amplitude`.
- Transition to IDLE exactly at `tick_idx > total_ticks`.

Integration (Jetson container, real arm):

- `/dance_throw` with default params completes in < 22s.
- The arm visibly sweeps before each throw and lands at one of the five
  preset angles.
- Three CSV logs appear per call, each with the new meta fields.
- Service rejects cleanly when `replay_path` is empty or the file is
  missing (no motion).
- Service rejects cleanly when joint feedback is stale.

## YAGNI'd

- No "dance only" service (without throw). If you want just the sweep
  for debugging, set duration low and use a no-op trajectory.
- No mapping from joint1 angle to per-angle recording (one recording
  retargeted for all four angles).
- No lookup-mode integration. Dance is manual-replay only.
- No seedable RNG. Demo run-to-run variation is the point.
- No live re-tuning of the damping schedule. 60/40 cos² is a constant.
