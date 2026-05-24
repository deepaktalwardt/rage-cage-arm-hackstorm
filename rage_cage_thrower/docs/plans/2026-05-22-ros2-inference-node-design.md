# `rage_cage_thrower` node: design

Build the ROS2 node that runs the trained throw policy against the real
AgileX PiPER arm. First milestone is dry-run-no-ball: validate plumbing,
motion sanity, and release-step timing without a ball loaded or a cup
placed. Same node carries forward to real throws once perception and
ball loading are in place — no refactor planned between milestones.

## Goal

Single Python rclpy node named `rage_cage_thrower`, running inside its
own container on the Jetson control box. Subscribes to `/joint_states`
and `/cup_pose`, loads the trained PPO policy, and streams 50Hz joint
position commands plus a one-shot gripper-open command at the right
moment in the throw. A `/throw_trigger` service kicks off one full
cycle (home → throw → settle) and blocks until done.

Arm-motion validation (controllers, velocity limits, e-stop, joint
mapping) is handled separately by piper_ros2's own tooling and is not
this node's job. This node only assumes a working position-command
interface exists.

## Target model

`models/random_stack_cup_thrower_no_ball_obs_v1/` (from
`origin/deepak/rm-ball-state-from-obs`). Ball state was removed from
the obs because it cannot be reliably estimated on the real arm at
sub-throw timescales. README reports 95% success in sim without ball
obs.

| Property | Value |
|---|---|
| Obs dim | 16 |
| Obs layout | `joint_pos(6), joint_vel(6), cup_xy(2), pedestal_height(1), release_countdown(1)` |
| Action dim | 6 |
| Action range | `[-1, 1]` per joint, integrated as `arm_target += action * 0.06` (rad/tick) |
| Control rate | 50 Hz exactly (`control_dt = 0.02`) |
| Release | Automatic at tick 45 (~0.9s windup), policy stops issuing actions afterward |
| Trained envelope | `cup_xy ∈ [0.75, 0.95] × [±0.10] m`, `pedestal ∈ [0, 0.15] m` |
| Required at inference | `policy.zip` + `vecnormalize.pkl` (obs normalization stats) |

The trained envelope is the contract — outside this box the policy
extrapolates and has no training distribution support.

## Architecture

Single Python rclpy node, class `RageCageThrower`, entry point
`real/rage_cage_thrower.py`. One model, one policy, one in-flight throw
at a time.

### Interfaces

Talks to the `piper_ros` driver (humble branch, single-arm launch:
`ros2 launch piper start_single_piper.launch.py`). That launch remaps
the driver's internal `joint_ctrl_single` to `/joint_states`, so
**`/joint_states` is the COMMAND topic, not feedback** — feedback lives
on `joint_states_feedback`. We match that convention rather than fight
it.

```
SUBSCRIBES:
  joint_states_feedback   sensor_msgs/JointState        from piper_ros
                                                        - 7 elements: joint1..6 + gripper
                                                        - we remap by name → drop gripper
                                                        - positions in radians, 200Hz
  /cup_pose               geometry_msgs/PoseStamped     from perception node
                                                        - latched (TRANSIENT_LOCAL QoS)
                                                        - point at the TOP of the cup
                                                        - arm-base frame

PUBLISHES:
  /joint_states           sensor_msgs/JointState        joint COMMANDS to driver, 50Hz
                                                        - name = [joint1..6, gripper]
                                                        - position[0:6] arm targets (rad)
                                                        - position[6]  gripper opening (m)
                                                        - velocity[6]  motor speed limit %
                                                          (only if motor_speed_limit > 0;
                                                          omitted ⇒ driver default 100%)
  /throw_state            std_msgs/String               IDLE / HOMING / THROWING / SETTLING

SERVICES:
  /throw_trigger          std_srvs/Trigger              blocking; response on cycle end
```

Gripper is **not** a separate topic — it's `position[6]` of every command
JointState. HOLD value before release, OPEN value at release and after.

### State machine

One trigger = one full pass:

```
IDLE ──/throw_trigger──► HOMING (joint-space linear interp current → rage_home, ~2s)
                            │
                            ▼
                         SETTLE_HOME (300ms; wait for joint_state to converge)
                            │
                            ▼
                         THROWING (45 ticks × 20ms = 0.9s; policy in loop)
                            │
                            ▼
                         RELEASE (last THROWING tick: arm_target from policy,
                                  gripper switches HOLD → OPEN in same JointState)
                            │
                            ▼
                         SETTLE_RELEASE (1s; re-publish last arm_target + OPEN
                                         gripper so the driver never sees a
                                         command-timeout)
                            │
                            ▼
                         return Trigger.Response(success=True) ──► IDLE
```

### Concurrency

`MultiThreadedExecutor`. The service callback signals a
`threading.Event` and waits on a `Future`. A 50Hz `rclpy.Timer`
callback drives the state machine and sets the Future when state
returns to IDLE. All publishing happens from the timer thread; the
service callback only orchestrates and blocks. Single-threaded executor
would deadlock (blocked service starves the timer).

## Conversions in and out of the policy

### Loaded once at node startup

```python
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

ppo = PPO.load("models/random_stack_cup_thrower_no_ball_obs_v1/policy.zip")
# StubEnv is a 20-line gym.Env with matching observation_space (16-d Box)
# and action_space (6-d Box [-1,1]). Avoids dragging MuJoCo + sim/ onto the
# Linux box just to satisfy VecNormalize.load's API.
dummy = DummyVecEnv([lambda: StubEnv()])
vn = VecNormalize.load(".../vecnormalize.pkl", dummy)
obs_mean, obs_var = vn.obs_rms.mean.copy(), vn.obs_rms.var.copy()
clip_obs, eps = float(vn.clip_obs), float(vn.epsilon)

ACTION_DELTA  = 0.06
RELEASE_STEP  = 45
ARM_JOINTS    = ("joint1","joint2","joint3","joint4","joint5","joint6")
JOINT_LOW     = np.array([-2.618, 0.0,   -2.697, -1.832, -1.22, -3.14])
JOINT_HIGH    = np.array([ 2.618, 3.14,   0.0,    1.832,  1.22,  3.14])
HOME_QPOS     = np.array([ 0.0,   1.57,  -1.3485, 0.0,   0.0,   0.0 ])
SIM_CUP_HEIGHT = 0.12   # NOT the real 0.115 — match training distribution
```

### Joint-state ingest (async callback)

```python
def on_joint_states(self, msg):
    # NEVER assume index order matches ARM_JOINTS — remap by name.
    idx = [msg.name.index(j) for j in ARM_JOINTS]
    self.latest_q    = np.array(msg.position, dtype=np.float32)[idx]
    self.latest_qdot = np.array(msg.velocity, dtype=np.float32)[idx]
    self.latest_q_t  = self.get_clock().now()
```

### Per-tick obs build (50Hz timer, only during THROWING)

```python
cup_xy     = np.array([self.cup_pose.x, self.cup_pose.y], dtype=np.float32)
pedestal_h = max(self.cup_pose.z - SIM_CUP_HEIGHT, 0.0)
countdown  = max(RELEASE_STEP - self.tick, 0) / RELEASE_STEP

obs = np.concatenate([
    self.latest_q,                                   # (6,)
    self.latest_qdot,                                # (6,)
    cup_xy,                                          # (2,)
    np.array([pedestal_h], dtype=np.float32),        # (1,)
    np.array([countdown],  dtype=np.float32),        # (1,)
]).astype(np.float32)                                # 16-d, order matches
                                                     # sim/env.py:_get_obs

normalized = np.clip(
    (obs - obs_mean) / np.sqrt(obs_var + eps),
    -clip_obs, clip_obs,
).astype(np.float32)

action, _ = ppo.predict(normalized, deterministic=True)   # (6,) in [-1, 1]
```

### Action → joint target

```python
action = np.clip(action, -1.0, 1.0)
self.arm_target = np.clip(
    self.arm_target + action * ACTION_DELTA,
    JOINT_LOW, JOINT_HIGH,
)
self.arm_cmd_pub.publish(
    Float64MultiArray(data=self.arm_target.tolist())     # joint1..joint6 order
)
self.tick += 1
```

### Release (one-shot at `tick == RELEASE_STEP`)

```python
self.publish_gripper_command(open=True)
# Keep re-publishing self.arm_target each tick through SETTLE_RELEASE
# so the controller never sees a command-timeout. No more policy calls.
```

### State-transition resets

| On entering… | Reset |
|---|---|
| `HOMING` | `self.tick = 0`; precompute 2s linear interp `current_q → HOME_QPOS` |
| `THROWING` | `self.arm_target = self.latest_q.copy()` — integrate deltas from where the arm actually is, not from a stale value |
| `IDLE` (after `SETTLE_RELEASE`) | stop publishing `/arm_command` so the controller holds passively |

### Five things that must match training exactly

Easy to get wrong silently:

1. **Obs ordering** — `joint_pos, joint_vel, cup_xy, pedestal_h, release_countdown`. Match `sim/env.py:_get_obs`.
2. **Joint-name → index remap** — `JointState.name` order is driver-dependent; always remap by name.
3. **VecNormalize** — without it the policy gets raw obs and produces garbage. Stats live in `vecnormalize.pkl` (~4KB) and ship alongside `policy.zip`.
4. **0.06 rad/tick action_delta** — policy outputs unitless `[-1,1]`; conversion factor matches `RageCageEnv.action_delta`.
5. **50Hz timer cadence** — `release_countdown` is normalized but tick counting is in real ticks. A jittery timer breaks release alignment.

## Homing, lifecycle, pre-flight

### Homing trajectory (joint-space linear interp, no IK)

```python
duration_s = 2.0
start_q   = self.latest_q.copy()
delta_q   = HOME_QPOS - start_q
# Stretch duration if any joint needs a big swing — keeps max homing
# velocity ≤ 1.5 rad/s regardless of where the arm started.
if float(np.max(np.abs(delta_q))) / duration_s > 1.5:
    duration_s = float(np.max(np.abs(delta_q))) / 1.5
n_ticks = int(duration_s / 0.02)

for k in range(1, n_ticks + 1):
    target = start_q + (k / n_ticks) * delta_q
    self.arm_cmd_pub.publish(Float64MultiArray(data=target.tolist()))
    # rclpy timer drives the 20ms cadence
```

After homing, `SETTLE_HOME` for 300ms — keep publishing `HOME_QPOS`.
Optional convergence check: `np.max(np.abs(self.latest_q - HOME_QPOS)) < 0.02 rad`.
If it does not converge within +1s of extra settle, abort the cycle with failure.

### Pre-flight checks (run when `/throw_trigger` fires, before HOMING)

| Check | Fail response |
|---|---|
| `self.state == IDLE` | "throw already in progress" |
| Latest `/joint_states` stamp < 100ms old | "joint_states stale or missing" |
| `/cup_pose` received at least once since startup | "no cup_pose yet" |
| (optional, ROS param) `cup_xy ∈ envelope`, `pedestal ∈ [0, 0.15]` | "cup outside trained workspace" |

Each returns `Trigger.Response(success=False, message=…)` immediately —
no state transition.

### Post-release SETTLE_RELEASE: 1.0s

Re-publish the last `arm_target` at 50Hz so the position controller
never times out. No policy calls. After 1s → `success=True` → IDLE.

### Failures that return `success=False` mid-cycle

- `/joint_states` stops flowing during HOMING or THROWING (>100ms gap)
- Homing fails to converge within 1s of extra settle
- (Caller-side) `std_srvs/Trigger` has no cancel; caller sets its own
  service timeout if needed

## Code layout & containerization

### File tree

```
real/                                  # all node code, mounted into the container
  __init__.py
  controller.py                        # Layer 1: ThrowController, state machine, model I/O
  stub_env.py                          # Layer 1: minimal gym.Env for VecNormalize.load
  rage_cage_thrower.py                 # Layer 2: rclpy node entry point (only rclpy import)
  docker/
    Dockerfile                         # FROM piper:jazzy + Python ML deps
    requirements.txt                   # stable-baselines3, torch (CPU), gymnasium, numpy
    compose.yaml                       # rage_cage_thrower service definition
  tests/
    test_controller_equivalence.py    # asserts ThrowController == play_policy.py
    test_state_machine.py             # asserts state transitions, homing shape
```

### Layering (for dev-time testability, not deployment split)

Layer 1 (`controller.py`, `stub_env.py`) has zero rclpy imports, so we
can run `uv run pytest real/tests/` on the Mac with the existing `uv`
env. This lets us validate the controller against the existing sim
*before* the container is available. The same files ship into the
container alongside Layer 2 — the whole node runs in one process inside
one container.

Layer 2 (`rage_cage_thrower.py`) is the only file that imports rclpy
and can therefore only be run inside the container. ~60 lines of
subscriber/publisher/service wiring that delegates to
`ThrowController.tick()`.

### Container

The team docker harness (`docker-harness-for-node/`) builds a shared
`piper:jazzy` base image with ROS Jazzy + piper_sdk + can-utils
preinstalled. We derive a per-node image rather than pollute the shared
one with ~500MB of ML dependencies — matches the "one container per
node" pattern already in use for `foxglove`, `realsense_d435i`, and
`ping_pong_detector`.

**`real/docker/Dockerfile`:**

```dockerfile
FROM piper:jazzy

COPY real/docker/requirements.txt /tmp/requirements.txt
# Noble (24.04) enforces PEP 668; --break-system-packages matches the
# pattern used by the shared harness Dockerfile.
RUN pip3 install --no-cache-dir --break-system-packages \
      --index-url https://pypi.org/simple \
      -r /tmp/requirements.txt

# Code + models come in via volume mount; see compose.yaml.
# WORKDIR /ws is inherited from the base image.
```

**`real/docker/requirements.txt`:**

```
stable-baselines3>=2.0
torch>=2.0
gymnasium>=0.29
numpy>=1.24
```

CPU torch is sufficient — the policy is a 256×256 MLP with sub-ms
inference. Jetson GPU torch is a separate nvidia-published install path
and not worth the wheels-juggling for what this node does.

**`real/docker/compose.yaml`** (sketch — full file goes alongside):

```yaml
services:
  rage_cage_thrower:
    build:
      context: ../..                          # repo root for the Dockerfile's COPY
      dockerfile: real/docker/Dockerfile
    image: rage_cage_thrower:jazzy
    container_name: rage_cage_thrower
    network_mode: host                        # share DDS with piper_ros
    runtime: nvidia                           # harmless if unused
    environment:
      ROS_DOMAIN_ID: ${ROS_DOMAIN_ID}         # must match piper_ros
      RMW_IMPLEMENTATION: rmw_cyclonedds_cpp  # must match piper_ros
    volumes:
      - ../../real:/ws/real
      - ../../models:/ws/models
    command: sleep infinity                   # dev mode — docker exec to launch
```

### Bringing it up

Prereq: the shared base image must exist (one-time, owned by the
harness):

```bash
cd docker-harness-for-node && docker compose build
```

Then build and launch our service:

```bash
docker compose -f real/docker/compose.yaml up -d --build
```

Dev iteration:

```bash
docker exec -it rage_cage_thrower bash
source /opt/ros/jazzy/setup.bash
python3 /ws/real/rage_cage_thrower.py
# in another exec shell:
ros2 service call /throw_trigger std_srvs/srv/Trigger
```

Once the node is stable, replace `command: sleep infinity` with the
auto-launch form so it starts with `docker compose up`.

### Cross-container contract

- `ROS_DOMAIN_ID` and `RMW_IMPLEMENTATION=rmw_cyclonedds_cpp` — both
  inherited from the harness `.env`, no override needed
- `network_mode: host` — required so DDS discovery reaches piper_ros
  without explicit peer config
- Model files live in `models/` at the repo root; bind-mounted at
  `/ws/models` inside the container. No need to bake them into the image.

### Why `stub_env.py`

`VecNormalize.load()` requires a `DummyVecEnv` to bind to. We don't
want to install MuJoCo + the full `sim/` package inside the container
just to recover the obs-normalization stats. A 20-line `gym.Env`
exposing the same `observation_space` (16-d Box) and `action_space`
(6-d Box `[-1,1]`) is enough.


## Out of scope (v0 — punted intentionally)

- Internal logging / rosbag — console-only for now; revisit if a throw
  goes weird and we need offline replay
- ROS2 launch file, parameters, composition — single-file script first
- Action server with feedback/cancel — service has no cancel; not
  needed for service-call dry runs
- Ball pickup — ball is preloaded by hand later, no autonomous pickup
- Perception node — separate work; assumed to publish `/cup_pose`
- Safety nets beyond the trained policy (speed scaling, workspace
  AABB, max-delta clamp) — arm-motion safety is handled outside this
  node by piper_ros2 limits + hardware e-stop

## Open items (resolve when piper_ros2 is up on the Linux box)

- Tune `GRIPPER_HOLD_M` / `GRIPPER_OPEN_M` in `real/controller.py` once
  a physical ping pong ball is loaded — sim used 0.022 / 0.035 m, but
  the real arm's grip-on-ball value may differ
- Confirm whether `auto_enable:=true` in `start_single_piper.launch.py`
  reliably brings the arm up enabled on container start — if not, call
  the driver's `/enable_srv` (or publish `True` to `/enable_flag`)
  before the first throw
- Decide whether perception will publish `/cup_pose` directly, or if we
  need a small bridge node that turns the existing
  `/PingPongDetector/bounding_box` (ball detector, not cup detector) into
  a cup pose. For dry runs, a constant publisher is enough
- Pick a sensible `motor_speed_limit` ROS param value for the first
  dry-run throws — full speed (`0` ⇒ driver default 100%) matches the
  trained policy but is aggressive; reduce to ~30 for safety, accepting
  the throw will undershoot
