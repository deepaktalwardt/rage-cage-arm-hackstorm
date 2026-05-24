# Action Smoothing & Realistic Joint Velocities

Context: while inspecting a v5 trained policy in `play_policy.py`, the arm was moving erratically and at speeds that aren't physically realizable on the real PiPER. We were issuing joint commands the real robot couldn't execute. This doc captures the diagnosis and the planned fix so we don't re-derive it.

## Symptom

- Arm visually thrashes even at `--speed 0.25` (4× slow-mo) playback.
- Joint targets command changes too large to be physical.
- PPO learning is harder than it should be — the policy maps tiny obs deltas to huge action deltas (high-Lipschitz, near-chaotic mean output).
- Trained policy would not transfer to the real PiPER without modification.

## Root cause

Two compounding factors.

### 1. Action mapping is full-range every step

Current `env.py`:

```python
self.data.ctrl[:6] = JOINT_MID + JOINT_HALF_RANGE * action[:6]
```

Each policy action `∈ [-1, 1]` is mapped to the **full physical joint range** every 20 ms control step. Joint 6 has range `[-3.14, 3.14]` rad = 6.28 rad swing. So if the policy outputs `0.3` for joint 6 one step and `-0.4` the next, the commanded target jumps **2.2 rad ≈ 125°** in 20 ms — which the actuator chases hard.

### 2. PD gains are tuned for stiff tracking, no implicit velocity cap

PiPER MJCF actuators (`piper.xml`) use `kp=80 kv=5` for major joints, `kp=10 kv=1.5` for the wrist. With small joint inertia (~0.005 kg·m²) this gives:

- Damping ratio ζ ≈ kv / (2·sqrt(kp·inertia)) ≈ 3.97 → strongly overdamped, *no* overshoot.
- But also no velocity ceiling: with `forcerange="-100 100"` (100 N·m peak torque, itself unrealistic — real PiPER motors are well under that), simulated joints can hit ~30 rad/s peaks. The real PiPER tops out around ~3 rad/s. **The policy is exploiting unrealistic actuator power.**

## Planned fix (three changes, applied together)

### A. Delta-action mapping

Replace the absolute mapping with a per-step delta-target:

```python
DELTA_MAX = 0.20  # rad per control step

self.data.ctrl[:6] = np.clip(
    self._last_ctrl + DELTA_MAX * action[:6],
    JOINT_LOW, JOINT_HIGH,
)
self._last_ctrl = self.data.ctrl[:6].copy()
```

State to maintain in env:
- `self._last_ctrl` — initialized in `reset()` from `self.data.ctrl[:6].copy()` (which is the noised home pose ctrl after keyframe + reset noise are applied).

Effect: target velocity capped at `DELTA_MAX / control_dt = 0.20 / 0.020 = 10 rad/s`. With the overdamped PD tracking tightly, actual joint velocity stays close to this. ~5 m/s tip velocity, plausible for a tabletop throw, executable on the real arm.

**No separate velocity actuator limit needed** — overdamped PD (ζ ≈ 4) doesn't overshoot, so capping the target velocity caps the actual velocity. We verified this is correct via the damping-ratio math above; the obvious alternative (`<velocity>` actuator with ctrlrange) would be redundant.

### B. Action smoothness penalty

Add a small per-step reward term:

```python
ACTION_SMOOTHNESS_LAMBDA = 0.001

# In _reward_and_term, after computing time penalty:
da = action[:6] - self._last_action[:6]
r -= ACTION_SMOOTHNESS_LAMBDA * float(np.dot(da, da))
self._last_action = action.copy()
```

State: `self._last_action` initialized to `np.zeros(7)` in reset.

Sizing: smooth policy with ||Δa||² ≈ 0.1 pays ~0.0001/step (negligible over 200 steps); a full reversal with ||Δa||² ≈ 6 pays ~0.006/step. Light pull, not a dominant signal. Belt-and-suspenders to the delta cap.

We deliberately exclude `action[6]` (release signal) from the smoothness term — it's binary in spirit and penalizing rapid change there could confuse the release decision.

### C. Plumbing

`_reward_and_term` needs to take `action` as an argument so it can compute the smoothness term:

```python
def _reward_and_term(self, action: np.ndarray): ...
```

And in `step`:
```python
reward, terminated, truncated, info = self._reward_and_term(action)
```

## Picking the delta cap

Rough mapping from joint velocity to tip velocity, assuming arm length ≈ 0.5 m:

| Throw style | Tip velocity | Joint velocity | DELTA_MAX |
|---|---|---|---|
| Gentle lob | ~3 m/s | ~6 rad/s | 0.12 rad/step |
| Confident throw | ~5 m/s | ~10 rad/s | **0.20 rad/step** ← start here |
| Forceful whip | ~8 m/s | ~16 rad/s | 0.32 rad/step |

Start at **0.20**. If the agent can't reach the far-corner cup position (0.75 m), bump to 0.30. If motion still looks unstable, drop to 0.15.

## Things considered and deliberately NOT included

- **Explicit `<velocity>` actuators or `actuatorvel` clamps in MJCF** — redundant given delta-action + overdamped PD.
- **Tightening `forcerange` in piper.xml** — `100 N·m` is unrealistic for PiPER, but reducing it is a separate sim-to-real concern. Doesn't help with the erratic-motion problem specifically. Address as a later iteration.
- **Lowering `kp/kv`** — changes arm dynamics globally, more invasive than needed.
- **Switching from `<position>` to `<velocity>` actuators** — much bigger refactor of the control stack. Higher risk for hackathon-scope.

## When this fix invalidates training

The action mapping is fundamentally different — a v5-or-earlier policy treats `action[i]=1` as "go to joint limit"; the new mapping treats it as "step 0.20 rad in that direction." Old weights are unusable. Fresh training run required after this change lands.
