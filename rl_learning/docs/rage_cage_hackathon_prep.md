# Rage Cage Robot — Hackathon Prep & Planning

**Use this doc for the first team meeting.** It breaks the project into workstreams, separates what we can do *before* the hackathon from what must happen *during*, lists what to bring, and flags the decisions we need to make as a team.

---

## 1. Project Summary

We're building a "drunk" rage cage robot: the AgileX PiPER 6-DOF arm watches a single stack of cups (1–10, within a 10×10 cm area) through an external camera and bounces a ping pong ball off the table into the stack. A "drunkenness" parameter (0–100%) makes it progressively wobbly and error-prone for comedic effect.

The technical approach is a **two-stage pipeline**: train an RL agent in simulation to discover successful throws, use its successful trajectories as demonstrations, then fine-tune a vision-language-action model (OpenVLA) to imitate them from camera input. We validate the whole loop in simulation first, then bridge to real hardware.

> **Headline message for the meeting:** Almost all of Phase 1 (simulation, RL, VLA fine-tuning) needs *zero physical hardware*. If we treat Phase 1 as pre-work, the 48 hours can be spent almost entirely on the sim-to-real gap and integration — which is where projects like this actually die.

---

## 2. Main Work Areas (Workstreams)

The project splits into four tracks that can run in parallel and map cleanly onto team roles.

### Track A — Simulation & RL
MuJoCo scene, PiPER URDF → MJCF conversion, Gym environment wrapper, reward design, RL training (Stable Baselines 3), demonstration dataset generation.

### Track B — VLA & ML
Cloud GPU account and pipeline, dataset formatting (LeRobot/HDF5), OpenVLA LoRA fine-tuning, closed-loop inference testing in simulation.

### Track C — Hardware & ROS2
Linux box, ROS2 + `piper_ros`, CAN-to-USB adapter, camera calibration, the ROS2 ↔ VLA bridge that turns model outputs into arm commands.

### Track D — Integration & Demo
Drunkenness UI, real-data collection tooling, end-to-end glue code, demo rehearsal and backup plan.

---

## 3. Pre-Hackathon Prep (Do This Ahead)

The key insight: **a sim-trained VLA can exist before the event starts.** Everything below needs no physical arm.

> **Note on hackathon rules:** Some events restrict how much can be pre-built versus prepped. Environment setup, learning, skeleton code, and tooling are almost always allowed; check your specific rules for the model-training portion. When in doubt, keep pre-built artifacts modular and be ready to re-run during the event.

### Track A — Simulation & RL (fully doable ahead)
- [ ] Install MuJoCo + Stable Baselines 3 on the Mac
- [ ] Obtain PiPER URDF from AgileX GitHub; convert to MJCF
- [ ] Fix joint limits, inertias, damping in the MJCF
- [ ] Build the scene: arm + table + cup stack primitives + ball
- [ ] Tune ping pong ball physics (restitution, mass) against reference video
- [ ] Set up offscreen camera in the scene (rough match to expected real viewpoint)
- [ ] Write the Gym environment wrapper (obs, action, reward, reset)
- [ ] Train the RL agent (PPO) until reliable throws
- [ ] Generate the demonstration dataset (~200 episodes, varied stack configs + drunkenness)

### Track B — VLA & ML (fully doable ahead)
- [ ] Create cloud GPU account (RunPod recommended); load $10–20 credit
- [ ] Test the fine-tuning pipeline end-to-end on dummy data
- [ ] Format the sim demo dataset (LeRobot or HDF5)
- [ ] Fine-tune OpenVLA (LoRA) on the sim dataset
- [ ] Validate closed-loop inference in simulation (VLA in the loop)
- [ ] Save the sim-trained checkpoint in two places (USB + HF Hub)

### Track C — Hardware & ROS2 (doable ahead except real-arm testing)
- [ ] Prep the Linux box: install ROS2, clone `piper_ros`
- [ ] Install + test CAN-to-USB drivers; confirm `candump` works (loopback if no arm)
- [ ] Write camera calibration scripts
- [ ] Code the ROS2 ↔ VLA bridge skeleton (subscribe camera, publish joint commands)
- [ ] Set up matching Python env on Linux (versions aligned with Mac)

### Track D — Integration & Demo (doable ahead)
- [ ] Build and test the drunkenness UI (slider 0–100%) against sim
- [ ] Write the real-data recording scripts (camera + joint states → HDF5)
- [ ] Draft the demo script and judge-facing narrative

**Definition of "ready for the hackathon":** a VLA that, in simulation, takes camera frames + a drunkenness prompt and lands the ball in the stack at a reasonable rate, plus a Linux box that can drive the arm — verified independently.

---

## 4. During-Hackathon Work (Needs the Arm / Is the Build)

Ordered roughly by dependency. The sim-to-real gap is the unpredictable part — give it the most buffer.

1. **Arm bring-up** — sanity-check PiPER motion at low speed, verify ROS2 topics flow
2. **Camera setup** — mount camera in a stable, repeatable position; run real calibration
3. **Viewpoint matching** — adjust or accept the gap between sim and real camera pose
4. **Real demo collection** — gather 20–30 successful real throws (scripted or teleop)
5. **Real-data fine-tuning** — fine-tune the sim VLA on real demos (cloud or Linux GPU)
6. **Sim-to-real debugging** — the hard part: visual gap, latency, backlash, exposure (biggest buffer here)
7. **Integration** — hook the drunkenness UI to the real arm; full pipeline live
8. **Demo rehearsal** — run the full path ≥3 times; record a backup video
9. **Polish** — slides, narrative, fallback ready

---

## 5. Things to Bring

### Compute & Power
- [ ] Mac (sim/ML work) + charger
- [ ] Linux box (arm control) + charger
- [ ] Power strip / extension cord
- [ ] **Mobile hotspot or backup internet** — critical; cloud GPU fine-tuning needs reliable internet and venue WiFi is always flaky

### Robot Interface
- [ ] CAN-to-USB adapter (+ a backup unit — single point of failure)
- [ ] USB webcam (the external camera)
- [ ] USB hub (you will run out of ports)
- [ ] Ethernet cable (Mac ↔ Linux file transfer, faster than WiFi)

### The Game Itself
- [ ] Ping pong balls (a dozen+; they get lost and crack)
- [ ] Cups for rage cage
- [ ] Tape / markers to mark the 10×10 cm region and table positions
- [ ] Known table surface if portable, or confirm the venue provides one

### Camera Mounting
- [ ] Tripod or clamp mount (stable, repeatable viewpoint is essential)
- [ ] Zip ties, gaffer tape

### Tools
- [ ] Small screwdriver set, zip ties, multitool — generic robot-wrangling kit

### Pre-Loaded Assets (prepare before leaving)
- [ ] All repos cloned + dependencies installed on **both** machines
- [ ] Sim-trained VLA checkpoint on a USB drive **and** on HF Hub (two copies)
- [ ] Cloud GPU account with credits already loaded
- [ ] Sim demo dataset, backed up

---

## 6. Open Questions / Decisions for the Meeting

- [ ] Is the PiPER arm provided by the hackathon, or do we bring it? Confirm the camera too.
- [ ] Who owns each track (A/B/C/D)?
- [ ] Cloud fine-tune provider: **RunPod vs Modal** — decide, and have one person set up the account ahead of time.
- [ ] Drunkenness implementation: **prompt-conditioned vs post-inference noise injection** — pick one to build first.
- [ ] Minimum viable demo if sim-to-real fails — agree on the fallback now (see below).
- [ ] Check the hackathon's pre-build rules for the training portion.

---

## 7. Fallback Plan (Decide Now, Not at Hour 40)

If the real arm fights us, a **live VLA running in MuJoCo with the drunkenness slider** is still a compelling judge demo. Having this as an explicit safety net changes how much risk we can take on the real-hardware side. Agree at the meeting that this is an acceptable floor, and keep the sim closed-loop demo working and runnable at all times.

---

## 8. Suggested Team Roles

| Person | Primary Track | Also Owns |
|---|---|---|
| TBD | Track A — Sim & RL | Reward design, demo data quality |
| TBD | Track B — VLA & ML | Cloud pipeline, checkpoint management |
| TBD | Track C — Hardware & ROS2 | CAN/ROS2 bring-up, camera calibration |
| TBD | Track D — Integration & Demo | **End-to-end owner** — keeps the full path working at all times; drunkenness UI; demo |

The Track D person is the integration owner: their job is to make sure a runnable end-to-end demo exists at every checkpoint, even a degraded one.

---

*Companion document: `rage_cage_vla_primer.md` (technical deep-dive on the pipeline). This doc is the planning layer; that one is the implementation reference.*
