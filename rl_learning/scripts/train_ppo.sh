#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
WORKFLOW_START=$SECONDS

# Default mode is a minimized train+eval smoke test. It proves that dependency
# sync, MuJoCo loading, Gymnasium validation, SB3 training, model saving, model
# loading, and GIF rendering all work. It is intentionally too short to solve
# the task. Set FULL_RUN=1 for a larger first training attempt.
FULL_RUN="${FULL_RUN:-0}"

if [[ "$FULL_RUN" == "1" ]]; then
  DEFAULT_TIMESTEPS=50000
  DEFAULT_N_ENVS=4
  DEFAULT_N_STEPS=2048
  DEFAULT_BATCH_SIZE=64
  DEFAULT_EPISODES=1
  DEFAULT_OUT="sim/_rl_out/ppo_thrower"
  DEFAULT_EVAL_OUT_DIR="sim/_rl_eval"
  DEFAULT_TRAIN_ROLLOUT_VIZ=1
  DEFAULT_TRAIN_ROLLOUT_VIZ_EVERY=100000
  DEFAULT_CURRICULUM=auto
  DEFAULT_CURRICULUM_EVAL_EVERY=100000
  DEFAULT_CURRICULUM_EVAL_EPISODES=8
else
  DEFAULT_TIMESTEPS=128
  DEFAULT_N_ENVS=1
  DEFAULT_N_STEPS=16
  DEFAULT_BATCH_SIZE=16
  DEFAULT_EPISODES=1
  DEFAULT_OUT="sim/_rl_out/ppo_smoke"
  DEFAULT_EVAL_OUT_DIR="sim/_rl_eval_smoke"
  DEFAULT_TRAIN_ROLLOUT_VIZ=0
  DEFAULT_TRAIN_ROLLOUT_VIZ_EVERY=0
  DEFAULT_CURRICULUM=manual
  DEFAULT_CURRICULUM_EVAL_EVERY=64
  DEFAULT_CURRICULUM_EVAL_EPISODES=2
fi

# TIMESTEPS is the target number of environment transitions SB3 should collect.
# Higher values give PPO more experience and are required for real learning, but
# runtime scales roughly linearly. PPO rounds up to a full rollout of
# N_ENVS * N_STEPS transitions, so requested TIMESTEPS may be exceeded.
TIMESTEPS="${TIMESTEPS:-$DEFAULT_TIMESTEPS}"

# N_ENVS controls how many MuJoCo environments run in parallel. More envs collect
# rollout data faster on multi-core machines and decorrelate experience, but use
# more CPU and memory. Total rollout size is N_ENVS * N_STEPS.
N_ENVS="${N_ENVS:-$DEFAULT_N_ENVS}"

# N_STEPS is how many actions each env runs before PPO updates the policy. Larger
# values make updates use more on-policy data and can stabilize learning, but
# each update takes longer and the first log/save is delayed. Smaller values are
# useful for smoke tests.
N_STEPS="${N_STEPS:-$DEFAULT_N_STEPS}"

# BATCH_SIZE is the minibatch size PPO uses when optimizing over a rollout.
# Larger batches are smoother but use more memory; smaller batches update more
# noisily. It should usually divide N_ENVS * N_STEPS cleanly.
BATCH_SIZE="${BATCH_SIZE:-$DEFAULT_BATCH_SIZE}"

# LR is PPO's optimizer learning rate. Higher values can learn faster but become
# unstable; lower values are slower and more conservative. 0.0003 is the common
# PPO baseline.
LR="${LR:-0.0003}"

# LR_SCHEDULE controls whether LR stays fixed or linearly decays to zero over
# training. Use "constant" for baseline comparisons and "linear" for longer runs
# where late-stage stability matters.
LR_SCHEDULE="${LR_SCHEDULE:-constant}"

# NET_ARCH controls the SB3 policy/value MLP preset. "default" is SB3's small
# baseline; "medium", "large", and "deep" are larger custom networks for
# capacity experiments.
NET_ARCH="${NET_ARCH:-default}"

# ACTIVATION controls the MLP nonlinearity. "tanh" matches SB3 PPO defaults;
# "relu" is useful to test with larger networks.
ACTIVATION="${ACTIVATION:-tanh}"

# REWARD_STAGE is the starting reward stage. Stage 1 mostly teaches any valid
# table bounce, stage 2 adds bounce location, stage 3 adds post-bounce cup
# approach, and stage 4 emphasizes cup entry/success.
REWARD_STAGE="${REWARD_STAGE:-1}"

# CURRICULUM=auto periodically evaluates the current policy on fixed-cup
# rollouts and promotes REWARD_STAGE when it crosses stage-specific thresholds.
# CURRICULUM=manual keeps REWARD_STAGE fixed for the whole run.
CURRICULUM="${CURRICULUM:-$DEFAULT_CURRICULUM}"

# CURRICULUM_EVAL_EVERY controls the promotion-check interval in environment
# timesteps. Checks happen at PPO rollout boundaries, so they may run after the
# exact target by up to N_ENVS * N_STEPS timesteps.
CURRICULUM_EVAL_EVERY="${CURRICULUM_EVAL_EVERY:-$DEFAULT_CURRICULUM_EVAL_EVERY}"

# CURRICULUM_EVAL_EPISODES controls how many deterministic fixed-cup eval
# episodes are used for each promotion check. More episodes make promotion less
# noisy but add overhead.
CURRICULUM_EVAL_EPISODES="${CURRICULUM_EVAL_EPISODES:-$DEFAULT_CURRICULUM_EVAL_EPISODES}"

# CURRICULUM_LOG receives the promotion-check CSV. Defaults to <out>/curriculum.csv
# (inside the run dir) when unset.
CURRICULUM_LOG="${CURRICULUM_LOG:-}"

# SEED controls reproducibility for env resets and PPO initialization. Change it
# when comparing robustness across random initializations.
SEED="${SEED:-0}"

# LOG_INTERVAL controls how often SB3 prints training metrics, in PPO
# iterations. Larger values reduce terminal noise during long experiments.
LOG_INTERVAL="${LOG_INTERVAL:-10}"

# EPISODES controls how many evaluation rollouts are rendered after training.
# More episodes give better success-rate signal and more GIFs, but eval time and
# disk usage scale linearly.
EPISODES="${EPISODES:-$DEFAULT_EPISODES}"

# OUT is the run directory. Training writes policy.zip, vecnormalize.pkl,
# curriculum.csv, grid.csv, best_R*.zip, training.json, checkpoints/,
# train_rollouts/, watch_rollouts/, and tb/ inside it. Use a new OUT to avoid
# overwriting previous runs.
OUT="${OUT:-$DEFAULT_OUT}"

# EVAL_OUT_DIR receives episode_*.gif files. GIFs can be several MB each.
EVAL_OUT_DIR="${EVAL_OUT_DIR:-$DEFAULT_EVAL_OUT_DIR}"

# TRAIN_ROLLOUT_VIZ=1 periodically renders one deterministic training rollout
# from the current policy. This is like a visual checkpoint: it writes a GIF and
# a CSV with per-step reward/cumulative reward. It adds overhead, so keep the
# interval coarse for long runs. GIFs land in <out>/train_rollouts/.
TRAIN_ROLLOUT_VIZ="${TRAIN_ROLLOUT_VIZ:-$DEFAULT_TRAIN_ROLLOUT_VIZ}"

# TRAIN_ROLLOUT_VIZ_EVERY is the target interval in environment timesteps.
# Snapshots are emitted at PPO rollout boundaries, so filenames may land after
# the requested interval by up to N_ENVS * N_STEPS timesteps.
TRAIN_ROLLOUT_VIZ_EVERY="${TRAIN_ROLLOUT_VIZ_EVERY:-$DEFAULT_TRAIN_ROLLOUT_VIZ_EVERY}"

# TRAIN_ROLLOUT_VIZ_STEPS caps policy steps in each rendered rollout. The env
# still appends passive post-release flight frames within its episode limit, so
# higher values mainly matter if an episode does not release/terminate quickly.
TRAIN_ROLLOUT_VIZ_STEPS="${TRAIN_ROLLOUT_VIZ_STEPS:-300}"

# FIXED_CUP=1 evaluates the nominal cup pose for easier visual comparison.
# FIXED_CUP=0 evaluates randomized cup positions, which is better for measuring
# generalization once the policy starts working.
FIXED_CUP="${FIXED_CUP:-1}"

# RAND_STAGE selects the initial randomization sub-stage R0..R3 of the
# cup-position curriculum. When set, training enables auto-promotion of
# the randomization range as the policy improves; per-stage best
# snapshots are saved alongside the periodic checkpoints. Leave unset
# to disable the randomization curriculum (legacy fixed-cup behavior).
RAND_STAGE="${RAND_STAGE:-}"

# RAND_EVAL_EPISODES sets how many episodes the range-mode eval runs to
# decide R-stage promotion. Larger values reduce promotion noise but add
# overhead. Default 16 matches the design plan.
RAND_EVAL_EPISODES="${RAND_EVAL_EPISODES:-16}"

# Z_STAGE selects the initial pedestal-height sub-stage Z0..Z3 of the
# stacked-cup curriculum (cup elevated by 0..15cm). Requires RAND_STAGE
# to also be set so the range-eval covers the (cup_xy × pedestal) box.
# Leave unset to disable.
Z_STAGE="${Z_STAGE:-}"

# SURGICAL_RESET_OBS_RMS_SLOTS resets specific obs_rms slot stats to
# (mean=0, var=1) after warm-load. Use "20" when warm-loading v34 into
# the v35 stacked-cup env (the cup_count slot at idx 20 now holds
# pedestal_height).
SURGICAL_RESET_OBS_RMS_SLOTS="${SURGICAL_RESET_OBS_RMS_SLOTS:-}"

# WARM_START_POLICY / WARM_START_VECNORMALIZE point at a saved policy +
# vecnormalize.pkl pair to load before training begins. Required when
# fine-tuning rather than training from scratch.
WARM_START_POLICY="${WARM_START_POLICY:-}"
WARM_START_VECNORMALIZE="${WARM_START_VECNORMALIZE:-}"

# ENT_COEF overrides the PPO entropy coefficient. The v22 baseline
# trained at 0.01; warm-starts may want a small bump (0.015) for
# cup-conditioned exploration. Empty = use train_rl default.
ENT_COEF="${ENT_COEF:-}"

# ACTION_DELTA controls the per-step joint-target delta (rad) at action=1.
# v36 / no_ball_obs_v1 used 0.06; the no_ball_obs_smooth_v1 design uses
# 0.05 for a tighter motion envelope. Empty = use train_rl default (0.06).
ACTION_DELTA="${ACTION_DELTA:-}"

# ACTION_FILTER_ALPHA controls the first-order low-pass filter on the
# (post-latency) action. 1.0 = no filter (back-compat). The smooth_v1
# design uses 0.6. Empty = use train_rl default (1.0).
ACTION_FILTER_ALPHA="${ACTION_FILTER_ALPHA:-}"

# ACTION_LATENCY_RANGE controls the per-episode-sampled action latency
# range as 'lo,hi' inclusive. Each reset() draws an int from [lo, hi]
# uniformly and resizes the action queue. '0,0' = no latency
# (back-compat). The latency_robust_v1 design uses '2,4' (40-80ms at
# 50Hz). Empty = use train_rl default ('0,0').
ACTION_LATENCY_RANGE="${ACTION_LATENCY_RANGE:-}"

# OBS_JOINT_POS_NOISE_STD applies gaussian noise (rad) to joint_pos
# values entering the obs history buffer. Physics state stays clean.
# 0.0 = no noise (back-compat). latency_robust_v1 uses 0.001. Empty =
# use train_rl default (0.0).
OBS_JOINT_POS_NOISE_STD="${OBS_JOINT_POS_NOISE_STD:-}"

# JOINT_POS_HISTORY_LEN controls the number of joint_pos frames in the
# obs (current + N-1 previous). 1 = current only (back-compat).
# latency_robust_v1 uses 4. Empty = use train_rl default (1).
JOINT_POS_HISTORY_LEN="${JOINT_POS_HISTORY_LEN:-}"

# ACTION_HISTORY_LEN controls the number of previous raw policy actions
# in the obs. 0 = no action history (back-compat). latency_robust_v1
# uses 4. Empty = use train_rl default (0).
ACTION_HISTORY_LEN="${ACTION_HISTORY_LEN:-}"

# PROFILE=1 runs a short PPO timing profile before training. This is enabled by
# default for FULL_RUN=1 and disabled by default for smoke runs, because the
# profile itself costs extra wall time. PROFILE_ROLLOUTS controls how many PPO
# rollout/update iterations are sampled for the estimate.
if [[ "$FULL_RUN" == "1" ]]; then
  DEFAULT_PROFILE=1
else
  DEFAULT_PROFILE=0
fi
PROFILE="${PROFILE:-$DEFAULT_PROFILE}"
PROFILE_ROLLOUTS="${PROFILE_ROLLOUTS:-2}"

if ! command -v uv >/dev/null 2>&1; then
  echo "error: uv is required. Install it from https://docs.astral.sh/uv/" >&2
  exit 1
fi

echo "==> PPO workflow config"
echo "mode=$([[ "$FULL_RUN" == "1" ]] && echo full || echo smoke)"
echo "TIMESTEPS=$TIMESTEPS N_ENVS=$N_ENVS N_STEPS=$N_STEPS BATCH_SIZE=$BATCH_SIZE"
echo "LR=$LR LR_SCHEDULE=$LR_SCHEDULE"
echo "NET_ARCH=$NET_ARCH ACTIVATION=$ACTIVATION REWARD_STAGE=$REWARD_STAGE CURRICULUM=$CURRICULUM"
echo "CURRICULUM_EVAL_EVERY=$CURRICULUM_EVAL_EVERY CURRICULUM_EVAL_EPISODES=$CURRICULUM_EVAL_EPISODES"
echo "SEED=$SEED EPISODES=$EPISODES FIXED_CUP=$FIXED_CUP PROFILE=$PROFILE LOG_INTERVAL=$LOG_INTERVAL"
echo "OUT=$OUT (run directory)"
echo "EVAL_OUT_DIR=$EVAL_OUT_DIR"
echo "TRAIN_ROLLOUT_VIZ=$TRAIN_ROLLOUT_VIZ TRAIN_ROLLOUT_VIZ_EVERY=$TRAIN_ROLLOUT_VIZ_EVERY"
echo "TRAIN_ROLLOUT_VIZ_STEPS=$TRAIN_ROLLOUT_VIZ_STEPS"
echo "RAND_STAGE=${RAND_STAGE:-(disabled)} RAND_EVAL_EPISODES=$RAND_EVAL_EPISODES"
echo "Z_STAGE=${Z_STAGE:-(disabled)}"
echo "SURGICAL_RESET_OBS_RMS_SLOTS=${SURGICAL_RESET_OBS_RMS_SLOTS:-(none)}"
echo "WARM_START_POLICY=${WARM_START_POLICY:-(none)}"
echo "WARM_START_VECNORMALIZE=${WARM_START_VECNORMALIZE:-(none)}"
echo "ENT_COEF=${ENT_COEF:-(default)}"
echo "ACTION_DELTA=${ACTION_DELTA:-(default 0.06)}"
echo "ACTION_FILTER_ALPHA=${ACTION_FILTER_ALPHA:-(default 1.0)}"
echo "ACTION_LATENCY_RANGE=${ACTION_LATENCY_RANGE:-(default 0,0)}"
echo "OBS_JOINT_POS_NOISE_STD=${OBS_JOINT_POS_NOISE_STD:-(default 0.0)}"
echo "JOINT_POS_HISTORY_LEN=${JOINT_POS_HISTORY_LEN:-(default 1)}"
echo "ACTION_HISTORY_LEN=${ACTION_HISTORY_LEN:-(default 0)}"
echo

echo "==> Syncing dependencies"
uv sync

echo "==> Running MuJoCo smoke test"
uv run python sim/smoke_test.py

echo "==> Checking Gymnasium env"
uv run python sim/env.py

if [[ "$PROFILE" == "1" ]]; then
  echo "==> Profiling PPO rollout/update timing"
  uv run python -m sim.profile_train \
    --n-envs "$N_ENVS" \
    --n-steps "$N_STEPS" \
    --batch-size "$BATCH_SIZE" \
    --lr "$LR" \
    --lr-schedule "$LR_SCHEDULE" \
    --net-arch "$NET_ARCH" \
    --activation "$ACTIVATION" \
    --reward-stage "$REWARD_STAGE" \
    --rollouts "$PROFILE_ROLLOUTS" \
    --seed "$SEED" \
    --estimate-timesteps "$TIMESTEPS"
fi

echo "==> Training PPO"
TRAIN_START=$SECONDS
TRAIN_ARGS=(
  --timesteps "$TIMESTEPS"
  --n-envs "$N_ENVS"
  --n-steps "$N_STEPS"
  --batch-size "$BATCH_SIZE"
  --lr "$LR"
  --lr-schedule "$LR_SCHEDULE"
  --net-arch "$NET_ARCH"
  --activation "$ACTIVATION"
  --reward-stage "$REWARD_STAGE"
  --curriculum "$CURRICULUM"
  --curriculum-eval-every "$CURRICULUM_EVAL_EVERY"
  --curriculum-eval-episodes "$CURRICULUM_EVAL_EPISODES"
  --log-interval "$LOG_INTERVAL"
  --seed "$SEED"
  --out "$OUT"
)

if [[ -n "$CURRICULUM_LOG" ]]; then
  TRAIN_ARGS+=(--curriculum-log "$CURRICULUM_LOG")
fi

if [[ "$TRAIN_ROLLOUT_VIZ" == "1" ]]; then
  TRAIN_ARGS+=(
    --train-rollout-viz-every "$TRAIN_ROLLOUT_VIZ_EVERY"
    --train-rollout-viz-steps "$TRAIN_ROLLOUT_VIZ_STEPS"
  )
  if [[ "$FIXED_CUP" == "1" ]]; then
    TRAIN_ARGS+=(--train-rollout-viz-fixed-cup)
  fi
fi

if [[ -n "$RAND_STAGE" ]]; then
  TRAIN_ARGS+=(--rand-stage "$RAND_STAGE" --rand-eval-episodes "$RAND_EVAL_EPISODES")
fi

if [[ -n "$Z_STAGE" ]]; then
  TRAIN_ARGS+=(--z-stage "$Z_STAGE")
fi

if [[ -n "$SURGICAL_RESET_OBS_RMS_SLOTS" ]]; then
  TRAIN_ARGS+=(--surgical-reset-obs-rms-slots "$SURGICAL_RESET_OBS_RMS_SLOTS")
fi

if [[ -n "$WARM_START_POLICY" ]]; then
  if [[ -z "$WARM_START_VECNORMALIZE" ]]; then
    echo "error: WARM_START_POLICY set but WARM_START_VECNORMALIZE is empty" >&2
    exit 1
  fi
  TRAIN_ARGS+=(
    --warm-start-policy "$WARM_START_POLICY"
    --warm-start-vecnormalize "$WARM_START_VECNORMALIZE"
  )
fi

if [[ -n "$ENT_COEF" ]]; then
  TRAIN_ARGS+=(--ent-coef "$ENT_COEF")
fi

if [[ -n "$ACTION_DELTA" ]]; then
  TRAIN_ARGS+=(--action-delta "$ACTION_DELTA")
fi

if [[ -n "$ACTION_FILTER_ALPHA" ]]; then
  TRAIN_ARGS+=(--action-filter-alpha "$ACTION_FILTER_ALPHA")
fi

if [[ -n "$ACTION_LATENCY_RANGE" ]]; then
  TRAIN_ARGS+=(--action-latency-range "$ACTION_LATENCY_RANGE")
fi

if [[ -n "$OBS_JOINT_POS_NOISE_STD" ]]; then
  TRAIN_ARGS+=(--obs-joint-pos-noise-std "$OBS_JOINT_POS_NOISE_STD")
fi

if [[ -n "$JOINT_POS_HISTORY_LEN" ]]; then
  TRAIN_ARGS+=(--joint-pos-history-len "$JOINT_POS_HISTORY_LEN")
fi

if [[ -n "$ACTION_HISTORY_LEN" ]]; then
  TRAIN_ARGS+=(--action-history-len "$ACTION_HISTORY_LEN")
fi

uv run python -m sim.train_rl \
  "${TRAIN_ARGS[@]}"
TRAIN_SECONDS=$((SECONDS - TRAIN_START))
FINAL_REWARD_STAGE="$REWARD_STAGE"
TRAINING_METADATA="${OUT}/training.json"
if [[ -f "$TRAINING_METADATA" ]]; then
  FINAL_REWARD_STAGE="$(uv run python -c 'import json, sys; print(json.load(open(sys.argv[1]))["final_reward_stage"])' "$TRAINING_METADATA")"
fi

EVAL_ARGS=(
  --model "${OUT}/policy.zip"
  --vecnormalize "${OUT}/vecnormalize.pkl"
  --episodes "$EPISODES"
  --seed "$SEED"
  --reward-stage "$FINAL_REWARD_STAGE"
  --out-dir "$EVAL_OUT_DIR"
)

if [[ "$FIXED_CUP" == "1" ]]; then
  EVAL_ARGS+=(--fixed-cup)
fi

echo "==> Evaluating PPO and rendering GIFs"
EVAL_START=$SECONDS
uv run python -m sim.eval_rl "${EVAL_ARGS[@]}"
EVAL_SECONDS=$((SECONDS - EVAL_START))
WORKFLOW_SECONDS=$((SECONDS - WORKFLOW_START))

echo
echo "Done."
echo "Run dir: ${OUT}/"
echo "Model: ${OUT}/policy.zip"
echo "VecNormalize stats: ${OUT}/vecnormalize.pkl"
echo "Training metadata: ${OUT}/training.json"
echo "Curriculum log: ${CURRICULUM_LOG:-${OUT}/curriculum.csv}"
echo "Final reward stage: ${FINAL_REWARD_STAGE}"
echo "TensorBoard logs: ${OUT}/tb/"
echo "Eval GIFs: ${EVAL_OUT_DIR}/episode_*.gif"
if [[ "$TRAIN_ROLLOUT_VIZ" == "1" ]]; then
  echo "Training rollout snapshots: ${OUT}/train_rollouts/train_rollout_*.gif"
  echo "Training rollout rewards: ${OUT}/train_rollouts/train_rollout_*.csv"
fi
echo "Actual training time: ${TRAIN_SECONDS}s"
echo "Actual eval time: ${EVAL_SECONDS}s"
echo "Actual total workflow time: ${WORKFLOW_SECONDS}s"
