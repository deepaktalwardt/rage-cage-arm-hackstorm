"""Profile PPO rollout collection vs network update time."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from statistics import mean
from types import MethodType

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

from sim.env import RageCageEnv
from sim.train_rl import (
    ACTIVATIONS,
    NET_ARCHS,
    _parse_latency_range,
    learning_rate_arg,
    policy_kwargs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--n-steps", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--lr-schedule", choices=("constant", "linear"), default="constant")
    parser.add_argument("--net-arch", choices=tuple(NET_ARCHS), default="default")
    parser.add_argument("--activation", choices=tuple(ACTIVATIONS), default="tanh")
    parser.add_argument("--reward-stage", type=int, choices=(1, 2, 3, 4), default=3)
    parser.add_argument("--rollouts", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--estimate-timesteps", type=int, default=50_000)
    parser.add_argument("--out-dir", type=Path, default=Path("sim/_rl_profile"))
    parser.add_argument("--action-filter-alpha", type=float, default=1.0)
    parser.add_argument("--action-latency-range", type=str, default="0,0")
    parser.add_argument("--obs-joint-pos-noise-std", type=float, default=0.0)
    parser.add_argument("--joint-pos-history-len", type=int, default=1)
    parser.add_argument("--action-history-len", type=int, default=0)
    return parser.parse_args()


def fmt(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.2f}s"
    minutes, rem = divmod(seconds, 60)
    return f"{int(minutes)}m {rem:.1f}s"


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    env = make_vec_env(
        RageCageEnv,
        n_envs=args.n_envs,
        seed=args.seed,
        env_kwargs={
            "reward_stage": args.reward_stage,
            "action_filter_alpha": args.action_filter_alpha,
            "action_latency_range": _parse_latency_range(args.action_latency_range),
            "obs_joint_pos_noise_std": args.obs_joint_pos_noise_std,
            "joint_pos_history_len": args.joint_pos_history_len,
            "action_history_len": args.action_history_len,
        },
    )
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=learning_rate_arg(args.lr, args.lr_schedule),
        policy_kwargs=policy_kwargs(args.net_arch, args.activation),
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        gamma=0.99,
        gae_lambda=0.95,
        verbose=0,
        seed=args.seed,
    )

    rollout_times: list[float] = []
    update_times: list[float] = []
    original_collect_rollouts = model.collect_rollouts
    original_train = model.train

    def timed_collect_rollouts(self: PPO, *collect_args, **collect_kwargs):
        start = time.perf_counter()
        result = original_collect_rollouts(*collect_args, **collect_kwargs)
        rollout_times.append(time.perf_counter() - start)
        return result

    def timed_train(self: PPO):
        start = time.perf_counter()
        result = original_train()
        update_times.append(time.perf_counter() - start)
        return result

    model.collect_rollouts = MethodType(timed_collect_rollouts, model)
    model.train = MethodType(timed_train, model)

    rollout_size = args.n_envs * args.n_steps
    total_timesteps = rollout_size * args.rollouts
    total_start = time.perf_counter()
    model.learn(total_timesteps=total_timesteps)
    total_time = time.perf_counter() - total_start

    env.close()

    rollout_mean = mean(rollout_times)
    update_mean = mean(update_times)
    iteration_mean = rollout_mean + update_mean
    measured_steps_per_second = total_timesteps / total_time
    estimated_seconds = args.estimate_timesteps / measured_steps_per_second

    print("PPO training profile")
    print(f"n_envs={args.n_envs}")
    print(f"n_steps={args.n_steps}")
    print(f"batch_size={args.batch_size}")
    print(f"lr={args.lr}")
    print(f"lr_schedule={args.lr_schedule}")
    print(f"net_arch={args.net_arch}")
    print(f"activation={args.activation}")
    print(f"reward_stage={args.reward_stage}")
    print(f"rollout_size={rollout_size} transitions")
    print(f"profiled_rollouts={args.rollouts}")
    print()
    for idx, (rollout_s, update_s) in enumerate(zip(rollout_times, update_times, strict=True), start=1):
        total_s = rollout_s + update_s
        rollout_pct = 100 * rollout_s / total_s
        update_pct = 100 * update_s / total_s
        print(
            f"iteration {idx}: rollout={fmt(rollout_s)} ({rollout_pct:.1f}%) "
            f"update={fmt(update_s)} ({update_pct:.1f}%) total={fmt(total_s)}"
        )
    print()
    print(f"mean_rollout_time={fmt(rollout_mean)}")
    print(f"mean_update_time={fmt(update_mean)}")
    print(f"mean_iteration_time={fmt(iteration_mean)}")
    print(f"measured_total_time={fmt(total_time)}")
    print(f"measured_steps_per_second={measured_steps_per_second:.0f}")
    print(
        f"estimated_time_for_{args.estimate_timesteps}_timesteps="
        f"{fmt(estimated_seconds)}"
    )


if __name__ == "__main__":
    main()
