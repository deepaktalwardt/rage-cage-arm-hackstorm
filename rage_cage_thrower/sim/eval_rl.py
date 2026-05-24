"""Evaluate a trained PPO policy and render rollout GIFs."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from sim.env import RageCageEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("sim/_rl_out/ppo_thrower.zip"))
    parser.add_argument(
        "--vecnormalize",
        type=Path,
        default=Path("sim/_rl_out/ppo_thrower.vecnormalize.pkl"),
    )
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--reward-stage", type=int, choices=(1, 2, 3, 4), default=3)
    parser.add_argument("--fixed-cup", action="store_true")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--out-dir", type=Path, default=Path("sim/_rl_eval"))
    return parser.parse_args()


def make_env(args: argparse.Namespace) -> RageCageEnv:
    return RageCageEnv(
        randomize_cup=not args.fixed_cup,
        reward_stage=args.reward_stage,
        render_mode="rgb_array",
        image_width=args.width,
        image_height=args.height,
    )


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    env = DummyVecEnv([lambda: make_env(args)])
    if args.vecnormalize.exists():
        env = VecNormalize.load(args.vecnormalize, env)
        env.training = False
        env.norm_reward = False

    model = PPO.load(args.model, env=env)

    for episode_idx in range(args.episodes):
        obs = env.reset()
        frames = []
        done = np.array([False])
        total_reward = 0.0
        final_info = {}

        while not done[0]:
            action, _ = model.predict(obs, deterministic=not args.stochastic)
            obs, reward, done, infos = env.step(action)
            total_reward += float(reward[0])
            final_info = infos[0]
            for passive_frame in final_info.get("passive_render_frames", []):
                frames.append(Image.fromarray(passive_frame))
            frame = env.env_method("render")[0]
            if frame is not None:
                frames.append(Image.fromarray(frame))

        out_path = args.out_dir / f"episode_{episode_idx:03d}.gif"
        if frames:
            frames[0].save(
                out_path,
                save_all=True,
                append_images=frames[1:],
                duration=20,
                loop=0,
            )
        print(
            f"episode={episode_idx} reward={total_reward:.2f} "
            f"success={final_info.get('success')} bounce_count={final_info.get('bounce_count')} "
            f"cup_dist={final_info.get('cup_dist'):.3f} gif={out_path}"
        )

    env.close()


if __name__ == "__main__":
    main()
