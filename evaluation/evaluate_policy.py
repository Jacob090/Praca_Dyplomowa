import argparse
import os

import numpy as np
from stable_baselines3 import SAC

from utils.wrappers import load_config, make_env, resolve_env_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-config", type=str, default="configs/env.yaml")
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--stage", type=str, default="place", choices=["reach", "grasp", "place"])
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    env_cfg = load_config(args.env_config)
    env_cfg = resolve_env_config(env_cfg, base_dir=os.getcwd())

    render_mode = "human" if args.render else None
    env = make_env(env_cfg, stage=args.stage, render_mode=render_mode)

    model = SAC.load(args.model_path, env=env)

    successes = 0
    episode_rewards = []
    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        done = False
        truncated = False
        total_reward = 0.0
        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            total_reward += float(reward)
            if args.render:
                env.render()
        episode_rewards.append(total_reward)
        if info.get("success", False):
            successes += 1

    success_rate = successes / float(args.episodes)
    print(f"Success rate: {success_rate:.2f}")
    print(f"Average reward: {np.mean(episode_rewards):.2f}")

    env.close()


if __name__ == "__main__":
    main()
