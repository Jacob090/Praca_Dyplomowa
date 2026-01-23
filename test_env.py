import argparse
import os

import numpy as np

from utils.wrappers import load_config, make_env, resolve_env_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-config", type=str, default="configs/env.yaml")
    parser.add_argument("--stage", type=str, default="reach", choices=["reach", "grasp", "place"])
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()

    env_cfg = load_config(args.env_config)
    env_cfg = resolve_env_config(env_cfg, base_dir=os.getcwd())
    render_mode = "human" if args.render else None
    env = make_env(env_cfg, stage=args.stage, render_mode=render_mode)

    obs, info = env.reset(seed=123)
    print("Initial info:", info)
    for _ in range(args.steps):
        action = env.action_space.sample().astype(np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        if args.render:
            env.render()
        if terminated or truncated:
            obs, info = env.reset()
    env.close()


if __name__ == "__main__":
    main()
