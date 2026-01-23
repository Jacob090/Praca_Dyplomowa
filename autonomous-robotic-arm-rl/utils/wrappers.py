import os
import yaml
import numpy as np
import gymnasium as gym

from envs.arm4dof_pick_place_env import Arm4DOFPickPlaceEnv


class ClipActionWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)

    def action(self, action):
        return np.clip(action, self.env.action_space.low, self.env.action_space.high)


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_env_config(config, base_dir):
    resolved = dict(config)
    scene_path = resolved.get("scene_path", "")
    if scene_path and not os.path.isabs(scene_path):
        resolved["scene_path"] = os.path.normpath(os.path.join(base_dir, scene_path))
    return resolved


def make_env(config, stage, render_mode=None):
    env = Arm4DOFPickPlaceEnv(config=config, stage=stage, render_mode=render_mode)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = ClipActionWrapper(env)
    return env
