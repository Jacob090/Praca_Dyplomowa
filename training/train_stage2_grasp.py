import argparse
import os

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

from utils.wrappers import load_config, make_env, resolve_env_config


def build_envs(env_cfg, stage):
    train_env = DummyVecEnv([lambda: make_env(env_cfg, stage)])
    eval_env = DummyVecEnv([lambda: make_env(env_cfg, stage)])
    train_env = VecMonitor(train_env)
    eval_env = VecMonitor(eval_env)
    return train_env, eval_env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-config", type=str, default="configs/env.yaml")
    parser.add_argument("--sac-config", type=str, default="configs/sac_stage2.yaml")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--pretrained", type=str, default=None)
    args = parser.parse_args()

    env_cfg = load_config(args.env_config)
    env_cfg = resolve_env_config(env_cfg, base_dir=os.getcwd())

    sac_cfg = load_config(args.sac_config)

    log_dir = sac_cfg["log_dir"]
    checkpoints_dir = sac_cfg["checkpoints_dir"]
    eval_dir = sac_cfg["eval_dir"]
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(eval_dir, exist_ok=True)

    train_env, eval_env = build_envs(env_cfg, stage="grasp")

    model_kwargs = dict(
        learning_rate=sac_cfg["sac"]["learning_rate"],
        buffer_size=sac_cfg["sac"]["buffer_size"],
        batch_size=sac_cfg["sac"]["batch_size"],
        tau=sac_cfg["sac"]["tau"],
        gamma=sac_cfg["sac"]["gamma"],
        train_freq=sac_cfg["sac"]["train_freq"],
        gradient_steps=sac_cfg["sac"]["gradient_steps"],
        learning_starts=sac_cfg["sac"]["learning_starts"],
        ent_coef=sac_cfg["sac"]["ent_coef"],
        target_entropy=sac_cfg["sac"]["target_entropy"],
        policy_kwargs=sac_cfg["policy_kwargs"],
        tensorboard_log=log_dir,
        seed=sac_cfg["seed"],
        device=sac_cfg["device"],
    )

    if args.resume:
        model = SAC.load(args.resume, env=train_env, device=sac_cfg["device"], tensorboard_log=log_dir)
        reset_num_timesteps = False
    else:
        pretrained = args.pretrained or sac_cfg["pretrained_path"]
        model = SAC.load(pretrained, env=train_env, device=sac_cfg["device"], tensorboard_log=log_dir)
        reset_num_timesteps = False

    checkpoint_callback = CheckpointCallback(
        save_freq=sac_cfg["checkpoint_freq"],
        save_path=checkpoints_dir,
        name_prefix="stage2_grasp",
    )
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=eval_dir,
        log_path=eval_dir,
        eval_freq=sac_cfg["eval_freq"],
        n_eval_episodes=sac_cfg["n_eval_episodes"],
        deterministic=True,
        render=False,
    )

    model.learn(
        total_timesteps=sac_cfg["total_timesteps"],
        callback=[checkpoint_callback, eval_callback],
        reset_num_timesteps=reset_num_timesteps,
        progress_bar=True,
    )

    model.save(os.path.join(log_dir, "final_model"))


if __name__ == "__main__":
    main()
