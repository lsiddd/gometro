"""
Training script for the Mini Metro RL agent.

Uses MaskablePPO from sb3-contrib with multiple parallel Go environments.

Usage:
    uv run python train.py [--n-envs 8] [--timesteps 5000000] [--city london]
"""
from __future__ import annotations

import argparse
import os

import numpy as np
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

from env import MiniMetroEnv, NUM_ACTIONS, OBS_DIM

BASE_PORT = 8765
CHECKPOINT_DIR = "checkpoints"
LOG_DIR = "tb_logs"


def make_env(rank: int, city: str, base_port: int):
    """Factory that creates a single environment for SubprocVecEnv."""
    def _init():
        env = MiniMetroEnv(port=base_port + rank, city=city, managed=True)
        return env
    return _init


def train(args: argparse.Namespace):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    print(f"Launching {args.n_envs} environments on ports {BASE_PORT}–{BASE_PORT + args.n_envs - 1}")
    vec_env = SubprocVecEnv(
        [make_env(i, args.city, BASE_PORT) for i in range(args.n_envs)]
    )
    vec_env = VecMonitor(vec_env)

    # Eval env on a separate port range.
    eval_env = SubprocVecEnv(
        [make_env(i, args.city, BASE_PORT + args.n_envs) for i in range(2)]
    )
    eval_env = VecMonitor(eval_env)

    model = MaskablePPO(
        "MlpPolicy",
        vec_env,
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        gamma=0.995,        # long-horizon game — discount future rewards slowly
        gae_lambda=0.95,
        learning_rate=3e-4,
        ent_coef=0.01,      # exploration encouragement
        clip_range=0.2,
        max_grad_norm=0.5,
        tensorboard_log=LOG_DIR,
        policy_kwargs={"net_arch": [256, 256]},
        verbose=1,
    )

    callbacks = [
        CheckpointCallback(
            save_freq=max(50_000 // args.n_envs, 1),
            save_path=CHECKPOINT_DIR,
            name_prefix="minimetro_ppo",
        ),
        MaskableEvalCallback(
            eval_env,
            n_eval_episodes=4,
            eval_freq=max(100_000 // args.n_envs, 1),
            best_model_save_path=CHECKPOINT_DIR,
            log_path=LOG_DIR,
            deterministic=True,
        ),
    ]

    model.learn(
        total_timesteps=args.timesteps,
        callback=callbacks,
        progress_bar=True,
    )

    final_path = os.path.join(CHECKPOINT_DIR, "minimetro_final")
    model.save(final_path)
    print(f"Training complete. Model saved to {final_path}.zip")

    vec_env.close()
    eval_env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--timesteps", type=int, default=5_000_000)
    parser.add_argument("--city", type=str, default="london")
    train(parser.parse_args())
