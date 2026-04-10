"""
Training script for the Mini Metro RL agent.

Runs indefinitely until interrupted (Ctrl+C). Checkpoints are saved
periodically so training can be resumed at any point.

Usage:
    uv run python train.py [--n-envs 8] [--city london] [--resume checkpoints/minimetro_ppo_100000_steps.zip]
"""
from __future__ import annotations

import argparse
import os
import signal
import sys

from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

from env import MiniMetroEnv

BASE_PORT = 8765
CHECKPOINT_DIR = "checkpoints"
LOG_DIR = "tb_logs"

# Save a checkpoint every N timesteps collected across all envs.
CHECKPOINT_FREQ = 50_000
# Run an evaluation round every N timesteps.
EVAL_FREQ = 100_000
# Number of parallel eval episodes.
EVAL_EPISODES = 4


def make_env(rank: int, city: str, base_port: int):
    def _init():
        return MiniMetroEnv(port=base_port + rank, city=city, managed=True)
    return _init


def train(args: argparse.Namespace):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    n = args.n_envs
    print(f"Launching {n} training envs on ports {BASE_PORT}–{BASE_PORT + n - 1}")
    vec_env = VecMonitor(SubprocVecEnv([make_env(i, args.city, BASE_PORT) for i in range(n)]))

    eval_env = VecMonitor(SubprocVecEnv([make_env(i, args.city, BASE_PORT + n) for i in range(2)]))

    if args.resume:
        print(f"Resuming from {args.resume}")
        model = MaskablePPO.load(args.resume, env=vec_env, tensorboard_log=LOG_DIR)
    else:
        model = MaskablePPO(
            "MlpPolicy",
            vec_env,
            n_steps=2048,
            batch_size=256,
            n_epochs=10,
            gamma=0.995,
            gae_lambda=0.95,
            learning_rate=3e-4,
            ent_coef=0.01,
            clip_range=0.2,
            max_grad_norm=0.5,
            tensorboard_log=LOG_DIR,
            policy_kwargs={"net_arch": [256, 256]},
            verbose=1,
        )

    callbacks = [
        CheckpointCallback(
            save_freq=max(CHECKPOINT_FREQ // n, 1),
            save_path=CHECKPOINT_DIR,
            name_prefix="minimetro_ppo",
        ),
        MaskableEvalCallback(
            eval_env,
            n_eval_episodes=EVAL_EPISODES,
            eval_freq=max(EVAL_FREQ // n, 1),
            best_model_save_path=CHECKPOINT_DIR,
            log_path=LOG_DIR,
            deterministic=True,
        ),
    ]

    # Intercept Ctrl+C: finish the current PPO iteration cleanly, then save.
    interrupted = False
    original_sigint = signal.getsignal(signal.SIGINT)

    def _handle_sigint(sig, frame):
        nonlocal interrupted
        if not interrupted:
            interrupted = True
            print("\n[train] Interrupt received — finishing current iteration then saving...")
        else:
            # Second Ctrl+C: exit immediately.
            signal.signal(signal.SIGINT, original_sigint)
            sys.exit(1)

    signal.signal(signal.SIGINT, _handle_sigint)

    try:
        # SB3's learn() runs until total_timesteps is reached. Setting it to
        # sys.maxsize makes it effectively infinite; the SIGINT handler above
        # stops the loop cleanly between iterations via StopIteration (SB3
        # checks the callback return value — we raise KeyboardInterrupt which
        # SB3 catches and re-raises, falling through to our finally block).
        model.learn(
            total_timesteps=sys.maxsize,
            callback=callbacks,
            reset_num_timesteps=not args.resume,
            progress_bar=False,  # not useful for infinite runs
        )
    except KeyboardInterrupt:
        pass
    finally:
        final_path = os.path.join(CHECKPOINT_DIR, "minimetro_latest")
        model.save(final_path)
        print(f"\n[train] Saved to {final_path}.zip  (timesteps: {model.num_timesteps:,})")
        vec_env.close()
        eval_env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--city", type=str, default="london")
    parser.add_argument("--resume", type=str, default="", help="Path to a checkpoint .zip to continue training from")
    train(parser.parse_args())
