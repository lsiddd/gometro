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

import numpy as np
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

from env import MiniMetroEnv
from models import MetroFeatureExtractor

BASE_PORT = 8765
CHECKPOINT_DIR = "checkpoints"
LOG_DIR = "tb_logs"

CHECKPOINT_FREQ  = 50_000
EVAL_FREQ        = 100_000
EVAL_EPISODES    = 16       # was 4 — more episodes → lower variance

# Learning-rate schedule: linear decay from LR_START to LR_END over
# LR_DECAY_STEPS total timesteps, then held constant at LR_END.
LR_START       = 3e-4
LR_END         = 1e-5
LR_DECAY_STEPS = 10_000_000


def linear_lr_schedule(progress_remaining: float) -> float:
    """Callable passed to MaskablePPO as learning_rate.

    SB3 calls this with progress_remaining ∈ [1.0, 0.0] where 1.0 is the
    start and 0.0 is total_timesteps. We map it to a linear decay from
    LR_START to LR_END over LR_DECAY_STEPS, then hold at LR_END.
    """
    # progress_remaining goes 1→0 over total_timesteps (sys.maxsize here).
    # We approximate the "fraction done" as 1 - progress_remaining, clamped.
    frac_done = min(1.0 - progress_remaining, 1.0)
    # Scale so the full decay happens over LR_DECAY_STEPS, not sys.maxsize.
    decay_frac = min(frac_done * sys.maxsize / LR_DECAY_STEPS, 1.0)
    lr = LR_START + decay_frac * (LR_END - LR_START)
    return float(np.clip(lr, LR_END, LR_START))


class SolverBaselineCallback(BaseCallback):
    """Periodically evaluates the heuristic solver alongside the RL agent and
    logs both scores to TensorBoard so progress is directly comparable.
    """

    def __init__(self, city: str, base_port: int, n_envs_offset: int,
                 eval_freq: int = EVAL_FREQ, n_episodes: int = 4,
                 verbose: int = 0):
        super().__init__(verbose)
        self.city = city
        self.base_port = base_port
        self.n_envs_offset = n_envs_offset
        self.eval_freq = eval_freq
        self.n_episodes = n_episodes
        self._last_eval = 0

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_eval < self.eval_freq:
            return True
        self._last_eval = self.num_timesteps
        self._run_solver_baseline()
        return True

    def _run_solver_baseline(self) -> None:
        import requests

        scores = []
        for ep in range(self.n_episodes):
            port = self.base_port + self.n_envs_offset + ep
            env = MiniMetroEnv(port=port, city=self.city, managed=True)
            try:
                obs, _ = env.reset()
                done = False
                while not done:
                    # Ask the server for the solver's preferred action.
                    try:
                        resp = requests.get(
                            f"http://localhost:{port}/solver_act", timeout=5
                        )
                        action = resp.json()["action"]
                    except Exception:
                        action = [0, 0, 0, 0]  # fallback NoOp
                    obs, _, done, _, info = env.step(np.array(action))
                scores.append(info.get("score", 0))
            finally:
                env.close()

        if scores:
            mean_score = float(np.mean(scores))
            self.logger.record("eval/solver_score_mean", mean_score)
            if self.verbose:
                print(f"[solver baseline] score={mean_score:.1f} "
                      f"over {len(scores)} episodes")


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
        model = MaskablePPO.load(
            args.resume, env=vec_env, tensorboard_log=LOG_DIR,
            learning_rate=linear_lr_schedule,
            verbose=1,
        )
    else:
        model = MaskablePPO(
            "MlpPolicy",
            vec_env,
            n_steps=4096,       # was 2048 — more data per update
            batch_size=512,     # was 256
            n_epochs=10,
            gamma=0.995,
            gae_lambda=0.95,
            learning_rate=linear_lr_schedule,
            ent_coef=0.01,
            clip_range=0.2,
            max_grad_norm=0.5,
            tensorboard_log=LOG_DIR,
            policy_kwargs={
                "features_extractor_class": MetroFeatureExtractor,
                "net_arch": [256, 256],
            },
            verbose=1,
        )

    # Solver baseline runs on ports after training + eval envs.
    solver_port_offset = n + 2
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
        SolverBaselineCallback(
            city=args.city,
            base_port=BASE_PORT,
            n_envs_offset=solver_port_offset,
            eval_freq=EVAL_FREQ,
            n_episodes=4,
        ),
    ]

    interrupted = False
    original_sigint = signal.getsignal(signal.SIGINT)

    def _handle_sigint(sig, frame):
        nonlocal interrupted
        if not interrupted:
            interrupted = True
            print("\n[train] Interrupt received — finishing current iteration then saving...")
        else:
            signal.signal(signal.SIGINT, original_sigint)
            sys.exit(1)

    signal.signal(signal.SIGINT, _handle_sigint)

    CHUNK = 500_000  # timesteps per learn() call — sets the progress bar horizon

    print("[train] All envs ready, starting learning loop...")
    first_chunk = True
    try:
        while not interrupted:
            model.learn(
                total_timesteps=CHUNK,
                callback=callbacks,
                reset_num_timesteps=first_chunk and not args.resume,
                progress_bar=True,
            )
            first_chunk = False
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
    parser.add_argument("--resume", type=str, default="",
                        help="Path to a checkpoint .zip to continue training from")
    train(parser.parse_args())
