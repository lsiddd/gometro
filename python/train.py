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

from constants import TRAIN_BASE_PORT
from env import MiniMetroEnv
from models import MetroFeatureExtractor
from policy import MetroPolicy

BASE_PORT = TRAIN_BASE_PORT
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


class LinearLRSchedule:
    """LR schedule that decays linearly from LR_START to LR_END over
    LR_DECAY_STEPS total environment steps, then holds at LR_END.

    Uses model.num_timesteps directly so it works correctly across multiple
    model.learn() chunks (where progress_remaining resets each chunk).
    Call `schedule.bind(model)` after model creation.
    """

    def __init__(self):
        self._model = None

    def bind(self, model) -> "LinearLRSchedule":
        self._model = model
        return self

    # Exclude the model reference from pickling (the model holds SubprocVecEnv
    # which contains AuthenticationString objects that cannot be serialised).
    # After loading a checkpoint, call bind(model) again to restore the link.
    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_model"] = None
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)

    def __call__(self, _progress_remaining: float) -> float:
        steps = self._model.num_timesteps if self._model is not None else 0
        decay_frac = min(steps / LR_DECAY_STEPS, 1.0)
        lr = LR_START + decay_frac * (LR_END - LR_START)
        return float(np.clip(lr, LR_END, LR_START))


lr_schedule = LinearLRSchedule()


class CurriculumCallback(BaseCallback):
    """Adjusts spawn-rate difficulty across all training environments based on
    recent episode survival.

    Difficulty is expressed as a ``spawn_rate_factor`` passed to the Go server
    on each episode reset: values > 1.0 stretch spawn intervals (fewer
    passengers per minute), making the game easier.

    Schedule (index 0 = easiest):
        [4.0, 3.0, 2.0, 1.5, 1.25, 1.0]

    Promotion rule: rolling mean weeks survived over the last ``window``
    completed episodes exceeds ``promote_threshold``  → advance one level.
    Demotion rule:  rolling mean falls below ``demote_threshold`` → retreat one
    level (floor = 0, ceiling = last index = normal difficulty).
    """

    SCHEDULE: list[float] = [4.0, 3.0, 2.0, 1.5, 1.25, 1.0]

    def __init__(
        self,
        vec_env,
        window: int = 50,
        promote_threshold: float = 10.0,
        demote_threshold: float = 4.0,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self._vec_env = vec_env
        self._window = window
        self._promote = promote_threshold
        self._demote = demote_threshold
        self._level = 0                   # index into SCHEDULE; 0 = easiest
        self._episode_weeks: list[float] = []
        self._set_difficulty(self._level)

    # ── SB3 callback interface ────────────────────────────────────────────────

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])
        for done, info in zip(dones, infos):
            if done:
                self._episode_weeks.append(float(info.get("week", 0)))

        if len(self._episode_weeks) >= self._window:
            self._episode_weeks = self._episode_weeks[-self._window:]
            mean_weeks = float(np.mean(self._episode_weeks))
            self.logger.record("curriculum/mean_weeks_survived", mean_weeks)
            self.logger.record("curriculum/level", self._level)
            self.logger.record(
                "curriculum/spawn_rate_factor", self.SCHEDULE[self._level]
            )

            prev_level = self._level
            max_level = len(self.SCHEDULE) - 1

            if mean_weeks >= self._promote and self._level < max_level:
                self._level += 1
                self._episode_weeks.clear()
                if self.verbose:
                    print(
                        f"[curriculum] ↑ level {prev_level}→{self._level}  "
                        f"(factor={self.SCHEDULE[self._level]:.2f}  "
                        f"mean_weeks={mean_weeks:.1f})"
                    )
                self._set_difficulty(self._level)

            elif mean_weeks < self._demote and self._level > 0:
                self._level -= 1
                self._episode_weeks.clear()
                if self.verbose:
                    print(
                        f"[curriculum] ↓ level {prev_level}→{self._level}  "
                        f"(factor={self.SCHEDULE[self._level]:.2f}  "
                        f"mean_weeks={mean_weeks:.1f})"
                    )
                self._set_difficulty(self._level)

        return True

    def _set_difficulty(self, level: int) -> None:
        factor = self.SCHEDULE[level]
        self._vec_env.env_method("set_difficulty", factor)


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
        from rl.proto import minimetro_pb2 as pb

        scores = []
        for ep in range(self.n_episodes):
            port = self.base_port + self.n_envs_offset + ep
            env = MiniMetroEnv(port=port, city=self.city, managed=True)
            try:
                obs, _ = env.reset()
                done = False
                while not done:
                    try:
                        action_resp = env._stub.SolverAct(pb.Empty(), timeout=5)
                        action = list(action_resp.action)
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
            learning_rate=lr_schedule,
            verbose=1,
        )
    else:
        model = MaskablePPO(
            MetroPolicy,
            vec_env,
            n_steps=16384,
            batch_size=2048,
            n_epochs=10,
            gamma=0.99,         # was 0.995 — shorter effective horizon speeds early credit assignment
            gae_lambda=0.95,
            learning_rate=lr_schedule,
            ent_coef=0.05,      # was 0.01 — more exploration in large MultiDiscrete space
            clip_range=0.2,
            max_grad_norm=0.5,
            tensorboard_log=LOG_DIR,
            policy_kwargs={
                "features_extractor_class": MetroFeatureExtractor,
                "net_arch": [256, 256],
            },
            verbose=1,
        )
    lr_schedule.bind(model)

    # Solver baseline runs on ports after training + eval envs.
    solver_port_offset = n + 2
    callbacks = [
        CurriculumCallback(vec_env, window=50, promote_threshold=10.0,
                           demote_threshold=4.0, verbose=1),
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
    parser.add_argument("--n-envs", type=int, default=16)
    parser.add_argument("--city", type=str, default="london")
    parser.add_argument("--resume", type=str, default="",
                        help="Path to a checkpoint .zip to continue training from")
    train(parser.parse_args())
