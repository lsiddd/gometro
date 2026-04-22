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
import time

# OMP_NUM_THREADS must be set before the OpenMP runtime is loaded (i.e., before
# `import torch` / `import numpy`).  torch.set_num_threads() covers the PyTorch
# intra-op pool, but the underlying BLAS (OpenBLAS / MKL) reads its thread count
# from the env at load time.  Setting both ensures all BLAS paths are covered.
_N_CPU = os.environ.get("TRAIN_TORCH_THREADS", str(min(os.cpu_count() or 1, 8)))
os.environ.setdefault("OMP_NUM_THREADS",      _N_CPU)
os.environ.setdefault("MKL_NUM_THREADS",      _N_CPU)
os.environ.setdefault("OPENBLAS_NUM_THREADS", _N_CPU)

import torch
import numpy as np

# Redundant with the env vars above but acts as a safety net for runtime
# changes and covers PyTorch's own thread pool (separate from BLAS threads).
torch.set_num_threads(int(_N_CPU))
# NOTE: set_num_interop_threads must NOT be called here — it raises RuntimeError
# if invoked after torch is already initialized (which it is via the imports above).

from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.vec_env import VecMonitor

from constants import TRAIN_BASE_PORT
from env import MiniMetroVecEnv
from models import MetroFeatureExtractor
from policy import MetroPolicy

BASE_PORT = TRAIN_BASE_PORT
CHECKPOINT_DIR = "checkpoints"
LOG_DIR = "tb_logs"

CHECKPOINT_FREQ  = 50_000
EVAL_FREQ        = 100_000
EVAL_EPISODES    = 16       # was 4 — more episodes → lower variance
TRACE_INTERVAL_S = 10.0

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


def trace(msg: str) -> None:
    print(f"[trace {time.strftime('%H:%M:%S')}] {msg}", flush=True)


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


class TraceCallback(BaseCallback):
    """Low-noise phase tracing for PPO collection/update/eval stalls."""

    def __init__(self, interval_s: float = TRACE_INTERVAL_S, verbose: int = 0):
        super().__init__(verbose)
        self.interval_s = interval_s
        self._last = 0.0
        self._rollout_started = 0.0
        self._last_steps = 0

    def _on_training_start(self) -> None:
        self._last = time.perf_counter()
        self._last_steps = self.num_timesteps
        trace(
            "training_start "
            f"torch_threads={torch.get_num_threads()} "
            f"interop_threads={torch.get_num_interop_threads()} "
            f"omp={os.environ.get('OMP_NUM_THREADS')} "
            f"mkl={os.environ.get('MKL_NUM_THREADS')} "
            f"openblas={os.environ.get('OPENBLAS_NUM_THREADS')}"
        )

    def _on_rollout_start(self) -> None:
        self._rollout_started = time.perf_counter()
        trace(f"rollout_start timesteps={self.num_timesteps:,}")

    def _on_step(self) -> bool:
        now = time.perf_counter()
        if now - self._last >= self.interval_s:
            delta_steps = self.num_timesteps - self._last_steps
            fps = delta_steps / max(now - self._last, 1e-9)
            trace(
                "rollout_progress "
                f"timesteps={self.num_timesteps:,} "
                f"delta_steps={delta_steps:,} "
                f"fps={fps:.1f}"
            )
            self._last = now
            self._last_steps = self.num_timesteps
        return True

    def _on_rollout_end(self) -> None:
        elapsed = time.perf_counter() - self._rollout_started
        trace(
            "rollout_end/update_start "
            f"timesteps={self.num_timesteps:,} "
            f"rollout_elapsed={elapsed:.2f}s"
        )

    def _on_training_end(self) -> None:
        trace(f"training_end timesteps={self.num_timesteps:,}")


class TracedMaskablePPO(MaskablePPO):
    def train(self) -> None:
        updates_before = getattr(self, "_n_updates", 0)
        t0 = time.perf_counter()
        buffer_size = getattr(self.rollout_buffer, "buffer_size", "?")
        n_envs = getattr(self.rollout_buffer, "n_envs", "?")
        trace(
            "ppo_update_start "
            f"timesteps={self.num_timesteps:,} "
            f"updates_before={updates_before} "
            f"buffer_size={buffer_size} n_envs={n_envs} "
            f"batch_size={self.batch_size} n_epochs={self.n_epochs} "
            f"torch_threads={torch.get_num_threads()}"
        )
        super().train()
        trace(
            "ppo_update_end "
            f"timesteps={self.num_timesteps:,} "
            f"updates_after={getattr(self, '_n_updates', '?')} "
            f"elapsed={time.perf_counter() - t0:.2f}s"
        )


class TracedCheckpointCallback(CheckpointCallback):
    def _on_step(self) -> bool:
        should_save = self.n_calls % self.save_freq == 0
        t0 = time.perf_counter()
        if should_save:
            trace(f"checkpoint_start n_calls={self.n_calls} timesteps={self.num_timesteps:,}")
        result = super()._on_step()
        if should_save:
            trace(f"checkpoint_end elapsed={time.perf_counter() - t0:.2f}s")
        return result


class TracedMaskableEvalCallback(MaskableEvalCallback):
    def _on_step(self) -> bool:
        should_eval = self.eval_freq > 0 and self.n_calls % self.eval_freq == 0
        t0 = time.perf_counter()
        if should_eval:
            trace(f"eval_start n_calls={self.n_calls} timesteps={self.num_timesteps:,}")
        result = super()._on_step()
        if should_eval:
            trace(f"eval_end elapsed={time.perf_counter() - t0:.2f}s")
        return result


def train(args: argparse.Namespace):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    n = args.n_envs
    print(f"Launching {n} training envs on port {BASE_PORT}")

    trace(
        "startup "
        f"n_envs={n} city={args.city} "
        f"torch_threads={torch.get_num_threads()} "
        f"trace_interval={args.trace_interval}s"
    )

    vec_env = MiniMetroVecEnv(
        n_envs=n,
        city=args.city,
        port=BASE_PORT,
        managed=True,
        trace_interval=args.trace_interval,
    )
    vec_env = VecMonitor(vec_env)

    eval_env = MiniMetroVecEnv(
        n_envs=2,
        city=args.city,
        port=BASE_PORT + 1,
        managed=True,
        trace_interval=args.trace_interval,
    )
    eval_env = VecMonitor(eval_env)

    if args.resume:
        print(f"Resuming from {args.resume}")
        model = TracedMaskablePPO.load(
            args.resume, env=vec_env, tensorboard_log=LOG_DIR,
            learning_rate=lr_schedule,
            verbose=1,
        )
    else:
        model = TracedMaskablePPO(
            MetroPolicy,
            vec_env,
            n_steps=2048,
            batch_size=2048,
            n_epochs=4,
            gamma=0.999,        # Increased from 0.99 to 0.999 for much longer effective horizon
            gae_lambda=0.95,
            learning_rate=lr_schedule,
            ent_coef=0.005,     # Decreased from 0.05 to reduce excessive random action noise
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

    callbacks = [
        TraceCallback(interval_s=args.trace_interval),
        CurriculumCallback(vec_env, window=50, promote_threshold=10.0,
                           demote_threshold=4.0, verbose=1),
        TracedCheckpointCallback(
            save_freq=max(CHECKPOINT_FREQ // n, 1),
            save_path=CHECKPOINT_DIR,
            name_prefix="minimetro_ppo",
        ),
        TracedMaskableEvalCallback(
            eval_env,
            n_eval_episodes=EVAL_EPISODES,
            eval_freq=max(EVAL_FREQ // n, 1),
            best_model_save_path=CHECKPOINT_DIR,
            log_path=LOG_DIR,
            deterministic=True,
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
            learn_start = time.perf_counter()
            trace(
                "learn_chunk_start "
                f"current_timesteps={model.num_timesteps:,} "
                f"target_chunk={CHUNK:,}"
            )
            model.learn(
                total_timesteps=CHUNK,
                callback=callbacks,
                reset_num_timesteps=first_chunk and not args.resume,
                progress_bar=True,
            )
            trace(
                "learn_chunk_end "
                f"timesteps={model.num_timesteps:,} "
                f"elapsed={time.perf_counter() - learn_start:.2f}s"
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
    parser.add_argument("--trace-interval", type=float, default=TRACE_INTERVAL_S,
                        help="Seconds between debug trace prints during training")
    train(parser.parse_args())
