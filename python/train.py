"""
Training script for the Mini Metro RL agent.

Runs indefinitely until interrupted (Ctrl+C). Checkpoints are saved
periodically so training can be resumed at any point.

Usage:
    uv run python train.py [--n-envs 8] [--city london] [--resume checkpoints/minimetro_ppo_100000_steps.zip]
"""
from __future__ import annotations

import argparse
import json
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
from sb3_contrib.common.maskable.buffers import MaskableRolloutBuffer
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.vec_env import VecMonitor

from constants import MASK_SIZE, NUM_ACTION_CATS, TRAIN_BASE_PORT
from env import MiniMetroFrameStack, MiniMetroVecEnv
from models import MetroFeatureExtractor
from policy import MetroPolicy

BASE_PORT = TRAIN_BASE_PORT
CHECKPOINT_DIR = "checkpoints"
LOG_DIR = "tb_logs"

CHECKPOINT_FREQ  = 50_000
EVAL_FREQ        = 100_000
EVAL_EPISODES    = 16       # was 4 — more episodes → lower variance
TRACE_INTERVAL_S = 10.0
FRAME_STACK      = 4

PPO_N_STEPS      = 1024
PPO_BATCH_SIZE   = 1024
PPO_N_EPOCHS     = 3
PPO_GAMMA        = 0.999
PPO_GAE_LAMBDA   = 0.97
PPO_CLIP_RANGE   = 0.15
PPO_MAX_GRAD_NORM = 0.5
PPO_TARGET_KL    = 0.03

# Learning-rate schedule: linear decay from LR_START to LR_END over
# LR_DECAY_STEPS total timesteps, then held constant at LR_END.
LR_START       = 3e-4
LR_END         = 1e-5
LR_DECAY_STEPS = 10_000_000

ENT_START       = 0.02
ENT_END         = 0.002
ENT_DECAY_STEPS = 5_000_000


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


def load_reward_config(path: str) -> dict[str, float] | None:
    if not path:
        return None
    with open(path, "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    if not isinstance(raw, dict):
        raise ValueError("--reward-config must point to a JSON object")
    return {str(k): float(v) for k, v in raw.items()}


class ConditionalMaskRolloutBuffer(MaskableRolloutBuffer):
    """Rollout buffer that stores the expanded conditional action mask."""

    def reset(self) -> None:
        self.mask_dims = MASK_SIZE
        self.action_masks = np.ones(
            (self.buffer_size, self.n_envs, self.mask_dims),
            dtype=np.bool_,
        )
        super(MaskableRolloutBuffer, self).reset()


def trace(msg: str) -> None:
    print(f"[trace {time.strftime('%H:%M:%S')}] {msg}", flush=True)


class CurriculumCallback(BaseCallback):
    """Adjusts spawn-rate difficulty across all training environments based on
    recent episode survival.

    Difficulty is expressed as a ``spawn_rate_factor`` passed to the Go server
    on each episode reset: values > 1.0 stretch spawn intervals (fewer
    passengers per minute), making the game easier.

    Schedule (index 0 = easiest):
        spawn-rate factors: [4.0, 3.0, 2.0, 1.5, 1.25, 1.0]
        complexity levels:   [0,   1,   2,   3,   4,    4]

    Promotion rule: rolling mean weeks survived over the last ``window``
    completed episodes exceeds ``promote_threshold``  → advance one level.
    Demotion rule:  rolling mean falls below ``demote_threshold`` → retreat one
    level (floor = 0, ceiling = last index = normal difficulty).
    """

    SCHEDULE: list[float] = [4.0, 3.0, 2.0, 1.5, 1.25, 1.0]
    COMPLEXITY: list[int] = [0, 1, 2, 3, 4, 4]

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
            self.logger.record("curriculum/complexity", self.COMPLEXITY[self._level])

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
        complexity = self.COMPLEXITY[level]
        self._vec_env.env_method("set_complexity", complexity)
        self._vec_env.env_method("set_difficulty", factor)


class InvalidActionRateCallback(BaseCallback):
    """Logs invalid-action rate inferred from the conditional action mask."""

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        if infos:
            invalid = [float(info.get("invalid_action", 0.0)) for info in infos]
            self.logger.record("rollout/invalid_action_rate", float(np.mean(invalid)))
        return True


class RolloutDiagnosticsCallback(BaseCallback):
    """Logs gameplay metrics that reveal whether reward gains map to better play."""

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        actions = self.locals.get("actions", None)
        if not infos:
            return True

        for key in ("score", "passengers_delivered", "week", "stations"):
            values = [float(info.get(key, 0.0)) for info in infos]
            self.logger.record(f"game/{key}", float(np.mean(values)))

        for key in ("queue_pressure", "overcrowd_pressure", "danger_count"):
            values = [float(info.get(key, 0.0)) for info in infos]
            self.logger.record(f"game/{key}", float(np.mean(values)))

        if actions is not None:
            actions_arr = np.asarray(actions)
            if actions_arr.ndim >= 2 and actions_arr.shape[1] > 0:
                cats = actions_arr[:, 0].astype(np.int64)
                self.logger.record("actions/noop_rate", float(np.mean(cats == 0)))
                for cat in range(NUM_ACTION_CATS):
                    self.logger.record(
                        f"actions/cat_{cat}_rate",
                        float(np.mean(cats == cat)),
                    )

        for key in (
            "reward_delivered",
            "reward_queue",
            "reward_queue_delta",
            "reward_overcrowd",
            "reward_overcrowd_delta",
            "reward_danger",
            "reward_noop",
        ):
            values = [float(info.get(key, 0.0)) for info in infos if key in info]
            if values:
                self.logger.record(f"reward_components/{key}", float(np.mean(values)))

        return True


class EntropyScheduleCallback(BaseCallback):
    """Linearly decays PPO entropy regularisation over global timesteps."""

    def __init__(
        self,
        start: float = ENT_START,
        end: float = ENT_END,
        decay_steps: int = ENT_DECAY_STEPS,
    ):
        super().__init__(verbose=0)
        self.start = start
        self.end = end
        self.decay_steps = decay_steps

    def _on_step(self) -> bool:
        frac = min(self.model.num_timesteps / max(self.decay_steps, 1), 1.0)
        ent_coef = self.start + frac * (self.end - self.start)
        self.model.ent_coef = float(np.clip(ent_coef, self.end, self.start))
        self.logger.record("train/ent_coef", self.model.ent_coef)
        return True


class DifficultySweepEvalCallback(BaseCallback):
    """Deterministic eval at every curriculum difficulty level."""

    def __init__(
        self,
        eval_env,
        eval_freq: int,
        n_eval_episodes: int,
        factors: list[float],
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.factors = factors

    def _on_step(self) -> bool:
        if self.eval_freq <= 0 or self.n_calls % self.eval_freq != 0:
            return True

        original = float(self.eval_env.get_attr("difficulty")[0])
        for factor in self.factors:
            rewards: list[float] = []
            lengths: list[int] = []
            scores: list[int] = []
            invalid_rates: list[float] = []
            self.eval_env.env_method("set_difficulty", factor)

            obs = self.eval_env.reset()
            active_rewards = np.zeros(self.eval_env.num_envs, dtype=np.float64)
            active_lengths = np.zeros(self.eval_env.num_envs, dtype=np.int64)
            active_invalid = np.zeros(self.eval_env.num_envs, dtype=np.float64)
            completed = 0
            while completed < self.n_eval_episodes:
                masks = np.asarray(self.eval_env.action_masks())
                actions, _ = self.model.predict(
                    obs, action_masks=masks, deterministic=True
                )
                self.eval_env.step_async(actions)
                obs, step_rewards, dones, infos = self.eval_env.step_wait()
                active_rewards += step_rewards
                active_lengths += 1
                for i, info in enumerate(infos):
                    active_invalid[i] += float(info.get("invalid_action", 0.0))
                    if dones[i]:
                        rewards.append(float(active_rewards[i]))
                        lengths.append(int(active_lengths[i]))
                        scores.append(int(info.get("score", 0)))
                        invalid_rates.append(
                            float(active_invalid[i] / max(active_lengths[i], 1))
                        )
                        active_rewards[i] = 0.0
                        active_lengths[i] = 0
                        active_invalid[i] = 0.0
                        completed += 1
                        if completed >= self.n_eval_episodes:
                            break

            tag = str(factor).replace(".", "_")
            self.logger.record(f"eval_difficulty_{tag}/mean_reward", float(np.mean(rewards)))
            self.logger.record(f"eval_difficulty_{tag}/mean_ep_length", float(np.mean(lengths)))
            self.logger.record(f"eval_difficulty_{tag}/mean_score", float(np.mean(scores)))
            self.logger.record(
                f"eval_difficulty_{tag}/invalid_action_rate",
                float(np.mean(invalid_rates)),
            )
            if self.verbose:
                print(
                    f"[eval:difficulty={factor:.2f}] "
                    f"reward={np.mean(rewards):.1f} "
                    f"len={np.mean(lengths):.1f} "
                    f"score={np.mean(scores):.1f} "
                    f"invalid={np.mean(invalid_rates):.3f}"
                )

        self.eval_env.env_method("set_difficulty", original)
        return True


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
    if args.frame_stack < 1:
        raise ValueError("--frame-stack must be >= 1")
    reward_config = load_reward_config(args.reward_config)
    checkpoint_dir = args.checkpoint_dir
    log_dir = args.log_dir
    base_port = args.base_port

    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    n = args.n_envs
    print(f"Launching {n} training envs on port {base_port}")

    trace(
        "startup "
        f"n_envs={n} city={args.city} "
        f"base_port={base_port} "
        f"torch_threads={torch.get_num_threads()} "
        f"trace_interval={args.trace_interval}s "
        f"frame_stack={args.frame_stack} "
        f"n_steps={PPO_N_STEPS} batch_size={PPO_BATCH_SIZE} "
        f"n_epochs={PPO_N_EPOCHS} clip={PPO_CLIP_RANGE} target_kl={PPO_TARGET_KL}"
    )

    vec_env = MiniMetroVecEnv(
        n_envs=n,
        city=args.city,
        port=base_port,
        managed=True,
        trace_interval=args.trace_interval,
    )
    if reward_config is not None:
        trace(f"applying_reward_config path={args.reward_config}")
        vec_env.set_reward_config(reward_config)
    vec_env = VecMonitor(vec_env)
    if args.frame_stack > 1:
        vec_env = MiniMetroFrameStack(vec_env, n_stack=args.frame_stack)

    # Keep the primary eval callback on a fixed benchmark. If curriculum also
    # mutates this env, `eval/mean_reward` becomes a moving target and can
    # trend downward even while the policy improves.
    eval_env = MiniMetroVecEnv(
        n_envs=2,
        city=args.city,
        port=base_port + 1,
        managed=True,
        trace_interval=args.trace_interval,
    )
    if reward_config is not None:
        eval_env.set_reward_config(reward_config)
    eval_env = VecMonitor(eval_env)
    if args.frame_stack > 1:
        eval_env = MiniMetroFrameStack(eval_env, n_stack=args.frame_stack)

    if args.resume:
        print(f"Resuming from {args.resume}")
        model = TracedMaskablePPO.load(
            args.resume, env=vec_env, tensorboard_log=log_dir,
            learning_rate=lr_schedule,
            rollout_buffer_class=ConditionalMaskRolloutBuffer,
            verbose=1,
        )
    else:
        model = TracedMaskablePPO(
            MetroPolicy,
            vec_env,
            n_steps=PPO_N_STEPS,
            batch_size=PPO_BATCH_SIZE,
            n_epochs=PPO_N_EPOCHS,
            gamma=PPO_GAMMA,
            gae_lambda=PPO_GAE_LAMBDA,
            learning_rate=lr_schedule,
            ent_coef=ENT_START,
            clip_range=PPO_CLIP_RANGE,
            max_grad_norm=PPO_MAX_GRAD_NORM,
            target_kl=PPO_TARGET_KL,
            rollout_buffer_class=ConditionalMaskRolloutBuffer,
            tensorboard_log=log_dir,
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
        EntropyScheduleCallback(),
        InvalidActionRateCallback(),
        RolloutDiagnosticsCallback(),
        TracedCheckpointCallback(
            save_freq=max(CHECKPOINT_FREQ // n, 1),
            save_path=checkpoint_dir,
            name_prefix="minimetro_ppo",
        ),
        TracedMaskableEvalCallback(
            eval_env,
            n_eval_episodes=EVAL_EPISODES,
            eval_freq=max(EVAL_FREQ // n, 1),
            best_model_save_path=checkpoint_dir,
            log_path=log_dir,
            deterministic=True,
        ),
        DifficultySweepEvalCallback(
            eval_env,
            n_eval_episodes=EVAL_EPISODES,
            eval_freq=max(EVAL_FREQ // n, 1),
            factors=CurriculumCallback.SCHEDULE,
            verbose=1,
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

    CHUNK = args.learn_chunk

    print("[train] All envs ready, starting learning loop...")
    first_chunk = True
    try:
        while not interrupted:
            if args.total_timesteps > 0 and model.num_timesteps >= args.total_timesteps:
                break
            chunk = CHUNK
            if args.total_timesteps > 0:
                chunk = min(chunk, args.total_timesteps - model.num_timesteps)
            learn_start = time.perf_counter()
            trace(
                "learn_chunk_start "
                f"current_timesteps={model.num_timesteps:,} "
                f"target_chunk={chunk:,}"
            )
            model.learn(
                total_timesteps=chunk,
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
        final_path = os.path.join(checkpoint_dir, "minimetro_latest")
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
    parser.add_argument("--frame-stack", type=int, default=FRAME_STACK,
                        help="Number of recent observations to stack for temporal context")
    parser.add_argument("--reward-config", type=str, default="",
                        help="Path to a JSON RewardConfig applied before training resets")
    parser.add_argument("--total-timesteps", type=int, default=0,
                        help="Stop after this many timesteps; 0 keeps training indefinitely")
    parser.add_argument("--learn-chunk", type=int, default=500_000,
                        help="Timesteps per learn() call")
    parser.add_argument("--checkpoint-dir", type=str, default=CHECKPOINT_DIR)
    parser.add_argument("--log-dir", type=str, default=LOG_DIR)
    parser.add_argument("--base-port", type=int, default=BASE_PORT)
    train(parser.parse_args())
