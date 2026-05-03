from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sb3_contrib import MaskablePPO

from constants import OBS_DIM, TRAIN_BASE_PORT
from env import MiniMetroFrameStack, MiniMetroVecEnv
from models import MetroFeatureExtractor

EVAL_PORT = TRAIN_BASE_PORT + 200
DEFAULT_CITIES = ["london"]
DEFAULT_COMPLEXITIES = [4]
DEFAULT_SPAWN_FACTORS = [1.0]
DEFAULT_SEEDS = [0]


@dataclass
class EpisodeMetrics:
    city: str
    complexity: int
    spawn_rate_factor: float
    episode: int
    score: float
    week: float
    passengers_delivered: float
    steps: int
    stations: float
    invalid_action_rate: float
    noop_rate: float
    danger_count_mean: float
    queue_pressure_mean: float
    overcrowd_pressure_mean: float


def infer_frame_stack(model: MaskablePPO) -> int:
    obs_shape = model.observation_space.shape
    if obs_shape and len(obs_shape) == 1 and obs_shape[0] % OBS_DIM == 0:
        return max(int(obs_shape[0]) // OBS_DIM, 1)
    return 1


def compute_fitness(
    summary: dict[str, float],
    fitness_version: str = "v2",
    week_weight: float = 1000.0,
    score_weight: float = 2.0,
    passenger_weight: float = 1.0,
    std_week_penalty: float = 250.0,
    invalid_action_penalty: float = 100.0,
    noop_penalty: float = 400.0,
    queue_penalty: float = 40.0,
    overcrowd_penalty: float = 120.0,
    danger_penalty: float = 100.0,
    zero_throughput_penalty: float = 500.0,
) -> float:
    return sum(
        compute_fitness_components(
            summary,
            fitness_version=fitness_version,
            week_weight=week_weight,
            score_weight=score_weight,
            passenger_weight=passenger_weight,
            std_week_penalty=std_week_penalty,
            invalid_action_penalty=invalid_action_penalty,
            noop_penalty=noop_penalty,
            queue_penalty=queue_penalty,
            overcrowd_penalty=overcrowd_penalty,
            danger_penalty=danger_penalty,
            zero_throughput_penalty=zero_throughput_penalty,
        ).values()
    )


def compute_fitness_components(
    summary: dict[str, float],
    fitness_version: str = "v2",
    week_weight: float = 1000.0,
    score_weight: float = 2.0,
    passenger_weight: float = 1.0,
    std_week_penalty: float = 250.0,
    invalid_action_penalty: float = 100.0,
    noop_penalty: float = 400.0,
    queue_penalty: float = 40.0,
    overcrowd_penalty: float = 120.0,
    danger_penalty: float = 100.0,
    zero_throughput_penalty: float = 500.0,
) -> dict[str, float]:
    if fitness_version == "v1":
        return {
            "week": week_weight * summary["mean_week"],
            "score": 1.0 * summary["mean_score"],
            "passengers": 0.5 * summary["mean_passengers_delivered"],
            "std_week": -200.0 * summary["std_week"],
            "invalid_action": -50.0 * summary["mean_invalid_action_rate"],
        }
    if fitness_version != "v2":
        raise ValueError(f"unknown fitness version: {fitness_version}")
    return {
        "week": week_weight * summary["mean_week"],
        "score": score_weight * summary["mean_score"],
        "passengers": passenger_weight * summary["mean_passengers_delivered"],
        "std_week": -std_week_penalty * summary["std_week"],
        "invalid_action": -invalid_action_penalty * summary["mean_invalid_action_rate"],
        "noop": -noop_penalty * summary["mean_noop_rate"],
        "queue": -queue_penalty * summary["mean_queue_pressure"],
        "overcrowd": -overcrowd_penalty * summary["mean_overcrowd_pressure"],
        "danger": -danger_penalty * summary["mean_danger_count"],
        "zero_throughput": -zero_throughput_penalty * summary["zero_throughput_rate"],
    }


def summarize_episodes(
    episodes: list[EpisodeMetrics],
    fitness_kwargs: dict[str, float] | None = None,
) -> dict[str, float]:
    if not episodes:
        raise ValueError("cannot summarize an empty episode list")

    def arr(field: str) -> np.ndarray:
        return np.asarray([float(getattr(ep, field)) for ep in episodes], dtype=np.float64)

    summary = {
        "episodes": float(len(episodes)),
        "mean_score": float(arr("score").mean()),
        "std_score": float(arr("score").std()),
        "mean_week": float(arr("week").mean()),
        "std_week": float(arr("week").std()),
        "mean_passengers_delivered": float(arr("passengers_delivered").mean()),
        "std_passengers_delivered": float(arr("passengers_delivered").std()),
        "mean_steps": float(arr("steps").mean()),
        "mean_stations": float(arr("stations").mean()),
        "mean_invalid_action_rate": float(arr("invalid_action_rate").mean()),
        "mean_noop_rate": float(arr("noop_rate").mean()),
        "mean_danger_count": float(arr("danger_count_mean").mean()),
        "mean_queue_pressure": float(arr("queue_pressure_mean").mean()),
        "mean_overcrowd_pressure": float(arr("overcrowd_pressure_mean").mean()),
        "zero_throughput_rate": float(
            np.mean(
                (arr("score") <= 0.0)
                | (arr("passengers_delivered") <= 0.0)
            )
        ),
    }
    summary["fitness_components"] = compute_fitness_components(summary, **(fitness_kwargs or {}))
    summary["fitness"] = float(sum(summary["fitness_components"].values()))
    return summary


def parse_csv(value: str, cast) -> list[Any]:
    return [cast(part.strip()) for part in value.split(",") if part.strip()]


def evaluate_setting(
    model: MaskablePPO,
    city: str,
    complexity: int,
    spawn_rate_factor: float,
    episodes: int,
    max_steps: int,
    port: int,
    deterministic: bool,
    seed: int,
) -> list[EpisodeMetrics]:
    n_stack = infer_frame_stack(model)
    env = MiniMetroVecEnv(n_envs=1, city=city, port=port, managed=True, seed=seed)
    try:
        env.env_method("set_complexity", complexity)
        env.env_method("set_difficulty", spawn_rate_factor)
        wrapped_env = MiniMetroFrameStack(env, n_stack=n_stack) if n_stack > 1 else env

        results: list[EpisodeMetrics] = []
        for ep_idx in range(episodes):
            obs = wrapped_env.reset()
            done = False
            steps = 0
            invalid_sum = 0.0
            noop_sum = 0.0
            danger_sum = 0.0
            queue_sum = 0.0
            overcrowd_sum = 0.0
            info: dict[str, Any] = {}

            while not done and steps < max_steps:
                mask = np.asarray(wrapped_env.action_masks())
                action, _ = model.predict(
                    obs,
                    action_masks=mask,
                    deterministic=deterministic,
                )
                wrapped_env.step_async(action)
                obs, _reward, dones, infos = wrapped_env.step_wait()
                info = infos[0]
                done = bool(dones[0])
                steps += 1

                invalid_sum += float(info.get("invalid_action", 0.0))
                noop_sum += float(info.get("action_category", -1) == 0)
                danger_sum += float(info.get("danger_count", 0.0))
                queue_sum += float(info.get("queue_pressure", 0.0))
                overcrowd_sum += float(info.get("overcrowd_pressure", 0.0))

            denom = max(steps, 1)
            results.append(
                EpisodeMetrics(
                    city=city,
                    complexity=complexity,
                    spawn_rate_factor=spawn_rate_factor,
                    episode=ep_idx,
                    score=float(info.get("score", 0.0)),
                    week=float(info.get("week", 0.0)),
                    passengers_delivered=float(info.get("passengers_delivered", 0.0)),
                    steps=steps,
                    stations=float(info.get("stations", 0.0)),
                    invalid_action_rate=invalid_sum / denom,
                    noop_rate=noop_sum / denom,
                    danger_count_mean=danger_sum / denom,
                    queue_pressure_mean=queue_sum / denom,
                    overcrowd_pressure_mean=overcrowd_sum / denom,
                )
            )
        return results
    finally:
        env.close()


def run_eval_suite(args: argparse.Namespace) -> dict[str, Any]:
    model = MaskablePPO.load(
        args.model,
        custom_objects={"MetroFeatureExtractor": MetroFeatureExtractor},
    )

    cities = parse_csv(args.cities, str)
    complexities = parse_csv(args.complexities, int)
    spawn_factors = parse_csv(args.spawn_factors, float)
    seeds = parse_csv(args.seeds, int) or DEFAULT_SEEDS

    all_episodes: list[EpisodeMetrics] = []
    setting_summaries: list[dict[str, Any]] = []
    fitness_kwargs = {
        "week_weight": args.week_weight,
        "score_weight": args.score_weight,
        "passenger_weight": args.passenger_weight,
        "std_week_penalty": args.std_week_penalty,
        "invalid_action_penalty": args.invalid_action_penalty,
        "noop_penalty": args.noop_penalty,
        "queue_penalty": args.queue_penalty,
        "overcrowd_penalty": args.overcrowd_penalty,
        "danger_penalty": args.danger_penalty,
        "zero_throughput_penalty": args.zero_throughput_penalty,
        "fitness_version": args.fitness_version,
    }
    setting_idx = 0
    for city in cities:
        for complexity in complexities:
            for spawn_factor in spawn_factors:
                for seed in seeds:
                    setting_port = args.port + setting_idx
                    episodes = evaluate_setting(
                        model=model,
                        city=city,
                        complexity=complexity,
                        spawn_rate_factor=spawn_factor,
                        episodes=args.episodes,
                        max_steps=args.max_steps,
                        port=setting_port,
                        deterministic=args.deterministic,
                        seed=seed,
                    )
                    all_episodes.extend(episodes)
                    setting_summaries.append({
                        "city": city,
                        "complexity": complexity,
                        "spawn_rate_factor": spawn_factor,
                        "seed": seed,
                        "summary": summarize_episodes(episodes, fitness_kwargs),
                    })
                    setting_idx += 1

    summary = summarize_episodes(all_episodes, fitness_kwargs)
    result = {
        "model": args.model,
        "deterministic": args.deterministic,
        "max_steps": args.max_steps,
        "fitness_weights": fitness_kwargs,
        "summary": summary,
        "settings": setting_summaries,
        "episodes": [asdict(ep) for ep in all_episodes],
    }

    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to a MaskablePPO checkpoint")
    parser.add_argument("--output", default="", help="Optional path for eval JSON")
    parser.add_argument("--episodes", type=int, default=32)
    parser.add_argument("--max-steps", type=int, default=4000)
    parser.add_argument("--cities", default=",".join(DEFAULT_CITIES))
    parser.add_argument("--complexities", default=",".join(map(str, DEFAULT_COMPLEXITIES)))
    parser.add_argument("--spawn-factors", default=",".join(map(str, DEFAULT_SPAWN_FACTORS)))
    parser.add_argument("--seeds", default=",".join(map(str, DEFAULT_SEEDS)))
    parser.add_argument("--port", type=int, default=EVAL_PORT)
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fitness-version", choices=["v1", "v2"], default="v2")
    parser.add_argument("--week-weight", type=float, default=1000.0)
    parser.add_argument("--score-weight", type=float, default=2.0)
    parser.add_argument("--passenger-weight", type=float, default=1.0)
    parser.add_argument("--std-week-penalty", type=float, default=250.0)
    parser.add_argument("--invalid-action-penalty", type=float, default=100.0)
    parser.add_argument("--noop-penalty", type=float, default=400.0)
    parser.add_argument("--queue-penalty", type=float, default=40.0)
    parser.add_argument("--overcrowd-penalty", type=float, default=120.0)
    parser.add_argument("--danger-penalty", type=float, default=100.0)
    parser.add_argument("--zero-throughput-penalty", type=float, default=500.0)
    args = parser.parse_args()

    result = run_eval_suite(args)
    print(json.dumps(result["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
