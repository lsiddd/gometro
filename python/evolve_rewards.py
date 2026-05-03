from __future__ import annotations

import argparse
import csv
import json
import math
import random
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ParamSpec:
    name: str
    low: float
    high: float
    scale: str = "log"


PARAM_SPECS = [
    ParamSpec("per_passenger", 0.5, 10.0),
    ParamSpec("queue_coeff", 0.001, 0.2),
    ParamSpec("queue_delta_coeff", 0.01, 2.0),
    ParamSpec("overcrowd_coeff", 0.05, 5.0),
    ParamSpec("overcrowd_delta_coeff", 0.05, 10.0),
    ParamSpec("danger_thresh", 0.5, 1.1, "linear"),
    ParamSpec("danger_penalty", 0.05, 5.0),
    ParamSpec("noop_critical_penalty", 0.0, 2.0, "linear"),
    ParamSpec("week_bonus", 1.0, 100.0),
    ParamSpec("terminal_penalty", 10.0, 500.0),
    ParamSpec("invalid_action", 0.1, 10.0),
]


@dataclass
class Individual:
    id: str
    generation: int
    config: dict[str, float]
    parents: list[str]
    origin: str
    metadata: dict[str, Any]
    fitness: float | None = None
    eval_summary: dict[str, Any] | None = None


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def sample_value(spec: ParamSpec, rng: random.Random) -> float:
    if spec.scale == "log":
        lo = math.log(spec.low)
        hi = math.log(spec.high)
        return math.exp(rng.uniform(lo, hi))
    return rng.uniform(spec.low, spec.high)


def sample_config(rng: random.Random) -> dict[str, float]:
    return {spec.name: sample_value(spec, rng) for spec in PARAM_SPECS}


def mutate_config(
    config: dict[str, float],
    rng: random.Random,
    sigma: float,
) -> dict[str, float]:
    specs = {spec.name: spec for spec in PARAM_SPECS}
    child: dict[str, float] = {}
    for name, value in config.items():
        spec = specs[name]
        if spec.scale == "log":
            mutated = math.exp(math.log(max(value, spec.low)) + rng.gauss(0.0, sigma))
        else:
            mutated = value + rng.gauss(0.0, sigma * (spec.high - spec.low))
        child[name] = clamp(mutated, spec.low, spec.high)
    return child


def crossover_config(
    parent_a: dict[str, float],
    parent_b: dict[str, float],
    rng: random.Random,
) -> dict[str, float]:
    return {
        spec.name: parent_a[spec.name] if rng.random() < 0.5 else parent_b[spec.name]
        for spec in PARAM_SPECS
    }


def select_elites(population: list[Individual], elite_count: int) -> list[Individual]:
    evaluated = [
        ind for ind in population
        if ind.fitness is not None and not ind.metadata.get("pruned", False)
    ]
    if not evaluated:
        evaluated = [ind for ind in population if ind.fitness is not None]
    return sorted(evaluated, key=lambda ind: ind.fitness or float("-inf"), reverse=True)[
        :elite_count
    ]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def git_metadata(repo_dir: Path) -> dict[str, Any]:
    def run_git(args: list[str]) -> str:
        return subprocess.check_output(
            ["git", *args],
            cwd=repo_dir,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()

    try:
        commit = run_git(["rev-parse", "HEAD"])
        branch = run_git(["branch", "--show-current"])
        dirty = subprocess.run(
            ["git", "diff", "--quiet"],
            cwd=repo_dir,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ).returncode != 0
        untracked = bool(run_git(["ls-files", "--others", "--exclude-standard"]))
        return {
            "commit": commit,
            "branch": branch,
            "dirty": dirty,
            "untracked": untracked,
        }
    except (OSError, subprocess.CalledProcessError):
        return {
            "commit": "",
            "branch": "",
            "dirty": None,
            "untracked": None,
        }


def read_eval_summary(path: Path) -> tuple[float, dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    summary = data["summary"]
    return float(summary["fitness"]), summary


def log_progress(message: str) -> None:
    print(f"[evolve {datetime.now().strftime('%H:%M:%S')}] {message}", flush=True)


def fmt_float(value: Any, digits: int = 3) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def summary_line(summary: dict[str, Any] | None) -> str:
    if not summary:
        return "summary=none"
    parts = [
        f"fitness={fmt_float(summary.get('fitness'))}",
        f"week={fmt_float(summary.get('mean_week'))}",
        f"score={fmt_float(summary.get('mean_score'))}",
        f"pax={fmt_float(summary.get('mean_passengers_delivered'))}",
        f"noop={fmt_float(summary.get('mean_noop_rate'))}",
        f"zero={fmt_float(summary.get('zero_throughput_rate'))}",
        f"invalid={fmt_float(summary.get('mean_invalid_action_rate'))}",
    ]
    if "fitness_delta_vs_default" in summary:
        parts.append(f"vs_default={fmt_float(summary.get('fitness_delta_vs_default'))}")
    return " ".join(parts)


def build_train_command(args: argparse.Namespace, ind_dir: Path, config_path: Path) -> list[str]:
    cmd = [
        sys.executable,
        "train.py",
        "--n-envs",
        str(args.n_envs),
        "--city",
        args.city,
        "--total-timesteps",
        str(args.train_timesteps),
        "--learn-chunk",
        str(args.learn_chunk),
        "--checkpoint-dir",
        str(ind_dir / "checkpoints"),
        "--log-dir",
        str(ind_dir / "tb_logs"),
        "--base-port",
        str(args.base_port),
        "--trace-interval",
        str(args.trace_interval),
        "--frame-stack",
        str(args.frame_stack),
    ]
    if config_path:
        cmd[6:6] = ["--reward-config", str(config_path)]
    return cmd


def build_eval_command(args: argparse.Namespace, ind_dir: Path, output_path: Path) -> list[str]:
    cmd = [
        sys.executable,
        "eval_suite.py",
        "--model",
        str(ind_dir / "checkpoints" / "minimetro_latest.zip"),
        "--output",
        str(output_path),
        "--episodes",
        str(args.eval_episodes),
        "--max-steps",
        str(args.eval_max_steps),
        "--cities",
        args.eval_cities,
        "--complexities",
        args.eval_complexities,
        "--spawn-factors",
        args.eval_spawn_factors,
        "--seeds",
        args.eval_seeds,
        "--port",
        str(args.eval_base_port),
    ]
    append_fitness_args(cmd, args)
    return cmd


def build_retrain_command(
    args: argparse.Namespace,
    finalist_dir: Path,
    config_path: Path,
) -> list[str]:
    return [
        sys.executable,
        "train.py",
        "--n-envs",
        str(args.retrain_n_envs or args.n_envs),
        "--city",
        args.city,
        "--reward-config",
        str(config_path),
        "--total-timesteps",
        str(args.retrain_timesteps),
        "--learn-chunk",
        str(args.retrain_learn_chunk or args.learn_chunk),
        "--checkpoint-dir",
        str(finalist_dir / "checkpoints"),
        "--log-dir",
        str(finalist_dir / "tb_logs"),
        "--base-port",
        str(args.retrain_base_port),
        "--trace-interval",
        str(args.trace_interval),
        "--frame-stack",
        str(args.frame_stack),
    ]


def build_retrain_eval_command(
    args: argparse.Namespace,
    finalist_dir: Path,
    output_path: Path,
) -> list[str]:
    cmd = [
        sys.executable,
        "eval_suite.py",
        "--model",
        str(finalist_dir / "checkpoints" / "minimetro_latest.zip"),
        "--output",
        str(output_path),
        "--episodes",
        str(args.retrain_eval_episodes or args.eval_episodes),
        "--max-steps",
        str(args.retrain_eval_max_steps or args.eval_max_steps),
        "--cities",
        args.retrain_eval_cities or args.eval_cities,
        "--complexities",
        args.retrain_eval_complexities or args.eval_complexities,
        "--spawn-factors",
        args.retrain_eval_spawn_factors or args.eval_spawn_factors,
        "--seeds",
        args.retrain_eval_seeds or args.eval_seeds,
        "--port",
        str(args.retrain_eval_base_port),
    ]
    append_fitness_args(cmd, args)
    return cmd


def append_fitness_args(cmd: list[str], args: argparse.Namespace) -> None:
    cmd.extend([
        "--fitness-version",
        args.fitness_version,
        "--week-weight",
        str(args.week_weight),
        "--score-weight",
        str(args.score_weight),
        "--passenger-weight",
        str(args.passenger_weight),
        "--std-week-penalty",
        str(args.std_week_penalty),
        "--invalid-action-penalty",
        str(args.invalid_action_penalty),
        "--noop-penalty",
        str(args.noop_penalty),
        "--queue-penalty",
        str(args.queue_penalty),
        "--overcrowd-penalty",
        str(args.overcrowd_penalty),
        "--danger-penalty",
        str(args.danger_penalty),
        "--zero-throughput-penalty",
        str(args.zero_throughput_penalty),
    ])


def append_event(run_dir: Path, event: str, payload: dict[str, Any]) -> None:
    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "event": event,
        **payload,
    }
    with (run_dir / "events.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, sort_keys=True) + "\n")


def append_csv(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    fields = list(row.keys())
    with path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def flatten_summary(prefix: str, summary: dict[str, Any] | None) -> dict[str, Any]:
    if not summary:
        return {}
    keys = [
        "fitness",
        "mean_week",
        "std_week",
        "mean_score",
        "mean_passengers_delivered",
        "mean_invalid_action_rate",
        "mean_noop_rate",
        "zero_throughput_rate",
        "mean_queue_pressure",
        "mean_overcrowd_pressure",
        "mean_danger_count",
    ]
    return {f"{prefix}_{key}": summary.get(key) for key in keys if key in summary}


def pruning_reason(args: argparse.Namespace, summary: dict[str, Any]) -> str:
    if summary.get("mean_noop_rate", 0.0) > args.prune_noop_rate:
        return "high_noop_rate"
    if summary.get("zero_throughput_rate", 0.0) > args.prune_zero_throughput_rate:
        return "high_zero_throughput"
    delta = summary.get("fitness_delta_vs_default")
    if delta is not None and delta < -args.prune_default_delta:
        return "below_default_delta"
    return ""


def write_leaderboard(run_dir: Path, individuals: list[Individual], top_k: int = 20) -> None:
    ranked = select_elites(individuals, min(top_k, len(individuals)))
    write_json(run_dir / "leaderboard.json", [asdict(ind) for ind in ranked])


def write_dashboard(run_dir: Path, individuals: list[Individual], baseline: dict[str, Any] | None) -> None:
    ranked = select_elites(individuals, min(20, len(individuals)))
    rows = "\n".join(
        "<tr>"
        f"<td>{ind.generation}</td><td>{ind.id}</td><td>{ind.fitness:.3f}</td>"
        f"<td>{(ind.eval_summary or {}).get('mean_week', '')}</td>"
        f"<td>{(ind.eval_summary or {}).get('mean_score', '')}</td>"
        f"<td>{(ind.eval_summary or {}).get('mean_noop_rate', '')}</td>"
        f"<td>{(ind.eval_summary or {}).get('zero_throughput_rate', '')}</td>"
        "</tr>"
        for ind in ranked
        if ind.fitness is not None
    )
    baseline_html = ""
    if baseline:
        baseline_html = (
            f"<p>Baseline fitness: <b>{baseline.get('fitness', 0.0):.3f}</b>, "
            f"week: {baseline.get('mean_week', '')}, score: {baseline.get('mean_score', '')}</p>"
        )
    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Evolution Dashboard</title>
<style>body{{font-family:sans-serif;margin:24px}}table{{border-collapse:collapse}}td,th{{border:1px solid #ccc;padding:6px 8px}}</style>
</head><body>
<h1>Evolution Dashboard</h1>
{baseline_html}
<h2>Leaderboard</h2>
<table><thead><tr><th>Generation</th><th>ID</th><th>Fitness</th><th>Week</th><th>Score</th><th>NoOp</th><th>Zero Throughput</th></tr></thead>
<tbody>{rows}</tbody></table>
</body></html>
"""
    (run_dir / "dashboard.html").write_text(html, encoding="utf-8")


def validate_guardrails(args: argparse.Namespace) -> list[str]:
    warnings: list[str] = []
    if args.eval_episodes < args.min_eval_episodes:
        msg = (
            f"eval episodes {args.eval_episodes} is below minimum "
            f"{args.min_eval_episodes}; selection may overfit noise"
        )
        if args.strict_guardrails:
            raise ValueError(msg)
        warnings.append(msg)
    if len([part for part in args.eval_cities.split(",") if part.strip()]) < args.min_eval_cities:
        msg = "eval cities count is below guardrail minimum"
        if args.strict_guardrails:
            raise ValueError(msg)
        warnings.append(msg)
    if args.retrain_top > 0:
        validation_cities = args.retrain_eval_cities or args.eval_cities
        validation_spawn = args.retrain_eval_spawn_factors or args.eval_spawn_factors
        if validation_cities == args.eval_cities and validation_spawn == args.eval_spawn_factors:
            warnings.append(
                "retrain validation uses the same city/spawn benchmark as evolution; "
                "consider --retrain-eval-cities or --retrain-eval-spawn-factors"
            )
    return warnings


def individual_manifest(
    args: argparse.Namespace,
    individual: Individual,
    ind_dir: Path,
    config_path: Path,
    eval_path: Path,
) -> dict[str, Any]:
    train_cmd = build_train_command(args, ind_dir, config_path)
    eval_cmd = build_eval_command(args, ind_dir, eval_path)
    return {
        "individual": asdict(individual),
        "paths": {
            "dir": str(ind_dir),
            "reward_config": str(config_path),
            "checkpoints": str(ind_dir / "checkpoints"),
            "tb_logs": str(ind_dir / "tb_logs"),
            "eval": str(eval_path),
            "train_log": str(ind_dir / "train.log"),
            "eval_log": str(ind_dir / "eval.log"),
        },
        "commands": {
            "train": train_cmd,
            "eval": eval_cmd,
        },
    }


def finalist_manifest(
    args: argparse.Namespace,
    source: Individual,
    finalist_dir: Path,
    config_path: Path,
    eval_path: Path,
) -> dict[str, Any]:
    return {
        "source_individual": asdict(source),
        "paths": {
            "dir": str(finalist_dir),
            "reward_config": str(config_path),
            "checkpoints": str(finalist_dir / "checkpoints"),
            "tb_logs": str(finalist_dir / "tb_logs"),
            "eval": str(eval_path),
            "train_log": str(finalist_dir / "retrain.log"),
            "eval_log": str(finalist_dir / "eval.log"),
        },
        "commands": {
            "train": build_retrain_command(args, finalist_dir, config_path),
            "eval": build_retrain_eval_command(args, finalist_dir, eval_path),
        },
    }


def run_command(cmd: list[str], cwd: Path, log_path: Path, label: str = "") -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    if label:
        log_progress(f"{label} start log={log_path}")
    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + " ".join(cmd) + "\n\n")
        log.flush()
        try:
            subprocess.run(cmd, cwd=cwd, stdout=log, stderr=subprocess.STDOUT, check=True)
        except subprocess.CalledProcessError:
            if label:
                log_progress(f"{label} failed elapsed={time.perf_counter() - started:.1f}s log={log_path}")
            raise
    if label:
        log_progress(f"{label} done elapsed={time.perf_counter() - started:.1f}s")


def evaluate_baseline(args: argparse.Namespace, run_dir: Path) -> dict[str, Any] | None:
    if args.skip_baseline or args.dry_run:
        log_progress("baseline skipped")
        return None
    baseline_dir = run_dir / "baseline_default"
    eval_path = baseline_dir / "eval.json"
    if eval_path.exists() and args.resume_run:
        _, summary = read_eval_summary(eval_path)
        log_progress(f"baseline reused {summary_line(summary)}")
        return summary
    append_event(run_dir, "baseline_start", {"dir": str(baseline_dir)})
    log_progress(f"baseline start dir={baseline_dir}")
    python_dir = Path(__file__).resolve().parent
    run_command(
        build_train_command(args, baseline_dir, None),
        cwd=python_dir,
        log_path=baseline_dir / "train.log",
        label="baseline train",
    )
    run_command(
        build_eval_command(args, baseline_dir, eval_path),
        cwd=python_dir,
        log_path=baseline_dir / "eval.log",
        label="baseline eval",
    )
    _, summary = read_eval_summary(eval_path)
    write_json(
        baseline_dir / "summary.json",
        {
            "summary": summary,
            "commands": {
                "train": build_train_command(args, baseline_dir, None),
                "eval": build_eval_command(args, baseline_dir, eval_path),
            },
        },
    )
    append_event(run_dir, "baseline_end", {"summary": summary})
    log_progress(f"baseline done {summary_line(summary)}")
    return summary


def retrain_finalist(args: argparse.Namespace, source: Individual, rank: int, run_dir: Path) -> dict[str, Any]:
    finalist_dir = run_dir / "retrain" / f"finalist_{rank:03d}"
    config_path = finalist_dir / "reward_config.json"
    eval_path = finalist_dir / "eval.json"
    write_json(config_path, source.config)
    write_json(finalist_dir / "manifest.json", finalist_manifest(args, source, finalist_dir, config_path, eval_path))

    if args.dry_run:
        result = {
            "rank": rank,
            "source": asdict(source),
            "fitness": 0.0,
            "summary": {"fitness": 0.0, "dry_run": True},
        }
    else:
        log_progress(
            f"retrain finalist_{rank:03d} source={source.id} "
            f"source_fitness={fmt_float(source.fitness)}"
        )
        python_dir = Path(__file__).resolve().parent
        run_command(
            build_retrain_command(args, finalist_dir, config_path),
            cwd=python_dir,
            log_path=finalist_dir / "retrain.log",
            label=f"retrain finalist_{rank:03d}",
        )
        run_command(
            build_retrain_eval_command(args, finalist_dir, eval_path),
            cwd=python_dir,
            log_path=finalist_dir / "eval.log",
            label=f"retrain eval finalist_{rank:03d}",
        )
        fitness, summary = read_eval_summary(eval_path)
        result = {
            "rank": rank,
            "source": asdict(source),
            "fitness": fitness,
            "summary": summary,
        }
        log_progress(f"retrain finalist_{rank:03d} done {summary_line(summary)}")
    write_json(finalist_dir / "result.json", result)
    write_json(finalist_dir / "manifest.json", finalist_manifest(args, source, finalist_dir, config_path, eval_path))
    return result


def retrain_finalists(args: argparse.Namespace, all_results: list[Individual], run_dir: Path) -> list[dict[str, Any]]:
    if args.retrain_top <= 0:
        log_progress("retrain skipped")
        return []
    finalists = select_elites(all_results, args.retrain_top)
    log_progress(
        "retrain start "
        + ", ".join(f"{ind.id}:{fmt_float(ind.fitness)}" for ind in finalists)
    )
    results = [
        retrain_finalist(args, source, rank, run_dir)
        for rank, source in enumerate(finalists)
    ]
    ranked = sorted(results, key=lambda item: item["fitness"], reverse=True)
    write_json(
        run_dir / "retrain" / "summary.json",
        {
            "best": ranked[0] if ranked else None,
            "finalists": ranked,
        },
    )
    return ranked


def evaluate_individual(
    args: argparse.Namespace,
    individual: Individual,
    run_dir: Path,
    baseline_summary: dict[str, Any] | None = None,
) -> Individual:
    ind_dir = run_dir / f"generation_{individual.generation:03d}" / individual.id
    config_path = ind_dir / "reward_config.json"
    eval_path = ind_dir / "eval.json"
    write_json(config_path, individual.config)
    write_json(ind_dir / "individual.json", asdict(individual))
    write_json(ind_dir / "manifest.json", individual_manifest(args, individual, ind_dir, config_path, eval_path))

    if not args.dry_run and args.resume_run and eval_path.exists():
        individual.fitness, individual.eval_summary = read_eval_summary(eval_path)
        log_progress(
            f"generation={individual.generation:03d} {individual.id} reused "
            f"{summary_line(individual.eval_summary)}"
        )
    elif not args.dry_run:
        log_progress(
            f"generation={individual.generation:03d} {individual.id} "
            f"start origin={individual.origin} parents={','.join(individual.parents) or '-'} "
            f"dir={ind_dir}"
        )
        append_event(
            run_dir,
            "individual_start",
            {
                "generation": individual.generation,
                "individual": individual.id,
                "origin": individual.origin,
            },
        )
        run_command(
            build_train_command(args, ind_dir, config_path),
            cwd=Path(__file__).resolve().parent,
            log_path=ind_dir / "train.log",
            label=f"generation={individual.generation:03d} {individual.id} train",
        )
        run_command(
            build_eval_command(args, ind_dir, eval_path),
            cwd=Path(__file__).resolve().parent,
            log_path=ind_dir / "eval.log",
            label=f"generation={individual.generation:03d} {individual.id} eval",
        )
        individual.fitness, individual.eval_summary = read_eval_summary(eval_path)
        if baseline_summary is not None:
            individual.eval_summary["fitness_delta_vs_default"] = (
                individual.eval_summary["fitness"] - baseline_summary["fitness"]
            )
            individual.eval_summary["mean_week_delta_vs_default"] = (
                individual.eval_summary["mean_week"] - baseline_summary["mean_week"]
            )
            individual.eval_summary["score_delta_vs_default"] = (
                individual.eval_summary["mean_score"] - baseline_summary["mean_score"]
            )
            individual.eval_summary["passenger_delta_vs_default"] = (
                individual.eval_summary["mean_passengers_delivered"]
                - baseline_summary["mean_passengers_delivered"]
            )
        reason = pruning_reason(args, individual.eval_summary)
        if reason:
            individual.metadata["pruned"] = True
            individual.metadata["prune_reason"] = reason
            append_event(
                run_dir,
                "individual_pruned",
                {
                    "generation": individual.generation,
                    "individual": individual.id,
                    "reason": reason,
                    "fitness": individual.fitness,
                },
            )
            log_progress(
                f"generation={individual.generation:03d} {individual.id} pruned "
                f"reason={reason} {summary_line(individual.eval_summary)}"
            )
        else:
            log_progress(
                f"generation={individual.generation:03d} {individual.id} done "
                f"{summary_line(individual.eval_summary)}"
            )
        append_event(
            run_dir,
            "individual_end",
            {
                "generation": individual.generation,
                "individual": individual.id,
                "fitness": individual.fitness,
                "summary": individual.eval_summary,
            },
        )
    else:
        individual.fitness = 0.0
        individual.eval_summary = {"fitness": 0.0, "dry_run": True}
        log_progress(f"generation={individual.generation:03d} {individual.id} dry_run")

    write_json(ind_dir / "individual.json", asdict(individual))
    write_json(ind_dir / "manifest.json", individual_manifest(args, individual, ind_dir, config_path, eval_path))
    row = {
        "generation": individual.generation,
        "id": individual.id,
        "origin": individual.origin,
        "fitness": individual.fitness,
        "pruned": individual.metadata.get("pruned", False),
        "prune_reason": individual.metadata.get("prune_reason", ""),
        **flatten_summary("eval", individual.eval_summary),
    }
    append_csv(run_dir / "individuals.csv", row)
    return individual


def initial_population(size: int, rng: random.Random) -> list[Individual]:
    return [
        Individual(
            id=f"individual_{idx:03d}",
            generation=0,
            config=sample_config(rng),
            parents=[],
            origin="random_initial",
            metadata={},
        )
        for idx in range(size)
    ]


def next_generation(
    previous: list[Individual],
    generation: int,
    population_size: int,
    elite_count: int,
    random_count: int,
    sigma: float,
    rng: random.Random,
) -> list[Individual]:
    elites = select_elites(previous, elite_count)
    if not elites:
        raise ValueError("cannot create next generation without evaluated elites")
    log_progress(
        f"generation={generation:03d} create_population "
        f"elites={','.join(f'{elite.id}:{fmt_float(elite.fitness)}' for elite in elites)} "
        f"random_injected={random_count} sigma={sigma:.3f}"
    )

    population: list[Individual] = []
    for idx, elite in enumerate(elites):
        population.append(
            Individual(
                id=f"individual_{idx:03d}",
                generation=generation,
                config=dict(elite.config),
                parents=[elite.id],
                origin="elite_copy",
                metadata={
                    "parent_fitness": elite.fitness,
                },
            )
        )

    while len(population) < max(population_size - random_count, len(population)):
        if len(elites) >= 2 and rng.random() < 0.5:
            a, b = rng.sample(elites, 2)
            config = crossover_config(a.config, b.config, rng)
            parents = [a.id, b.id]
            origin = "crossover_mutation"
            metadata = {
                "parent_fitness": [a.fitness, b.fitness],
                "mutation_sigma": sigma,
            }
        else:
            parent = rng.choice(elites)
            config = dict(parent.config)
            parents = [parent.id]
            origin = "mutation"
            metadata = {
                "parent_fitness": parent.fitness,
                "mutation_sigma": sigma,
            }
        before_mutation = dict(config)
        config = mutate_config(config, rng, sigma=sigma)
        metadata["mutation_delta"] = {
            name: config[name] - before_mutation[name] for name in config
        }
        population.append(
            Individual(
                id=f"individual_{len(population):03d}",
                generation=generation,
                config=config,
                parents=parents,
                origin=origin,
                metadata=metadata,
            )
        )

    while len(population) < population_size:
        population.append(
            Individual(
                id=f"individual_{len(population):03d}",
                generation=generation,
                config=sample_config(rng),
                parents=[],
                origin="random_injected",
                metadata={},
            )
        )
    return population


def run_evolution(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    existing_run_config = run_dir / "run_config.json"
    if args.resume_run and existing_run_config.exists():
        stored = json.loads(existing_run_config.read_text(encoding="utf-8")).get("args", {})
        for key, value in stored.items():
            if key not in {"run_dir", "resume_run"}:
                setattr(args, key, value)
        args.resume_run = True
    rng = random.Random(args.seed)
    python_dir = Path(__file__).resolve().parent
    repo_dir = python_dir.parent
    run_config = {
        "args": vars(args),
        "command": [sys.executable, *sys.argv],
        "python_executable": sys.executable,
        "git": git_metadata(repo_dir),
        "guardrail_warnings": validate_guardrails(args),
    }
    write_json(
        run_dir / "run_config.json",
        run_config,
    )
    append_event(run_dir, "run_start", run_config)
    log_progress(
        f"run start dir={run_dir} population={args.population} generations={args.generations} "
        f"train_timesteps={args.train_timesteps} eval_episodes={args.eval_episodes} "
        f"fitness={args.fitness_version}"
    )
    log_progress(
        f"artifacts dashboard={run_dir / 'dashboard.html'} "
        f"leaderboard={run_dir / 'leaderboard.json'} events={run_dir / 'events.jsonl'}"
    )
    for warning in run_config["guardrail_warnings"]:
        log_progress(f"guardrail warning: {warning}")
    baseline_summary = evaluate_baseline(args, run_dir)

    population = initial_population(args.population, rng)
    all_results: list[Individual] = []

    for gen in range(args.generations):
        log_progress(f"generation={gen:03d} start population={len(population)}")
        append_event(run_dir, "generation_start", {"generation": gen, "population": len(population)})
        evaluated = [
            evaluate_individual(args, individual, run_dir, baseline_summary)
            for individual in population
        ]
        all_results.extend(evaluated)
        elites = select_elites(evaluated, args.elites)
        fitness_values = [ind.fitness for ind in evaluated if ind.fitness is not None]
        generation_row = {
            "generation": gen,
            "best_fitness": elites[0].fitness if elites else None,
            "mean_fitness": sum(fitness_values) / max(len(fitness_values), 1),
            "evaluated": len(evaluated),
            "pruned": sum(1 for ind in evaluated if ind.metadata.get("pruned", False)),
            "best_id": elites[0].id if elites else "",
        }
        append_csv(run_dir / "generations.csv", generation_row)
        write_json(
            run_dir / f"generation_{gen:03d}" / "summary.json",
            {
                "generation": gen,
                "best": asdict(elites[0]) if elites else None,
                "individuals": [asdict(ind) for ind in evaluated],
                "baseline": baseline_summary,
            },
        )
        write_leaderboard(run_dir, all_results)
        write_dashboard(run_dir, all_results, baseline_summary)
        append_event(run_dir, "generation_end", generation_row)
        best_text = f"{elites[0].id}:{fmt_float(elites[0].fitness)}" if elites else "-"
        log_progress(
            f"generation={gen:03d} done best={best_text} "
            f"mean_fitness={fmt_float(generation_row['mean_fitness'])} "
            f"pruned={generation_row['pruned']}/{generation_row['evaluated']}"
        )
        if gen + 1 >= args.generations:
            break
        population = next_generation(
            previous=evaluated,
            generation=gen + 1,
            population_size=args.population,
            elite_count=args.elites,
            random_count=args.random_individuals,
            sigma=args.sigma,
            rng=rng,
        )

    best = select_elites(all_results, 1)
    retrain_results = retrain_finalists(args, all_results, run_dir)
    for result in retrain_results:
        append_csv(
            run_dir / "finalists.csv",
            {
                "rank": result.get("rank"),
                "fitness": result.get("fitness"),
                **flatten_summary("eval", result.get("summary")),
            },
        )
    write_json(
        run_dir / "final_summary.json",
        {
            "best": asdict(best[0]) if best else None,
            "best_retrained": retrain_results[0] if retrain_results else None,
            "baseline": baseline_summary,
            "individuals": [asdict(ind) for ind in all_results],
            "retrain": retrain_results,
        },
    )
    write_leaderboard(run_dir, all_results)
    write_dashboard(run_dir, all_results, baseline_summary)
    append_event(run_dir, "run_end", {"best": asdict(best[0]) if best else None})
    if best:
        log_progress(f"run done best={best[0].id} {summary_line(best[0].eval_summary)}")
    else:
        log_progress("run done best=-")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", default="evolution_runs/reward_search")
    parser.add_argument("--population", type=int, default=12)
    parser.add_argument("--generations", type=int, default=10)
    parser.add_argument("--elites", type=int, default=3)
    parser.add_argument("--random-individuals", type=int, default=2)
    parser.add_argument("--sigma", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume-run", action="store_true")
    parser.add_argument("--skip-baseline", action="store_true")

    parser.add_argument("--train-timesteps", type=int, default=500_000)
    parser.add_argument("--learn-chunk", type=int, default=500_000)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--city", default="london")
    parser.add_argument("--base-port", type=int, default=8765)
    parser.add_argument("--trace-interval", type=float, default=30.0)
    parser.add_argument("--frame-stack", type=int, default=4)

    parser.add_argument("--eval-episodes", type=int, default=32)
    parser.add_argument("--eval-max-steps", type=int, default=4000)
    parser.add_argument("--eval-cities", default="london")
    parser.add_argument("--eval-complexities", default="4")
    parser.add_argument("--eval-spawn-factors", default="1.0")
    parser.add_argument("--eval-seeds", default="101,202,303,404")
    parser.add_argument("--eval-base-port", type=int, default=8965)
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
    parser.add_argument("--prune-noop-rate", type=float, default=0.85)
    parser.add_argument("--prune-zero-throughput-rate", type=float, default=0.75)
    parser.add_argument("--prune-default-delta", type=float, default=500.0)
    parser.add_argument("--min-eval-episodes", type=int, default=16)
    parser.add_argument("--min-eval-cities", type=int, default=1)
    parser.add_argument("--strict-guardrails", action="store_true")

    parser.add_argument("--retrain-top", type=int, default=0)
    parser.add_argument("--retrain-timesteps", type=int, default=20_000_000)
    parser.add_argument("--retrain-learn-chunk", type=int, default=0)
    parser.add_argument("--retrain-n-envs", type=int, default=0)
    parser.add_argument("--retrain-base-port", type=int, default=9165)
    parser.add_argument("--retrain-eval-episodes", type=int, default=0)
    parser.add_argument("--retrain-eval-max-steps", type=int, default=0)
    parser.add_argument("--retrain-eval-cities", default="")
    parser.add_argument("--retrain-eval-complexities", default="")
    parser.add_argument("--retrain-eval-spawn-factors", default="")
    parser.add_argument("--retrain-eval-seeds", default="")
    parser.add_argument("--retrain-eval-base-port", type=int, default=9365)
    args = parser.parse_args()

    if args.population < 1:
        raise ValueError("--population must be >= 1")
    if args.elites < 1 or args.elites > args.population:
        raise ValueError("--elites must be in [1, population]")
    if args.random_individuals < 0 or args.random_individuals > args.population:
        raise ValueError("--random-individuals must be in [0, population]")
    if args.retrain_top < 0:
        raise ValueError("--retrain-top must be >= 0")

    run_evolution(args)


if __name__ == "__main__":
    main()
