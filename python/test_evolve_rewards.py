import argparse
import json
import random
from pathlib import Path

from evolve_rewards import (
    PARAM_SPECS,
    Individual,
    build_eval_command,
    build_retrain_command,
    build_retrain_eval_command,
    build_train_command,
    evaluate_individual,
    crossover_config,
    git_metadata,
    initial_population,
    individual_manifest,
    retrain_finalists,
    mutate_config,
    next_generation,
    sample_config,
    select_elites,
    validate_guardrails,
)


def add_guardrail_args(args):
    args.fitness_version = "v2"
    args.week_weight = 1000.0
    args.score_weight = 2.0
    args.passenger_weight = 1.0
    args.std_week_penalty = 250.0
    args.invalid_action_penalty = 100.0
    args.noop_penalty = 400.0
    args.queue_penalty = 40.0
    args.overcrowd_penalty = 120.0
    args.danger_penalty = 100.0
    args.zero_throughput_penalty = 500.0
    args.prune_noop_rate = 0.85
    args.prune_zero_throughput_rate = 0.75
    args.prune_default_delta = 500.0
    args.eval_seeds = "101,202"
    args.retrain_eval_seeds = ""
    args.resume_run = False
    args.skip_baseline = True
    args.min_eval_episodes = 16
    args.min_eval_cities = 1
    args.strict_guardrails = False
    return args


def test_sample_and_mutate_config_stay_within_ranges():
    rng = random.Random(7)
    config = sample_config(rng)
    mutated = mutate_config(config, rng, sigma=0.5)

    for spec in PARAM_SPECS:
        assert spec.low <= config[spec.name] <= spec.high
        assert spec.low <= mutated[spec.name] <= spec.high


def test_crossover_uses_parent_values():
    rng = random.Random(1)
    parent_a = {spec.name: spec.low for spec in PARAM_SPECS}
    parent_b = {spec.name: spec.high for spec in PARAM_SPECS}

    child = crossover_config(parent_a, parent_b, rng)

    for spec in PARAM_SPECS:
        assert child[spec.name] in {parent_a[spec.name], parent_b[spec.name]}


def test_select_elites_and_next_generation_preserve_size():
    rng = random.Random(3)
    population = initial_population(4, rng)
    for idx, individual in enumerate(population):
        individual.fitness = float(idx)

    elites = select_elites(population, 2)
    assert [elite.fitness for elite in elites] == [3.0, 2.0]

    next_pop = next_generation(
        previous=population,
        generation=1,
        population_size=5,
        elite_count=2,
        random_count=1,
        sigma=0.2,
        rng=rng,
    )
    assert len(next_pop) == 5
    assert all(ind.generation == 1 for ind in next_pop)
    assert next_pop[0].parents == [elites[0].id]
    assert next_pop[0].origin == "elite_copy"
    assert "mutation_delta" in next_pop[2].metadata


def test_build_commands_point_to_individual_artifacts(tmp_path):
    args = add_guardrail_args(argparse.Namespace(
        n_envs=2,
        city="london",
        train_timesteps=1024,
        learn_chunk=1024,
        base_port=8765,
        trace_interval=99.0,
        frame_stack=4,
        eval_episodes=3,
        eval_max_steps=100,
        eval_cities="london",
        eval_complexities="4",
        eval_spawn_factors="1.0",
        eval_base_port=8965,
        retrain_timesteps=2048,
        retrain_learn_chunk=0,
        retrain_n_envs=0,
        retrain_base_port=9165,
        retrain_eval_episodes=0,
        retrain_eval_max_steps=0,
        retrain_eval_cities="",
        retrain_eval_complexities="",
        retrain_eval_spawn_factors="",
        retrain_eval_base_port=9365,
    ))
    ind_dir = tmp_path / "individual_000"
    config_path = ind_dir / "reward_config.json"

    train_cmd = build_train_command(args, ind_dir, config_path)
    eval_cmd = build_eval_command(args, ind_dir, ind_dir / "eval.json")

    assert "train.py" in train_cmd
    assert "--reward-config" in train_cmd
    assert str(config_path) in train_cmd
    assert str(ind_dir / "checkpoints") in train_cmd
    assert "eval_suite.py" in eval_cmd
    assert str(ind_dir / "checkpoints" / "minimetro_latest.zip") in eval_cmd
    assert "--std-week-penalty" in eval_cmd

    retrain_cmd = build_retrain_command(args, ind_dir, config_path)
    retrain_eval_cmd = build_retrain_eval_command(args, ind_dir, ind_dir / "eval.json")
    assert "--total-timesteps" in retrain_cmd
    assert "2048" in retrain_cmd
    assert str(ind_dir / "checkpoints" / "minimetro_latest.zip") in retrain_eval_cmd


def test_individual_manifest_records_paths_and_commands(tmp_path):
    args = add_guardrail_args(argparse.Namespace(
        n_envs=2,
        city="london",
        train_timesteps=1024,
        learn_chunk=1024,
        base_port=8765,
        trace_interval=99.0,
        frame_stack=4,
        eval_episodes=3,
        eval_max_steps=100,
        eval_cities="london",
        eval_complexities="4",
        eval_spawn_factors="1.0",
        eval_base_port=8965,
        retrain_timesteps=2048,
        retrain_learn_chunk=0,
        retrain_n_envs=0,
        retrain_base_port=9165,
        retrain_eval_episodes=0,
        retrain_eval_max_steps=0,
        retrain_eval_cities="",
        retrain_eval_complexities="",
        retrain_eval_spawn_factors="",
        retrain_eval_base_port=9365,
    ))
    individual = Individual(
        id="individual_000",
        generation=0,
        config=sample_config(random.Random(1)),
        parents=[],
        origin="random_initial",
        metadata={},
    )
    manifest = individual_manifest(
        args,
        individual,
        tmp_path / "individual_000",
        tmp_path / "individual_000" / "reward_config.json",
        tmp_path / "individual_000" / "eval.json",
    )

    assert manifest["individual"]["origin"] == "random_initial"
    assert manifest["paths"]["reward_config"].endswith("reward_config.json")
    assert manifest["commands"]["train"][1] == "train.py"
    assert manifest["commands"]["eval"][1] == "eval_suite.py"


def test_evaluate_individual_dry_run_writes_manifest(tmp_path):
    args = add_guardrail_args(argparse.Namespace(
        dry_run=True,
        n_envs=2,
        city="london",
        train_timesteps=1024,
        learn_chunk=1024,
        base_port=8765,
        trace_interval=99.0,
        frame_stack=4,
        eval_episodes=3,
        eval_max_steps=100,
        eval_cities="london",
        eval_complexities="4",
        eval_spawn_factors="1.0",
        eval_base_port=8965,
        retrain_timesteps=2048,
        retrain_learn_chunk=0,
        retrain_n_envs=0,
        retrain_base_port=9165,
        retrain_eval_episodes=0,
        retrain_eval_max_steps=0,
        retrain_eval_cities="",
        retrain_eval_complexities="",
        retrain_eval_spawn_factors="",
        retrain_eval_base_port=9365,
    ))
    individual = Individual(
        id="individual_000",
        generation=0,
        config=sample_config(random.Random(1)),
        parents=[],
        origin="random_initial",
        metadata={},
    )

    evaluated = evaluate_individual(args, individual, tmp_path)
    manifest_path = tmp_path / "generation_000" / "individual_000" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert evaluated.fitness == 0.0
    assert manifest["individual"]["eval_summary"]["dry_run"] is True


def test_retrain_finalists_dry_run_writes_summary(tmp_path):
    args = add_guardrail_args(argparse.Namespace(
        dry_run=True,
        retrain_top=2,
        retrain_timesteps=2048,
        retrain_learn_chunk=0,
        retrain_n_envs=0,
        n_envs=2,
        city="london",
        learn_chunk=1024,
        retrain_base_port=9165,
        trace_interval=99.0,
        frame_stack=4,
        retrain_eval_episodes=0,
        eval_episodes=3,
        retrain_eval_max_steps=0,
        eval_max_steps=100,
        retrain_eval_cities="",
        eval_cities="london",
        retrain_eval_complexities="",
        eval_complexities="4",
        retrain_eval_spawn_factors="",
        eval_spawn_factors="1.0",
        retrain_eval_base_port=9365,
    ))
    rng = random.Random(5)
    individuals = [
        Individual(
            id=f"individual_{idx:03d}",
            generation=0,
            config=sample_config(rng),
            parents=[],
            origin="random_initial",
            metadata={},
            fitness=float(idx),
        )
        for idx in range(3)
    ]

    results = retrain_finalists(args, individuals, tmp_path)
    summary = json.loads((tmp_path / "retrain" / "summary.json").read_text(encoding="utf-8"))

    assert len(results) == 2
    assert summary["best"]["summary"]["dry_run"] is True
    assert (tmp_path / "retrain" / "finalist_000" / "manifest.json").exists()


def test_validate_guardrails_warns_or_raises_for_weak_eval():
    args = argparse.Namespace(
        eval_episodes=4,
        min_eval_episodes=16,
        eval_cities="london",
        min_eval_cities=1,
        strict_guardrails=False,
        retrain_top=0,
        retrain_eval_cities="",
        eval_spawn_factors="1.0",
        retrain_eval_spawn_factors="",
    )

    warnings = validate_guardrails(args)
    assert "below minimum" in warnings[0]

    args.strict_guardrails = True
    try:
        validate_guardrails(args)
    except ValueError as exc:
        assert "below minimum" in str(exc)
    else:
        raise AssertionError("strict guardrails should raise")


def test_git_metadata_has_expected_keys():
    metadata = git_metadata(Path(__file__).resolve().parents[1])
    assert {"commit", "branch", "dirty", "untracked"} <= set(metadata)
