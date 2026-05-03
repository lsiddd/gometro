import pytest

from eval_suite import (
    EpisodeMetrics,
    compute_fitness,
    compute_fitness_components,
    parse_csv,
    summarize_episodes,
)


def test_summarize_episodes_and_fitness_prioritize_external_metrics():
    episodes = [
        EpisodeMetrics(
            city="london",
            complexity=4,
            spawn_rate_factor=1.0,
            episode=0,
            score=100,
            week=2,
            passengers_delivered=50,
            steps=10,
            stations=6,
            invalid_action_rate=0.0,
            noop_rate=0.2,
            danger_count_mean=0.1,
            queue_pressure_mean=1.0,
            overcrowd_pressure_mean=0.5,
        ),
        EpisodeMetrics(
            city="london",
            complexity=4,
            spawn_rate_factor=1.0,
            episode=1,
            score=200,
            week=4,
            passengers_delivered=80,
            steps=20,
            stations=9,
            invalid_action_rate=0.1,
            noop_rate=0.4,
            danger_count_mean=0.3,
            queue_pressure_mean=2.0,
            overcrowd_pressure_mean=0.7,
        ),
    ]

    summary = summarize_episodes(episodes)

    assert summary["episodes"] == 2.0
    assert summary["mean_score"] == 150.0
    assert summary["mean_week"] == 3.0
    assert summary["mean_invalid_action_rate"] == 0.05
    assert summary["zero_throughput_rate"] == 0.0
    assert "noop" in summary["fitness_components"]
    assert summary["fitness"] == compute_fitness(summary)

    custom_summary = summarize_episodes(
        episodes,
        {
            "week_weight": 1.0,
            "fitness_version": "v2",
            "score_weight": 0.0,
            "passenger_weight": 0.0,
            "std_week_penalty": 0.0,
            "invalid_action_penalty": 0.0,
            "noop_penalty": 0.0,
            "queue_penalty": 0.0,
            "overcrowd_penalty": 0.0,
            "danger_penalty": 0.0,
            "zero_throughput_penalty": 0.0,
        },
    )
    assert custom_summary["fitness"] == 3.0


def test_fitness_v2_penalizes_passive_policies():
    active = {
        "mean_week": 2.0,
        "mean_score": 50.0,
        "mean_passengers_delivered": 50.0,
        "std_week": 0.0,
        "mean_invalid_action_rate": 0.0,
        "mean_noop_rate": 0.0,
        "mean_queue_pressure": 1.0,
        "mean_overcrowd_pressure": 0.1,
        "mean_danger_count": 0.0,
        "zero_throughput_rate": 0.0,
    }
    passive = {
        **active,
        "mean_score": 0.0,
        "mean_passengers_delivered": 0.0,
        "mean_noop_rate": 1.0,
        "zero_throughput_rate": 1.0,
    }

    assert compute_fitness(active) > compute_fitness(passive)
    assert compute_fitness_components(passive)["zero_throughput"] < 0


def test_summarize_episodes_rejects_empty_list():
    with pytest.raises(ValueError, match="empty episode list"):
        summarize_episodes([])


def test_parse_csv_casts_and_ignores_empty_parts():
    assert parse_csv("london, paris,,tokyo", str) == ["london", "paris", "tokyo"]
    assert parse_csv("4,3", int) == [4, 3]
    assert parse_csv("1.25,1.0", float) == [1.25, 1.0]
