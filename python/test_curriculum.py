import numpy as np

from constants import NUM_ACTION_CATS
from train import (
    ENT_END,
    ENT_START,
    EntropyScheduleCallback,
    CurriculumCallback,
    RolloutDiagnosticsCallback,
)


class FakeVecEnv:
    def __init__(self):
        self.calls = []

    def env_method(self, method_name, *args, **kwargs):
        self.calls.append((method_name, args))
        return [None]


def test_curriculum_sets_complexity_before_spawn_rate():
    env = FakeVecEnv()
    callback = CurriculumCallback(env, verbose=0)

    assert env.calls[:2] == [
        ("set_complexity", (0,)),
        ("set_difficulty", (4.0,)),
    ]

    callback._set_difficulty(3)
    assert env.calls[-2:] == [
        ("set_complexity", (3,)),
        ("set_difficulty", (1.5,)),
    ]


class FakeLogger:
    def __init__(self):
        self.records = {}

    def record(self, key, value):
        self.records[key] = value


class FakeModel:
    def __init__(self, timesteps):
        self.num_timesteps = timesteps
        self.ent_coef = None
        self.logger = FakeLogger()


def test_entropy_schedule_decays_and_clips():
    callback = EntropyScheduleCallback(start=0.02, end=0.002, decay_steps=100)
    callback.model = FakeModel(50)

    assert callback._on_step()
    assert callback.model.ent_coef == 0.011
    assert callback.model.logger.records["train/ent_coef"] == 0.011

    callback.model.num_timesteps = 500
    assert callback._on_step()
    assert callback.model.ent_coef == ENT_END

    callback.model.num_timesteps = 0
    assert callback._on_step()
    assert callback.model.ent_coef == ENT_START


def test_rollout_diagnostics_logs_game_and_action_metrics():
    callback = RolloutDiagnosticsCallback()
    callback.model = FakeModel(0)
    callback.locals = {
        "infos": [
            {
                "score": 10,
                "passengers_delivered": 4,
                "week": 2,
                "stations": 5,
                "queue_pressure": 1.5,
                "overcrowd_pressure": 0.25,
                "danger_count": 1,
            },
            {
                "score": 20,
                "passengers_delivered": 6,
                "week": 3,
                "stations": 7,
                "queue_pressure": 2.5,
                "overcrowd_pressure": 0.75,
                "danger_count": 0,
            },
        ],
        "actions": np.array([[0, 0, 0, 0], [7, 0, 0, 0]], dtype=np.int64),
    }

    assert callback._on_step()
    records = callback.model.logger.records
    assert records["game/score"] == 15.0
    assert records["game/queue_pressure"] == 2.0
    assert records["actions/noop_rate"] == 0.5
    assert records["actions/cat_0_rate"] == 0.5
    assert records["actions/cat_7_rate"] == 0.5
    assert f"actions/cat_{NUM_ACTION_CATS - 1}_rate" in records
