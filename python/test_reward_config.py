import json

import pytest

from env import MiniMetroVecEnv, reward_config_request
from train import load_reward_config


REWARD_CONFIG = {
    "per_passenger": 3.0,
    "queue_coeff": 0.03,
    "queue_delta_coeff": 0.2,
    "overcrowd_coeff": 0.75,
    "overcrowd_delta_coeff": 2.0,
    "danger_thresh": 0.8,
    "danger_penalty": 0.5,
    "noop_critical_penalty": 0.25,
    "week_bonus": 20.0,
    "terminal_penalty": 100.0,
    "invalid_action": 1.0,
}


def test_reward_config_request_requires_exact_fields():
    req = reward_config_request(REWARD_CONFIG)
    assert req.per_passenger == 3.0
    assert req.noop_critical_penalty == 0.25

    missing = dict(REWARD_CONFIG)
    del missing["queue_coeff"]
    with pytest.raises(ValueError, match="missing fields"):
        reward_config_request(missing)

    unknown = dict(REWARD_CONFIG)
    unknown["extra"] = 1
    with pytest.raises(ValueError, match="unknown fields"):
        reward_config_request(unknown)


def test_vec_env_set_reward_config_stores_and_sends_request():
    env = MiniMetroVecEnv.__new__(MiniMetroVecEnv)
    env.num_envs = 2
    env._channel = object()
    calls = []
    env._set_server_reward_config = calls.append

    assert env.env_method("set_reward_config", REWARD_CONFIG) == [None, None]

    assert env.reward_config == REWARD_CONFIG
    assert len(calls) == 1
    assert calls[0].terminal_penalty == 100.0


def test_load_reward_config_reads_json_as_floats(tmp_path):
    path = tmp_path / "reward_config.json"
    path.write_text(json.dumps(REWARD_CONFIG), encoding="utf-8")

    loaded = load_reward_config(str(path))

    assert loaded == REWARD_CONFIG
    assert all(isinstance(value, float) for value in loaded.values())
