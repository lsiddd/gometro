"""
Integration tests for the Go rl_server ↔ Python boundary.

These tests start a real rl_server subprocess (managed=True) and exercise the
full reset/step/close cycle.  They are skipped automatically when the
rl_server binary is absent so that the test suite stays green in environments
where Go has not been compiled yet.

Build the binary first:
    go build -o rl_server ./cmd/rl_server/
Then run:
    cd python && uv run pytest test_integration.py -v
"""
from __future__ import annotations

import os
import time

import numpy as np
import pytest
import requests

from constants import OBS_DIM, ACTION_DIMS
from env import MiniMetroEnv

_BINARY = os.path.join(os.path.dirname(__file__), "..", "rl_server")
_BASE_PORT = 18700  # well outside the default training range


def _binary_exists() -> bool:
    return os.path.isfile(os.path.abspath(_BINARY))


skip_no_binary = pytest.mark.skipif(
    not _binary_exists(),
    reason="rl_server binary not found — run: go build -o rl_server ./cmd/rl_server/",
)


# ── fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def env():
    """One managed env for the entire test module (server starts once)."""
    e = MiniMetroEnv(port=_BASE_PORT, city="london", managed=True)
    yield e
    e.close()


# ── /info validation ──────────────────────────────────────────────────────────

@skip_no_binary
def test_info_obs_dim_matches_constants(env):
    """Server-reported obs_dim must match OBS_DIM in constants.py.

    Uses the env fixture to ensure the server is running before the request.
    """
    resp = requests.get(f"http://localhost:{_BASE_PORT}/info", timeout=5)
    assert resp.status_code == 200
    data = resp.json()
    assert data["obs_dim"] == OBS_DIM, (
        f"server obs_dim={data['obs_dim']} != constants.OBS_DIM={OBS_DIM}"
    )


@skip_no_binary
def test_info_action_dims_match_constants(env):
    resp = requests.get(f"http://localhost:{_BASE_PORT}/info", timeout=5)
    data = resp.json()
    assert data["action_dims"] == list(ACTION_DIMS), (
        f"server action_dims={data['action_dims']} != constants.ACTION_DIMS={list(ACTION_DIMS)}"
    )


# ── reset ─────────────────────────────────────────────────────────────────────

@skip_no_binary
def test_reset_obs_shape(env):
    obs, info = env.reset()
    assert obs.shape == (OBS_DIM,), f"obs shape: want ({OBS_DIM},), got {obs.shape}"


@skip_no_binary
def test_reset_obs_in_unit_range(env):
    obs, _ = env.reset()
    assert obs.dtype == np.float32
    assert np.all(obs >= 0.0), "obs values must be ≥ 0"
    assert np.all(obs <= 1.0), "obs values must be ≤ 1"


@skip_no_binary
def test_reset_mask_length(env):
    env.reset()
    mask = env.action_masks()
    expected_len = sum(ACTION_DIMS)
    assert len(mask) == expected_len, (
        f"mask length: want {expected_len}, got {len(mask)}"
    )


@skip_no_binary
def test_reset_mask_has_valid_actions(env):
    env.reset()
    mask = env.action_masks()
    assert mask.any(), "at least one action must be valid after reset"


# ── step ──────────────────────────────────────────────────────────────────────

@skip_no_binary
def test_step_noop_returns_correct_shapes(env):
    env.reset()
    noop = np.array([0, 0, 0, 0])  # ActNoOp
    obs, reward, done, truncated, info = env.step(noop)

    assert obs.shape == (OBS_DIM,)
    assert isinstance(reward, float)
    assert isinstance(done, bool)
    assert truncated is False
    assert isinstance(info, dict)


@skip_no_binary
def test_step_obs_in_unit_range(env):
    env.reset()
    noop = np.array([0, 0, 0, 0])
    obs, _, _, _, _ = env.step(noop)

    assert np.all(obs >= 0.0), "obs after step must be ≥ 0"
    assert np.all(obs <= 1.0), "obs after step must be ≤ 1"


@skip_no_binary
def test_step_mask_refreshes(env):
    env.reset()
    mask_before = env.action_masks().copy()
    noop = np.array([0, 0, 0, 0])
    env.step(noop)
    mask_after = env.action_masks()
    # Mask is not guaranteed to change, but it must remain a valid bool array.
    assert mask_after.dtype == bool
    assert len(mask_after) == len(mask_before)


@skip_no_binary
def test_multiple_steps_no_crash(env):
    env.reset()
    noop = np.array([0, 0, 0, 0])
    for _ in range(20):
        obs, reward, done, _, _ = env.step(noop)
        if done:
            env.reset()


# ── reset idempotency ─────────────────────────────────────────────────────────

@skip_no_binary
def test_double_reset_is_safe(env):
    obs1, _ = env.reset()
    obs2, _ = env.reset()
    # Two resets produce valid obs; they need not be identical (RNG).
    assert obs1.shape == obs2.shape == (OBS_DIM,)
