"""
MiniMetroEnv — Gymnasium wrapper around the Go rl_server HTTP API.

Each environment instance manages its own Go subprocess so that multiple
environments can run in parallel (via stable-baselines3's SubprocVecEnv)
without port conflicts.
"""
from __future__ import annotations

import os
import signal
import subprocess
import time
from typing import Any

import numpy as np
import requests
import gymnasium as gym
from gymnasium import spaces

# Path to the compiled rl_server binary (relative to this script).
_DEFAULT_BINARY = os.path.join(os.path.dirname(__file__), "..", "rl_server")

OBS_DIM = 514
NUM_ACTIONS = 1845


class MiniMetroEnv(gym.Env):
    """Single Mini Metro game environment backed by a Go HTTP server.

    Parameters
    ----------
    port:
        TCP port for the Go rl_server.  Each parallel env should use a
        different port (e.g. base_port + worker_index).
    city:
        City name passed to the Go server on reset (e.g. "london").
    binary:
        Path to the rl_server executable.  Defaults to ../rl_server relative
        to this file.
    managed:
        If True, this env launches and terminates the Go subprocess.
        Set to False when the server is started externally.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        port: int = 8765,
        city: str = "london",
        binary: str = _DEFAULT_BINARY,
        managed: bool = True,
    ):
        super().__init__()
        self.port = port
        self.city = city
        self.binary = binary
        self.managed = managed
        self._proc: subprocess.Popen | None = None
        self._base_url = f"http://localhost:{port}"

        self.observation_space = spaces.Box(
            low=-1.0, high=10.0, shape=(OBS_DIM,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(NUM_ACTIONS)

        # Latest mask, updated after every reset/step.
        self._mask: np.ndarray = np.ones(NUM_ACTIONS, dtype=bool)

        if managed:
            self._start_server()

    # ── gymnasium interface ───────────────────────────────────────────────────

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        city = (options or {}).get("city", self.city)
        resp = self._post("/reset", {"city": city})
        obs = np.array(resp["obs"], dtype=np.float32)
        self._mask = np.array(resp["mask"], dtype=bool)
        return obs, {}

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        resp = self._post("/step", {"action": int(action)})
        obs = np.array(resp["obs"], dtype=np.float32)
        reward = float(resp["reward"])
        done = bool(resp["done"])
        self._mask = np.array(resp["mask"], dtype=bool)
        info = resp.get("info", {})
        return obs, reward, done, False, info  # truncated=False

    def action_masks(self) -> np.ndarray:
        """Return the current boolean action mask (for sb3-contrib MaskablePPO)."""
        return self._mask.copy()

    def close(self):
        if self.managed and self._proc is not None:
            self._proc.send_signal(signal.SIGTERM)
            self._proc.wait(timeout=5)
            self._proc = None

    # ── helpers ───────────────────────────────────────────────────────────────

    def _start_server(self):
        binary = os.path.abspath(self.binary)
        if not os.path.isfile(binary):
            raise FileNotFoundError(
                f"rl_server binary not found at {binary}. "
                "Run: go build -o rl_server ./cmd/rl_server/"
            )
        self._proc = subprocess.Popen(
            [binary, "--port", str(self.port)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        # Wait for the server to be ready.
        deadline = time.time() + 10.0
        while time.time() < deadline:
            try:
                requests.get(f"{self._base_url}/info", timeout=0.5)
                return
            except requests.exceptions.ConnectionError:
                time.sleep(0.1)
        raise RuntimeError(
            f"rl_server on port {self.port} did not start within 10 s"
        )

    def _post(self, path: str, body: dict) -> dict[str, Any]:
        resp = requests.post(f"{self._base_url}{path}", json=body, timeout=30)
        resp.raise_for_status()
        return resp.json()
