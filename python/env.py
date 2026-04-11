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

from constants import OBS_DIM, ACTION_DIMS, validate_server_constants
from http_schemas import parse_reset_response, parse_step_response

# Path to the compiled rl_server binary (relative to this script).
_DEFAULT_BINARY = os.path.join(os.path.dirname(__file__), "..", "rl_server")


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

        # All observation features are normalised to [0, 1] by BuildObservation.
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(OBS_DIM,), dtype=np.float32
        )
        self.action_space = spaces.MultiDiscrete(ACTION_DIMS)

        # Latest mask, updated after every reset/step.
        self._mask: np.ndarray = np.ones(sum(ACTION_DIMS), dtype=bool)

        if managed:
            self._start_server()

    # ── gymnasium interface ───────────────────────────────────────────────────

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        # The Go simulation is deterministic given its own RNG; the Python seed
        # is forwarded to the Go server as a hint for reproducibility, but the
        # server currently ignores it (Go uses its own global rand source).
        city = (options or {}).get("city", self.city)
        resp = parse_reset_response(self._post("/reset", {"city": city}))
        obs = np.array(resp["obs"], dtype=np.float32)
        self._mask = np.array(resp["mask"], dtype=bool)
        return obs, {}

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        resp = parse_step_response(self._post("/step", {"action": action.tolist()}))
        obs = np.array(resp["obs"], dtype=np.float32)
        reward = float(resp["reward"])
        done = bool(resp["done"])
        self._mask = np.array(resp["mask"], dtype=bool)
        return obs, reward, done, False, resp["info"]  # truncated=False

    def action_masks(self) -> np.ndarray:
        """Return the current boolean action mask (for sb3-contrib MaskablePPO)."""
        return self._mask.copy()

    def close(self):
        if self.managed and self._proc is not None:
            self._proc.send_signal(signal.SIGTERM)
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                self._proc.wait()
            self._proc = None
        if hasattr(self, "_log_file") and self._log_file:
            self._log_file.close()
            self._log_file = None

    # ── helpers ───────────────────────────────────────────────────────────────

    def _start_server(self):
        binary = os.path.abspath(self.binary)
        if not os.path.isfile(binary):
            raise FileNotFoundError(
                f"rl_server binary not found at {binary}. "
                "Run: go build -o rl_server ./cmd/rl_server/"
            )
        log_path = f"/tmp/rl_server_{self.port}.log"
        self._log_file = open(log_path, "w")
        self._proc = subprocess.Popen(
            [binary, "--port", str(self.port)],
            stdout=self._log_file,
            stderr=self._log_file,
        )
        # Wait for the server to be ready.
        deadline = time.time() + 10.0
        while time.time() < deadline:
            try:
                resp = requests.get(f"{self._base_url}/info", timeout=0.5)
                validate_server_constants(resp.json())
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
