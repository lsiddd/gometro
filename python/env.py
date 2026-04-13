"""
MiniMetroEnv — Gymnasium wrapper around the Go rl_server gRPC API.

Each environment instance manages its own Go subprocess so that multiple
environments can run in parallel (via stable-baselines3's SubprocVecEnv)
without port conflicts.

Transport: gRPC with bidirectional streaming for RunEpisode.
  - Reset()      → unary RPC   (once per episode)
  - RunEpisode() → bidi stream (all steps within an episode share one stream)

This eliminates the per-step TCP connection overhead of the old HTTP/JSON
transport and replaces JSON float serialization with protobuf packed arrays.
"""
from __future__ import annotations

import os
import queue
import signal
import subprocess
import threading
import time
from typing import Any

import grpc
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from constants import OBS_DIM, ACTION_DIMS, validate_server_constants
from rl.proto import minimetro_pb2 as pb
from rl.proto import minimetro_pb2_grpc as pb_grpc

# Path to the compiled rl_server binary (relative to this script).
_DEFAULT_BINARY = os.path.join(os.path.dirname(__file__), "..", "rl_server")

_MASK_SIZE = sum(ACTION_DIMS)  # = 73


class MiniMetroEnv(gym.Env):
    """Single Mini Metro game environment backed by a Go gRPC server.

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
        self._log_file = None

        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(OBS_DIM,), dtype=np.float32
        )
        self.action_space = spaces.MultiDiscrete(ACTION_DIMS)

        self._mask: np.ndarray = np.ones(_MASK_SIZE, dtype=bool)
        # Curriculum learning difficulty: spawn_rate_factor > 1.0 = easier.
        # Updated between episodes by CurriculumCallback via set_difficulty().
        self.difficulty: float = 1.0

        # gRPC channel and stubs — created in _connect().
        self._channel: grpc.Channel | None = None
        self._stub: pb_grpc.RLEnvStub | None = None

        # Bidirectional stream state for RunEpisode.
        self._stream = None
        self._action_queue: queue.Queue = queue.Queue()
        self._stream_thread: threading.Thread | None = None

        if managed:
            self._start_server()
        else:
            self._connect()

    # ── gymnasium interface ───────────────────────────────────────────────────

    def set_difficulty(self, spawn_rate_factor: float) -> None:
        """Set the curriculum difficulty for subsequent episodes.

        spawn_rate_factor > 1.0 stretches spawn intervals (easier);
        1.0 = normal game difficulty.  Called by CurriculumCallback via
        SubprocVecEnv.env_method().
        """
        self.difficulty = spawn_rate_factor

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        city = (options or {}).get("city", self.city)
        spawn_rate_factor = (options or {}).get(
            "spawn_rate_factor", getattr(self, "difficulty", 1.0)
        )

        # Close any open stream from the previous episode before resetting.
        self._close_stream()

        resp = self._stub.Reset(
            pb.ResetRequest(city=city, spawn_rate_factor=spawn_rate_factor)
        )
        obs = np.array(resp.obs, dtype=np.float32)
        self._mask = np.array(resp.mask, dtype=bool)

        # Open a new RunEpisode stream for the upcoming episode steps.
        self._open_stream()

        return obs, {}

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        # Send action into the stream's action queue.
        self._action_queue.put(action.tolist())
        # Block until the stream thread delivers the StepResponse.
        resp: pb.StepResponse = self._resp_queue.get()

        obs = np.array(resp.obs, dtype=np.float32)
        self._mask = np.array(resp.mask, dtype=bool)
        done = resp.done
        info = {
            "score":                resp.score,
            "passengers_delivered": resp.passengers_delivered,
            "week":                 resp.week,
            "stations":             resp.stations,
            "game_over":            resp.game_over,
            "in_upgrade_modal":     resp.in_upgrade_modal,
        }
        return obs, resp.reward, done, False, info  # truncated=False

    def action_masks(self) -> np.ndarray:
        """Return the current boolean action mask (for sb3-contrib MaskablePPO)."""
        return self._mask.copy()

    def close(self):
        self._close_stream()
        if self._channel is not None:
            self._channel.close()
            self._channel = None
            self._stub = None
        if self.managed and self._proc is not None:
            self._proc.send_signal(signal.SIGTERM)
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                self._proc.wait()
            self._proc = None
        if self._log_file is not None:
            self._log_file.close()
            self._log_file = None

    # ── gRPC stream management ────────────────────────────────────────────────

    def _open_stream(self):
        """Open a new RunEpisode bidirectional stream for one episode."""
        self._resp_queue: queue.Queue = queue.Queue()

        def _action_iter():
            """Generator that yields ActionRequests from the action queue."""
            while True:
                action = self._action_queue.get()
                if action is None:  # sentinel: close stream
                    return
                yield pb.ActionRequest(action=action)

        self._stream = self._stub.RunEpisode(_action_iter())

        def _reader():
            """Background thread that reads StepResponses and enqueues them."""
            try:
                for resp in self._stream:
                    self._resp_queue.put(resp)
            except grpc.RpcError:
                pass  # stream closed cleanly

        self._stream_thread = threading.Thread(target=_reader, daemon=True)
        self._stream_thread.start()

    def _close_stream(self):
        """Signal the action generator to stop and wait for the reader thread."""
        if self._stream_thread is not None:
            self._action_queue.put(None)  # sentinel
            self._stream_thread.join(timeout=3)
            self._stream = None
            self._stream_thread = None

    # ── helpers ───────────────────────────────────────────────────────────────

    def _connect(self):
        """Open the gRPC channel and validate server constants."""
        self._channel = grpc.insecure_channel(f"localhost:{self.port}")
        self._stub = pb_grpc.RLEnvStub(self._channel)

        # Validate that Go and Python constants agree.
        info_resp = self._stub.Info(pb.Empty())
        validate_server_constants({
            "obs_dim":      info_resp.obs_dim,
            "action_dims":  list(info_resp.action_dims),
            "global_dim":   info_resp.global_dim,
            "station_dim":  info_resp.station_dim,
            "num_stations": info_resp.num_stations,
            "line_dim":     info_resp.line_dim,
            "num_lines":    info_resp.num_lines,
        })

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
        # Wait for the gRPC server to become ready.
        deadline = time.time() + 10.0
        while time.time() < deadline:
            try:
                channel = grpc.insecure_channel(f"localhost:{self.port}")
                stub = pb_grpc.RLEnvStub(channel)
                stub.Info(pb.Empty(), timeout=0.5)
                channel.close()
                break
            except grpc.RpcError:
                time.sleep(0.1)
        else:
            raise RuntimeError(
                f"rl_server on port {self.port} did not start within 10 s"
            )
        self._connect()
