from __future__ import annotations

import os
import queue
import signal
import subprocess
import threading
import time
from typing import Any, Mapping

import grpc
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from stable_baselines3.common.vec_env import VecEnv, VecEnvWrapper
from stable_baselines3.common.vec_env.base_vec_env import VecEnvStepReturn

from constants import (
    OBS_DIM,
    ACTION_DIMS,
    MASK_SIZE,
    GLOBAL_DIM,
    STATION_DIM,
    NUM_STATIONS,
    COND_LINE_OFFSET,
    COND_STATION_OFFSET,
    COND_OPTION_OFFSET,
    MAX_LINE_SLOTS,
    MAX_STATION_SLOTS,
    NUM_ACTION_CATS,
    NUM_OPTIONS,
    validate_server_constants,
)
from rl.proto import minimetro_pb2 as pb
from rl.proto import minimetro_pb2_grpc as pb_grpc

_DEFAULT_BINARY = os.path.join(os.path.dirname(__file__), "..", "rl_server")
_MASK_SIZE = MASK_SIZE
_STEP_TIMEOUT_S = float(os.environ.get("MINIMETRO_STEP_TIMEOUT", "30"))
_STATION_OFFSET = GLOBAL_DIM
_PAX_RATIO_IDX = 3
_OVERCROWD_IDX = 4

_REWARD_CONFIG_FIELDS = (
    "per_passenger",
    "queue_coeff",
    "queue_delta_coeff",
    "overcrowd_coeff",
    "overcrowd_delta_coeff",
    "danger_thresh",
    "danger_penalty",
    "noop_critical_penalty",
    "week_bonus",
    "terminal_penalty",
    "invalid_action",
)


def _actions_valid_for_masks(actions: np.ndarray, masks: np.ndarray) -> np.ndarray:
    """Validate [act, line, station, option] against conditional masks."""
    actions = np.asarray(actions, dtype=np.int64).reshape((-1, len(ACTION_DIMS)))
    masks = np.asarray(masks, dtype=bool).reshape((actions.shape[0], _MASK_SIZE))
    valid = np.ones(actions.shape[0], dtype=bool)
    for i, (act, line, station, option) in enumerate(actions):
        if not (0 <= act < NUM_ACTION_CATS and masks[i, act]):
            valid[i] = False
            continue
        line_off = COND_LINE_OFFSET + act * MAX_LINE_SLOTS + line
        station_off = (
            COND_STATION_OFFSET
            + ((act * MAX_LINE_SLOTS + line) * MAX_STATION_SLOTS)
            + station
        )
        option_off = COND_OPTION_OFFSET + act * NUM_OPTIONS + option
        if not (
            0 <= line < MAX_LINE_SLOTS
            and 0 <= station < MAX_STATION_SLOTS
            and 0 <= option < NUM_OPTIONS
            and masks[i, line_off]
            and masks[i, station_off]
            and masks[i, option_off]
        ):
            valid[i] = False
    return valid


def reward_config_request(config: Mapping[str, Any]) -> pb.RewardConfigRequest:
    """Build a typed protobuf request from a reward-config mapping."""
    missing = [field for field in _REWARD_CONFIG_FIELDS if field not in config]
    if missing:
        raise ValueError(f"reward config missing fields: {', '.join(missing)}")
    unknown = sorted(set(config) - set(_REWARD_CONFIG_FIELDS))
    if unknown:
        raise ValueError(f"reward config has unknown fields: {', '.join(unknown)}")
    return pb.RewardConfigRequest(
        **{field: float(config[field]) for field in _REWARD_CONFIG_FIELDS}
    )

class MiniMetroVecEnv(VecEnv):
    def __init__(
        self,
        n_envs: int,
        port: int = 8765,
        city: str = "london",
        binary: str = _DEFAULT_BINARY,
        managed: bool = True,
        trace_interval: float = 10.0,
        seed: int = 0,
    ):
        observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(OBS_DIM,), dtype=np.float32
        )
        action_space = spaces.MultiDiscrete(ACTION_DIMS)
        super().__init__(n_envs, observation_space, action_space)

        self.port = port
        self.city = city
        self.binary = binary
        self.managed = managed
        self.trace_interval = trace_interval
        self.seed = int(seed)
        self._proc: subprocess.Popen | None = None
        self._log_file = None

        self._mask = np.ones((n_envs, _MASK_SIZE), dtype=bool)
        self.difficulty = 1.0
        self.complexity = 4
        self.reward_config: dict[str, float] | None = None

        self._channel: grpc.Channel | None = None
        self._stub: pb_grpc.RLEnvStub | None = None
        self._control_stub: pb_grpc.ControlStub | None = None

        self._stream = None
        self._action_queue: queue.Queue = queue.Queue()
        self._stream_thread: threading.Thread | None = None
        self._stream_error: grpc.RpcError | None = None
        self._step_count = 0
        self._last_trace = time.perf_counter()
        self._last_step_count = 0
        self._step_wait_acc = 0.0
        self._step_wait_max = 0.0
        self._done_count = 0

        if managed:
            self._start_server()
        else:
            self._connect()

        self._actions_buffer = None

    def env_method(self, method_name: str, *args, indices=None, **kwargs) -> list[Any]:
        if method_name == "set_difficulty":
            self.difficulty = args[0]
            self._trace(f"set_difficulty factor={self.difficulty:.3f}")
            self._set_server_difficulty(self.difficulty)
            return [None] * self.num_envs
        if method_name == "set_complexity":
            self.complexity = int(args[0])
            self._trace(f"set_complexity level={self.complexity}")
            self._set_server_complexity(self.complexity)
            return [None] * self.num_envs
        if method_name == "set_reward_config":
            self.set_reward_config(args[0])
            return [None] * self.num_envs
        if method_name == "action_masks":
            return list(self._mask)
        raise NotImplementedError(method_name)

    def env_is_wrapped(self, wrapper_class: type, indices=None) -> list[bool]:
        return [False] * self.num_envs

    def get_attr(self, attr_name: str, indices=None) -> list[Any]:
        if attr_name == "difficulty":
            return [self.difficulty] * self.num_envs
        if attr_name == "complexity":
            return [self.complexity] * self.num_envs
        if attr_name == "reward_config":
            return [self.reward_config] * self.num_envs
        return [None] * self.num_envs

    def set_attr(self, attr_name: str, value: Any, indices=None) -> None:
        if attr_name == "difficulty":
            self.difficulty = value
            self._set_server_difficulty(self.difficulty)
        if attr_name == "complexity":
            self.complexity = int(value)
            self._set_server_complexity(self.complexity)
        if attr_name == "reward_config":
            self.set_reward_config(value)

    def action_masks(self) -> list[np.ndarray]:
        return list(self._mask)

    def set_reward_config(self, config: Mapping[str, Any]) -> None:
        request = reward_config_request(config)
        self.reward_config = {
            field: float(getattr(request, field)) for field in _REWARD_CONFIG_FIELDS
        }
        self._set_server_reward_config(request)

    def reset(self) -> np.ndarray:
        t0 = time.perf_counter()
        self._close_stream()

        resp = self._stub.ResetVector(
            pb.VectorResetRequest(
                num_envs=self.num_envs,
                city=self.city,
                spawn_rate_factor=self.difficulty,
                seed=self.seed,
            )
        )
        obs = np.array(resp.obs, dtype=np.float32).reshape(self.num_envs, OBS_DIM)
        self._mask = np.array(resp.mask, dtype=bool).reshape(self.num_envs, _MASK_SIZE)

        self._open_stream()
        self._trace(
            "reset "
            f"elapsed={time.perf_counter() - t0:.3f}s "
            f"obs_shape={obs.shape} mask_shape={self._mask.shape}"
        )
        return obs

    def step_async(self, actions: np.ndarray) -> None:
        self._actions_buffer = actions

    def step_wait(self) -> VecEnvStepReturn:
        prev_mask = self._mask.copy()
        actions_for_info = np.asarray(self._actions_buffer, dtype=np.int64).reshape(
            self.num_envs, len(ACTION_DIMS)
        )
        valid_actions = _actions_valid_for_masks(actions_for_info, prev_mask)
        flat_actions = self._actions_buffer.flatten().tolist()
        t0 = time.perf_counter()
        self._action_queue.put(flat_actions)

        try:
            resp = self._resp_queue.get(timeout=_STEP_TIMEOUT_S)
        except queue.Empty as exc:
            raise RuntimeError(
                "timed out waiting for rl_server step response "
                f"after {_STEP_TIMEOUT_S:.1f}s; {self._server_state()}"
            ) from exc
        if isinstance(resp, grpc.RpcError):
            raise RuntimeError(
                "rl_server stream failed while waiting for step response: "
                f"code={resp.code()} details={resp.details()!r}; {self._server_state()}"
            ) from resp
        wait_s = time.perf_counter() - t0
        self._step_count += 1
        self._step_wait_acc += wait_s
        self._step_wait_max = max(self._step_wait_max, wait_s)

        obs = np.array(resp.obs, dtype=np.float32).reshape(self.num_envs, OBS_DIM)
        self._mask = np.array(resp.mask, dtype=bool).reshape(self.num_envs, _MASK_SIZE)
        rewards = np.array(resp.reward, dtype=np.float32)
        dones = np.array(resp.done, dtype=bool)
        self._done_count += int(dones.sum())

        if resp.terminal_obs:
            terminal_obs = np.array(resp.terminal_obs, dtype=np.float32).reshape(self.num_envs, OBS_DIM)
        else:
            terminal_obs = None

        score = resp.score
        pax = resp.passengers_delivered
        week = resp.week
        stations = resp.stations
        game_over = resp.game_over
        in_modal = resp.in_upgrade_modal

        infos = []
        for i in range(self.num_envs):
            station_slice = obs[
                i,
                _STATION_OFFSET:_STATION_OFFSET + NUM_STATIONS * STATION_DIM,
            ].reshape(NUM_STATIONS, STATION_DIM)
            valid_station = station_slice[:, 6] > 0
            queue_pressure = float(station_slice[valid_station, _PAX_RATIO_IDX].sum())
            overcrowd_pressure = float(
                np.square(station_slice[valid_station, _OVERCROWD_IDX]).sum()
            )
            danger_count = float(
                np.sum(station_slice[valid_station, _OVERCROWD_IDX] > 0.80)
            )
            info = {
                "score": score[i],
                "passengers_delivered": pax[i],
                "week": week[i],
                "stations": stations[i],
                "action_category": int(actions_for_info[i, 0]),
                "queue_pressure": queue_pressure,
                "overcrowd_pressure": overcrowd_pressure,
                "danger_count": danger_count,
                "game_over": game_over[i],
                "in_upgrade_modal": in_modal[i],
                "valid_action": bool(valid_actions[i]),
                "invalid_action": float(not valid_actions[i]),
            }
            if dones[i] and terminal_obs is not None:
                info["terminal_observation"] = terminal_obs[i]
            infos.append(info)

        self._maybe_trace_step()
        return obs, rewards, dones, infos

    def close(self):
        self._trace("close")
        self._close_stream()
        if self._channel is not None:
            self._channel.close()
            self._channel = None
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

    def _open_stream(self):
        self._trace("open_stream")
        self._resp_queue: queue.Queue = queue.Queue()

        def _action_iter():
            while True:
                action = self._action_queue.get()
                if action is None:
                    return
                yield pb.VectorActionRequest(actions=action)

        self._stream = self._stub.RunVectorEpisode(_action_iter())
        self._stream_error = None

        def _reader():
            try:
                for resp in self._stream:
                    self._resp_queue.put(resp)
            except grpc.RpcError as exc:
                self._stream_error = exc
                self._trace(f"stream_reader_rpc_error code={exc.code()} details={exc.details()!r}")
                self._resp_queue.put(exc)

        self._stream_thread = threading.Thread(target=_reader, daemon=True)
        self._stream_thread.start()

    def _close_stream(self):
        if self._stream_thread is not None:
            self._trace(
                "close_stream "
                f"action_q={self._action_queue.qsize()} "
                f"resp_q={getattr(self, '_resp_queue', queue.Queue()).qsize()}"
            )
            self._action_queue.put(None)
            self._stream_thread.join(timeout=3)
            if self._stream_thread.is_alive():
                self._trace("close_stream timeout thread_still_alive=True")
            self._stream = None
            self._stream_thread = None

    def _connect(self):
        self._channel = grpc.insecure_channel(f"localhost:{self.port}", options=[
            ('grpc.max_send_message_length', 50 * 1024 * 1024),
            ('grpc.max_receive_message_length', 50 * 1024 * 1024)
        ])
        self._stub = pb_grpc.RLEnvStub(self._channel)
        self._control_stub = pb_grpc.ControlStub(self._channel)

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
        self._trace("connect ok")

    def _set_server_difficulty(self, difficulty: float) -> None:
        if self._channel is None:
            return
        t0 = time.perf_counter()
        self._control_stub.SetDifficulty(
            pb.ResetRequest(spawn_rate_factor=float(difficulty)),
            timeout=5,
        )
        self._trace(f"set_server_difficulty elapsed={time.perf_counter() - t0:.3f}s")

    def _set_server_complexity(self, level: int) -> None:
        if self._channel is None:
            return
        t0 = time.perf_counter()
        self._control_stub.SetComplexity(
            pb.ResetRequest(spawn_rate_factor=float(level)),
            timeout=5,
        )
        self._trace(f"set_server_complexity elapsed={time.perf_counter() - t0:.3f}s")

    def _set_server_reward_config(self, request: pb.RewardConfigRequest) -> None:
        if self._channel is None:
            return
        t0 = time.perf_counter()
        self._control_stub.SetRewardConfig(request, timeout=5)
        self._trace(f"set_server_reward_config elapsed={time.perf_counter() - t0:.3f}s")

    def _start_server(self):
        binary = os.path.abspath(self.binary)
        if not os.path.isfile(binary):
            raise FileNotFoundError(f"rl_server binary not found at {binary}.")

        log_path = f"/tmp/rl_server_vector_{self.port}.log"
        self._log_file = open(log_path, "w")
        self._proc = subprocess.Popen(
            [binary, "--port", str(self.port)],
            stdout=self._log_file,
            stderr=self._log_file,
        )
        self._trace(f"server_start pid={self._proc.pid} log={log_path}")
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
            raise RuntimeError(f"rl_server on port {self.port} did not start within 10 s")
        self._connect()

    def _trace(self, msg: str) -> None:
        print(f"[env:{self.port} {time.strftime('%H:%M:%S')}] {msg}", flush=True)

    def _maybe_trace_step(self) -> None:
        now = time.perf_counter()
        if now - self._last_trace < self.trace_interval:
            return
        delta_steps = self._step_count - self._last_step_count
        avg_wait = self._step_wait_acc / max(delta_steps, 1)
        proc_state = "external"
        if self._proc is not None:
            poll = self._proc.poll()
            proc_state = f"pid={self._proc.pid} returncode={poll}"
        self._trace(
            "step_wait "
            f"steps={self._step_count:,} delta_steps={delta_steps:,} "
            f"avg_wait={avg_wait:.4f}s max_wait={self._step_wait_max:.4f}s "
            f"dones={self._done_count:,} "
            f"action_q={self._action_queue.qsize()} resp_q={self._resp_queue.qsize()} "
            f"{proc_state}"
        )
        self._last_trace = now
        self._last_step_count = self._step_count
        self._step_wait_acc = 0.0
        self._step_wait_max = 0.0

    def _server_state(self) -> str:
        if self._proc is None:
            return "server=external"
        return f"server_pid={self._proc.pid} returncode={self._proc.poll()}"


class MiniMetroEnv(gym.Env):
    """Single-env Gymnasium wrapper backed by the vector gRPC implementation."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        port: int = 8765,
        city: str = "london",
        binary: str = _DEFAULT_BINARY,
        managed: bool = True,
        trace_interval: float = 10.0,
        seed: int = 0,
    ):
        super().__init__()
        self._vec = MiniMetroVecEnv(
            n_envs=1,
            port=port,
            city=city,
            binary=binary,
            managed=managed,
            trace_interval=trace_interval,
            seed=seed,
        )
        self.observation_space = self._vec.observation_space
        self.action_space = self._vec.action_space
        self._stub = self._vec._stub

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self._vec.seed = int(seed)
        obs = self._vec.reset()
        return obs[0], {}

    def step(self, action):
        action_arr = np.asarray(action, dtype=np.int32)
        if action_arr.shape == ():
            # Compatibility with older smoke scripts that sampled an index from
            # the flat mask instead of a full MultiDiscrete action.
            action_idx = int(action_arr)
            if 0 <= action_idx < ACTION_DIMS[0]:
                action_arr = np.array([action_idx, 0, 0, 0], dtype=np.int32)
            else:
                action_arr = np.array([0, 0, 0, 0], dtype=np.int32)
        action = action_arr.reshape(1, len(ACTION_DIMS))
        self._vec.step_async(action)
        obs, rewards, dones, infos = self._vec.step_wait()
        return obs[0], float(rewards[0]), bool(dones[0]), False, infos[0]

    def action_masks(self) -> np.ndarray:
        return self._vec._mask[0].copy()

    def close(self):
        self._vec.close()


class MiniMetroFrameStack(VecEnvWrapper):
    """Frame stacking wrapper for flat Mini Metro observations.

    SB3's generic VecFrameStack zero-fills earlier frames after reset. Repeating
    the initial observation keeps the stacked input in-distribution from the
    first decision of each episode.
    """

    def __init__(self, venv: VecEnv, n_stack: int):
        if n_stack < 1:
            raise ValueError("n_stack must be >= 1")
        if len(venv.observation_space.shape) != 1:
            raise ValueError("MiniMetroFrameStack expects flat Box observations")
        self.n_stack = n_stack
        self.obs_dim = int(venv.observation_space.shape[0])
        low = np.tile(venv.observation_space.low, n_stack)
        high = np.tile(venv.observation_space.high, n_stack)
        observation_space = spaces.Box(low=low, high=high, dtype=venv.observation_space.dtype)
        super().__init__(venv, observation_space=observation_space)
        self._stacked_obs = np.zeros(
            (self.num_envs, self.obs_dim * self.n_stack),
            dtype=venv.observation_space.dtype,
        )

    def reset(self) -> np.ndarray:
        obs = self.venv.reset()
        self._stacked_obs = np.tile(obs, (1, self.n_stack))
        return self._stacked_obs.copy()

    def step_wait(self) -> VecEnvStepReturn:
        obs, rewards, dones, infos = self.venv.step_wait()
        prev_stacked = self._stacked_obs.copy()
        for i in range(self.num_envs):
            if dones[i]:
                terminal_obs = infos[i].get("terminal_observation")
                if terminal_obs is not None:
                    infos[i]["terminal_observation"] = np.concatenate(
                        [prev_stacked[i, self.obs_dim:], terminal_obs.astype(prev_stacked.dtype)]
                    )
                self._stacked_obs[i] = np.tile(obs[i], self.n_stack)
            else:
                self._stacked_obs[i, :-self.obs_dim] = self._stacked_obs[i, self.obs_dim:]
                self._stacked_obs[i, -self.obs_dim:] = obs[i]
        return self._stacked_obs.copy(), rewards, dones, infos

    def action_masks(self) -> list[np.ndarray]:
        return self.venv.action_masks()
