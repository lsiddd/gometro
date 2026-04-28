"""
Inference server for online play.

Loads a trained checkpoint and serves action decisions over gRPC
(Inference service defined in rl/proto/minimetro.proto).
The Go game binary calls this server via the --rl-client flag.

Two backends are supported, selected automatically by file extension:
  .zip  — original MaskablePPO / PyTorch path (slower, no extra setup needed)
  .onnx — ONNX Runtime path (~50x faster for batch_size=1)

To produce an .onnx file from a .zip checkpoint:
    uv run python export_onnx.py --model checkpoints/best_model.zip

Usage:
    # ONNX backend (recommended for play)
    uv run python infer.py --model actor.onnx --port 9000

    # PyTorch backend (fallback / debugging)
    uv run python infer.py --model checkpoints/best_model.zip --port 9000

    # Terminal 2 — game with RL agent
    ./minimetro-go --rl-client localhost:9000
"""
from __future__ import annotations

import argparse
from concurrent import futures
from threading import Lock
from typing import Callable

import grpc
import numpy as np

from rl.proto import minimetro_pb2 as pb
from rl.proto import minimetro_pb2_grpc as pb_grpc


# ---------------------------------------------------------------------------
# Backend factories
# ---------------------------------------------------------------------------

def _make_sb3_predictor(model_path: str) -> tuple[Callable[[np.ndarray, np.ndarray], list[int]], int]:
    from sb3_contrib import MaskablePPO
    from models import MetroFeatureExtractor

    custom_objects = {"MetroFeatureExtractor": MetroFeatureExtractor}
    model = MaskablePPO.load(model_path, custom_objects=custom_objects)
    expected_obs_dim = int(model.observation_space.shape[0])

    def predict(obs: np.ndarray, mask: np.ndarray) -> list[int]:
        action, _ = model.predict(obs, action_masks=mask, deterministic=True)
        return action[0].tolist()

    print(f"PyTorch backend loaded: {model_path}")
    return predict, expected_obs_dim


def _make_onnx_predictor(model_path: str) -> tuple[Callable[[np.ndarray, np.ndarray], list[int]], int]:
    import onnxruntime as ort

    sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    obs_dim = sess.get_inputs()[0].shape[1]
    expected_obs_dim = int(obs_dim) if isinstance(obs_dim, int) else 0

    def predict(obs: np.ndarray, mask: np.ndarray) -> list[int]:
        out = sess.run(["action"], {
            "obs":  obs.astype(np.float32),
            "mask": mask.astype(np.float32),
        })[0]
        return out[0].tolist()

    print(f"ONNX Runtime backend loaded: {model_path}")
    return predict, expected_obs_dim


# ---------------------------------------------------------------------------
# gRPC servicer
# ---------------------------------------------------------------------------

class InferenceServicer(pb_grpc.InferenceServicer):
    def __init__(
        self,
        predict: Callable[[np.ndarray, np.ndarray], list[int]],
        expected_obs_dim: int,
    ) -> None:
        self._predict = predict
        self._expected_obs_dim = expected_obs_dim
        self._stack: np.ndarray | None = None
        self._last_episode_marker: tuple[float, float] | None = None
        self._lock = Lock()

    def Act(self, request: pb.ActRequest, context: grpc.ServicerContext) -> pb.ActionResponse:
        obs  = np.array(request.obs,  dtype=np.float32).reshape(1, -1)
        mask = np.array(request.mask, dtype=np.float32).reshape(1, -1)
        obs = self._stack_obs(obs)
        action = self._predict(obs, mask)
        return pb.ActionResponse(action=action)

    def _stack_obs(self, obs: np.ndarray) -> np.ndarray:
        if self._expected_obs_dim <= obs.shape[1]:
            return obs
        if self._expected_obs_dim % obs.shape[1] != 0:
            raise ValueError(
                f"model expects obs_dim={self._expected_obs_dim}, got raw obs_dim={obs.shape[1]}"
            )

        marker = (float(obs[0, 0]), float(obs[0, 2]))  # week, score
        with self._lock:
            reset_stack = self._stack is None
            if self._last_episode_marker is not None:
                prev_week, prev_score = self._last_episode_marker
                reset_stack = reset_stack or marker[0] < prev_week or marker[1] < prev_score

            if reset_stack:
                n_stack = self._expected_obs_dim // obs.shape[1]
                self._stack = np.repeat(obs, n_stack, axis=1)
            else:
                self._stack = np.concatenate([self._stack[:, obs.shape[1]:], obs], axis=1)
            self._last_episode_marker = marker
            return self._stack.copy()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def serve(model_path: str, port: int) -> None:
    if model_path.endswith(".onnx"):
        predictor, expected_obs_dim = _make_onnx_predictor(model_path)
    else:
        predictor, expected_obs_dim = _make_sb3_predictor(model_path)

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    pb_grpc.add_InferenceServicer_to_server(
        InferenceServicer(predictor, expected_obs_dim), server
    )
    server.add_insecure_port(f"[::]:{port}")
    server.start()
    print(f"Inference gRPC server ready at :{port}")
    server.wait_for_termination()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True,
                        help="Path to .zip (PyTorch) or .onnx (ONNX Runtime) checkpoint")
    parser.add_argument("--port", type=int, default=9000)
    args = parser.parse_args()
    serve(args.model, args.port)
