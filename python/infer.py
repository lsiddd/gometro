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
from typing import Callable

import grpc
import numpy as np

from rl.proto import minimetro_pb2 as pb
from rl.proto import minimetro_pb2_grpc as pb_grpc


# ---------------------------------------------------------------------------
# Backend factories
# ---------------------------------------------------------------------------

def _make_sb3_predictor(model_path: str) -> Callable[[np.ndarray, np.ndarray], list[int]]:
    from sb3_contrib import MaskablePPO
    from models import MetroFeatureExtractor

    custom_objects = {"MetroFeatureExtractor": MetroFeatureExtractor}
    model = MaskablePPO.load(model_path, custom_objects=custom_objects)

    def predict(obs: np.ndarray, mask: np.ndarray) -> list[int]:
        action, _ = model.predict(obs, action_masks=mask, deterministic=True)
        return action[0].tolist()

    print(f"PyTorch backend loaded: {model_path}")
    return predict


def _make_onnx_predictor(model_path: str) -> Callable[[np.ndarray, np.ndarray], list[int]]:
    import onnxruntime as ort

    sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])

    def predict(obs: np.ndarray, mask: np.ndarray) -> list[int]:
        out = sess.run(["action"], {
            "obs":  obs.astype(np.float32),
            "mask": mask.astype(np.float32),
        })[0]
        return out[0].tolist()

    print(f"ONNX Runtime backend loaded: {model_path}")
    return predict


# ---------------------------------------------------------------------------
# gRPC servicer
# ---------------------------------------------------------------------------

class InferenceServicer(pb_grpc.InferenceServicer):
    def __init__(self, predict: Callable[[np.ndarray, np.ndarray], list[int]]) -> None:
        self._predict = predict

    def Act(self, request: pb.ActRequest, context: grpc.ServicerContext) -> pb.ActionResponse:
        obs  = np.array(request.obs,  dtype=np.float32).reshape(1, -1)
        mask = np.array(request.mask, dtype=np.float32).reshape(1, -1)
        action = self._predict(obs, mask)
        return pb.ActionResponse(action=action)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def serve(model_path: str, port: int) -> None:
    if model_path.endswith(".onnx"):
        predictor = _make_onnx_predictor(model_path)
    else:
        predictor = _make_sb3_predictor(model_path)

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    pb_grpc.add_InferenceServicer_to_server(InferenceServicer(predictor), server)
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
