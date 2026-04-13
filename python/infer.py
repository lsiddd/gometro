"""
Inference server for online play.

Loads a trained checkpoint and serves action decisions over HTTP.
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
    ./minimetro-go --rl-client http://localhost:9000
"""
from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Callable

import numpy as np

from http_schemas import parse_act_request


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
        # mask must be float32 (1.0=valid, 0.0=invalid)
        out = sess.run(["action"], {
            "obs":  obs.astype(np.float32),
            "mask": mask.astype(np.float32),
        })[0]
        return out[0].tolist()

    print(f"ONNX Runtime backend loaded: {model_path}")
    return predict


# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------

class InferenceHandler(BaseHTTPRequestHandler):
    """HTTP handler that wraps a loaded policy backend."""

    predict: Callable[[np.ndarray, np.ndarray], list[int]]  # set by serve()

    def do_POST(self):  # noqa: N802
        if self.path != "/act":
            self.send_error(404)
            return

        length = int(self.headers.get("Content-Length", 0))
        try:
            raw = json.loads(self.rfile.read(length))
            req = parse_act_request(raw)
        except (json.JSONDecodeError, ValueError) as exc:
            self.send_error(400, str(exc))
            return

        obs  = np.array(req["obs"],  dtype=np.float32).reshape(1, -1)
        mask = np.array(req["mask"], dtype=np.float32).reshape(1, -1)

        action_array = InferenceHandler.predict(obs, mask)

        resp = json.dumps({"action": action_array}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(resp)))
        self.end_headers()
        self.wfile.write(resp)

    def log_message(self, fmt, *args):  # suppress per-request logs
        pass


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def serve(model_path: str, port: int) -> None:
    if model_path.endswith(".onnx"):
        predictor = _make_onnx_predictor(model_path)
    else:
        predictor = _make_sb3_predictor(model_path)

    InferenceHandler.predict = staticmethod(predictor)
    server = HTTPServer(("localhost", port), InferenceHandler)
    print(f"Inference server ready at http://localhost:{port}/act")
    server.serve_forever()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True,
                        help="Path to .zip (PyTorch) or .onnx (ONNX Runtime) checkpoint")
    parser.add_argument("--port", type=int, default=9000)
    args = parser.parse_args()
    serve(args.model, args.port)
