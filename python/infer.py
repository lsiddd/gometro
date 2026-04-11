"""
Inference server for online play.

Loads a trained MaskablePPO checkpoint and serves action decisions over HTTP.
The Go game binary calls this server via the --rl-client flag.

Usage:
    # Terminal 1 — inference server
    uv run python infer.py --model checkpoints/minimetro_final.zip --port 9000

    # Terminal 2 — game with RL agent
    ./minimetro-go --rl-client http://localhost:9000
"""
from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
from sb3_contrib import MaskablePPO

from models import MetroFeatureExtractor
from http_schemas import parse_act_request


class InferenceHandler(BaseHTTPRequestHandler):
    """HTTP handler that wraps a loaded PPO policy."""

    model: MaskablePPO  # class-level, set before server start

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

        obs = np.array(req["obs"], dtype=np.float32).reshape(1, -1)
        mask = np.array(req["mask"], dtype=bool).reshape(1, -1)

        action, _ = self.model.predict(obs, action_masks=mask, deterministic=True)
        action_array = action[0].tolist()

        resp = json.dumps({"action": action_array}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(resp)))
        self.end_headers()
        self.wfile.write(resp)

    def log_message(self, fmt, *args):  # suppress per-request logs
        pass


def serve(model_path: str, port: int):
    # Register MetroFeatureExtractor dynamically if required by load,
    # though SB3 pickle usually handles it if imported.
    custom_objects = {"MetroFeatureExtractor": MetroFeatureExtractor}
    model = MaskablePPO.load(model_path, custom_objects=custom_objects)
    InferenceHandler.model = model
    server = HTTPServer(("localhost", port), InferenceHandler)
    print(f"Inference server ready at http://localhost:{port}/act")
    print(f"Model: {model_path}")
    server.serve_forever()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to .zip checkpoint")
    parser.add_argument("--port", type=int, default=9000)
    args = parser.parse_args()
    serve(args.model, args.port)
