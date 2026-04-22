from __future__ import annotations

import argparse

import numpy as np
from sb3_contrib import MaskablePPO

from constants import TRAIN_BASE_PORT
from env import MiniMetroEnv
from models import MetroFeatureExtractor

# Evaluation runs on a port well above any training/pretrain range to avoid
# conflicts when running alongside an active training session.
EVAL_PORT = TRAIN_BASE_PORT + 100


def evaluate(model_path: str, n_episodes: int, port: int) -> None:
    print(f"Loading model from {model_path}...")
    model = MaskablePPO.load(
        model_path,
        custom_objects={"MetroFeatureExtractor": MetroFeatureExtractor},
    )
    env = MiniMetroEnv(port=port, managed=True)

    scores = []
    try:
        for ep in range(n_episodes):
            obs, _ = env.reset()
            done = False
            ep_reward = 0.0
            steps = 0
            while not done and steps < 2000:
                mask = env.action_masks()
                action, _ = model.predict(obs, action_masks=mask, deterministic=True)
                obs, reward, done, _trunc, info = env.step(action)
                ep_reward += reward
                steps += 1
            scores.append(info.get("score", 0))
            print(
                f"Episode {ep + 1}/{n_episodes}  steps={steps}"
                f"  score={info.get('score', 0)}"
                f"  passengers={info.get('passengers_delivered', 0)}"
            )
    finally:
        env.close()

    print(f"Average score: {np.mean(scores):.1f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="checkpoints/best_model.zip")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--port", type=int, default=EVAL_PORT)
    args = parser.parse_args()
    evaluate(args.model, args.episodes, args.port)
