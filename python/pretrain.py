"""
Behavioral Cloning pre-training from the heuristic solver.

Collects (obs, action) demonstrations by running the solver through
complete episodes, then trains the MaskablePPO policy network via
supervised cross-entropy loss on each action dimension.

The resulting checkpoint is a warm-started policy that already knows the
solver's basic strategies (connect isolated stations, close loops, deploy
trains) — greatly accelerating subsequent PPO fine-tuning.

Usage:
    uv run python pretrain.py [--episodes 50] [--city london] [--epochs 20]
                              [--batch-size 512] [--lr 1e-3]
                              [--out checkpoints/pretrain_bc.zip]

Workflow:
    just pretrain
    just resume checkpoint=checkpoints/pretrain_bc.zip
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from torch.utils.data import DataLoader, TensorDataset

from constants import ACTION_DIMS, PRETRAIN_BASE_PORT
from env import MiniMetroEnv
from models import MetroFeatureExtractor
from rl.proto import minimetro_pb2 as pb

BASE_PORT    = PRETRAIN_BASE_PORT
CHECKPOINT_DIR = "checkpoints"


# ── Demo collection ──────────────────────────────────────────────────────────

def collect_demos(city: str, n_episodes: int, port: int) -> tuple[np.ndarray, np.ndarray]:
    """Run the solver for n_episodes and collect (obs, action) pairs.

    Returns
    -------
    obs_arr : float32 array of shape (N, OBS_DIM)
    act_arr : int32 array of shape (N, 4)
    """
    obs_list: list[np.ndarray] = []
    act_list: list[list[int]] = []

    env = MiniMetroEnv(port=port, city=city, managed=True)
    try:
        for ep in range(n_episodes):
            obs, _ = env.reset()
            done = False
            ep_steps = 0
            while not done:
                # Get solver action via gRPC SolverAct RPC.
                try:
                    action_resp = env._stub.SolverAct(pb.Empty(), timeout=5)
                    action = list(action_resp.action)
                except Exception:
                    action = [0, 0, 0, 0]

                obs_list.append(obs.copy())
                act_list.append(action)

                obs, _, done, _, info = env.step(np.array(action))
                ep_steps += 1

            print(f"  ep {ep+1}/{n_episodes}  steps={ep_steps}  "
                  f"score={info.get('score', 0)}  "
                  f"pax={info.get('passengers_delivered', 0)}")
    finally:
        env.close()

    obs_arr = np.array(obs_list, dtype=np.float32)
    act_arr = np.array(act_list, dtype=np.int32)
    return obs_arr, act_arr


# ── Behavioral Cloning ───────────────────────────────────────────────────────

def bc_train(
    model: MaskablePPO,
    obs_arr: np.ndarray,
    act_arr: np.ndarray,
    n_epochs: int,
    batch_size: int,
    lr: float,
) -> None:
    """Train the policy network with supervised cross-entropy loss.

    MaskablePPO uses an `ActorCriticPolicy` where `policy.action_net`
    produces the flat logits for the MultiDiscrete action space. We split
    those logits per dimension and minimise cross-entropy against the
    solver's choices.
    """
    policy = model.policy
    policy.train()

    obs_t = torch.tensor(obs_arr, dtype=torch.float32)
    # act_arr columns: [actCat, lineIdx, stationIdx, opt]
    act_t = torch.tensor(act_arr, dtype=torch.long)

    dataset = TensorDataset(obs_t, act_t)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)

    # Boundaries for slicing the flat logits into each action dimension.
    # ACTION_DIMS = [14, 7, 50, 2]
    boundaries = [0] + list(np.cumsum(ACTION_DIMS))

    n_total = len(obs_arr)
    for epoch in range(n_epochs):
        total_loss = 0.0
        n_batches = 0
        for obs_b, act_b in loader:
            obs_b  = obs_b.to(policy.device)
            act_b  = act_b.to(policy.device)

            # Extract features then logits.
            features  = policy.extract_features(obs_b)
            latent_pi = policy.mlp_extractor.forward_actor(features)
            logits    = policy.action_net(latent_pi)   # (B, sum(ACTION_DIMS))

            loss = torch.tensor(0.0, device=policy.device)
            for dim_i, (lo, hi) in enumerate(zip(boundaries[:-1], boundaries[1:])):
                dim_logits = logits[:, lo:hi]           # (B, n_choices)
                dim_labels = act_b[:, dim_i]            # (B,)
                loss = loss + F.cross_entropy(dim_logits, dim_labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            optimizer.step()

            total_loss += loss.item()
            n_batches  += 1

        avg_loss = total_loss / max(n_batches, 1)
        print(f"  epoch {epoch+1}/{n_epochs}  loss={avg_loss:.4f}")

    policy.eval()


# ── Main ─────────────────────────────────────────────────────────────────────

def main(args: argparse.Namespace) -> None:
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    print(f"=== Collecting {args.episodes} solver episodes on port {BASE_PORT} ===")
    t0 = time.time()
    obs_arr, act_arr = collect_demos(args.city, args.episodes, BASE_PORT)
    print(f"Collected {len(obs_arr):,} steps in {time.time()-t0:.1f}s")

    # Save raw demos in case we want to reuse them.
    demo_path = os.path.join(CHECKPOINT_DIR, "solver_demos.npz")
    np.savez_compressed(demo_path, obs=obs_arr, act=act_arr)
    print(f"Demos saved to {demo_path}")

    print(f"\n=== Building fresh model for BC ===")
    # We need a throwaway env just to initialise the model.
    def _make_env():
        return MiniMetroEnv(port=BASE_PORT + 1, city=args.city, managed=True)
    vec_env = VecMonitor(DummyVecEnv([_make_env]))

    model = MaskablePPO(
        "MlpPolicy",
        vec_env,
        n_steps=4096,
        batch_size=512,
        n_epochs=10,
        gamma=0.995,
        gae_lambda=0.95,
        learning_rate=1e-4,
        ent_coef=0.01,
        clip_range=0.2,
        max_grad_norm=0.5,
        policy_kwargs={
            "features_extractor_class": MetroFeatureExtractor,
            "net_arch": [256, 256],
        },
        verbose=0,
    )

    print(f"\n=== Behavioral Cloning: {args.epochs} epochs, "
          f"batch={args.batch_size}, lr={args.lr} ===")
    bc_train(model, obs_arr, act_arr, args.epochs, args.batch_size, args.lr)

    out_path = args.out
    model.save(out_path)
    vec_env.close()
    print(f"\nPre-trained model saved to {out_path}.zip")
    print("Next step: just resume checkpoint=checkpoints/pretrain_bc.zip")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Behavioural cloning pre-training from the heuristic solver"
    )
    parser.add_argument("--episodes",   type=int,   default=50)
    parser.add_argument("--city",       type=str,   default="london")
    parser.add_argument("--epochs",     type=int,   default=20)
    parser.add_argument("--batch-size", type=int,   default=512)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--out",        type=str,
                        default=os.path.join(CHECKPOINT_DIR, "pretrain_bc"))
    main(parser.parse_args())
