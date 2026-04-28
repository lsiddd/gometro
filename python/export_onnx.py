"""
Export the actor network from a trained MaskablePPO checkpoint to ONNX.

The exported model takes (obs, mask) and returns the greedy action directly,
so the inference server needs no PyTorch dependency at runtime.

Inputs:
    obs   – float32[batch, OBS_DIM]
    mask  – float32[batch, MASK_SIZE]          1.0=valid, 0.0=invalid

Output:
    action – int64[batch, 4]                   [category, line, station, option]

Usage:
    uv run python export_onnx.py --model checkpoints/best_model.zip
    uv run python export_onnx.py --model checkpoints/best_model.zip --output actor.onnx
"""
from __future__ import annotations

import argparse

import numpy as np
import torch
import torch.nn as nn

from sb3_contrib import MaskablePPO

from constants import (
    ACTION_DIMS,
    MASK_SIZE,
    COND_LINE_OFFSET,
    COND_STATION_OFFSET,
    COND_OPTION_OFFSET,
    MAX_LINE_SLOTS,
    MAX_STATION_SLOTS,
    NUM_OPTIONS,
)
from models import MetroFeatureExtractor


class MaskedActor(nn.Module):
    """Deterministic actor wrapper for flat MultiDiscrete action heads.

    Applies action masking by setting logits of invalid actions to -1e9 before
    argmax, matching the behaviour of MaskablePPO.predict(..., deterministic=True).
    Used when policy.action_net is a plain nn.Linear.
    """

    def __init__(self, policy: nn.Module) -> None:
        super().__init__()
        self.features_extractor = policy.features_extractor
        self.mlp_extractor = policy.mlp_extractor
        self.action_net = policy.action_net
        self.splits = list(ACTION_DIMS)

    def forward(self, obs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            obs:  float32 [B, OBS_DIM]
            mask: float32 [B, MASK_SIZE]  — 1.0 valid, 0.0 invalid
        Returns:
            int64 [B, 4]
        """
        features = self.features_extractor(obs)
        latent_pi = self.mlp_extractor.forward_actor(features)
        logits = self.action_net(latent_pi)                    # [B, 73]
        mask = mask[:, :sum(self.splits)]
        masked = logits + (mask - 1.0) * 1e9                  # -1e9 for invalid
        parts = torch.split(masked, self.splits, dim=-1)
        actions = [p.argmax(dim=-1, keepdim=True) for p in parts]
        return torch.cat(actions, dim=-1)                      # [B, 4]


class AutoregressiveActor(nn.Module):
    """Deterministic actor wrapper for AutoregressiveActionNet.

    Executes the 4 heads sequentially (deterministic argmax) with
    embedding-based conditioning, matching MetroPolicy.get_actions(deterministic=True).
    """

    def __init__(self, policy: nn.Module) -> None:
        super().__init__()
        self.features_extractor = policy.features_extractor
        self.mlp_extractor = policy.mlp_extractor
        self.action_net = policy.action_net  # AutoregressiveActionNet

    def forward(self, obs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            obs:  float32 [B, OBS_DIM]
            mask: float32 [B, MASK_SIZE]  — 1.0 valid, 0.0 invalid
        Returns:
            int64 [B, 4]
        """
        features  = self.features_extractor(obs)
        station_embeddings = getattr(self.features_extractor, "last_station_embeddings", None)
        latent_pi = self.mlp_extractor.forward_actor(features)

        net      = self.action_net
        mask_bool = mask.bool()
        context  = latent_pi
        sampled  = []
        offset   = 0

        for i in range(len(net.action_dims)):
            if i == 2 and station_embeddings is not None:
                query = net.station_query(context)
                query = torch.nn.functional.normalize(query, dim=-1)
                keys = torch.nn.functional.normalize(station_embeddings, dim=-1)
                logits = torch.einsum("bd,bnd->bn", query, keys) * (query.shape[-1] ** 0.5)
            else:
                logits = net.heads[i](context)
            if i == 1:
                act = sampled[0].squeeze(-1)
                mask_table = mask_bool[
                    :, COND_LINE_OFFSET:COND_STATION_OFFSET
                ].reshape(-1, net.action_dims[0], MAX_LINE_SLOTS)
                mask_slice = mask_table[
                    torch.arange(mask_table.shape[0], device=act.device), act
                ]
            elif i == 2:
                act = sampled[0].squeeze(-1)
                line = sampled[1].squeeze(-1)
                mask_table = mask_bool[
                    :, COND_STATION_OFFSET:COND_OPTION_OFFSET
                ].reshape(-1, net.action_dims[0], MAX_LINE_SLOTS, MAX_STATION_SLOTS)
                mask_slice = mask_table[
                    torch.arange(mask_table.shape[0], device=act.device), act, line
                ]
            elif i == 3:
                act = sampled[0].squeeze(-1)
                mask_table = mask_bool[
                    :, COND_OPTION_OFFSET:COND_OPTION_OFFSET + net.action_dims[0] * NUM_OPTIONS
                ].reshape(-1, net.action_dims[0], NUM_OPTIONS)
                mask_slice = mask_table[
                    torch.arange(mask_table.shape[0], device=act.device), act
                ]
            else:
                mask_slice = mask_bool[:, offset:offset + net.action_dims[i]]
            logits     = logits.masked_fill(~mask_slice, -1e9)
            offset    += net.action_dims[i]
            a_i        = logits.argmax(dim=-1)
            sampled.append(a_i.unsqueeze(-1))
            if i < len(net.embeds):
                context = torch.cat([context, net.embeds[i](a_i)], dim=-1)

        return torch.cat(sampled, dim=-1)                      # [B, 4]


def _make_actor(policy: nn.Module) -> nn.Module:
    """Return the appropriate ONNX-exportable actor wrapper for this policy."""
    from policy import AutoregressiveActionNet
    if isinstance(policy.action_net, AutoregressiveActionNet):
        return AutoregressiveActor(policy)
    return MaskedActor(policy)


def export(model_path: str, output_path: str) -> None:
    custom_objects = {"MetroFeatureExtractor": MetroFeatureExtractor}
    model = MaskablePPO.load(model_path, custom_objects=custom_objects, device="cpu")
    model.policy.eval()

    actor = _make_actor(model.policy)
    actor.eval()

    mask_dim = MASK_SIZE
    obs_shape = model.observation_space.shape
    dummy_obs  = torch.zeros((1, *obs_shape), dtype=torch.float32)
    dummy_mask = torch.ones(1,  mask_dim,  dtype=torch.float32)

    torch.onnx.export(
        actor,
        (dummy_obs, dummy_mask),
        output_path,
        input_names=["obs", "mask"],
        output_names=["action"],
        dynamic_axes={
            "obs":    {0: "batch"},
            "mask":   {0: "batch"},
            "action": {0: "batch"},
        },
        opset_version=18,
        do_constant_folding=True,
    )
    print(f"Exported ONNX actor → {output_path}")

    # Smoke-test: reload with ONNX Runtime and verify shape/dtype.
    import onnxruntime as ort

    sess = ort.InferenceSession(output_path, providers=["CPUExecutionProvider"])
    out = sess.run(["action"], {
        "obs":  dummy_obs.numpy(),
        "mask": dummy_mask.numpy(),
    })[0]
    assert out.shape == (1, 4), f"unexpected output shape {out.shape}"
    print(f"Verification passed — action: {out[0].tolist()}")

    # Latency benchmark: 1000 calls.
    import time
    obs_np  = dummy_obs.numpy()
    mask_np = dummy_mask.numpy()
    t0 = time.perf_counter()
    for _ in range(1000):
        sess.run(["action"], {"obs": obs_np, "mask": mask_np})
    elapsed = time.perf_counter() - t0
    print(f"Mean latency: {elapsed:.3f} s / 1000 calls = {elapsed:.4f} ms/call")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",  required=True,          help="Path to .zip checkpoint")
    parser.add_argument("--output", default="actor.onnx",   help="Output .onnx path")
    args = parser.parse_args()
    export(args.model, args.output)
