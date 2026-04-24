"""
Autoregressive policy for Mini Metro.

Factorises the MultiDiscrete action as a chain of conditional categoricals:
    π(a₁,a₂,a₃,a₄) = π₁(a₁) · π₂(a₂|a₁) · π₃(a₃|a₁,a₂) · π₄(a₄|a₁,a₂,a₃)

Conditioning is via learned embeddings: head i receives
    concat(latent_pi, embed(a₀), …, embed(aᵢ₋₁))
as input.

Benefits over the flat MultiDiscrete head:
  - Gradients flow to each head independently.
  - The network can learn that "add train to line 3" means line-3 station
    choices matter, before even looking at the station dimension.
  - Effective search space is linear in |A| rather than multiplicative.

Usage in train.py:
    from policy import MetroPolicy
    model = MaskablePPO(MetroPolicy, vec_env, policy_kwargs={"net_arch": [256, 256]}, ...)
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from stable_baselines3.common.type_aliases import Schedule

from constants import (
    ACTION_DIMS,
    BASE_MASK_SIZE,
    COND_LINE_OFFSET,
    COND_STATION_OFFSET,
    COND_OPTION_OFFSET,
    MAX_LINE_SLOTS,
    MAX_STATION_SLOTS,
    NUM_ACTION_CATS,
    NUM_OPTIONS,
)


# ---------------------------------------------------------------------------
# Action network
# ---------------------------------------------------------------------------

class AutoregressiveActionNet(nn.Module):
    """
    Four sequential heads with embedding-based conditioning.

    Head i: Linear(latent_dim + i*embed_dim,  action_dims[i])
    Each head's input is the original latent_pi concatenated with the
    embeddings of all previously sampled (or given) action indices.
    """

    def __init__(self, latent_dim: int, action_dims: list[int], embed_dim: int = 16):
        super().__init__()
        self.action_dims = list(action_dims)
        self.latent_dim  = latent_dim
        self.embed_dim   = embed_dim
        n = len(action_dims)

        # Head i input size grows by embed_dim per preceding head.
        self.heads = nn.ModuleList([
            nn.Linear(latent_dim + i * embed_dim, action_dims[i])
            for i in range(n)
        ])
        # One embedding table per head, except the last (no successor to condition).
        self.embeds = nn.ModuleList([
            nn.Embedding(action_dims[i], embed_dim)
            for i in range(n - 1)
        ])

    def forward(self, latent_pi: torch.Tensor) -> torch.Tensor:
        """Thin wrapper for SB3 compatibility — returns first-head logits only.

        The full autoregressive computation happens inside
        AutoregressiveDistribution, which holds a reference to this module.
        This method is never called during training (evaluate_actions bypasses
        it); it is kept so that the module is a valid nn.Module.
        """
        return self.heads[0](latent_pi)


# ---------------------------------------------------------------------------
# Distribution
# ---------------------------------------------------------------------------

class AutoregressiveDistribution:
    """
    Autoregressive factorised distribution over MultiDiscrete actions.

    Compatible with the interface expected by MaskableActorCriticPolicy:
        apply_masking(masks)
        log_prob(actions)   → [B]
        entropy()           → [B]
        get_actions(deterministic) → [B, 4]
    """

    def __init__(self, action_net: AutoregressiveActionNet, latent_pi: torch.Tensor):
        self.action_net = action_net
        self.latent_pi  = latent_pi
        self._masks: Optional[list[torch.Tensor]] = None
        # Cached per-head Categorical distributions, set during log_prob().
        self._cached_dists: Optional[list[Categorical]] = None

    # ------------------------------------------------------------------
    # Masking
    # ------------------------------------------------------------------

    def apply_masking(self, masks: Optional[np.ndarray | torch.Tensor]) -> None:
        """Split flat boolean mask into base and conditional head masks.

        The first BASE_MASK_SIZE entries are the standard independent
        MultiDiscrete masks. Extra sections carry per-action-category masks for
        line, station and option heads. Older masks still work because the base
        masks are used as a fallback.
        """
        if masks is None:
            self._masks = None
            self._cond_line_masks = None
            self._cond_station_masks = None
            self._cond_option_masks = None
            return
        if isinstance(masks, np.ndarray):
            masks = torch.as_tensor(masks, device=self.latent_pi.device)
        masks = masks.bool()
        self._masks = []
        offset = 0
        for d in self.action_net.action_dims:
            self._masks.append(masks[:, offset:offset + d])
            offset += d
        self._cond_line_masks = None
        self._cond_station_masks = None
        self._cond_option_masks = None
        if masks.shape[1] >= COND_OPTION_OFFSET + NUM_ACTION_CATS * NUM_OPTIONS:
            b = masks.shape[0]
            self._cond_line_masks = masks[
                :, COND_LINE_OFFSET:COND_STATION_OFFSET
            ].reshape(b, NUM_ACTION_CATS, MAX_LINE_SLOTS)
            self._cond_station_masks = masks[
                :, COND_STATION_OFFSET:COND_OPTION_OFFSET
            ].reshape(b, NUM_ACTION_CATS, MAX_LINE_SLOTS, MAX_STATION_SLOTS)
            self._cond_option_masks = masks[
                :, COND_OPTION_OFFSET:COND_OPTION_OFFSET + NUM_ACTION_CATS * NUM_OPTIONS
            ].reshape(b, NUM_ACTION_CATS, NUM_OPTIONS)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _masked_logits(self, raw_logits: torch.Tensor, head_idx: int) -> torch.Tensor:
        if self._masks is not None:
            raw_logits = raw_logits.masked_fill(~self._masks[head_idx], -1e9)
        return raw_logits

    def _conditional_mask(
        self,
        head_idx: int,
        act_cat: torch.Tensor,
        line_idx: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        table = None
        if head_idx == 1:
            table = self._cond_line_masks
        elif head_idx == 2:
            table = self._cond_station_masks
        elif head_idx == 3:
            table = self._cond_option_masks
        if table is None:
            return None
        batch_idx = torch.arange(act_cat.shape[0], device=act_cat.device)
        if head_idx == 2 and line_idx is not None:
            return table[batch_idx, act_cat.long(), line_idx.long()]
        return table[batch_idx, act_cat.long()]

    def _masked_logits_for_context(
        self,
        raw_logits: torch.Tensor,
        head_idx: int,
        act_cat: Optional[torch.Tensor],
        line_idx: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if head_idx > 0 and act_cat is not None:
            cond = self._conditional_mask(head_idx, act_cat, line_idx)
            if cond is not None:
                return raw_logits.masked_fill(~cond, -1e9)
        return self._masked_logits(raw_logits, head_idx)

    def _build_context(self, latent_pi: torch.Tensor, prev_actions: list[torch.Tensor]) -> torch.Tensor:
        """Concatenate latent_pi with embeddings of all previous actions."""
        parts = [latent_pi]
        net = self.action_net
        for i, a in enumerate(prev_actions):
            parts.append(net.embeds[i](a.long()))
        return torch.cat(parts, dim=-1)

    # ------------------------------------------------------------------
    # Core methods
    # ------------------------------------------------------------------

    def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        """Sum of conditional log-probs, differentiable w.r.t. network params.

        Caches per-head Categorical distributions for the subsequent entropy()
        call (evaluate_actions always calls log_prob before entropy).
        """
        net     = self.action_net
        latent  = self.latent_pi
        actions = actions.long()
        total   = torch.zeros(actions.shape[0], device=latent.device)
        dists   = []
        context = latent

        for i in range(len(net.action_dims)):
            act_cat = actions[:, 0] if i > 0 else None
            line_idx = actions[:, 1] if i > 1 else None
            logits = self._masked_logits_for_context(
                net.heads[i](context), i, act_cat, line_idx
            )
            dist   = Categorical(logits=logits)
            dists.append(dist)
            total  = total + dist.log_prob(actions[:, i])

            if i < len(net.embeds):
                context = torch.cat([context, net.embeds[i](actions[:, i])], dim=-1)

        self._cached_dists = dists
        return total

    def entropy(self) -> torch.Tensor:
        """Sum of per-head entropies.

        When called right after log_prob (the normal PPO path), uses the
        cached distributions whose conditioning is the batch's own actions,
        giving a reasonable per-head entropy estimate. Falls back to an
        unconditional computation otherwise.
        """
        if self._cached_dists is not None:
            return sum(d.entropy() for d in self._cached_dists)

        # Fallback: zero-conditioning context for each head.
        net     = self.action_net
        latent  = self.latent_pi
        context = latent
        total   = torch.zeros(latent.shape[0], device=latent.device)
        for i in range(len(net.action_dims)):
            logits = self._masked_logits(net.heads[i](context), i)
            total  = total + Categorical(logits=logits).entropy()
            if i < len(net.embeds):
                zero_emb = torch.zeros(latent.shape[0], net.embed_dim, device=latent.device)
                context  = torch.cat([context, zero_emb], dim=-1)
        return total

    def get_actions(self, deterministic: bool = False) -> torch.Tensor:
        """Sample (or take the mode of) each head sequentially."""
        net     = self.action_net
        context = self.latent_pi
        sampled = []

        for i in range(len(net.action_dims)):
            act_cat = sampled[0].squeeze(-1) if i > 0 and sampled else None
            line_idx = sampled[1].squeeze(-1) if i > 1 and len(sampled) > 1 else None
            logits = self._masked_logits_for_context(
                net.heads[i](context), i, act_cat, line_idx
            )
            dist   = Categorical(logits=logits)
            a_i    = dist.mode if deterministic else dist.sample()
            sampled.append(a_i.unsqueeze(-1))

            if i < len(net.embeds):
                context = torch.cat([context, net.embeds[i](a_i)], dim=-1)

        return torch.cat(sampled, dim=-1)   # [B, 4]


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------

class MetroPolicy(MaskableActorCriticPolicy):
    """
    Drop-in replacement for 'MlpPolicy' in MaskablePPO.

    Installs AutoregressiveActionNet after the standard _build() and
    overrides _get_action_dist_from_latent to return the autoregressive
    distribution. The action space must remain MultiDiscrete([14,7,50,2]).

    Example usage in train.py::

        from policy import MetroPolicy
        model = MaskablePPO(
            MetroPolicy, vec_env,
            policy_kwargs={"net_arch": [256, 256]},
            ...
        )
    """

    EMBED_DIM: int = 16

    def _build(self, lr_schedule: Schedule) -> None:
        # Standard build: constructs mlp_extractor, value_net, optimizer.
        super()._build(lr_schedule)

        # Replace the flat Linear action head with the autoregressive net.
        latent_dim_pi = self.mlp_extractor.latent_dim_pi
        self.action_net = AutoregressiveActionNet(
            latent_dim  = latent_dim_pi,
            action_dims = list(ACTION_DIMS),
            embed_dim   = self.EMBED_DIM,
        )

        # Recreate the optimizer so it tracks the new action_net parameters.
        self.optimizer = self.optimizer_class(
            self.parameters(),
            lr=lr_schedule(1),
            **self.optimizer_kwargs,
        )

    def _get_action_dist_from_latent(
        self, latent_pi: torch.Tensor
    ) -> AutoregressiveDistribution:
        return AutoregressiveDistribution(self.action_net, latent_pi)
