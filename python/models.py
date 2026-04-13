import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from constants import GLOBAL_DIM, STATION_DIM, NUM_STATIONS, LINE_DIM, NUM_LINES

# Embedding dimensions — decoupled from the observation dims for easy tuning.
_D_STATION  = 128  # station node embedding
_D_LINE     = 128  # line embedding
_MP_ROUNDS  = 3    # message-passing rounds over the shared-line adjacency


class MetroFeatureExtractor(BaseFeaturesExtractor):
    """
    Graph-aware feature extractor for Mini Metro.

    Architecture
    ============
    1. Station path
       - Per-station MLP: Linear(STATION_DIM → D_S) + ReLU + Linear(D_S → D_S) + ReLU.
         Applied independently to each station slot (no cross-station mixing yet).
       - _MP_ROUNDS rounds of message passing via a soft adjacency built from
         the topology tensor: adj[i,j] = lines shared by stations i and j.
         Each round: update MLP(concat(h, mean-normalised neighbour agg)).
         Multiple rounds propagate information through the real transit graph,
         replacing the dense O(N²) Transformer attention with topology-guided
         O(N·D_S) aggregation.

    2. Line path
       - Linear(LINE_DIM → D_L) per line.
       - Line-station context: for each line, mean-pool the post-MP station
         embeddings weighted by topology membership, projected to D_L.
       - Line representation: concat(line_embed, line_context) → [B, L, 2*D_L].

    3. Readout
       - station_pool: mean over all station slots → [B, D_S].
       - line_flat: reshape line representations → [B, L * 2*D_L].

    4. MLP head
       - Input: concat(global[GLOBAL_DIM], station_pool[D_S], line_flat[L*2*D_L])
       - Shape: (15 + 128 + 7*256) = 1935 → features_dim (768)
    """

    def __init__(self, observation_space: spaces.Box, features_dim: int = 768):
        super().__init__(observation_space, features_dim)

        self.global_dim   = GLOBAL_DIM
        self.station_dim  = STATION_DIM
        self.num_stations = NUM_STATIONS
        self.line_dim     = LINE_DIM
        self.num_lines    = NUM_LINES
        D_S = _D_STATION
        D_L = _D_LINE

        # --- Station path ---
        # Per-station MLP replaces the Transformer encoder. Cross-station
        # relational reasoning is handled entirely by the MP rounds below,
        # which use the actual line topology as the adjacency structure.
        # A dense Transformer here would redo that work at O(N²) cost.
        self.station_embed = nn.Sequential(
            nn.Linear(STATION_DIM, D_S),
            nn.ReLU(),
            nn.Linear(D_S, D_S),
            nn.ReLU(),
        )

        # Separate update MLP per message-passing round so each round can
        # learn a distinct aggregation function (weight-untied).
        self.mp_updates = nn.ModuleList([
            nn.Sequential(nn.Linear(D_S * 2, D_S), nn.ReLU())
            for _ in range(_MP_ROUNDS)
        ])

        # --- Line path ---
        self.line_embed = nn.Linear(LINE_DIM, D_L)
        self.line_station_proj = nn.Linear(D_S, D_L)

        # --- MLP head ---
        # global(15) + station_pool(D_S) + line_flat(NUM_LINES * 2*D_L)
        combined_dim = GLOBAL_DIM + D_S + NUM_LINES * D_L * 2
        self.mlp = nn.Sequential(
            nn.Linear(combined_dim, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        b = observations.shape[0]
        N = self.num_stations
        L = self.num_lines
        D_S = _D_STATION
        D_L = _D_LINE

        # Unpack flat observation vector
        idx = 0
        globals_ft = observations[:, idx:idx + self.global_dim]
        idx += self.global_dim

        stations_ft = observations[:, idx:idx + N * self.station_dim].view(b, N, self.station_dim)
        idx += N * self.station_dim

        lines_ft = observations[:, idx:idx + L * self.line_dim].view(b, L, self.line_dim)
        idx += L * self.line_dim

        topology = observations[:, idx:idx + L * N].view(b, L, N)  # [B, L, N] binary

        # ── Station path ─────────────────────────────────────────────────────
        h = self.station_embed(stations_ft)       # [B, N, D_S]

        # Shared-line adjacency (computed once, reused across MP rounds).
        # adj[b,i,j] = number of lines connecting stations i and j.
        topo_t = topology.permute(0, 2, 1).float()   # [B, N, L]
        adj    = topo_t @ topology.float()            # [B, N, N]
        row_sum  = adj.sum(dim=-1, keepdim=True).clamp(min=1.0)
        adj_norm = adj / row_sum                      # [B, N, N]  row-normalised

        # _MP_ROUNDS rounds of neighbourhood aggregation (weight-untied).
        for mp_update in self.mp_updates:
            agg = adj_norm @ h                        # [B, N, D_S]
            h   = mp_update(torch.cat([h, agg], dim=-1))  # [B, N, D_S]

        # Global station readout
        station_pool = h.mean(dim=1)                  # [B, D_S]

        # ── Line path ────────────────────────────────────────────────────────
        l_emb = self.line_embed(lines_ft)             # [B, L, D_L]

        topo_count   = topology.float().sum(dim=-1, keepdim=True).clamp(min=1.0)
        topo_norm    = topology.float() / topo_count  # [B, L, N]
        line_ctx_raw = topo_norm @ h                  # [B, L, D_S]
        line_ctx     = self.line_station_proj(line_ctx_raw)  # [B, L, D_L]

        line_repr = torch.cat([l_emb, line_ctx], dim=-1)  # [B, L, 2*D_L]
        line_flat = line_repr.reshape(b, L * D_L * 2)     # [B, L*2*D_L]

        # ── Final MLP ────────────────────────────────────────────────────────
        combined = torch.cat([globals_ft, station_pool, line_flat], dim=1)
        return self.mlp(combined)
