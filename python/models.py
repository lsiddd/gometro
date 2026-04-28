import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from constants import GLOBAL_DIM, STATION_DIM, NUM_STATIONS, LINE_DIM, NUM_LINES

torch.set_float32_matmul_precision("highest")

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
       - Line-role context: the endpoint/middle/loop role tensor produces a
         second station context, projected to D_L.
       - Line representation: concat(line_embed, line_context, role_context)
         → [B, L, 3*D_L].

    3. Readout
       - station_pool: mean over all station slots → [B, D_S].
       - line_flat: reshape line representations → [B, L * 3*D_L].

    4. MLP head
       - Input: concat(global[GLOBAL_DIM], station_pool[D_S], line_flat[L*3*D_L])
    """

    def __init__(self, observation_space: spaces.Box, features_dim: int = 768):
        super().__init__(observation_space, features_dim)

        self.global_dim   = GLOBAL_DIM
        self.station_dim  = STATION_DIM
        self.num_stations = NUM_STATIONS
        self.line_dim     = LINE_DIM
        self.num_lines    = NUM_LINES

        self.obs_dim = (
            GLOBAL_DIM
            + NUM_STATIONS * STATION_DIM
            + NUM_LINES * LINE_DIM
            + 2 * NUM_STATIONS * NUM_LINES
        )

        shape = observation_space.shape
        self.n_stack = 1
        self.channels_first = False
        self.flat_stack = False

        if len(shape) == 2:
            if shape[0] == self.obs_dim:
                self.n_stack = shape[1]
                self.channels_first = False
                self.flat_stack = False
            elif shape[1] == self.obs_dim:
                self.n_stack = shape[0]
                self.channels_first = True
                self.flat_stack = False
        elif len(shape) == 1:
            if shape[0] % self.obs_dim != 0:
                raise ValueError(
                    f"Observation dim {shape[0]} is not a multiple of base obs dim {self.obs_dim}"
                )
            self.n_stack = shape[0] // self.obs_dim
            self.channels_first = False
            self.flat_stack = True

        D_S = _D_STATION
        D_L = _D_LINE

        self.station_embed = nn.Sequential(
            nn.Linear(STATION_DIM * self.n_stack, D_S),
            nn.ReLU(),
            nn.Linear(D_S, D_S),
            nn.ReLU(),
        )

        self.mp_updates = nn.ModuleList([
            nn.Sequential(nn.Linear(D_S * 2, D_S), nn.ReLU())
            for _ in range(_MP_ROUNDS)
        ])

        self.line_embed = nn.Linear(LINE_DIM * self.n_stack, D_L)
        self.line_station_proj = nn.Linear(D_S, D_L)
        self.line_role_proj = nn.Linear(D_S, D_L)
        self.last_station_embeddings: torch.Tensor | None = None

        combined_dim = (GLOBAL_DIM * self.n_stack) + D_S + NUM_LINES * D_L * 3
        self.mlp = nn.Sequential(
            nn.Linear(combined_dim, features_dim),
            nn.ReLU(),
        )

    # torch.compile can diverge slightly on this CPU-heavy graph due to matmul
    # kernel choices; training does not rely on compiling the extractor.
    @torch._dynamo.disable
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        b = observations.shape[0]
        N = self.num_stations
        L = self.num_lines
        D_S = _D_STATION
        D_L = _D_LINE

        if self.n_stack > 1:
            if self.channels_first:
                obs_seq = observations.view(b, self.n_stack, self.obs_dim)
            elif self.flat_stack:
                obs_seq = observations.view(b, self.n_stack, self.obs_dim)
            else:
                raw = observations.view(b, self.obs_dim, self.n_stack)
                obs_seq = raw.permute(0, 2, 1)
        else:
            obs_seq = observations.unsqueeze(1)

        idx = 0

        globals_ft = obs_seq[:, :, idx:idx + self.global_dim].reshape(b, -1)
        idx += self.global_dim

        stations_ft = obs_seq[:, :, idx:idx + N * self.station_dim]
        stations_ft = (
            stations_ft.view(b, self.n_stack, N, self.station_dim)
            .permute(0, 2, 1, 3)
            .reshape(b, N, -1)
        )
        idx += N * self.station_dim

        lines_ft = obs_seq[:, :, idx:idx + L * self.line_dim]
        lines_ft = (
            lines_ft.view(b, self.n_stack, L, self.line_dim)
            .permute(0, 2, 1, 3)
            .reshape(b, L, -1)
        )
        idx += L * self.line_dim

        # Topology is static relative to instantaneous validity, take most recent frame.
        # Membership is binary and drives graph connectivity. Role stores
        # endpoint/middle/loop position and is used only as a line context.
        membership = obs_seq[:, -1, idx:idx + L * N].view(b, L, N)
        idx += L * N
        role = obs_seq[:, -1, idx:idx + L * N].view(b, L, N)

        # ── Station path ─────────────────────────────────────────────────────
        h = self.station_embed(stations_ft)       # [B, N, D_S]

        topo_t = membership.permute(0, 2, 1).float()   # [B, N, L]
        adj    = topo_t @ membership.float()            # [B, N, N]
        row_sum  = adj.sum(dim=-1, keepdim=True).clamp(min=1.0)
        adj_norm = adj / row_sum                      # [B, N, N]

        for mp_update in self.mp_updates:
            agg = adj_norm @ h                        # [B, N, D_S]
            h   = mp_update(torch.cat([h, agg], dim=-1))  # [B, N, D_S]

        self.last_station_embeddings = h
        station_pool = h.mean(dim=1)                  # [B, D_S]

        # ── Line path ────────────────────────────────────────────────────────
        l_emb = self.line_embed(lines_ft)             # [B, L, D_L]

        topo_count   = membership.float().sum(dim=-1, keepdim=True).clamp(min=1.0)
        topo_norm    = membership.float() / topo_count  # [B, L, N]
        line_ctx_raw = topo_norm @ h                  # [B, L, D_S]
        line_ctx     = self.line_station_proj(line_ctx_raw)  # [B, L, D_L]

        role_count   = role.float().sum(dim=-1, keepdim=True).clamp(min=1.0)
        role_norm    = role.float() / role_count
        role_ctx_raw = role_norm @ h
        role_ctx     = self.line_role_proj(role_ctx_raw)

        line_repr = torch.cat([l_emb, line_ctx, role_ctx], dim=-1)  # [B, L, 3*D_L]
        line_flat = line_repr.reshape(b, L * D_L * 3)              # [B, L*3*D_L]

        # ── Final MLP ────────────────────────────────────────────────────────
        combined = torch.cat([globals_ft, station_pool, line_flat], dim=1)
        return self.mlp(combined)
