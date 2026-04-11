import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from constants import GLOBAL_DIM, STATION_DIM, NUM_STATIONS, LINE_DIM, NUM_LINES

class MetroFeatureExtractor(BaseFeaturesExtractor):
    """
    Custom Feature Extractor for Mini Metro.
    Uses a Transformer Encoder to process stations (since they are essentially
    an unordered set of slots on the map).
    """

    def __init__(self, observation_space: spaces.Box, features_dim: int = 512):
        super().__init__(observation_space, features_dim)
        
        self.global_dim   = GLOBAL_DIM
        self.station_dim  = STATION_DIM
        self.num_stations = NUM_STATIONS
        self.line_dim     = LINE_DIM
        self.num_lines    = NUM_LINES
        self.topology_dim = self.num_lines * self.num_stations
        
        # Embedding and Transformer for Stations
        self.station_embedding = nn.Linear(self.station_dim, 64)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=64, nhead=4, dim_feedforward=256, batch_first=True
        )
        self.station_transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        # Lines embedding
        self.line_embedding = nn.Linear(self.line_dim, 64)
        
        # Sizes after flattening
        s_flat_dim = self.num_stations * 64   # 3200
        l_flat_dim = self.num_lines * 64      # 448
        
        combined_dim = self.global_dim + s_flat_dim + l_flat_dim + self.topology_dim
        
        self.mlp = nn.Sequential(
            nn.Linear(combined_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        b = observations.shape[0]
        
        idx = 0
        globals_ft = observations[:, idx : idx + self.global_dim]
        idx += self.global_dim
        
        stations_ft = observations[:, idx : idx + self.num_stations * self.station_dim]
        stations_ft = stations_ft.view(b, self.num_stations, self.station_dim)
        idx += self.num_stations * self.station_dim
        
        lines_ft = observations[:, idx : idx + self.num_lines * self.line_dim]
        lines_ft = lines_ft.view(b, self.num_lines, self.line_dim)
        idx += self.num_lines * self.line_dim
        
        topology_ft = observations[:, idx : idx + self.topology_dim]
        
        # Process Stations
        s_emb = self.station_embedding(stations_ft)
        s_out = self.station_transformer(s_emb)
        s_flat = s_out.reshape(b, -1)
        
        # Process Lines
        l_emb = self.line_embedding(lines_ft)
        l_flat = l_emb.reshape(b, -1)
        
        # Concatenate
        combined = torch.cat([globals_ft, s_flat, l_flat, topology_ft], dim=1)
        
        return self.mlp(combined)
