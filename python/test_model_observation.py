import numpy as np
import torch
from gymnasium import spaces

from constants import (
    GLOBAL_DIM,
    LINE_DIM,
    NUM_LINES,
    NUM_STATIONS,
    OBS_DIM,
    STATION_DIM,
)
from models import MetroFeatureExtractor


def test_feature_extractor_accepts_split_membership_and_role_topology():
    obs_space = spaces.Box(low=0.0, high=1.0, shape=(OBS_DIM,), dtype=np.float32)
    extractor = MetroFeatureExtractor(obs_space, features_dim=64)

    obs = torch.zeros((2, OBS_DIM), dtype=torch.float32)
    topology_base = GLOBAL_DIM + NUM_STATIONS * STATION_DIM + NUM_LINES * LINE_DIM
    membership_base = topology_base
    role_base = membership_base + NUM_LINES * NUM_STATIONS

    obs[:, membership_base + 0] = 1.0
    obs[:, membership_base + 1] = 1.0
    obs[:, role_base + 0] = 0.5
    obs[:, role_base + 1] = 1.0

    out = extractor(obs)

    assert out.shape == (2, 64)
    assert extractor.last_station_embeddings is not None
    assert extractor.last_station_embeddings.shape == (2, NUM_STATIONS, 128)
    assert torch.isfinite(out).all()


def test_feature_extractor_accepts_flat_frame_stack():
    n_stack = 4
    obs_space = spaces.Box(
        low=0.0,
        high=1.0,
        shape=(OBS_DIM * n_stack,),
        dtype=np.float32,
    )
    extractor = MetroFeatureExtractor(obs_space, features_dim=64)

    obs = torch.zeros((2, OBS_DIM * n_stack), dtype=torch.float32)
    out = extractor(obs)

    assert extractor.n_stack == n_stack
    assert out.shape == (2, 64)
    assert torch.isfinite(out).all()
