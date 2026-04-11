"""
Canonical constants for the Mini Metro RL environment.

These mirror the Go constants in rl/actions.go and rl/obs.go.
MiniMetroEnv validates them against the live server on startup via GET /info,
so a mismatch between Go and Python is caught immediately rather than causing
silent shape errors during training.
"""

MAX_LINE_SLOTS    = 7
MAX_STATION_SLOTS = 50
NUM_ACTION_CATS   = 14  # action-category dimension of the MultiDiscrete space
NUM_OPTIONS       = 2   # head/tail option dimension

GLOBAL_DIM   = 15
STATION_DIM  = 16  # 9 base + 7 passenger-demand dims
LINE_DIM     = 7
NUM_LINES    = MAX_LINE_SLOTS
NUM_STATIONS = MAX_STATION_SLOTS

OBS_DIM = (
    GLOBAL_DIM
    + NUM_STATIONS * STATION_DIM
    + NUM_LINES * LINE_DIM
    + NUM_LINES * NUM_STATIONS
)
ACTION_DIMS = [NUM_ACTION_CATS, MAX_LINE_SLOTS, MAX_STATION_SLOTS, NUM_OPTIONS]

# Expected value from /info — used for validation.
_EXPECTED_INFO = {
    "obs_dim":      OBS_DIM,
    "action_dims":  ACTION_DIMS,
    "global_dim":   GLOBAL_DIM,
    "station_dim":  STATION_DIM,
    "num_stations": NUM_STATIONS,
    "line_dim":     LINE_DIM,
    "num_lines":    NUM_LINES,
}


def validate_server_constants(info: dict) -> None:
    """Raise ValueError if the Go server's constants differ from this module.

    Call this after the server starts, passing the parsed JSON from GET /info.
    """
    mismatches = []
    for key, expected in _EXPECTED_INFO.items():
        actual = info.get(key)
        if actual != expected:
            mismatches.append(f"  {key}: Go={actual!r} Python={expected!r}")
    if mismatches:
        raise ValueError(
            "Go/Python constant mismatch — recompile the server or update "
            "python/constants.py:\n" + "\n".join(mismatches)
        )
