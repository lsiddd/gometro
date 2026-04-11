"""
Shared request/response schemas for the Go rl_server ↔ Python HTTP boundary.

Defines TypedDicts that document the contract for each endpoint and provides
lightweight validators used by both the inference server (infer.py) and the
training environment (env.py).  Centralising this avoids each file silently
drifting to a different understanding of the same JSON structure.

Endpoints covered:
    POST /act      — inference server (infer.py serves; rl/client.go calls)
    GET  /info     — env startup validation (env.py calls; rl/server.go serves)
    POST /reset    — env reset (env.py calls; rl/server.go serves)
    POST /step     — env step  (env.py calls; rl/server.go serves)
"""
from __future__ import annotations

from typing import Any
from typing_extensions import TypedDict


# ── /act ─────────────────────────────────────────────────────────────────────

class ActRequest(TypedDict):
    obs: list[float]
    mask: list[bool]


class ActResponse(TypedDict):
    action: list[int]


def parse_act_request(body: dict[str, Any]) -> ActRequest:
    """Validate and return a parsed /act request body.

    Raises ValueError with a descriptive message when required keys are absent.
    """
    missing = [k for k in ("obs", "mask") if k not in body]
    if missing:
        raise ValueError(f"Missing required keys in /act request: {', '.join(missing)}")
    return ActRequest(obs=body["obs"], mask=body["mask"])


# ── /reset ────────────────────────────────────────────────────────────────────

class ResetResponse(TypedDict):
    obs: list[float]
    mask: list[bool]


def parse_reset_response(body: dict[str, Any]) -> ResetResponse:
    missing = [k for k in ("obs", "mask") if k not in body]
    if missing:
        raise ValueError(f"Missing required keys in /reset response: {', '.join(missing)}")
    return ResetResponse(obs=body["obs"], mask=body["mask"])


# ── /step ─────────────────────────────────────────────────────────────────────

class StepResponse(TypedDict):
    obs: list[float]
    reward: float
    done: bool
    mask: list[bool]
    info: dict[str, Any]


def parse_step_response(body: dict[str, Any]) -> StepResponse:
    missing = [k for k in ("obs", "reward", "done", "mask") if k not in body]
    if missing:
        raise ValueError(f"Missing required keys in /step response: {', '.join(missing)}")
    return StepResponse(
        obs=body["obs"],
        reward=body["reward"],
        done=body["done"],
        mask=body["mask"],
        info=body.get("info", {}),
    )
