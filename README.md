# gometro

A [Mini Metro](https://www.dinopoloclub.com/minimetro/) clone written in Go, built as both a fully playable game and a reinforcement learning research platform. The game engine runs at 60 Hz via [Ebitengine](https://ebitengine.org/); a Python training stack connects over gRPC to train neural network agents to play it.

---

## What It Is

Players draw metro lines between stations, deploy trains, and manage passenger flow across a procedurally populated city. The city expands each week, overcrowded stations end the game, and the trade-offs between line topology, capacity, and coverage compound quickly.

Beyond interactive play, gometro ships a complete RL pipeline: a Go gRPC server exposes the game as a Gymnasium environment, a graph-aware PyTorch policy learns to play it with PPO, and a curriculum scheduler adjusts difficulty as the agent improves.

---

## Features

### Game
- Multiple city maps with distinct river layouts and station configurations
- Expandable metro lines with per-line color coding
- Weekly upgrade system: new trains, carriages, tunnels, bridges
- Simulation speeds: 1×, 2×, 4× fast-forward
- BFS pathfinding rebuilt lazily on topology changes

### Heuristic Solver
A built-in rule-based solver handles the game autonomously. Its decision tree operates on eight priority tiers:

1. **Emergency evacuation** — ghost lines (temporary rapid-evacuation routes with automatic timeout) when any station exceeds 85% capacity
2. **Isolation repair** — connect unlinked stations
3. **Train deployment** — use available spare trains
4. **Load rebalancing** — move trains between over/under-provisioned lines
5. **Capacity upgrades** — add carriages to bottleneck trains
6. **Loop closure** — convert terminal lines to loops to eliminate endpoint starvation
7. **Interchange placement** — strategic hub upgrades
8. **Simulated annealing** — continuous background topology search

The SA optimizer runs in a goroutine on a 2-second cycle, deep-copies game state, perturbs topology with Metropolis acceptance, evaluates candidates by simulating 6 000 frames (~10 s of game time), and scores with a multi-objective cost function (angle penalties, type-coverage rewards, loop bonuses, geographic constraints).

### RL Training Pipeline
- **Transport**: bidirectional gRPC streaming — one TCP connection per episode, no per-step overhead
- **Observation**: 1 214-float normalized vector (global state, 50 station slots, 7 line slots, topology adjacency matrix)
- **Action space**: `MultiDiscrete[14, 7, 50, 2]` with action masking to exclude illegal moves at every step
- **Feature extractor**: graph-aware MLP with 3 rounds of topology-guided message passing over stations; line embeddings conditioned on station context
- **Policy**: autoregressive action factorization — each dimension conditions on previous choices, giving better gradient flow than a flat joint distribution
- **Algorithm**: MaskablePPO (stable-baselines3)
- **Curriculum**: `SpawnRateFactor` adjusts passenger and station spawn rates across six difficulty levels; a rolling-window callback promotes or demotes automatically
- **Parallelism**: 12+ workers, each spawning its own headless Go subprocess on a distinct port
- **Inference**: HTTP server backed by PyTorch or ONNX runtime (ONNX ~50× faster for single-sample inference); the live game calls it for AI-driven play

---

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                     Go Game Engine                        │
│  Ebitengine loop → systems/ → state/ → rendering/        │
│                                                          │
│  rl/ gRPC server  ←─────────────────────────────────┐   │
└───────────────────────────────────────────┬──────────┘   │
                                            │ protobuf      │
┌──────────────────────────────────────────▼──────────────┐│
│                  Python Training Stack                   ││
│                                                         ││
│  train.py  →  MaskablePPO  →  MetroFeatureExtractor     ││
│                ↓                   ↓                    ││
│         AutoregressiveNet    message passing            ││
│                ↓                                        ││
│  curriculum callback  →  checkpoint / ONNX export       ││
└─────────────────────────────────────────────────────────┘│
                                                            │
  infer.py (HTTP /act)  ←────────────────────────────────┘
```

The Go side never imports Python. The Python side never links the game engine. Communication is protobuf over gRPC.

---

## Requirements

**Game only:**
- Go 1.21+
- Ebitengine system libs (Linux): `libgl1-mesa-dev xorg-dev`

**RL training:**
- Python 3.11+ with [uv](https://github.com/astral-sh/uv)
- The Go binary in `PATH` (training spawns headless subprocesses)

---

## Building & Running

```bash
# Build and run the game
go build -o gometro .
./gometro

# Or run directly
go run .
```

### With the just task runner

```bash
just build        # compile Go binary
just run          # run game interactively
just train        # start RL training (spawns Go workers + Python PPO)
just infer        # start inference server
```

---

## Controls

| Action | Input |
|---|---|
| Draw / extend a line | Left-click drag between stations |
| Remove a segment | Right-click on a line segment |
| Select a line | Left-click on a line button (UI) |
| Fast-forward | Speed buttons in the UI |
| Toggle AI solver | Solver button in the UI |

---

## Training the Agent

```bash
# Start training from scratch (12 parallel workers, curriculum enabled)
just train

# Resume from latest checkpoint
just train --resume

# Monitor with TensorBoard
tensorboard --logdir runs/
```

Training configuration lives in `python/train.py`. Key hyperparameters:

| Parameter | Default |
|---|---|
| Workers | 12 |
| Algorithm | MaskablePPO |
| Learning rate | 3e-4 → 1e-5 (linear decay over 10 M steps) |
| Curriculum levels | `[4.0, 3.0, 2.0, 1.5, 1.25, 1.0]` spawn rate |
| Checkpoint interval | 50 000 steps |

To export a trained model to ONNX for fast inference:

```bash
uv run python/export_onnx.py --checkpoint runs/best_model.zip
```

---

## Project Structure

```
.
├── main.go              # Entry point and game loop wiring
├── config/              # City definitions, line colors, river layouts
├── components/          # Data types: Station, Train, Line, Passenger, River
├── state/               # Global GameState with deep-copy support
├── systems/             # Game logic: update loop, input, UI, pathfinding, solver, SA optimizer
├── rendering/           # Ebitengine draw calls
├── rl/                  # gRPC server and Gymnasium environment wrapper
│   └── proto/           # minimetro.proto — shared contract between Go and Python
├── python/
│   ├── train.py         # PPO training loop with curriculum callback
│   ├── env.py           # MiniMetroEnv (Gymnasium-compatible)
│   ├── models.py        # MetroFeatureExtractor + AutoregressiveActionNet
│   ├── policy.py        # MaskablePPO policy wiring
│   ├── infer.py         # Inference HTTP server (PyTorch / ONNX)
│   └── pretrain.py      # Behavioral cloning from solver demonstrations
└── assets/              # Fonts and static resources
```

---

## License

MIT
