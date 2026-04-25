# gometro

A [Mini Metro](https://www.dinopoloclub.com/minimetro/) clone written in Go — fully playable as a desktop game and doubling as a reinforcement learning research platform.

The game engine runs at 60 Hz via [Ebitengine](https://ebitengine.org/). A Python training stack connects over gRPC to train autoregressive neural network agents to play it from scratch, using curriculum learning, action masking, and graph-aware message passing.

---

## Table of Contents

- [What is this?](#what-is-this)
- [Game Features](#game-features)
- [RL Pipeline](#rl-pipeline)
  - [Observation Space](#observation-space)
  - [Action Space & Conditional Masking](#action-space--conditional-masking)
  - [Feature Extractor](#feature-extractor)
  - [Policy: Autoregressive Action Factorization](#policy-autoregressive-action-factorization)
  - [Curriculum Learning](#curriculum-learning)
  - [Training Infrastructure](#training-infrastructure)
- [Architecture](#architecture)
- [Requirements](#requirements)
- [Installation](#installation)
- [Running the Game](#running-the-game)
- [Controls](#controls)
- [Training an Agent](#training-an-agent)
- [Exporting to ONNX](#exporting-to-onnx)
- [Project Structure](#project-structure)
- [License](#license)

---

## What is this?

Mini Metro is a puzzle game where you draw subway lines between stations, deploy trains, and keep passenger flow from overcrowding. The city grows every week, stations accumulate passengers of different types, and trains can only carry passengers heading toward their matching destination type. A single overcrowded station ends the game.

**gometro** reimplements that game engine in Go and wraps it in a gRPC server so a Python PPO agent can interact with it like any Gymnasium environment — but at much higher throughput than a Python-native simulator would allow. The Go engine and Python training stack share zero code and communicate exclusively through protobuf messages over a persistent TCP stream.

The project is designed to be a clean reference for building RL environments around non-trivial game engines, with a focus on:

- **Structured action spaces** with illegal-action masking at every step
- **Autoregressive policies** where later action dimensions condition on earlier ones
- **Graph-structured observations** processed with message passing rather than flat MLPs
- **Curriculum learning** that adjusts environment difficulty based on agent progress

---

## Game Features

- Multiple city maps with distinct river layouts, station densities, and geographic constraints
- Up to 7 metro lines, each color-coded and independently manageable
- Trains deployable per line, carriages addable per train
- Weekly upgrade system: choose between new lines, extra trains, bridge/tunnel tokens, or interchange stations
- Simulation speeds: 1×, 2×, 4× fast-forward
- BFS pathfinding rebuilt lazily on topology changes — passengers reroute when lines are restructured
- River crossing constraints — bridge tokens required to span rivers, driving non-trivial routing decisions

---

## RL Pipeline

### Observation Space

Each step produces a 1 214-dimensional float32 vector:

| Segment | Dimensions | Content |
|---|---|---|
| Global state | 15 | Week, score, passengers delivered, available resources (trains, carriages, bridges, interchanges), spawn rate factor |
| Station slots | 50 × 16 = 800 | Per-station: type one-hot, occupancy, overcrowd timer, interchange flag, 7 passenger-demand type counts |
| Line slots | 7 × 7 = 49 | Per-line: active flag, station count, train count, carriage count, loop flag, mean occupancy, resource usage |
| Topology matrix | 50 × 7 = 350 | Binary adjacency: `topology[s][l] = 1` if station `s` is on line `l` |

All values are normalized to `[0, 1]`. Station and line slots are zero-padded to fixed sizes, so the observation shape is constant regardless of how many stations or lines are currently active.

---

### Action Space & Conditional Masking

The action space is `MultiDiscrete([14, 7, 50, 2])`:

| Head | Dimension | Meaning |
|---|---|---|
| `act_cat` | 14 | Action category (NoOp, AddEndpoint, RemoveEndpoint, CloseLoop, OpenLoop, SwapEndpoint, InsertIntoLoop, DeployTrain, AddCarriage, UpgradeInterchange, ChooseUpgrade, …) |
| `line_idx` | 7 | Which line to act on |
| `station_idx` | 50 | Which station to act on |
| `option` | 2 | Head vs. tail endpoint |

A naive independent mask per head (73 bits total) would still permit many structurally invalid combinations — e.g. marking line 2 as valid for `AddEndpoint` while station 17 is already on that line. **gometro** uses an expanded conditional mask that encodes validity of each `(act_cat, line, station, option)` combination explicitly.

**Mask layout** (5 099 bits total):

```
[0,  73)              base categorical masks    — backwards compatible
[73, 171)             per-action line masks     — 14 × 7
[171, 5071)           per-action+line station   — 14 × 7 × 50
[5071, 5099)          per-action option masks   — 14 × 2
```

The Go server computes this mask by iterating over the actual game state: for each action category, it only sets line/station/option bits for combinations that are structurally reachable given the current topology, resource counts, and loop state. The Python policy reads the conditional sections and applies them in the autoregressive sampling loop — the line head is conditioned on `act_cat`, the station head on `(act_cat, line)`, and the option head on `act_cat`.

This design eliminates the invalid-action problem almost entirely without requiring the policy to learn which moves are legal from experience.

---

### Feature Extractor

`MetroFeatureExtractor` processes the observation through three parallel paths before feeding into the policy MLP:

**1. Station path with message passing**

Each station's 16-dimensional slot is projected to a 128-dimensional embedding with a 2-layer MLP. Then 3 rounds of message passing run over a soft adjacency built from the topology matrix: `adj[i,j] = number of lines shared by stations i and j`. Each round aggregates mean-normalized neighbor embeddings and updates station representations with a small update MLP.

This propagates real connectivity information through the station graph — a congested station passes signal to its neighbors, which pass it further, so the policy can reason about downstream effects of local decisions.

**2. Line path with station context**

Each line's 7-dimensional slot is projected to 128 dimensions. Then a line-station context is computed by mean-pooling post-message-passing station embeddings weighted by topology membership, and projected to 128 dimensions. The final line representation is `concat(line_embed, line_context)` — 256 dimensions per line.

**3. Readout and MLP head**

Station embeddings are mean-pooled to a single 128-dimensional vector. Line representations are flattened to `7 × 256 = 1792` dimensions. These are concatenated with the 15-dimensional global state, giving a `1935`-dimensional input to a final projection MLP that outputs a 768-dimensional feature vector.

---

### Policy: Autoregressive Action Factorization

The joint action distribution is factored as:

```
P(act, line, station, option) =
    P(act) · P(line | act) · P(station | act, line) · P(option | act)
```

Each factor is a separate head over a shared latent representation. Sampling proceeds left to right: `act_cat` is sampled first, then `line_idx` conditioned on the sampled category, then `station_idx` conditioned on both, and finally `option`. At each step, the conditional mask section for that head and the previously sampled values is applied to the logits before sampling.

During the log-probability computation (for PPO's policy gradient), the same conditioning is applied — so the policy gradient flows through the correct conditional distributions rather than an incorrectly factorized joint.

This factorization gives much better gradient flow than a flat joint distribution over all `14 × 7 × 50 × 2 = 9800` combinations, and better credit assignment than an independent MultiDiscrete policy that ignores the structured dependencies between heads.

---

### Curriculum Learning

Training starts at 4× the normal spawn rate (slow and easy) and progressively tightens to 1× (normal game difficulty) as the agent improves. The schedule has 6 levels:

| Level | Spawn Rate Factor | Description |
|---|---|---|
| 0 | 4.0× | Very slow spawning, forgiving |
| 1 | 3.0× | |
| 2 | 2.0× | |
| 3 | 1.5× | |
| 4 | 1.25× | |
| 5 | 1.0× | Real game difficulty |

`CurriculumCallback` tracks a rolling window of episode rewards (default: last 50). When the mean exceeds `promote_threshold` (10.0), it advances one level. When it drops below `demote_threshold` (4.0), it drops one level. Promotion and demotion are propagated to both the training and evaluation environments simultaneously.

`DifficultySweepEvalCallback` runs deterministic evaluation at every curriculum level on a fixed frequency, logging per-level mean reward, episode length, score, and invalid-action rate to TensorBoard. This gives a complete picture of the agent's capability across the difficulty spectrum, not just at the current training level.

---

### Training Infrastructure

| Component | Detail |
|---|---|
| Algorithm | MaskablePPO (stable-baselines3-contrib) |
| Rollout buffer | `ConditionalMaskRolloutBuffer` — pre-allocates `(buffer_size, n_envs, 5099)` masks, bypassing SB3's dimension inference |
| Learning rate | Linear decay 3e-4 → 1e-5 over 10 M steps |
| Workers | 12 parallel headless Go subprocesses (configurable) |
| Transport | Bidirectional gRPC streaming — one TCP connection per worker per episode, no per-step connection overhead |
| Checkpoints | Every 50 000 steps; latest checkpoint auto-detected by `just resume` |
| Monitoring | TensorBoard; `InvalidActionRateCallback` logs `rollout/invalid_action_rate` every step |
| Inference | gRPC server (`infer.py`) backed by PyTorch or ONNX; live game calls it via `--rl-client` flag |

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        Go Game Engine                            │
│                                                                  │
│  main.go  →  Ebitengine 60 Hz loop                               │
│                  │                                               │
│              systems/                                            │
│                game.go        — update, overcrowd check          │
│                simulation.go  — train movement, passenger board  │
│                spawning.go    — station + passenger spawning      │
│                upgrades.go    — weekly upgrade choices            │
│                input.go       — mouse / keyboard                 │
│                ui.go          — HUD rendering                    │
│                graph/         — BFS pathfinding                  │
│                  │                                               │
│  rl/                                                             │
│    grpc_service.go  — vectorized gRPC server (n workers)         │
│    env.go           — per-env step, reward, done detection       │
│    actions.go       — action application + conditional mask      │
│    obs.go           — observation encoding                       │
│                  │                                               │
│           minimetro.proto  (shared contract)                     │
└──────────────────────────────┬───────────────────────────────────┘
                               │ protobuf / gRPC (TCP)
┌──────────────────────────────▼───────────────────────────────────┐
│                      Python Training Stack                       │
│                                                                  │
│  env.py          — MiniMetroVecEnv (SB3 VecEnv wrapper)          │
│  models.py       — MetroFeatureExtractor (graph MP + line ctx)   │
│  policy.py       — AutoregressiveDistribution + MetroPolicy      │
│  train.py        — MaskablePPO + curriculum + callbacks          │
│  export_onnx.py  — export MaskedActor / AutoregressiveActor      │
│  infer.py        — gRPC inference server for live play           │
│  pretrain.py     — behavioral cloning from solver demos          │
└──────────────────────────────────────────────────────────────────┘
```

The Go engine and Python stack share zero source code. The only contract between them is `rl/proto/minimetro.proto`. `env.py` validates at startup that its compiled-in constants (`OBS_DIM`, `ACTION_DIMS`, `MASK_SIZE`) match what the live server reports via `Info()`.

---

## Requirements

**Game only:**
- Go 1.26.1+
- Ebitengine system libs (Linux): `libgl1-mesa-dev xorg-dev`

**RL training:**
- Python 3.12+ with [uv](https://github.com/astral-sh/uv)
- The Go RL server binary in the repo root (`rl_server`) — `just build` compiles it
- `protoc` + `protoc-gen-go` + `protoc-gen-go-grpc` (only if modifying the proto file)

---

## Installation

```bash
git clone https://github.com/lsiddd/gometro.git
cd gometro

# Install system dependencies (Linux)
sudo apt install libgl1-mesa-dev xorg-dev

# Install Go dependencies
go mod download

# Install Python dependencies
cd python && uv sync
```

---

## Running the Game

```bash
# Direct
go run .

# Via just
just run
```

To play with the trained AI agent driving the decisions:

```bash
# Start the inference server
just infer

# Then launch the game (the game reads the --rl-client flag at startup)
./gometro --rl-client
```

---

## Controls

| Action | Input |
|---|---|
| Draw / extend a line | Left-click drag from station to station |
| Remove a line segment | Right-click on a segment |
| Select active line | Click a line button in the bottom bar |
| Fast-forward | Speed buttons (1×, 2×, 4×) in the top bar |
| Toggle AI solver | Solver button in the UI |

---

## Training an Agent

```bash
# Compile Go binaries and start training from scratch
just train

# Resume from the most recent checkpoint
just resume

# Resume from a specific checkpoint
just resume checkpoints/minimetro_step_500000

# Monitor training progress
just tensorboard
# → open http://localhost:6006
```

Key hyperparameters live in `python/train.py`. The most impactful ones:

| Parameter | Default | Notes |
|---|---|---|
| `--n-envs` | 12 | Parallel Go workers; higher = more throughput, more RAM |
| `--city` | `london` | Map to train on |
| Learning rate | 3e-4 → 1e-5 | Linear decay over 10 M timesteps |
| `n_steps` | 2048 | Rollout length per worker before each PPO update |
| `batch_size` | 256 | Minibatch size for PPO updates |
| `ent_coef` | 0.005 | Entropy coefficient; increase to encourage exploration |
| Curriculum promote | 10.0 | Rolling-window mean reward to advance one difficulty level |
| Curriculum demote | 4.0 | Rolling-window mean reward to drop one level |

Reward shaping coefficients (in `rl/env.go`):

| Signal | Value | Rationale |
|---|---|---|
| Per passenger delivered | +5.0 | Dense throughput signal keeps credit assignment local |
| Overcrowd pressure | −0.02 × fraction | Mild congestion cost that does not swamp step rewards |
| Danger zone (>80% full) | −0.1 per station | Early warning while terminal loss remains the dominant punishment |
| Week survival bonus | +20.0 | Explicit milestone reward once per week |
| Terminal penalty | −100.0 | Clear failure signal without crushing the return scale |
| Invalid action | −1.0 | Small penalty because the action mask already removes most invalid moves |

---

## Exporting to ONNX

Once training produces a good checkpoint, export it to ONNX for ~50× faster single-sample inference (no Python runtime overhead on the hot path):

```bash
cd python
uv run python export_onnx.py --checkpoint checkpoints/best_model.zip --output model.onnx
```

The exported model accepts `(obs: float32[B, 1214], mask: float32[B, 5099])` and returns `action: int64[B, 4]`. The inference server (`infer.py`) automatically uses the ONNX runtime if an `.onnx` file is passed.

---

## Project Structure

```
.
├── main.go                  # Entry point, Ebitengine game loop wiring
├── cmd/
│   └── rl_server/           # Headless binary entry point (--tags headless)
├── config/                  # City definitions, spawn parameters, colors, river layouts
├── components/              # Core data types: Station, Train, Line, Passenger, River
├── state/                   # GameState with deep-copy support for parallel envs
├── systems/
│   ├── game.go              # Main update loop, overcrowd detection
│   ├── simulation.go        # Train movement, passenger boarding/alighting
│   ├── spawning.go          # Passenger and station spawn logic
│   ├── upgrades.go          # Weekly upgrade generation and application
│   ├── input.go             # Mouse and keyboard handling
│   ├── ui.go                # HUD, line buttons, speed controls
│   └── graph/               # BFS pathfinding, graph manager
├── rendering/               # Ebitengine draw calls (stations, trains, lines, UI)
├── rl/
│   ├── grpc_service.go      # Vectorized gRPC server (RunVectorEpisode)
│   ├── env.go               # Per-env step execution, reward computation
│   ├── actions.go           # Action application, conditional mask construction
│   ├── obs.go               # Observation vector encoding
│   └── proto/
│       └── minimetro.proto  # Shared contract between Go server and Python client
├── python/
│   ├── train.py             # PPO training loop, curriculum, callbacks
│   ├── env.py               # MiniMetroVecEnv — SB3-compatible VecEnv wrapper
│   ├── models.py            # MetroFeatureExtractor + AutoregressiveActionNet
│   ├── policy.py            # AutoregressiveDistribution, MetroPolicy
│   ├── constants.py         # Canonical dimension constants (mirrored from Go)
│   ├── export_onnx.py       # ONNX export for MaskedActor / AutoregressiveActor
│   ├── infer.py             # gRPC inference server for live-play AI
│   └── pretrain.py          # Behavioral cloning from built-in solver demos
└── assets/                  # Fonts and static resources
```

---

## License

MIT
