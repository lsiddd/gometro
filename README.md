# gometro

A Mini Metro clone written in Go using the [Ebitengine](https://ebitengine.org/) 2D game framework.

## About

gometro replicates the core mechanics of Mini Metro: players draw metro lines between stations, assign trains, and manage passenger flow across a procedurally populated city map. The game runs at a fixed 60 Hz simulation step with optional fast-forward (1×, 2×, 4×) and ends when any station exceeds its passenger capacity.

A built-in **Solver** (AI player) can take over line management autonomously, making topology decisions at configurable intervals and deploying ghost lines to evacuate critically overcrowded stations.

## Features

- Multiple city maps with distinct river layouts and station configurations
- Expandable metro lines with per-line color coding
- Weekly upgrade system (new trains, carriages, tunnels, bridges)
- Autonomous solver / AI player toggle
- BFS-based pathfinding graph rebuilt lazily on topology changes

## Requirements

- Go 1.21+
- A system supported by Ebitengine (Linux/X11 or Wayland, macOS, Windows)

On Linux, Ebitengine requires the following libraries:

```
libgl1-mesa-dev xorg-dev
```

## Building & Running

```bash
go build -o gometro .
./gometro
```

Or run directly:

```bash
go run .
```

## Controls

| Action | Input |
|---|---|
| Draw / extend a line | Left-click drag between stations |
| Remove a segment | Right-click on a line segment |
| Select a line | Left-click on a line button (UI) |
| Fast-forward | Speed buttons in the UI |
| Toggle AI solver | Solver button in the UI |

## Project Structure

```
.
├── main.go              # Entry point and game loop wiring
├── config/              # City definitions, line colors, river layouts
├── components/          # Data types: Station, Train, Line, Passenger, River
├── state/               # Global GameState
├── systems/             # Game logic: update loop, input, UI, pathfinding, solver
├── rendering/           # Ebitengine draw calls
└── assets/              # Fonts and static resources
```

## License

MIT
