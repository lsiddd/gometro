package systems

import (
	"minimetro-go/state"
)

const (
	// SimDeltaMs is the fixed time step for each simulated frame (~60 Hz).
	SimDeltaMs = 16.67
	// SimScreenWidth and SimScreenHeight are nominal dimensions passed to
	// game.Update() during simulation. Station positions are already set;
	// these values only affect the station-spawning code, which is disabled.
	SimScreenWidth  = 800.0
	SimScreenHeight = 600.0
)

// Rollout deep-copies gs and advances it by `frames` steps using the exact
// same game engine that drives the real game. Each step calls Game.Update()
// with SimDeltaMs as the time delta, so physics, passenger boarding,
// overcrowding, and train movement are all identical to live play.
//
// SpawnStationsEnabled is forced to false on the copy so that no new stations
// appear during evaluation, keeping the rollout focused on current topology.
//
// The original gs is never mutated. Each call creates an independent Game and
// GraphManager, making concurrent rollouts safe.
func Rollout(gs *state.GameState, frames int) *state.GameState {
	cp := gs.DeepCopy()

	// Each rollout gets its own Game + GraphManager so concurrent SA goroutines
	// never share mutable caches.
	localGame := NewGame()
	localGame.Initialized = true

	now := cp.SimTimeMs
	for i := 0; i < frames; i++ {
		now += SimDeltaMs
		cp.SimTimeMs = now // mirrors the main-loop increment that precedes game.Update()
		localGame.Update(cp, SimDeltaMs, SimScreenWidth, SimScreenHeight, now)
		if cp.GameOver {
			break
		}
	}
	return cp
}

