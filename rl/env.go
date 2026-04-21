// Package rl provides a reinforcement-learning environment wrapper around the
// Mini Metro game engine. It exposes a gymnasium-compatible interface (Reset,
// Step, ActionMask) via an HTTP JSON server so that Python training scripts can
// drive the simulation without any rendering dependency.
package rl

import (
	"minimetro-go/config"
	"minimetro-go/state"
	"minimetro-go/systems"
)

// actionIntervalFrames controls how many game frames are simulated between
// consecutive RL steps. At SimDeltaMs = 16.67 ms, 18 frames ≈ 300 ms of
// game time, matching the solver's default decision interval.
const actionIntervalFrames = 18

// RLEnv is a single instance of the game environment used for RL training.
// It is not safe for concurrent use; each training worker should own its own
// RLEnv (or communicate through the HTTP server, which serialises requests).
type RLEnv struct {
	gs   *state.GameState
	game *systems.Game

	// Reward tracking: compare current tick against previous values.
	prevDelivered int
	prevWeek      int

	// Upgrade modal state.
	inUpgradeModal bool
	upgradeChoices []string // up to 2 choices produced by GenerateUpgradeChoices

}

// NewRLEnv allocates an RLEnv. Call Reset before the first Step.
func NewRLEnv() *RLEnv {
	return &RLEnv{}
}

// Reset initialises a new episode for the given city (e.g. "london").
// spawnRateFactor is a curriculum-learning multiplier (>1.0 = slower spawns =
// easier); pass 1.0 for normal difficulty.
// Returns the initial observation and action mask.
func (e *RLEnv) Reset(city string, spawnRateFactor float64) (obs []float32, mask []bool) {
	e.gs = state.NewGameState()
	e.gs.SelectedCity = city
	if spawnRateFactor > 0 {
		e.gs.SpawnRateFactor = spawnRateFactor
	}
	e.game = systems.NewGame()
	e.game.InitGame(e.gs, systems.SimScreenWidth, systems.SimScreenHeight)

	e.prevDelivered = 0
	e.prevWeek = e.gs.Week
	e.inUpgradeModal = false
	e.upgradeChoices = nil

	obs = BuildObservation(e)
	mask = BuildActionMaskMulti(e)
	return
}

// Step advances the environment by one decision interval.
//
// If an upgrade modal is pending, the agent must supply a ChooseUpgrade action
// (offChooseUpgrade + 0 or 1). Any other action during a modal is silently
// treated as a NoOp.
//
// Returns:
//
//	obs    – next observation vector (length ObsDim)
//	reward – shaped reward for this transition
//	done   – true when the episode is over (game over)
//	mask   – action mask for the next step
func (e *RLEnv) Step(action []int) (obs []float32, reward float64, done bool, mask []bool) {
	gs := e.gs

	validAction := true

	if e.inUpgradeModal {
		// Resolve upgrade choice. Only ActChooseUpgrade is valid here;
		// any other action is penalised and the modal stays open so the
		// agent is forced to pick a valid upgrade on the next step.
		if len(action) == 4 && action[0] == ActChooseUpgrade {
			choiceIdx := action[3]
			if choiceIdx < len(e.upgradeChoices) {
				systems.ApplyUpgrade(gs, e.upgradeChoices[choiceIdx])
			}
			e.inUpgradeModal = false
			e.upgradeChoices = nil
			gs.Paused = false
		} else {
			validAction = false
		}
	} else {
		validAction = ApplyRLAction(e, action)
	}

	// Advance simulation.
	gameOverResult := e.runFrames()
	if gameOverResult {
		done = true
	}

	reward = e.computeReward(done)
	if !validAction {
		reward -= rewardInvalidAction
	}
	e.prevDelivered = gs.PassengersDelivered
	e.prevWeek = gs.Week

	obs = BuildObservation(e)
	mask = BuildActionMaskMulti(e)
	return
}

// runFrames advances the game by actionIntervalFrames steps and returns true
// when the game ends. When "show_upgrades" is encountered mid-run, the loop
// stops early and inUpgradeModal is set for the next Step call.
func (e *RLEnv) runFrames() (gameOver bool) {
	gs := e.gs
	now := gs.SimTimeMs
	for i := 0; i < actionIntervalFrames; i++ {
		now += systems.SimDeltaMs
		gs.SimTimeMs = now
		result := e.game.Update(gs, systems.SimDeltaMs, systems.SimScreenWidth, systems.SimScreenHeight, now)
		if result == "show_upgrades" {
			e.upgradeChoices = e.game.GenerateUpgradeChoices(gs)
			e.inUpgradeModal = true
			// Game.Update returns early on week boundary; resume next step.
			return false
		}
		if result == "game_over" || gs.GameOver {
			return true
		}
	}
	return false
}

// Reward shaping coefficients. All tunable values are centralised here so
// experiments require no recompile of unrelated code — change a constant, run.
const (
	rewardPerPassenger   = 1.0    // dense delivery signal per passenger delivered
	rewardOvercrowdCoeff = 0.05   // continuous penalty × Σ(overcrowdProgress/OvercrowdTime)
	rewardDangerThresh   = 0.80   // overcrowd fraction at which a station is "in danger"
	rewardDangerPenalty  = 0.5    // penalty per station above danger threshold
	rewardWeekBonus      = 2.0    // bonus for surviving each new week (was 0.1 — too weak vs passenger signal)
	rewardTerminalPenalty = 1000.0 // subtracted on game-over to strongly discourage loss
	rewardInvalidAction  = 0.1    // penalty for an invalid MultiDiscrete combination
)

// computeReward returns the shaped reward for the last transition.
//
//	+rewardPerPassenger   × passengers delivered since last step
//	-rewardOvercrowdCoeff × Σ(overcrowdProgress/OvercrowdTime)
//	-rewardDangerPenalty  per station above rewardDangerThresh overcrowd
//	+rewardWeekBonus      per new week completed
//	-rewardTerminalPenalty on game over
func (e *RLEnv) computeReward(done bool) float64 {
	gs := e.gs

	delivered := float64(gs.PassengersDelivered - e.prevDelivered)

	var overcrowdSum float64
	var dangerCount float64
	for _, s := range gs.Stations {
		effectiveLimit := float64(config.OvercrowdTime)
		if s.OvercrowdIsGrace {
			effectiveLimit += config.OvercrowdGraceExtra
		}
		frac := s.OvercrowdProgress / effectiveLimit
		overcrowdSum += frac
		if frac > rewardDangerThresh {
			dangerCount++
		}
	}

	weekBonus := 0.0
	if gs.Week > e.prevWeek {
		weekBonus = rewardWeekBonus
	}

	terminal := 0.0
	if done {
		terminal = rewardTerminalPenalty
	}

	return delivered*rewardPerPassenger - overcrowdSum*rewardOvercrowdCoeff - dangerCount*rewardDangerPenalty + weekBonus - terminal
}

// Info returns a map of diagnostic values for the current step. Included in
// the HTTP /step response as the "info" field.
func (e *RLEnv) Info() map[string]any {
	gs := e.gs
	return map[string]any{
		"week":                gs.Week,
		"score":               gs.Score,
		"passengers_delivered": gs.PassengersDelivered,
		"stations":            len(gs.Stations),
		"game_over":           gs.GameOver,
		"in_upgrade_modal":    e.inUpgradeModal,
	}
}
