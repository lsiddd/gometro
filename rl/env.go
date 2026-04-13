// Package rl provides a reinforcement-learning environment wrapper around the
// Mini Metro game engine. It exposes a gymnasium-compatible interface (Reset,
// Step, ActionMask) via an HTTP JSON server so that Python training scripts can
// drive the simulation without any rendering dependency.
package rl

import (
	"minimetro-go/config"
	"minimetro-go/state"
	"minimetro-go/systems"
	"minimetro-go/systems/graph"
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

	// Pending solver actions: when the solver performs a compound operation
	// (e.g. startNewLineFor adds 2 stations + 1 train atomically), we break it
	// into individual RL steps and queue the overflow here.
	pendingSolverActions [][]int
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
	e.pendingSolverActions = nil

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

// InferSolverAction runs the solver on a deep copy of the current state and
// returns the RL action [actCat, lineIdx, stationIdx, opt] that best describes
// what the solver did, without modifying the live game state.
//
// Because the solver can perform compound operations (e.g. startNewLineFor adds
// 2 stations + 1 train atomically), this function may queue multiple follow-up
// actions in e.pendingSolverActions and drain them on successive calls before
// re-running the solver.
func (e *RLEnv) InferSolverAction() []int {
	// Drain any previously queued actions first.
	if len(e.pendingSolverActions) > 0 {
		act := e.pendingSolverActions[0]
		e.pendingSolverActions = e.pendingSolverActions[1:]
		return act
	}

	// ── Snapshot key state fields (IDs only, no pointers) ───────────────────
	type lineSnap struct {
		stationIDs []int
		trainCount int
		carriages  map[int]int // train ID → carriage count
		isLoop     bool
	}
	snap := make([]lineSnap, len(e.gs.Lines))
	for li, l := range e.gs.Lines {
		ids := make([]int, len(l.Stations))
		for k, s := range l.Stations {
			ids[k] = s.ID
		}
		carr := make(map[int]int, len(l.Trains))
		for _, t := range l.Trains {
			carr[t.ID] = t.CarriageCount
		}
		n := len(l.Stations)
		snap[li] = lineSnap{
			stationIDs: ids,
			trainCount: len(l.Trains),
			carriages:  carr,
			isLoop:     l.Active && n > 2 && l.Stations[0] == l.Stations[n-1],
		}
	}
	snapInterchanges := make(map[int]bool, len(e.gs.Stations))
	for _, st := range e.gs.Stations {
		if st.IsInterchange {
			snapInterchanges[st.ID] = true
		}
	}

	// Build a reverse-map: station ID → index in original gs.Stations.
	stIDToIdx := make(map[int]int, len(e.gs.Stations))
	for idx, st := range e.gs.Stations {
		stIDToIdx[st.ID] = idx
	}

	// ── Run solver on a throwaway deep copy ─────────────────────────────────
	gsCopy := e.gs.DeepCopy()
	gm := graph.NewGraphManager()
	gm.GetGraph(gsCopy)

	solver := systems.NewSolver()
	solver.Enabled = true
	// Pass SimTimeMs + a large offset so nowMs-lastRunMs always exceeds
	// runInterval (300 ms), guaranteeing the solver executes this tick.
	solver.Update(gsCopy, gm, gsCopy.SimTimeMs+10000.0)

	// ── Collect ALL detected changes, ordered by solver priority ─────────────
	// Station additions come first (startNewLineFor adds 2 stations + a train;
	// we queue all additions so lines get fully built across successive steps).
	var actions [][]int

	for li, l := range gsCopy.Lines {
		if li >= len(snap) {
			break
		}
		s := snap[li]
		n := len(l.Stations)
		isLoop := l.Active && n > 2 && l.Stations[0] == l.Stations[n-1]

		// Swap endpoint? (same length but a head/tail changed)
		// Must be checked before add/remove to avoid misclassifying a swap
		// as one remove + one add, which would corrupt BC labels.
		if len(l.Stations) == len(s.stationIDs) && len(s.stationIDs) >= 2 {
			prevSet := make(map[int]bool, len(s.stationIDs))
			for _, id := range s.stationIDs {
				prevSet[id] = true
			}
			for _, st := range l.Stations {
				if !prevSet[st.ID] {
					// This station is new — a swap occurred.
					si, ok := stIDToIdx[st.ID]
					if !ok {
						continue
					}
					atHead := 0
					if len(l.Stations) > 0 && l.Stations[0].ID == st.ID {
						atHead = 1
					}
					actions = append(actions, []int{ActSwapEndpoint, li, si, atHead})
				}
			}
		}

		// Station(s) added?
		if len(l.Stations) > len(s.stationIDs) {
			prevSet := make(map[int]bool, len(s.stationIDs))
			for _, id := range s.stationIDs {
				prevSet[id] = true
			}
			for _, st := range l.Stations {
				if prevSet[st.ID] {
					continue
				}
				si, ok := stIDToIdx[st.ID]
				if !ok {
					continue
				}
				atHead := 0
				if len(l.Stations) > 0 && l.Stations[0].ID == st.ID {
					atHead = 1
				}
				actions = append(actions, []int{ActAddEndpoint, li, si, atHead})
			}
		}

		// Loop closed?
		if isLoop && !s.isLoop {
			actions = append(actions, []int{ActCloseLoop, li, 0, 0})
		}

		// Loop opened?
		if !isLoop && s.isLoop {
			actions = append(actions, []int{ActOpenLoop, li, 0, 0})
		}

		// Train deployed? (queued after stations so line is active first)
		if len(l.Trains) > s.trainCount {
			actions = append(actions, []int{ActDeployTrain, li, 0, 0})
		}

		// Carriage added?
		for _, t := range l.Trains {
			if prev, ok := s.carriages[t.ID]; ok && t.CarriageCount > prev {
				actions = append(actions, []int{ActAddCarriage, li, 0, 0})
			}
		}

		// Station removed?
		if len(l.Stations) < len(s.stationIDs) && len(s.stationIDs) >= 3 {
			curSet := make(map[int]bool, len(l.Stations))
			for _, st := range l.Stations {
				curSet[st.ID] = true
			}
			for _, id := range s.stationIDs {
				if curSet[id] {
					continue
				}
				atHead := 0
				if len(s.stationIDs) > 0 && s.stationIDs[0] == id {
					atHead = 1
				}
				actions = append(actions, []int{ActRemoveEndpoint, li, 0, atHead})
			}
		}
	}

	// Interchange upgraded?
	for si, st := range gsCopy.Stations {
		if st.IsInterchange && !snapInterchanges[st.ID] {
			origSi := si
			if idx, ok := stIDToIdx[st.ID]; ok {
				origSi = idx
			}
			actions = append(actions, []int{ActUpgradeInterchange, 0, origSi, 0})
		}
	}

	if len(actions) == 0 {
		return []int{ActNoOp, 0, 0, 0}
	}

	// Return the first action, queue the rest.
	if len(actions) > 1 {
		e.pendingSolverActions = actions[1:]
	}
	return actions[0]
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
