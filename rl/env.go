// Package rl provides a reinforcement-learning environment wrapper around the
// Mini Metro game engine. It exposes a gymnasium-compatible interface (Reset,
// Step, ActionMask) via an HTTP JSON server so that Python training scripts can
// drive the simulation without any rendering dependency.
package rl

import (
	"errors"
	"math"

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
	gs        *state.GameState
	game      *systems.Game
	cfg       ComplexityConfig
	rewardCfg RewardConfig

	// Reward tracking: compare current tick against previous values.
	prevDelivered         int
	prevWeek              int
	prevQueuePressure     float64
	prevOvercrowdPressure float64
	consecutiveNoOp       int
	lastReward            RewardBreakdown
	lastValid             bool
	lastActionCategory    int

	// Upgrade modal state.
	inUpgradeModal bool
	upgradeChoices []string // up to 2 choices produced by GenerateUpgradeChoices

}

type ComplexityConfig struct {
	Level             int
	RiversEnabled     bool
	UpgradesEnabled   bool
	StationSpawnLimit int
}

func ComplexityForLevel(level int) ComplexityConfig {
	switch {
	case level <= 0:
		return ComplexityConfig{Level: 0, RiversEnabled: false, UpgradesEnabled: false, StationSpawnLimit: 6}
	case level == 1:
		return ComplexityConfig{Level: 1, RiversEnabled: false, UpgradesEnabled: false, StationSpawnLimit: 10}
	case level == 2:
		return ComplexityConfig{Level: 2, RiversEnabled: false, UpgradesEnabled: true, StationSpawnLimit: 16}
	case level == 3:
		return ComplexityConfig{Level: 3, RiversEnabled: true, UpgradesEnabled: true, StationSpawnLimit: 24}
	default:
		return ComplexityConfig{Level: 4, RiversEnabled: true, UpgradesEnabled: true, StationSpawnLimit: 0}
	}
}

// NewRLEnv allocates an RLEnv. Call Reset before the first Step.
func NewRLEnv() *RLEnv {
	return &RLEnv{
		cfg:       ComplexityForLevel(4),
		rewardCfg: DefaultRewardConfig(),
	}
}

// Reset initialises a new episode for the given city (e.g. "london").
// spawnRateFactor is a curriculum-learning multiplier (>1.0 = slower spawns =
// easier); pass 1.0 for normal difficulty.
// Returns the initial observation and action mask.
func (e *RLEnv) Reset(city string, spawnRateFactor float64) (obs []float32, mask []bool) {
	return e.ResetWithSeed(city, spawnRateFactor, 0)
}

func (e *RLEnv) ResetWithSeed(city string, spawnRateFactor float64, seed int64) (obs []float32, mask []bool) {
	e.gs = state.NewGameState()
	if seed != 0 {
		e.gs.SetSeed(seed)
	}
	e.gs.SelectedCity = city
	e.game = systems.NewGame()
	e.game.InitGame(e.gs, systems.SimScreenWidth, systems.SimScreenHeight)
	e.applyComplexity()
	if spawnRateFactor > 0 {
		e.gs.SpawnRateFactor = spawnRateFactor
	}

	e.prevDelivered = 0
	e.prevWeek = e.gs.Week
	e.prevQueuePressure, e.prevOvercrowdPressure, _ = e.rewardStatePressure()
	e.consecutiveNoOp = 0
	e.lastReward = RewardBreakdown{}
	e.lastValid = true
	e.lastActionCategory = ActNoOp
	e.inUpgradeModal = false
	e.upgradeChoices = nil

	obs = BuildObservation(e)
	mask = BuildActionMaskMulti(e)
	return
}

func (e *RLEnv) SetComplexity(level int) {
	e.cfg = ComplexityForLevel(level)
}

func (e *RLEnv) applyComplexity() {
	e.gs.UpgradesEnabled = e.cfg.UpgradesEnabled
	e.gs.StationSpawnLimit = e.cfg.StationSpawnLimit
	if !e.cfg.RiversEnabled {
		e.gs.Rivers = nil
		e.gs.Bridges = 0
	}
}

type RewardConfig struct {
	PerPassenger        float64
	QueueCoeff          float64
	QueueDeltaCoeff     float64
	OvercrowdCoeff      float64
	OvercrowdDeltaCoeff float64
	DangerThresh        float64
	DangerPenalty       float64
	NoOpCriticalPenalty float64
	WeekBonus           float64
	TerminalPenalty     float64
	InvalidAction       float64
}

func DefaultRewardConfig() RewardConfig {
	return RewardConfig{
		PerPassenger:        3.0,
		QueueCoeff:          0.03,
		QueueDeltaCoeff:     0.20,
		OvercrowdCoeff:      0.75,
		OvercrowdDeltaCoeff: 2.00,
		DangerThresh:        0.80,
		DangerPenalty:       0.5,
		NoOpCriticalPenalty: 0.25,
		WeekBonus:           20.0,
		TerminalPenalty:     100.0,
		InvalidAction:       1.0,
	}
}

func (e *RLEnv) SetRewardConfig(cfg RewardConfig) error {
	if err := ValidateRewardConfig(cfg); err != nil {
		return err
	}
	e.rewardCfg = cfg
	return nil
}

func ValidateRewardConfig(cfg RewardConfig) error {
	values := []struct {
		name  string
		value float64
	}{
		{"PerPassenger", cfg.PerPassenger},
		{"QueueCoeff", cfg.QueueCoeff},
		{"QueueDeltaCoeff", cfg.QueueDeltaCoeff},
		{"OvercrowdCoeff", cfg.OvercrowdCoeff},
		{"OvercrowdDeltaCoeff", cfg.OvercrowdDeltaCoeff},
		{"DangerThresh", cfg.DangerThresh},
		{"DangerPenalty", cfg.DangerPenalty},
		{"NoOpCriticalPenalty", cfg.NoOpCriticalPenalty},
		{"WeekBonus", cfg.WeekBonus},
		{"TerminalPenalty", cfg.TerminalPenalty},
		{"InvalidAction", cfg.InvalidAction},
	}
	for _, item := range values {
		if math.IsNaN(item.value) || math.IsInf(item.value, 0) {
			return errors.New("reward config contains NaN or Inf")
		}
		if item.value < 0 {
			return errors.New("reward config contains negative coefficient")
		}
	}
	if cfg.DangerThresh < 0.1 || cfg.DangerThresh > 1.5 {
		return errors.New("reward config DangerThresh must be in [0.1, 1.5]")
	}
	if cfg.TerminalPenalty > 1000 {
		return errors.New("reward config TerminalPenalty must be <= 1000")
	}
	return nil
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
	if len(action) == 4 {
		e.lastActionCategory = action[0]
	} else {
		e.lastActionCategory = -1
	}

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

	isNoOp := len(action) == 4 && action[0] == ActNoOp
	if validAction && isNoOp && !e.inUpgradeModal {
		e.consecutiveNoOp++
	} else if validAction {
		e.consecutiveNoOp = 0
	}

	reward = e.computeReward(done)
	if validAction && isNoOp {
		noopPenalty := e.computeNoOpPenalty()
		reward -= noopPenalty
		e.lastReward.NoOp = noopPenalty
	}
	if !validAction {
		reward -= e.rewardCfg.InvalidAction
		e.lastReward.InvalidAction = e.rewardCfg.InvalidAction
	}
	e.lastReward.Total = reward
	e.lastValid = validAction
	e.prevDelivered = gs.PassengersDelivered
	e.prevWeek = gs.Week
	e.prevQueuePressure, e.prevOvercrowdPressure, _ = e.rewardStatePressure()

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

// RewardBreakdown stores the components of the most recent transition reward.
type RewardBreakdown struct {
	Delivered      float64
	Queue          float64
	QueueDelta     float64
	Overcrowd      float64
	OvercrowdDelta float64
	Danger         float64
	Week           float64
	Terminal       float64
	InvalidAction  float64
	NoOp           float64
	Total          float64
}

// computeReward returns the shaped reward for the last transition.
//
//	+PerPassenger        × passengers delivered since last step
//	-QueueCoeff          × Σ(waiting passengers / station capacity)
//	-QueueDeltaCoeff     × queue pressure growth, positive when it shrinks
//	-OvercrowdCoeff      × Σ(overcrowdProgress/OvercrowdTime)²
//	-OvercrowdDeltaCoeff × overcrowd risk growth, positive when it shrinks
//	-DangerPenalty       per station above DangerThresh overcrowd
//	+WeekBonus           per new week completed
//	-TerminalPenalty     on game over
func (e *RLEnv) computeReward(done bool) float64 {
	gs := e.gs

	delivered := float64(gs.PassengersDelivered - e.prevDelivered)
	queuePressure, overcrowdPressure, dangerCount := e.rewardStatePressure()
	queueDelta := queuePressure - e.prevQueuePressure
	overcrowdDelta := overcrowdPressure - e.prevOvercrowdPressure

	weekBonus := 0.0
	if gs.Week > e.prevWeek {
		weekBonus = e.rewardCfg.WeekBonus
	}

	terminal := 0.0
	if done {
		terminal = e.rewardCfg.TerminalPenalty
	}

	e.lastReward = RewardBreakdown{
		Delivered:      delivered * e.rewardCfg.PerPassenger,
		Queue:          queuePressure * e.rewardCfg.QueueCoeff,
		QueueDelta:     queueDelta * e.rewardCfg.QueueDeltaCoeff,
		Overcrowd:      overcrowdPressure * e.rewardCfg.OvercrowdCoeff,
		OvercrowdDelta: overcrowdDelta * e.rewardCfg.OvercrowdDeltaCoeff,
		Danger:         dangerCount * e.rewardCfg.DangerPenalty,
		Week:           weekBonus,
		Terminal:       terminal,
	}
	return e.lastReward.Delivered -
		e.lastReward.Queue -
		e.lastReward.QueueDelta -
		e.lastReward.Overcrowd -
		e.lastReward.OvercrowdDelta -
		e.lastReward.Danger +
		e.lastReward.Week -
		e.lastReward.Terminal
}

func (e *RLEnv) rewardStatePressure() (queuePressure, overcrowdPressure, dangerCount float64) {
	cityCfg := config.Cities[e.gs.SelectedCity]
	for _, s := range e.gs.Stations {
		capacity := float64(s.Capacity(cityCfg.StationCapacity))
		if capacity <= 0 {
			continue
		}
		queuePressure += math.Min(float64(len(s.Passengers))/capacity, 3.0)

		effectiveLimit := float64(config.OvercrowdTime)
		if s.OvercrowdIsGrace {
			effectiveLimit += config.OvercrowdGraceExtra
		}
		if effectiveLimit <= 0 {
			continue
		}
		frac := math.Max(s.OvercrowdProgress/effectiveLimit, 0)
		if frac > e.rewardCfg.DangerThresh {
			dangerCount++
		}
		frac = math.Min(frac, 1.5)
		overcrowdPressure += frac * frac
	}
	return
}

func (e *RLEnv) computeNoOpPenalty() float64 {
	if e.consecutiveNoOp < 2 {
		return 0
	}
	queuePressure, overcrowdPressure, dangerCount := e.rewardStatePressure()
	if dangerCount == 0 && overcrowdPressure < 0.25 && queuePressure < 2.0 {
		return 0
	}
	scale := 1.0 + 0.25*float64(e.consecutiveNoOp-2)
	if scale > 3.0 {
		scale = 3.0
	}
	return e.rewardCfg.NoOpCriticalPenalty * scale
}

// Info returns a map of diagnostic values for the current step. Included in
// the HTTP /step response as the "info" field.
func (e *RLEnv) Info() map[string]any {
	gs := e.gs
	queuePressure, overcrowdPressure, dangerCount := e.rewardStatePressure()
	return map[string]any{
		"week":                   gs.Week,
		"score":                  gs.Score,
		"passengers_delivered":   gs.PassengersDelivered,
		"stations":               len(gs.Stations),
		"action_category":        e.lastActionCategory,
		"queue_pressure":         queuePressure,
		"overcrowd_pressure":     overcrowdPressure,
		"danger_count":           dangerCount,
		"consecutive_noop":       e.consecutiveNoOp,
		"game_over":              gs.GameOver,
		"in_upgrade_modal":       e.inUpgradeModal,
		"valid_action":           e.lastValid,
		"reward_delivered":       e.lastReward.Delivered,
		"reward_queue":           e.lastReward.Queue,
		"reward_queue_delta":     e.lastReward.QueueDelta,
		"reward_overcrowd":       e.lastReward.Overcrowd,
		"reward_overcrowd_delta": e.lastReward.OvercrowdDelta,
		"reward_danger":          e.lastReward.Danger,
		"reward_week":            e.lastReward.Week,
		"reward_terminal":        e.lastReward.Terminal,
		"reward_invalid":         e.lastReward.InvalidAction,
		"reward_noop":            e.lastReward.NoOp,
	}
}
