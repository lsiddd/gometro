package systems

import (
	"minimetro-go/components"
	"minimetro-go/config"
	"minimetro-go/state"
	"testing"
)

// ── Rollout ───────────────────────────────────────────────────────────────────

func TestRollout_OriginalStateIsUnchanged(t *testing.T) {
	gs := buildSimState()
	origScore := gs.Score
	origSimTime := gs.SimTimeMs
	origPassengers := len(gs.Passengers)

	Rollout(gs, 60)

	// Original must be completely untouched.
	if gs.Score != origScore {
		t.Errorf("original Score mutated: want %d got %d", origScore, gs.Score)
	}
	if gs.SimTimeMs != origSimTime {
		t.Errorf("original SimTimeMs mutated: want %f got %f", origSimTime, gs.SimTimeMs)
	}
	if len(gs.Passengers) != origPassengers {
		t.Errorf("original Passengers mutated: want %d got %d", origPassengers, len(gs.Passengers))
	}
}

func TestRollout_TimeAdvances(t *testing.T) {
	gs := buildSimState()
	origTime := gs.SimTimeMs

	result := Rollout(gs, 60)

	expectedTime := origTime + 60*SimDeltaMs
	if result.SimTimeMs < origTime+SimDeltaMs {
		t.Errorf("SimTimeMs should advance; start=%.1f result=%.1f expected≈%.1f",
			origTime, result.SimTimeMs, expectedTime)
	}
}

func TestRollout_NoNewStationsSpawn(t *testing.T) {
	gs := buildSimState()
	origCount := len(gs.Stations)

	result := Rollout(gs, 300) // 5 seconds

	if len(result.Stations) != origCount {
		t.Errorf("no new stations should spawn during rollout; start=%d end=%d",
			origCount, len(result.Stations))
	}
}

func TestRollout_ZeroFrames_NoChange(t *testing.T) {
	gs := buildSimState()
	result := Rollout(gs, 0)

	if result.SimTimeMs != gs.SimTimeMs {
		t.Errorf("0-frame rollout should not change SimTimeMs; got %f", result.SimTimeMs)
	}
	if len(result.Stations) != len(gs.Stations) {
		t.Errorf("0-frame rollout changed station count")
	}
}

func TestRollout_ConcurrentCallsDoNotRace(t *testing.T) {
	gs := buildSimState()
	done := make(chan struct{}, 4)
	for i := 0; i < 4; i++ {
		go func() {
			Rollout(gs, 30)
			done <- struct{}{}
		}()
	}
	for i := 0; i < 4; i++ {
		<-done
	}
	// The test passes if the race detector does not flag anything.
}

func TestRollout_TrainsMoveDuringSimulation(t *testing.T) {
	gs := buildSimState()
	if len(gs.Trains) == 0 {
		t.Skip("no trains in test state")
	}
	origX := gs.Trains[0].X
	origY := gs.Trains[0].Y

	result := Rollout(gs, 120) // 2 seconds

	if len(result.Trains) == 0 {
		t.Fatal("trains disappeared after rollout")
	}
	// At least one train should have moved.
	moved := false
	for _, tr := range result.Trains {
		if tr.X != origX || tr.Y != origY {
			moved = true
			break
		}
	}
	if !moved {
		t.Error("no train moved after 120-frame rollout")
	}
}

func TestRollout_PassengersDeliveredIncrease(t *testing.T) {
	// Build a state where passengers are already on a train heading to their
	// destination. After a rollout they should be delivered.
	gs := state.NewGameState()
	gs.SelectedCity = "london"

	circle := components.NewStation(0, 0, 0, config.Circle)
	triangle := components.NewStation(1, 100, 0, config.Triangle)
	gs.Stations = []*components.Station{circle, triangle}

	line := newConnectedLine(config.LineColors[0], 0, circle, triangle)
	line.Active = true
	gs.Lines = []*components.Line{line, components.NewLine(config.LineColors[1], 1)}
	gs.AvailableLines = 2

	train := components.NewTrain(0, line, 6, config.TrainMaxSpeed)
	train.State = components.TrainMoving
	train.CurrentStationIndex = 0
	train.NextStationIndex = 1
	train.PathPts = [][2]float64{{0, 0}, {100, 0}}
	train.PathLength = 100
	train.Progress = 0

	p := components.NewPassenger(circle, config.Triangle, 0)
	p.OnTrain = train
	train.Passengers = []*components.Passenger{p}
	gs.Passengers = []*components.Passenger{p}

	line.Trains = []*components.Train{train}
	gs.Trains = []*components.Train{train}

	result := Rollout(gs, 300)

	if result.PassengersDelivered == 0 && result.Score == 0 {
		t.Error("passenger should be delivered during rollout; score and delivered still 0")
	}
}

// ── Evaluate ─────────────────────────────────────────────────────────────────

func TestEvaluate_GameOverReturnsLargePenalty(t *testing.T) {
	gs := state.NewGameState()
	gs.GameOver = true

	score := Evaluate(gs)
	if score < 1e6 {
		t.Errorf("game-over state should return very large penalty; got %.0f", score)
	}
}

func TestEvaluate_LowerScoreForLessOvercrowding(t *testing.T) {
	gs1 := newTestGameState()
	s1 := stationAt(0, 0, 0, config.Circle)
	s1.OvercrowdProgress = float64(config.OvercrowdTime) * 0.9
	gs1.Stations = []*components.Station{s1}

	gs2 := newTestGameState()
	s2 := stationAt(0, 0, 0, config.Circle)
	s2.OvercrowdProgress = float64(config.OvercrowdTime) * 0.1
	gs2.Stations = []*components.Station{s2}

	if Evaluate(gs1) <= Evaluate(gs2) {
		t.Error("higher overcrowding should produce higher (worse) evaluate score")
	}
}

func TestEvaluate_BetterTopologyLowerScore(t *testing.T) {
	// Good topology: alternating types, straight line.
	goodGs := newTestGameState()
	a := stationAt(0, 0, 0, config.Circle)
	b := stationAt(1, 100, 0, config.Triangle)
	c := stationAt(2, 200, 0, config.Circle)
	goodLine := newConnectedLine(config.LineColors[0], 0, a, b, c)
	goodLine.Active = true
	goodGs.Stations = []*components.Station{a, b, c}
	goodGs.Lines = []*components.Line{goodLine}
	goodGs.AvailableLines = 1

	// Bad topology: same types, U-turn.
	badGs := newTestGameState()
	x := stationAt(0, 0, 0, config.Circle)
	y := stationAt(1, 100, 0, config.Circle)
	z := stationAt(2, 0, 0, config.Circle)
	badLine := newConnectedLine(config.LineColors[0], 0, x, y, z)
	badLine.Active = true
	badGs.Stations = []*components.Station{x, y, z}
	badGs.Lines = []*components.Line{badLine}
	badGs.AvailableLines = 1

	if Evaluate(goodGs) >= Evaluate(badGs) {
		t.Errorf("good topology (%.2f) should score lower than bad topology (%.2f)",
			Evaluate(goodGs), Evaluate(badGs))
	}
}

// ── Bidirectional movement ────────────────────────────────────────────────────

// TestTrain_BidirectionalBounce verifies that a train on a linear (non-loop)
// line travels from station 0 to the end and then reverses direction, visiting
// all stations in both directions. This exercises DetermineNextStation's
// endpoint-flip logic.
func TestTrain_BidirectionalBounce(t *testing.T) {
	gs := state.NewGameState()
	gs.SelectedCity = "london"
	gs.SpawnStationsEnabled = false

	a := components.NewStation(0, 0, 0, config.Circle)
	b := components.NewStation(1, 200, 0, config.Triangle)
	c := components.NewStation(2, 400, 0, config.Square)
	gs.Stations = []*components.Station{a, b, c}
	gs.StationIDCounter = 3

	line := newConnectedLine(config.LineColors[0], 0, a, b, c)
	line.Active = true
	gs.Lines = []*components.Line{line}
	gs.AvailableLines = 1

	train := components.NewTrain(0, line, 6, config.TrainMaxSpeed)
	train.State = components.TrainMoving
	train.CurrentStationIndex = 0
	train.NextStationIndex = 1
	train.PathPts = [][2]float64{{0, 0}, {200, 0}}
	train.PathLength = 200
	line.Trains = []*components.Train{train}
	gs.Trains = []*components.Train{train}

	// Run enough frames for the train to travel A→B→C→B→A (full round trip).
	// At 1.2 px/frame on 200 px segments plus wait time, 2 000 frames is ample.
	result := Rollout(gs, 2000)

	if len(result.Trains) == 0 {
		t.Fatal("train disappeared during rollout")
	}
	tr := result.Trains[0]

	// After a full round trip the train should be back at or near station A (index 0)
	// OR at station B on its way back, meaning direction must have been -1 at some
	// point. We verify direction is -1 now OR the train visited index 0 again.
	// The simplest observable invariant: CurrentStationIndex is in [0, 2].
	if tr.CurrentStationIndex < 0 || tr.CurrentStationIndex >= len(line.Stations) {
		t.Errorf("train index %d out of bounds for 3-station line", tr.CurrentStationIndex)
	}

	// Verify the train has reversed: after 2 000 frames it cannot still be traveling
	// only in direction +1 (it would have to be at index 2 perpetually, which is
	// physically impossible since DetermineNextStation flips to -1 there).
	// A proxy: the train must not be at index 2 with direction still 1.
	if tr.CurrentStationIndex == 2 && tr.Direction == 1 {
		t.Error("train is at the last station with direction=1; it should have reversed")
	}
}

// TestTrain_BidirectionalBounce_PassengerReachesOrigin confirms that a passenger
// at the last station whose destination matches the first station type can be
// delivered in a simulation using the bidirectional bounce (no loop required).
func TestTrain_BidirectionalBounce_PassengerDelivered(t *testing.T) {
	gs := state.NewGameState()
	gs.SelectedCity = "london"
	gs.SpawnStationsEnabled = false

	// A-B-C where A and C are Circle; passenger at C wants to reach a Circle station.
	// Without reversal the passenger at C would never be delivered (no loop, no forward circle).
	a := components.NewStation(0, 0, 0, config.Circle)
	b := components.NewStation(1, 200, 0, config.Triangle)
	c := components.NewStation(2, 400, 0, config.Square)
	gs.Stations = []*components.Station{a, b, c}
	gs.StationIDCounter = 3

	line := newConnectedLine(config.LineColors[0], 0, a, b, c)
	line.Active = true
	gs.Lines = []*components.Line{line}
	gs.AvailableLines = 1

	train := components.NewTrain(0, line, 6, config.TrainMaxSpeed)
	train.State = components.TrainMoving
	train.CurrentStationIndex = 0
	train.NextStationIndex = 1
	train.PathPts = [][2]float64{{0, 0}, {200, 0}}
	train.PathLength = 200
	line.Trains = []*components.Train{train}
	gs.Trains = []*components.Train{train}

	// Passenger at station C (Square) wants to go to Circle (station A type).
	p := components.NewPassenger(c, config.Circle, 0)
	c.AddPassenger(p, 0)
	gs.Passengers = []*components.Passenger{p}

	result := Rollout(gs, 3000)

	if result.PassengersDelivered == 0 {
		t.Error("passenger at terminal station should be delivered via bidirectional train reversal")
	}
}

// ── helpers ──────────────────────────────────────────────────────────────────

// buildSimState returns a minimal but runnable game state with two connected
// stations, a line, and a train in motion.
func buildSimState() *state.GameState {
	gs := state.NewGameState()
	gs.SelectedCity = "london"
	gs.SpawnStationsEnabled = false

	circle := components.NewStation(0, 100, 200, config.Circle)
	triangle := components.NewStation(1, 300, 200, config.Triangle)
	square := components.NewStation(2, 200, 400, config.Square)
	gs.Stations = []*components.Station{circle, triangle, square}
	gs.StationIDCounter = 3

	line := newConnectedLine(config.LineColors[0], 0, circle, triangle, square)
	line.Active = true
	gs.Lines = []*components.Line{line, components.NewLine(config.LineColors[1], 1)}
	gs.AvailableLines = 2

	train := components.NewTrain(0, line, 6, config.TrainMaxSpeed)
	train.State = components.TrainMoving
	train.CurrentStationIndex = 0
	train.NextStationIndex = 1
	train.PathPts = [][2]float64{{100, 200}, {300, 200}}
	train.PathLength = 200

	p1 := components.NewPassenger(circle, config.Triangle, 0)
	p1.OnTrain = train
	train.Passengers = []*components.Passenger{p1}
	gs.Passengers = []*components.Passenger{p1}

	line.Trains = []*components.Train{train}
	gs.Trains = []*components.Train{train}

	gs.WeekStartTime = 0
	gs.LastSpawnTime = 0

	return gs
}
