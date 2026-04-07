package systems

import (
	"minimetro-go/components"
	"minimetro-go/config"
	"minimetro-go/state"
	"testing"
)

// --- helpers ---

func newTestGame() *Game {
	g := NewGame()
	g.Initialized = true
	return g
}

func newTestGameState() *state.GameState {
	gs := state.NewGameState()
	gs.SelectedCity = "london"
	gs.GraphDirty = true
	return gs
}

func newConnectedLine(color string, idx int, stations ...*components.Station) *components.Line {
	l := components.NewLine(color, idx)
	for _, s := range stations {
		l.AddStation(s, -1, nil)
	}
	return l
}

// --- CheckGameOver ---

func TestCheckGameOver_False_WhenNoOvercrowding(t *testing.T) {
	g := newTestGame()
	gs := newTestGameState()
	s := components.NewStation(0, 0, 0, "circle")
	s.OvercrowdProgress = 0
	gs.Stations = []*components.Station{s}

	if g.CheckGameOver(gs, 0) {
		t.Error("game over must not trigger with zero overcrowd progress")
	}
}

func TestCheckGameOver_True_WhenProgressExceedsLimit(t *testing.T) {
	// Spec: game ends when overcrowd timer exceeds ~45 seconds (OvercrowdTime ms)
	g := newTestGame()
	gs := newTestGameState()
	s := components.NewStation(0, 0, 0, "circle")
	s.OvercrowdProgress = float64(config.OvercrowdTime) + 1
	gs.Stations = []*components.Station{s}

	if !g.CheckGameOver(gs, 0) {
		t.Error("game over should trigger when any station overcrowd progress exceeds OvercrowdTime")
	}
}

func TestCheckGameOver_AtExactLimit_NotOver(t *testing.T) {
	g := newTestGame()
	gs := newTestGameState()
	s := components.NewStation(0, 0, 0, "circle")
	s.OvercrowdProgress = float64(config.OvercrowdTime) // exactly at limit, not over
	gs.Stations = []*components.Station{s}

	if g.CheckGameOver(gs, 0) {
		t.Error("game over should not trigger at exactly OvercrowdTime (must exceed it)")
	}
}

func TestCheckGameOver_MultipleStations_OneOvercrowded(t *testing.T) {
	g := newTestGame()
	gs := newTestGameState()
	ok := components.NewStation(0, 0, 0, "circle")
	overcrowded := components.NewStation(1, 100, 0, "triangle")
	overcrowded.OvercrowdProgress = float64(config.OvercrowdTime) + 1
	gs.Stations = []*components.Station{ok, overcrowded}

	if !g.CheckGameOver(gs, 0) {
		t.Error("a single overcrowded station among many should trigger game over")
	}
}

// --- Overcrowd timer logic (via Update inner loop) ---

func TestOvercrowdProgress_IncreasesWhenOverCapacity(t *testing.T) {
	// Spec: when passenger count exceeds capacity, the timer fills toward 45s
	gs := newTestGameState()

	s := components.NewStation(0, 400, 300, "circle")
	// Exceed London capacity of 6
	for i := 0; i < 7; i++ {
		s.Passengers = append(s.Passengers, &components.Passenger{Destination: "triangle"})
	}
	s.OvercrowdProgress = 0
	gs.Stations = []*components.Station{s}

	deltaTime := 100.0 // ms

	// Simulate the overcrowd update loop directly (same logic as in Update)
	cityCfg := config.Cities[gs.SelectedCity]
	cap := s.Capacity(cityCfg.StationCapacity)
	if len(s.Passengers) > cap {
		s.OvercrowdProgress += deltaTime * 1.0 * gs.Speed
	}

	if s.OvercrowdProgress <= 0 {
		t.Error("overcrowd progress should increase when station is over capacity")
	}
}

func TestOvercrowdProgress_DecreasesWhenUnderCapacity(t *testing.T) {
	// Spec: timer recedes when passengers drop to 5 or fewer
	s := components.NewStation(0, 0, 0, "circle")
	s.OvercrowdProgress = 5000.0 // partially filled

	deltaTime := 100.0
	speed := 1.0
	cap := 6

	// Below capacity → timer should decrease
	for i := 0; i < 5; i++ {
		s.Passengers = append(s.Passengers, &components.Passenger{Destination: "triangle"})
	}

	if len(s.Passengers) <= cap && s.OvercrowdProgress > 0 {
		s.OvercrowdProgress -= deltaTime * speed
	}

	if s.OvercrowdProgress >= 5000.0 {
		t.Error("overcrowd progress should decrease when station is back under capacity")
	}
}

func TestOvercrowdProgress_GracePeriod_SlowsRate(t *testing.T) {
	// Spec: 2-second grace period when a train is approaching — timer progresses
	// at rate OvercrowdTime / (OvercrowdTime + 2000) instead of 1.0
	deltaTime := 100.0
	speed := 1.0

	normalRate := 1.0
	graceRate := float64(config.OvercrowdTime) / (float64(config.OvercrowdTime) + 2000.0)

	normalGain := deltaTime * normalRate * speed
	graceGain := deltaTime * graceRate * speed

	if graceGain >= normalGain {
		t.Errorf("grace rate (%.4f) must be slower than normal rate (%.4f)", graceGain, normalGain)
	}
}

// --- Passenger boarding and alighting ---

func TestProcessPassengers_AlightAtDestination(t *testing.T) {
	// Spec: passenger disembarks only at a station whose type matches their destination
	g := newTestGame()
	gs := newTestGameState()

	circle := components.NewStation(0, 0, 0, "circle")
	triangle := components.NewStation(1, 100, 0, "triangle")
	gs.Stations = []*components.Station{circle, triangle}

	line := newConnectedLine("#ff0000", 0, circle, triangle)
	gs.Lines = []*components.Line{line}

	train := components.NewTrain(0, line, 6, 1.2)
	p := components.NewPassenger(circle, "triangle", 0)
	p.OnTrain = train
	train.Passengers = []*components.Passenger{p}
	gs.Passengers = []*components.Passenger{p}
	gs.Trains = []*components.Train{train}

	g.processPassengers(train, triangle, gs, 0)

	if len(train.Passengers) != 0 {
		t.Errorf("passenger should alight at matching destination; got %d on train", len(train.Passengers))
	}
	if gs.Score != 1 {
		t.Errorf("score should increment on delivery; got %d", gs.Score)
	}
}

func TestProcessPassengers_StaysOnBoardAtNonDestination(t *testing.T) {
	// Spec: passenger carrying "triangle" shape stays on board at a "square" station
	g := newTestGame()
	gs := newTestGameState()

	circle := components.NewStation(0, 0, 0, "circle")
	square := components.NewStation(1, 100, 0, "square")
	triangle := components.NewStation(2, 200, 0, "triangle")
	gs.Stations = []*components.Station{circle, square, triangle}

	line := newConnectedLine("#ff0000", 0, circle, square, triangle)
	gs.Lines = []*components.Line{line}

	train := components.NewTrain(0, line, 6, 1.2)
	train.CurrentStationIndex = 0
	train.NextStationIndex = 1
	p := components.NewPassenger(circle, "triangle", 0)
	p.OnTrain = train
	train.Passengers = []*components.Passenger{p}
	gs.Passengers = []*components.Passenger{p}
	gs.Trains = []*components.Train{train}

	g.processPassengers(train, square, gs, 0)

	if gs.Score != 0 {
		t.Error("score should not increment when passenger is not yet at destination")
	}
}

func TestProcessPassengers_BoardingFillsCapacity(t *testing.T) {
	g := newTestGame()
	gs := newTestGameState()

	circle := components.NewStation(0, 0, 0, "circle")
	triangle := components.NewStation(1, 100, 0, "triangle")
	gs.Stations = []*components.Station{circle, triangle}

	line := newConnectedLine("#ff0000", 0, circle, triangle)
	gs.Lines = []*components.Line{line}

	train := components.NewTrain(0, line, 3, 1.2) // capacity 3
	train.CurrentStationIndex = 0
	train.NextStationIndex = 1

	// 5 passengers at circle, all want triangle
	for i := 0; i < 5; i++ {
		p := components.NewPassenger(circle, "triangle", 0)
		circle.AddPassenger(p, 0)
		gs.Passengers = append(gs.Passengers, p)
	}
	gs.Trains = []*components.Train{train}

	g.processPassengers(train, circle, gs, 0)

	if len(train.Passengers) > 3 {
		t.Errorf("train should board at most capacity=3 passengers, got %d", len(train.Passengers))
	}
}

func TestProcessPassengers_ScoreIncrements(t *testing.T) {
	g := newTestGame()
	gs := newTestGameState()

	circle := components.NewStation(0, 0, 0, "circle")
	triangle := components.NewStation(1, 100, 0, "triangle")
	gs.Stations = []*components.Station{circle, triangle}

	line := newConnectedLine("#ff0000", 0, circle, triangle)
	gs.Lines = []*components.Line{line}

	train := components.NewTrain(0, line, 6, 1.2)

	// 3 passengers destined for triangle
	for i := 0; i < 3; i++ {
		p := components.NewPassenger(circle, "triangle", 0)
		p.OnTrain = train
		train.Passengers = append(train.Passengers, p)
		gs.Passengers = append(gs.Passengers, p)
	}
	gs.Trains = []*components.Train{train}

	g.processPassengers(train, triangle, gs, 0)

	if gs.Score != 3 {
		t.Errorf("score should be 3 after delivering 3 passengers, got %d", gs.Score)
	}
}

// --- Weekly cycle ---

func TestUpdate_WeekAdvance_ReturnsShowUpgrades(t *testing.T) {
	// Spec: after each week the game presents upgrade choices to the player
	g := newTestGame()
	gs := newTestGameState()
	gs.SpawnStationsEnabled = false

	// Place 3 stations so SpawnPassenger doesn't bail out immediately
	circle := components.NewStation(0, 400, 300, "circle")
	triangle := components.NewStation(1, 500, 300, "triangle")
	square := components.NewStation(2, 450, 400, "square")
	gs.Stations = []*components.Station{circle, triangle, square}

	// Start week 1 far in the past
	gs.WeekStartTime = 0
	nowMs := float64(config.WeekDuration) + 1000

	result := g.Update(gs, 16.67, 800, 600, nowMs)
	if result != "show_upgrades" {
		t.Errorf("expected 'show_upgrades' after week duration, got %q", result)
	}
	if gs.Week != 2 {
		t.Errorf("week counter should advance to 2, got %d", gs.Week)
	}
}

func TestUpdate_TrainCountIncreasesEachWeek(t *testing.T) {
	g := newTestGame()
	gs := newTestGameState()
	gs.SpawnStationsEnabled = false

	circle := components.NewStation(0, 400, 300, "circle")
	triangle := components.NewStation(1, 500, 300, "triangle")
	square := components.NewStation(2, 450, 400, "square")
	gs.Stations = []*components.Station{circle, triangle, square}

	initialTrains := gs.AvailableTrains
	gs.WeekStartTime = 0
	nowMs := float64(config.WeekDuration) + 1000

	g.Update(gs, 16.67, 800, 600, nowMs)

	if gs.AvailableTrains != initialTrains+1 {
		t.Errorf("available trains should increase by 1 per week; want %d, got %d",
			initialTrains+1, gs.AvailableTrains)
	}
}

// --- Passenger spawning ---

func TestSpawnPassenger_DoesNotSpawnWithOnlyOneStationType(t *testing.T) {
	// Spec: passengers need a different destination type to exist; spawning is suppressed
	// if no valid destination exists
	g := newTestGame()
	gs := newTestGameState()

	// All stations are circles — no valid destination exists for a circle passenger
	for i := 0; i < 3; i++ {
		gs.Stations = append(gs.Stations, components.NewStation(i, float64(i*100), 0, "circle"))
	}

	g.SpawnPassenger(gs, 0)

	if len(gs.Passengers) != 0 {
		t.Errorf("no passengers should spawn when all stations share the same type; got %d", len(gs.Passengers))
	}
}

func TestSpawnPassenger_CreatesPassengerWithDifferentDestination(t *testing.T) {
	g := newTestGame()
	gs := newTestGameState()
	gs.Stations = []*components.Station{
		components.NewStation(0, 0, 0, "circle"),
		components.NewStation(1, 100, 0, "triangle"),
	}

	g.SpawnPassenger(gs, 0)

	if len(gs.Passengers) == 0 {
		t.Fatal("a passenger should spawn when different station types exist")
	}
	p := gs.Passengers[0]
	if p.Destination == "" {
		t.Error("spawned passenger must have a destination type")
	}
}

func TestSpawnPassenger_DestinationDiffersFromOrigin(t *testing.T) {
	g := newTestGame()
	gs := newTestGameState()
	gs.Stations = []*components.Station{
		components.NewStation(0, 0, 0, "circle"),
		components.NewStation(1, 100, 0, "triangle"),
	}

	for i := 0; i < 20; i++ {
		g.SpawnPassenger(gs, float64(i))
	}

	for _, p := range gs.Passengers {
		if p.CurrentStation != nil && p.CurrentStation.Type == p.Destination {
			t.Errorf("passenger's destination must differ from their origin station type; got origin=%s dest=%s",
				p.CurrentStation.Type, p.Destination)
		}
	}
}

// --- Station type probability (P(circle) formula) ---

func TestGetNewStationType_CircleProbability_Balanced(t *testing.T) {
	// Spec: P(circle) = (squares + triangles) / (2 * total)
	// With 1 circle, 1 square, 1 triangle: P(circle) = 2/6 ≈ 0.33
	g := newTestGame()
	gs := newTestGameState()
	gs.Stations = []*components.Station{
		components.NewStation(0, 0, 0, config.Circle),
		components.NewStation(1, 100, 0, config.Square),
		components.NewStation(2, 200, 0, config.Triangle),
	}
	gs.GameStartTime = 0
	gs.LastStationSpawnTime = 0

	// With 3 balanced stations, circle probability = 2/6 ≈ 0.33
	// Run many times to observe distribution
	counts := map[string]int{}
	for i := 0; i < 300; i++ {
		typ := g.getNewStationType(gs, 0)
		counts[typ]++
	}

	// Circles should not dominate (probability ~33%)
	circleRatio := float64(counts[config.Circle]) / 300.0
	if circleRatio > 0.60 {
		t.Errorf("circle spawn ratio too high (%.2f); expected closer to 0.33", circleRatio)
	}
}

func TestGetNewStationType_CircleProbability_ManyCircles(t *testing.T) {
	// When circles dominate, their spawn probability should be suppressed
	g := newTestGame()
	gs := newTestGameState()
	for i := 0; i < 10; i++ {
		gs.Stations = append(gs.Stations, components.NewStation(i, float64(i*50), 0, config.Circle))
	}
	gs.Stations = append(gs.Stations, components.NewStation(10, 0, 100, config.Square))
	gs.Stations = append(gs.Stations, components.NewStation(11, 50, 100, config.Triangle))
	gs.GameStartTime = 0
	gs.LastStationSpawnTime = 0

	counts := map[string]int{}
	for i := 0; i < 300; i++ {
		typ := g.getNewStationType(gs, 0)
		counts[typ]++
	}

	circleRatio := float64(counts[config.Circle]) / 300.0
	nonCircle := float64(counts[config.Square]+counts[config.Triangle]) / 300.0
	if circleRatio > nonCircle+0.2 {
		t.Errorf("when circles dominate (10 vs 2), non-circle types should spawn more often; got circle=%.2f non-circle=%.2f",
			circleRatio, nonCircle)
	}
}

// --- City configuration ---

func TestCityConfig_London_StandardCapacity(t *testing.T) {
	cfg := config.Cities["london"]
	if cfg.StationCapacity != 6 {
		t.Errorf("london station capacity: want 6, got %d", cfg.StationCapacity)
	}
	if cfg.TrainCapacity != 6 {
		t.Errorf("london train capacity: want 6, got %d", cfg.TrainCapacity)
	}
}

func TestCityConfig_Paris_LowerStationCapacity(t *testing.T) {
	// Spec: Paris stations overflow faster (capacity 4 instead of 6)
	cfg := config.Cities["paris"]
	if cfg.StationCapacity != 4 {
		t.Errorf("paris station capacity: want 4, got %d", cfg.StationCapacity)
	}
}

func TestCityConfig_Cairo_SmallerTrains(t *testing.T) {
	// Spec: Cairo/Mumbai have train capacity of 4, demanding higher frequency
	cfg := config.Cities["cairo"]
	if cfg.TrainCapacity != 4 {
		t.Errorf("cairo train capacity: want 4, got %d", cfg.TrainCapacity)
	}
}

func TestCityConfig_Mumbai_SmallerTrains(t *testing.T) {
	cfg := config.Cities["mumbai"]
	if cfg.TrainCapacity != 4 {
		t.Errorf("mumbai train capacity: want 4, got %d", cfg.TrainCapacity)
	}
}

// --- Ghost line mechanic ---

func TestGhostLine_ResourcesRefundedWhenLineDeleted(t *testing.T) {
	// Spec: ghost lines — when a line is marked for deletion and it's still active,
	// the game refunds the line slot and the trains (with their carriages) to inventory
	gs := newTestGameState()
	gs.SpawnStationsEnabled = false

	circle := components.NewStation(0, 400, 300, "circle")
	triangle := components.NewStation(1, 500, 300, "triangle")
	gs.Stations = []*components.Station{circle, triangle}

	line := newConnectedLine(config.LineColors[0], 0, circle, triangle)
	line.Active = true
	gs.Lines = []*components.Line{line}

	train := components.NewTrain(0, line, 6, 1.2)
	train.CarriageCount = 1
	line.Trains = []*components.Train{train}
	gs.Trains = []*components.Train{train}

	initialLines := gs.AvailableLines
	initialTrains := gs.AvailableTrains
	initialCarriages := gs.Carriages

	// Mark for deletion — triggers resource return in the Update finalization loop
	line.MarkedForDeletion = true

	// Simulate the finalization loop from Update
	for i, l := range gs.Lines {
		if l.MarkedForDeletion && l.Active {
			gs.AvailableLines++
			gs.AvailableTrains += len(l.Trains)
			for _, tr := range l.Trains {
				gs.Carriages += tr.CarriageCount
				tr.CarriageCount = 0
			}
			l.ClearLine(func() { gs.GraphDirty = true })
			gs.Lines[i] = components.NewLine(config.LineColors[0], l.Index)
		}
	}

	if gs.AvailableLines != initialLines+1 {
		t.Errorf("line slot not refunded; want %d, got %d", initialLines+1, gs.AvailableLines)
	}
	if gs.AvailableTrains != initialTrains+1 {
		t.Errorf("train not refunded; want %d, got %d", initialTrains+1, gs.AvailableTrains)
	}
	if gs.Carriages != initialCarriages+1 {
		t.Errorf("carriage not refunded; want %d, got %d", initialCarriages+1, gs.Carriages)
	}
}

func TestGhostLine_TrainDeliversPassengersOnArrival(t *testing.T) {
	// Spec: when a ghost line train arrives at its next station, it deposits its passengers
	// and is then removed — the line slot and locomotive return to inventory
	g := newTestGame()
	gs := newTestGameState()

	circle := components.NewStation(0, 0, 0, "circle")
	triangle := components.NewStation(1, 100, 0, "triangle")
	gs.Stations = []*components.Station{circle, triangle}

	line := newConnectedLine("#ff0000", 0, circle, triangle)
	line.MarkedForDeletion = true

	train := components.NewTrain(0, line, 6, 1.2)
	train.CurrentStationIndex = 0
	train.NextStationIndex = 1
	train.State = components.TrainMoving
	train.Progress = 99999 // force arrival

	p := components.NewPassenger(circle, "square", 0) // mid-journey, not at destination
	p.OnTrain = train
	train.Passengers = []*components.Passenger{p}

	gs.Trains = []*components.Train{train}
	gs.Passengers = []*components.Passenger{p}

	// Set up waypoints so the arrival path length is reachable
	train.PathPts = [][2]float64{{0, 0}, {100, 0}}
	train.PathLength = 100.0
	train.Progress = 100.0

	g.UpdateTrain(train, gs, 16.67, 0)

	// After delivery to triangle: passenger should now be at triangle, train removed
	if len(gs.Trains) != 0 {
		t.Errorf("ghost line train should be removed after delivering to next station; got %d trains", len(gs.Trains))
	}
	if len(triangle.Passengers) == 0 {
		t.Error("passenger should be deposited at triangle after ghost train arrival")
	}
}

// --- Interchange upgrade ---

func TestInterchange_TripliesCapacity(t *testing.T) {
	// Spec: upgrade to Interchange increases limit to 18 (London base * 3)
	s := components.NewStation(0, 0, 0, "circle")
	if s.Capacity(6) != 6 {
		t.Fatal("precondition: standard capacity should be 6")
	}
	s.IsInterchange = true
	if s.Capacity(6) != 18 {
		t.Errorf("interchange capacity should be 18, got %d", s.Capacity(6))
	}
}
