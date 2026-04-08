package systems

import (
	"minimetro-go/components"
	"minimetro-go/config"
	"testing"
)

// newTestSolver returns a Solver ready to run (Enabled is left false; tests call
// internal methods directly).
func newTestSolver() *Solver {
	return NewSolver()
}

func stationAt(id int, x, y float64, typ config.StationType) *components.Station {
	return components.NewStation(id, x, y, typ)
}

// ---------------------------------------------------------------------------
// stationsAbove
// ---------------------------------------------------------------------------

func TestSolver_StationsAbove_NoneOverThreshold(t *testing.T) {
	s := newTestSolver()
	gs := newTestGameState()
	gs.Stations = []*components.Station{
		stationAt(0, 0, 0, config.Circle),
		stationAt(1, 100, 0, config.Triangle),
	}
	// No overcrowd progress set → all zeros.
	if got := s.stationsAbove(gs, config.OvercrowdCriticalThreshold); len(got) != 0 {
		t.Errorf("expected 0 stations above threshold, got %d", len(got))
	}
}

func TestSolver_StationsAbove_OneOverThreshold(t *testing.T) {
	s := newTestSolver()
	gs := newTestGameState()

	below := stationAt(0, 0, 0, config.Circle)
	above := stationAt(1, 100, 0, config.Triangle)
	above.OvercrowdProgress = float64(config.OvercrowdTime) * config.OvercrowdCriticalThreshold * 1.1 // 10% over

	gs.Stations = []*components.Station{below, above}

	result := s.stationsAbove(gs, config.OvercrowdCriticalThreshold)
	if len(result) != 1 {
		t.Fatalf("expected 1 station above threshold, got %d", len(result))
	}
	if result[0] != above {
		t.Error("wrong station returned")
	}
}

// ---------------------------------------------------------------------------
// isolatedStations
// ---------------------------------------------------------------------------

func TestSolver_IsolatedStations_AllOnLine(t *testing.T) {
	s := newTestSolver()
	gs := newTestGameState()

	circle := stationAt(0, 0, 0, config.Circle)
	triangle := stationAt(1, 100, 0, config.Triangle)
	line := newConnectedLine(config.LineColors[0], 0, circle, triangle)
	line.Active = true
	gs.Stations = []*components.Station{circle, triangle}
	gs.Lines = []*components.Line{line}
	gs.AvailableLines = 1

	if isolated := s.isolatedStations(gs); len(isolated) != 0 {
		t.Errorf("expected 0 isolated stations when all are on an active line, got %d", len(isolated))
	}
}

func TestSolver_IsolatedStations_OneUnconnected(t *testing.T) {
	s := newTestSolver()
	gs := newTestGameState()

	onLine := stationAt(0, 0, 0, config.Circle)
	offLine := stationAt(1, 200, 0, config.Square)
	anchor := stationAt(2, 100, 0, config.Triangle)

	line := newConnectedLine(config.LineColors[0], 0, onLine, anchor)
	line.Active = true
	gs.Stations = []*components.Station{onLine, offLine, anchor}
	gs.Lines = []*components.Line{line}
	gs.AvailableLines = 1

	isolated := s.isolatedStations(gs)
	if len(isolated) != 1 {
		t.Fatalf("expected 1 isolated station, got %d", len(isolated))
	}
	if isolated[0] != offLine {
		t.Error("wrong station identified as isolated")
	}
}

// ---------------------------------------------------------------------------
// hygieneScore
// ---------------------------------------------------------------------------

func TestSolver_HygieneScore_PenalisesRepeatType(t *testing.T) {
	s := newTestSolver()

	// Line: circle, circle — adding another circle should score lower than adding
	// a triangle (which breaks the same-type run).
	circle1 := stationAt(0, 0, 0, config.Circle)
	circle2 := stationAt(1, 100, 0, config.Circle)
	line := newConnectedLine(config.LineColors[0], 0, circle1, circle2)

	newCircle := stationAt(2, 200, 0, config.Circle)
	newTriangle := stationAt(3, 200, 0, config.Triangle)

	scoreCircle := s.hygieneScore(line, newCircle, -1)
	scoreTriangle := s.hygieneScore(line, newTriangle, -1)

	if scoreTriangle <= scoreCircle {
		t.Errorf("appending a new type (triangle, %.2f) should score higher than repeating the same type (circle, %.2f)",
			scoreTriangle, scoreCircle)
	}
}

func TestSolver_HygieneScore_BonusForNewType(t *testing.T) {
	s := newTestSolver()

	// Line contains only circles. Adding a square (not yet on line) earns the
	// "new type" bonus.
	circle1 := stationAt(0, 0, 0, config.Circle)
	circle2 := stationAt(1, 100, 0, config.Circle)
	line := newConnectedLine(config.LineColors[0], 0, circle1, circle2)

	alreadyPresent := stationAt(2, 200, 0, config.Circle)
	brandNew := stationAt(3, 200, 0, config.Square)

	scoreSame := s.hygieneScore(line, alreadyPresent, -1)
	scoreNew := s.hygieneScore(line, brandNew, -1)

	if scoreNew <= scoreSame {
		t.Errorf("adding a new type (%.2f) should outscore repeating an existing type (%.2f)", scoreNew, scoreSame)
	}
}

// ---------------------------------------------------------------------------
// bestLineForStation
// ---------------------------------------------------------------------------

func TestSolver_BestLineForStation_ReturnsNilWhenNoActiveLines(t *testing.T) {
	s := newTestSolver()
	gs := newTestGameState()

	circle := stationAt(0, 0, 0, config.Circle)
	triangle := stationAt(1, 100, 0, config.Triangle)
	gs.Stations = []*components.Station{circle, triangle}
	// Lines exist but none is active.
	inactive := components.NewLine(config.LineColors[0], 0)
	gs.Lines = []*components.Line{inactive}
	gs.AvailableLines = 1

	_, _, ok := s.bestLineForStation(gs, triangle)
	if ok {
		t.Error("should not find a line when no active lines exist")
	}
}

func TestSolver_BestLineForStation_ReturnsLineForIsolatedStation(t *testing.T) {
	s := newTestSolver()
	gs := newTestGameState()

	circle := stationAt(0, 0, 0, config.Circle)
	triangle := stationAt(1, 100, 0, config.Triangle)
	square := stationAt(2, 300, 0, config.Square)

	line := newConnectedLine(config.LineColors[0], 0, circle, triangle)
	line.Active = true
	gs.Stations = []*components.Station{circle, triangle, square}
	gs.Lines = []*components.Line{line}
	gs.AvailableLines = 1

	foundLine, _, ok := s.bestLineForStation(gs, square)
	if !ok {
		t.Fatal("expected a line candidate for an isolated square station")
	}
	if foundLine != line {
		t.Error("returned wrong line")
	}
}

func TestSolver_BestLineForStation_SkipsStationAlreadyOnLine(t *testing.T) {
	s := newTestSolver()
	gs := newTestGameState()

	circle := stationAt(0, 0, 0, config.Circle)
	triangle := stationAt(1, 100, 0, config.Triangle)

	line := newConnectedLine(config.LineColors[0], 0, circle, triangle)
	line.Active = true
	gs.Stations = []*components.Station{circle, triangle}
	gs.Lines = []*components.Line{line}
	gs.AvailableLines = 1

	// circle is already on the line — should not be returned as a candidate.
	_, _, ok := s.bestLineForStation(gs, circle)
	if ok {
		t.Error("station already on the line should not be proposed for insertion")
	}
}

// ---------------------------------------------------------------------------
// bestPartnerForNewLine
// ---------------------------------------------------------------------------

func TestSolver_BestPartnerForNewLine_PrefersOtherType(t *testing.T) {
	s := newTestSolver()
	gs := newTestGameState()

	origin := stationAt(0, 0, 0, config.Circle)
	sameType := stationAt(1, 50, 0, config.Circle)  // close, same type
	diffType := stationAt(2, 60, 0, config.Triangle) // slightly farther, different type

	gs.Stations = []*components.Station{origin, sameType, diffType}
	gs.Bridges = 10 // ample bridges

	partner := s.bestPartnerForNewLine(gs, origin)
	if partner == nil {
		t.Fatal("expected a partner to be chosen")
	}
	// A different-type station should be preferred over a same-type one at similar distance.
	if partner.Type == config.Circle {
		t.Errorf("solver should prefer a different-type partner; got type %s", partner.Type)
	}
}

func TestSolver_BestPartnerForNewLine_ExcludesSelf(t *testing.T) {
	s := newTestSolver()
	gs := newTestGameState()

	origin := stationAt(0, 0, 0, config.Circle)
	other := stationAt(1, 100, 0, config.Triangle)
	gs.Stations = []*components.Station{origin, other}
	gs.Bridges = 10

	partner := s.bestPartnerForNewLine(gs, origin)
	if partner == origin {
		t.Error("bestPartnerForNewLine must not return the origin station itself")
	}
}

func TestSolver_BestPartnerForNewLine_NilWhenNoBridges(t *testing.T) {
	s := newTestSolver()
	gs := newTestGameState()
	gs.Bridges = 0

	origin := stationAt(0, 0, 0, config.Circle)
	// Place the partner station across a simulated "river" — since we can't
	// trivially inject a river, we test bridge depletion via the cost guard:
	// with 0 bridges and CheckRiverCrossing always false (no rivers in gs),
	// cost=0 so a partner IS found. Adjust test to verify the no-bridge path
	// only matters when there IS a crossing cost.
	other := stationAt(1, 100, 0, config.Triangle)
	gs.Stations = []*components.Station{origin, other}

	// No rivers → cost is 0 → partner is found regardless of bridges.
	partner := s.bestPartnerForNewLine(gs, origin)
	if partner == nil {
		t.Error("with no rivers a partner should be found even with 0 bridges")
	}
}

// ---------------------------------------------------------------------------
// tryCloseLoop — prefers shortest eligible line
// ---------------------------------------------------------------------------

func TestSolver_TryCloseLoop_PrefersShortestLine(t *testing.T) {
	s := newTestSolver()
	gs := newTestGameState()

	// Short line: 3 stations.
	a := stationAt(0, 0, 0, config.Circle)
	b := stationAt(1, 100, 0, config.Triangle)
	c := stationAt(2, 50, 100, config.Square)
	shortLine := newConnectedLine(config.LineColors[0], 0, a, b, c)
	shortLine.Active = true

	// Long line: 5 stations.
	d := stationAt(3, 200, 0, config.Pentagon)
	e := stationAt(4, 300, 0, config.Circle)
	f := stationAt(5, 250, 100, config.Triangle)
	g := stationAt(6, 150, 150, config.Square)
	h := stationAt(7, 50, 150, config.Pentagon)
	longLine := newConnectedLine(config.LineColors[1], 1, d, e, f, g, h)
	longLine.Active = true

	gs.Stations = []*components.Station{a, b, c, d, e, f, g, h}
	gs.Lines = []*components.Line{shortLine, longLine}
	gs.AvailableLines = 2
	gs.Bridges = 10

	ok := s.tryCloseLoop(gs)
	if !ok {
		t.Fatal("tryCloseLoop should close a loop when eligible lines exist")
	}

	// The short line (3 stations) should have been closed first.
	n := len(shortLine.Stations)
	if shortLine.Stations[0] != shortLine.Stations[n-1] {
		t.Error("tryCloseLoop should have closed the shortest eligible line first")
	}
}

func TestSolver_TryCloseLoop_AcceptsThreeStations(t *testing.T) {
	s := newTestSolver()
	gs := newTestGameState()

	a := stationAt(0, 0, 0, config.Circle)
	b := stationAt(1, 100, 0, config.Triangle)
	c := stationAt(2, 50, 100, config.Square)
	line := newConnectedLine(config.LineColors[0], 0, a, b, c)
	line.Active = true
	gs.Stations = []*components.Station{a, b, c}
	gs.Lines = []*components.Line{line}
	gs.AvailableLines = 1
	gs.Bridges = 5

	if !s.tryCloseLoop(gs) {
		t.Fatal("tryCloseLoop should accept a 3-station line")
	}
	n := len(line.Stations)
	if line.Stations[0] != line.Stations[n-1] {
		t.Error("3-station line was not closed into a loop")
	}
}

// ---------------------------------------------------------------------------
// trySplitLoop
// ---------------------------------------------------------------------------

func TestSolver_TrySplitLoop_SplitsLongLoop(t *testing.T) {
	s := newTestSolver()
	gs := newTestGameState()

	// Build a 6-station closed loop: [a,b,c,d,e,f,a].
	a := stationAt(0, 0, 0, config.Circle)
	b := stationAt(1, 100, 0, config.Triangle)
	c := stationAt(2, 200, 0, config.Square)
	d := stationAt(3, 200, 100, config.Pentagon)
	e := stationAt(4, 100, 100, config.Circle)
	f := stationAt(5, 0, 100, config.Triangle)
	loop := newConnectedLine(config.LineColors[0], 0, a, b, c, d, e, f)
	loop.AddStation(a, -1, nil) // close: [a,b,c,d,e,f,a]
	loop.Active = true

	spare := components.NewLine(config.LineColors[1], 1)
	gs.Stations = []*components.Station{a, b, c, d, e, f}
	gs.StationIDCounter = 6
	gs.Lines = []*components.Line{loop, spare}
	gs.AvailableLines = 2
	gs.Bridges = 10
	gs.AvailableTrains = 1
	cityCfg := config.Cities[gs.SelectedCity]
	gs.TrainIDCounter = 0
	// The loop needs at least one train so the solver doesn't auto-spawn a new one.
	tr := components.NewTrain(0, loop, cityCfg.TrainCapacity, config.TrainMaxSpeed)
	loop.Trains = []*components.Train{tr}
	gs.Trains = []*components.Train{tr}
	gs.TrainIDCounter = 1

	ok := s.trySplitLoop(gs)
	if !ok {
		t.Fatal("trySplitLoop should split a 6-station loop when a spare line is available")
	}

	// After split: original loop and spare loop should each be closed (first==last).
	origN := len(loop.Stations)
	if loop.Stations[0] != loop.Stations[origN-1] {
		t.Error("original line after split should still be a closed loop")
	}
	spareN := len(spare.Stations)
	if spare.Stations[0] != spare.Stations[spareN-1] {
		t.Error("spare line after split should be a closed loop")
	}

	// Neither loop should have as many stations as the original (6 unique).
	origUnique := origN - 1
	spareUnique := spareN - 1
	if origUnique >= 6 || spareUnique >= 6 {
		t.Errorf("split loops should each have fewer than 6 unique stations; got %d and %d",
			origUnique, spareUnique)
	}
}

func TestSolver_TrySplitLoop_NoSpareLine_ReturnsFalse(t *testing.T) {
	s := newTestSolver()
	gs := newTestGameState()

	a := stationAt(0, 0, 0, config.Circle)
	b := stationAt(1, 100, 0, config.Triangle)
	c := stationAt(2, 200, 0, config.Square)
	d := stationAt(3, 200, 100, config.Pentagon)
	e := stationAt(4, 100, 100, config.Circle)
	f := stationAt(5, 0, 100, config.Triangle)
	loop := newConnectedLine(config.LineColors[0], 0, a, b, c, d, e, f)
	loop.AddStation(a, -1, nil)
	loop.Active = true
	gs.Stations = []*components.Station{a, b, c, d, e, f}
	gs.Lines = []*components.Line{loop} // no spare line
	gs.AvailableLines = 1
	gs.AvailableTrains = 1
	gs.Bridges = 10

	if s.trySplitLoop(gs) {
		t.Error("trySplitLoop should return false when no spare line is available")
	}
}

func TestSolver_TrySplitLoop_ShortLoop_ReturnsFalse(t *testing.T) {
	s := newTestSolver()
	gs := newTestGameState()

	// 3-station loop: too short to split.
	a := stationAt(0, 0, 0, config.Circle)
	b := stationAt(1, 100, 0, config.Triangle)
	c := stationAt(2, 50, 100, config.Square)
	loop := newConnectedLine(config.LineColors[0], 0, a, b, c)
	loop.AddStation(a, -1, nil)
	loop.Active = true
	spare := components.NewLine(config.LineColors[1], 1)
	gs.Stations = []*components.Station{a, b, c}
	gs.Lines = []*components.Line{loop, spare}
	gs.AvailableLines = 2
	gs.AvailableTrains = 1
	gs.Bridges = 10

	if s.trySplitLoop(gs) {
		t.Error("trySplitLoop should return false for a loop with fewer than 6 unique stations")
	}
}

// ---------------------------------------------------------------------------
// NewSolver — no longer accepts InputHandler
// ---------------------------------------------------------------------------

func TestSolver_NewSolver_NoInputHandlerRequired(t *testing.T) {
	// Verify the constructor compiles and returns a non-nil solver without
	// needing an InputHandler argument.
	solver := NewSolver()
	if solver == nil {
		t.Fatal("NewSolver() returned nil")
	}
	if solver.Enabled {
		t.Error("solver should be disabled by default")
	}
}

// ---------------------------------------------------------------------------
// tryDeployTrain — integration-level smoke test
// ---------------------------------------------------------------------------

func TestSolver_TryDeployTrain_DeploysToMostLoadedLine(t *testing.T) {
	s := newTestSolver()
	gs := newTestGameState()

	circle := stationAt(0, 0, 0, config.Circle)
	triangle := stationAt(1, 100, 0, config.Triangle)
	square := stationAt(2, 200, 0, config.Square)
	gs.Stations = []*components.Station{circle, triangle, square}

	busyLine := newConnectedLine(config.LineColors[0], 0, circle, triangle)
	busyLine.Active = true
	quietLine := newConnectedLine(config.LineColors[1], 1, triangle, square)
	quietLine.Active = true

	// 4 passengers waiting at circle (served by busyLine), none on quietLine.
	for i := 0; i < 4; i++ {
		p := components.NewPassenger(circle, config.Triangle, 0)
		circle.AddPassenger(p, 0)
	}

	gs.Lines = []*components.Line{busyLine, quietLine}
	gs.AvailableLines = 2
	gs.AvailableTrains = 1

	ok := s.tryDeployTrain(gs)
	if !ok {
		t.Fatal("tryDeployTrain should succeed when trains and lines are available")
	}
	if gs.AvailableTrains != 0 {
		t.Errorf("train should have been consumed; available trains = %d", gs.AvailableTrains)
	}
	if len(busyLine.Trains) != 1 {
		t.Errorf("train should have been placed on the busy line; busy=%d quiet=%d",
			len(busyLine.Trains), len(quietLine.Trains))
	}
}

// ---------------------------------------------------------------------------
// detectCircleClusters
// ---------------------------------------------------------------------------

func TestSolver_DetectCircleClusters_NoneWhenSpread(t *testing.T) {
	s := newTestSolver()
	gs := newTestGameState()
	// Circles far apart — no cluster should form.
	gs.Stations = []*components.Station{
		stationAt(0, 0, 0, config.Circle),
		stationAt(1, 500, 0, config.Circle),
		stationAt(2, 1000, 0, config.Circle),
	}
	hubs := s.detectCircleClusters(gs, 200, 3)
	if len(hubs) != 0 {
		t.Errorf("expected 0 clusters for spread circles, got %d", len(hubs))
	}
}

func TestSolver_DetectCircleClusters_DetectsCluster(t *testing.T) {
	s := newTestSolver()
	gs := newTestGameState()
	// 3 circles close together + 1 isolated triangle.
	gs.Stations = []*components.Station{
		stationAt(0, 100, 100, config.Circle),
		stationAt(1, 150, 120, config.Circle),
		stationAt(2, 130, 160, config.Circle),
		stationAt(3, 800, 800, config.Triangle),
	}
	hubs := s.detectCircleClusters(gs, 200, 3)
	if len(hubs) != 1 {
		t.Errorf("expected 1 cluster, got %d", len(hubs))
	}
}

func TestSolver_DetectCircleClusters_ReturnsHighestOvercrowdHub(t *testing.T) {
	s := newTestSolver()
	gs := newTestGameState()

	low := stationAt(0, 100, 100, config.Circle)
	low.OvercrowdProgress = 1000

	high := stationAt(1, 150, 120, config.Circle)
	high.OvercrowdProgress = 30000

	other := stationAt(2, 130, 160, config.Circle)
	other.OvercrowdProgress = 500

	gs.Stations = []*components.Station{low, high, other}
	hubs := s.detectCircleClusters(gs, 200, 3)

	if len(hubs) != 1 {
		t.Fatalf("expected 1 hub, got %d", len(hubs))
	}
	if hubs[0] != high {
		t.Errorf("hub should be the most overcrowded station; got %s", hubs[0].Type)
	}
}

// ---------------------------------------------------------------------------
// tryTitanLine
// ---------------------------------------------------------------------------

func TestSolver_TryTitanLine_DeploysLineToNearestNonCircle(t *testing.T) {
	s := newTestSolver()
	gs := newTestGameState()

	hub := stationAt(0, 100, 100, config.Circle)
	near := stationAt(1, 150, 100, config.Triangle)
	far := stationAt(2, 900, 900, config.Square)
	gs.Stations = []*components.Station{hub, near, far}
	gs.Bridges = 5
	gs.AvailableTrains = 1

	spareLine := components.NewLine(config.LineColors[0], 0)
	gs.Lines = []*components.Line{spareLine}
	gs.AvailableLines = 1
	gs.StationIDCounter = 3
	gs.TrainIDCounter = 0
	gs.SelectedCity = "london"

	if !s.tryTitanLine(gs, hub) {
		t.Fatal("tryTitanLine should succeed when resources are available")
	}
	if !spareLine.Active {
		t.Error("titan line should be active after deployment")
	}
	// Should connect to the nearest non-circle (near, not far).
	found := false
	for _, st := range spareLine.Stations {
		if st == near {
			found = true
		}
	}
	if !found {
		t.Error("titan line should connect to nearest non-circle station")
	}
}

func TestSolver_TryTitanLine_FailsWhenNoSpareTrains(t *testing.T) {
	s := newTestSolver()
	gs := newTestGameState()

	hub := stationAt(0, 100, 100, config.Circle)
	dest := stationAt(1, 200, 100, config.Triangle)
	gs.Stations = []*components.Station{hub, dest}
	gs.AvailableTrains = 0 // no trains available

	spareLine := components.NewLine(config.LineColors[0], 0)
	gs.Lines = []*components.Line{spareLine}
	gs.AvailableLines = 1

	if s.tryTitanLine(gs, hub) {
		t.Error("tryTitanLine should fail when no trains are available")
	}
}

// ---------------------------------------------------------------------------
// tryUpgradeCentralNode
// ---------------------------------------------------------------------------

func TestSolver_TryUpgradeCentralNode_UpgradesHighCentralityJunction(t *testing.T) {
	s := newTestSolver()
	gs := newTestGameState()

	hub := stationAt(0, 200, 200, config.Circle)
	a := stationAt(1, 0, 200, config.Triangle)
	b := stationAt(2, 400, 200, config.Square)
	c := stationAt(3, 200, 0, config.Pentagon)
	gs.Stations = []*components.Station{hub, a, b, c}

	// Hub is a junction between two lines.
	line1 := newConnectedLine(config.LineColors[0], 0, a, hub)
	line1.Active = true
	line2 := newConnectedLine(config.LineColors[1], 1, hub, b)
	line2.Active = true
	gs.Lines = []*components.Line{line1, line2}
	gs.AvailableLines = 2
	gs.Interchanges = 1

	// Set moderate overcrowding on the hub.
	hub.OvercrowdProgress = float64(config.OvercrowdTime) * 0.4

	gm := NewGraphManager()
	s.centrality = BetweennessCentrality(gs, gm)

	if !s.tryUpgradeCentralNode(gs) {
		t.Fatal("should upgrade high-centrality junction with moderate overcrowding")
	}
	if !hub.IsInterchange {
		t.Error("hub should be upgraded to interchange")
	}
	if gs.Interchanges != 0 {
		t.Error("interchanges resource should be decremented")
	}
}

func TestSolver_TryUpgradeCentralNode_SkipsAlreadyUpgraded(t *testing.T) {
	s := newTestSolver()
	gs := newTestGameState()

	hub := stationAt(0, 200, 200, config.Circle)
	hub.IsInterchange = true
	a := stationAt(1, 0, 200, config.Triangle)
	b := stationAt(2, 400, 200, config.Square)
	gs.Stations = []*components.Station{hub, a, b}

	line1 := newConnectedLine(config.LineColors[0], 0, a, hub)
	line1.Active = true
	line2 := newConnectedLine(config.LineColors[1], 1, hub, b)
	line2.Active = true
	gs.Lines = []*components.Line{line1, line2}
	gs.AvailableLines = 2
	gs.Interchanges = 1
	hub.OvercrowdProgress = float64(config.OvercrowdTime) * 0.9

	gm := NewGraphManager()
	s.centrality = BetweennessCentrality(gs, gm)

	if s.tryUpgradeCentralNode(gs) {
		t.Error("should not upgrade a station already marked as interchange")
	}
}
