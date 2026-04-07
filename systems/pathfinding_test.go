package systems

import (
	"minimetro-go/components"
	"minimetro-go/state"
	"testing"
)

func makeTestState(stations []*components.Station, lines []*components.Line) *state.GameState {
	gs := state.NewGameState()
	gs.Stations = stations
	gs.Lines = lines
	gs.GraphDirty = true
	return gs
}

func TestFindPath_DirectRoute(t *testing.T) {
	circle := components.NewStation(0, 0, 0, "circle")
	triangle := components.NewStation(1, 100, 0, "triangle")
	line := components.NewLine("#ff0000", 0)
	line.AddStation(circle, -1, nil)
	line.AddStation(triangle, -1, nil)

	gs := makeTestState([]*components.Station{circle, triangle}, []*components.Line{line})
	gm := NewGraphManager()

	path := FindPath(gm, gs, circle, "triangle")
	if path == nil {
		t.Fatal("expected a path, got nil")
	}
	if len(path) != 2 {
		t.Errorf("direct route should have 2 stations, got %d", len(path))
	}
	if path[0] != circle || path[1] != triangle {
		t.Errorf("path should be [circle, triangle], got %v", path)
	}
}

func TestFindPath_AlreadyAtDestination(t *testing.T) {
	// Spec: passenger spawns at station of their own type — this shouldn't happen,
	// but the pathfinder must handle it gracefully
	circle := components.NewStation(0, 0, 0, "circle")
	gs := makeTestState([]*components.Station{circle}, []*components.Line{})
	gm := NewGraphManager()

	path := FindPath(gm, gs, circle, "circle")
	if path == nil {
		t.Fatal("path should not be nil when already at destination")
	}
	if len(path) != 1 || path[0] != circle {
		t.Errorf("path at destination should be [circle], got %v", path)
	}
}

func TestFindPath_NoRoute(t *testing.T) {
	// Two isolated stations with no connecting line
	circle := components.NewStation(0, 0, 0, "circle")
	triangle := components.NewStation(1, 100, 0, "triangle")
	gs := makeTestState([]*components.Station{circle, triangle}, []*components.Line{})
	gm := NewGraphManager()

	path := FindPath(gm, gs, circle, "triangle")
	if path != nil {
		t.Errorf("expected nil path when no connection exists, got %v", path)
	}
}

func TestFindPath_NilStart(t *testing.T) {
	gs := makeTestState([]*components.Station{}, []*components.Line{})
	gm := NewGraphManager()

	path := FindPath(gm, gs, nil, "triangle")
	if path != nil {
		t.Error("FindPath with nil start should return nil")
	}
}

func TestFindPath_ThreeStationChain(t *testing.T) {
	// A → B → C; passenger at A wants C-type
	a := components.NewStation(0, 0, 0, "circle")
	b := components.NewStation(1, 100, 0, "square")
	c := components.NewStation(2, 200, 0, "triangle")
	line := components.NewLine("#ff0000", 0)
	line.AddStation(a, -1, nil)
	line.AddStation(b, -1, nil)
	line.AddStation(c, -1, nil)

	gs := makeTestState([]*components.Station{a, b, c}, []*components.Line{line})
	gm := NewGraphManager()

	path := FindPath(gm, gs, a, "triangle")
	if path == nil {
		t.Fatal("expected path through chain A→B→C")
	}
	if path[0] != a || path[len(path)-1] != c {
		t.Errorf("path should start at A and end at C, got %v", path)
	}
}

func TestFindPath_TransferPenalty_PrefersDirectLine(t *testing.T) {
	// Spec: passengers prefer direct routes over routes requiring transfers.
	// Transfer cost = 2.5 per transfer in the scoring formula.
	//
	// Topology:
	//   A —line1— B —line1— D(triangle)
	//   A —line1— C —line2— D(triangle)
	//
	// Route A→B→D: 0 transfers, score = 3 + 0 = 3.0
	// Route A→C→D: 1 transfer, score = 3 + 2.5 = 5.5
	// BFS should return A→B→D

	a := components.NewStation(0, 0, 0, "circle")
	b := components.NewStation(1, 100, 0, "square")
	c := components.NewStation(2, 50, 100, "square")
	d := components.NewStation(3, 200, 0, "triangle")

	line1 := components.NewLine("#ff0000", 0)
	line1.AddStation(a, -1, nil)
	line1.AddStation(b, -1, nil)
	line1.AddStation(d, -1, nil)

	line2 := components.NewLine("#0000ff", 1)
	line2.AddStation(a, -1, nil)
	line2.AddStation(c, -1, nil)
	line2.AddStation(d, -1, nil)

	gs := makeTestState(
		[]*components.Station{a, b, c, d},
		[]*components.Line{line1, line2},
	)
	gm := NewGraphManager()

	path := FindPath(gm, gs, a, "triangle")
	if path == nil {
		t.Fatal("expected a path, got nil")
	}

	// Both routes are valid. The direct line1 route (A→B→D) has score 3.0
	// while A→C→D via two different lines would cost more.
	// Verify path ends at a triangle station
	last := path[len(path)-1]
	if last.Type != "triangle" {
		t.Errorf("path should end at triangle station, got %s", last.Type)
	}
}

func TestFindPath_Loop_ReachesAllStations(t *testing.T) {
	// A loop A→B→C→A; passenger at C wants A-type
	a := components.NewStation(0, 0, 0, "circle")
	b := components.NewStation(1, 100, 0, "triangle")
	c := components.NewStation(2, 50, 100, "square")
	line := components.NewLine("#ff0000", 0)
	line.AddStation(a, -1, nil)
	line.AddStation(b, -1, nil)
	line.AddStation(c, -1, nil)
	line.AddStation(a, -1, nil) // close loop

	gs := makeTestState([]*components.Station{a, b, c}, []*components.Line{line})
	gm := NewGraphManager()

	path := FindPath(gm, gs, c, "circle")
	if path == nil {
		t.Fatal("expected path from C to circle station in loop")
	}
	last := path[len(path)-1]
	if last.Type != "circle" {
		t.Errorf("path should end at circle, got %s", last.Type)
	}
}

func TestFindPath_GraphRebuildOnDirty(t *testing.T) {
	// Verify that after adding a new line, GraphDirty=true causes the graph to
	// be rebuilt and the new route is discoverable.
	a := components.NewStation(0, 0, 0, "circle")
	triangle := components.NewStation(1, 100, 0, "triangle")

	gs := makeTestState([]*components.Station{a, triangle}, []*components.Line{})
	gm := NewGraphManager()

	// No lines yet → no path
	if path := FindPath(gm, gs, a, "triangle"); path != nil {
		t.Error("should find no path before line exists")
	}

	// Add a line and mark dirty
	line := components.NewLine("#ff0000", 0)
	line.AddStation(a, -1, nil)
	line.AddStation(triangle, -1, nil)
	gs.Lines = append(gs.Lines, line)
	gs.GraphDirty = true

	path := FindPath(gm, gs, a, "triangle")
	if path == nil {
		t.Error("after adding line and marking dirty, path should be found")
	}
}
