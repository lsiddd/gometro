package state

import (
	"minimetro-go/components"
	"minimetro-go/config"
	"testing"
)

// ── helpers ──────────────────────────────────────────────────────────────────

func buildLine(idx int, stations ...*components.Station) *components.Line {
	l := components.NewLine(config.LineColors[idx], idx)
	for _, s := range stations {
		l.AddStation(s, -1, nil)
	}
	return l
}

func buildTestState() *GameState {
	gs := NewGameState()
	gs.SelectedCity = "london"

	circle := components.NewStation(0, 100, 200, config.Circle)
	triangle := components.NewStation(1, 300, 200, config.Triangle)
	square := components.NewStation(2, 200, 400, config.Square)
	gs.Stations = []*components.Station{circle, triangle, square}
	gs.StationIDCounter = 3

	line := buildLine(0, circle, triangle)
	line.Active = true
	gs.Lines = []*components.Line{line, components.NewLine(config.LineColors[1], 1)}
	gs.AvailableLines = 2

	train := components.NewTrain(0, line, 6, config.TrainMaxSpeed)
	train.CarriageCount = 1
	line.Trains = []*components.Train{train}
	gs.Trains = []*components.Train{train}
	gs.TrainIDCounter = 1

	p := components.NewPassenger(circle, config.Triangle, 0)
	p.CurrentStation = circle
	p.Path = []*components.Station{circle, triangle}
	circle.Passengers = []*components.Passenger{p}
	gs.Passengers = []*components.Passenger{p}

	return gs
}

// ── independence ─────────────────────────────────────────────────────────────

func TestDeepCopy_ModifyingCopyDoesNotAffectOriginal(t *testing.T) {
	gs := buildTestState()
	cp := gs.DeepCopy()

	// Mutate a station in the copy.
	cp.Stations[0].OvercrowdProgress = 9999
	if gs.Stations[0].OvercrowdProgress != 0 {
		t.Error("mutating copy station leaked into original")
	}

	// Mutate a line in the copy.
	cp.Lines[0].MarkedForDeletion = true
	if gs.Lines[0].MarkedForDeletion {
		t.Error("mutating copy line leaked into original")
	}

	// Mutate a train in the copy.
	cp.Trains[0].CarriageCount = 99
	if gs.Trains[0].CarriageCount == 99 {
		t.Error("mutating copy train leaked into original")
	}
}

func TestDeepCopy_ModifyingOriginalDoesNotAffectCopy(t *testing.T) {
	gs := buildTestState()
	cp := gs.DeepCopy()

	gs.Stations[0].X = -999
	if cp.Stations[0].X == -999 {
		t.Error("mutating original station leaked into copy")
	}

	gs.Score = 999
	if cp.Score == 999 {
		t.Error("mutating original score leaked into copy")
	}
}

// ── pointer remapping ────────────────────────────────────────────────────────

func TestDeepCopy_StationRefsInsideLineAreRemapped(t *testing.T) {
	gs := buildTestState()
	cp := gs.DeepCopy()

	lineStations := cp.Lines[0].Stations
	if len(lineStations) == 0 {
		t.Fatal("copy line has no stations")
	}
	for _, ls := range lineStations {
		found := false
		for _, cs := range cp.Stations {
			if ls == cs {
				found = true
				break
			}
		}
		if !found {
			t.Errorf("line station %p is not in copy.Stations — pointer remapping failed", ls)
		}
	}
}

func TestDeepCopy_TrainLineRefIsRemapped(t *testing.T) {
	gs := buildTestState()
	cp := gs.DeepCopy()

	if len(cp.Trains) == 0 {
		t.Fatal("no trains in copy")
	}
	trainLine := cp.Trains[0].Line
	if trainLine == nil {
		t.Fatal("copy train.Line is nil")
	}
	found := false
	for _, l := range cp.Lines {
		if l == trainLine {
			found = true
			break
		}
	}
	if !found {
		t.Error("train.Line in copy points outside copy.Lines — pointer remapping failed")
	}
}

func TestDeepCopy_PassengerCurrentStationIsRemapped(t *testing.T) {
	gs := buildTestState()
	cp := gs.DeepCopy()

	if len(cp.Passengers) == 0 {
		t.Fatal("no passengers in copy")
	}
	p := cp.Passengers[0]
	if p.CurrentStation == nil {
		t.Fatal("copy passenger.CurrentStation is nil")
	}
	found := false
	for _, s := range cp.Stations {
		if s == p.CurrentStation {
			found = true
			break
		}
	}
	if !found {
		t.Error("passenger.CurrentStation in copy points outside copy.Stations")
	}
}

func TestDeepCopy_PassengerPathStationsAreRemapped(t *testing.T) {
	gs := buildTestState()
	cp := gs.DeepCopy()

	p := cp.Passengers[0]
	if len(p.Path) == 0 {
		t.Fatal("passenger path is empty in copy")
	}
	for _, ps := range p.Path {
		if ps == nil {
			continue
		}
		found := false
		for _, s := range cp.Stations {
			if s == ps {
				found = true
				break
			}
		}
		if !found {
			t.Errorf("path station %p not found in copy.Stations", ps)
		}
	}
}

func TestDeepCopy_StationPassengersAreRemapped(t *testing.T) {
	gs := buildTestState()
	cp := gs.DeepCopy()

	if len(cp.Stations[0].Passengers) == 0 {
		t.Fatal("station has no passengers in copy")
	}
	stPax := cp.Stations[0].Passengers[0]
	found := false
	for _, p := range cp.Passengers {
		if p == stPax {
			found = true
			break
		}
	}
	if !found {
		t.Error("station passenger in copy is not in copy.Passengers")
	}
}

// ── simulation config ─────────────────────────────────────────────────────────

func TestDeepCopy_SimulationDefaults(t *testing.T) {
	gs := buildTestState()
	gs.Paused = true
	gs.SpawnStationsEnabled = true

	cp := gs.DeepCopy()
	if cp.Paused {
		t.Error("copy should never be paused (simulation runs unpaused)")
	}
	if cp.SpawnStationsEnabled {
		t.Error("copy should have SpawnStationsEnabled=false for deterministic evaluation")
	}
	if !cp.GraphDirty {
		t.Error("copy should have GraphDirty=true to force graph rebuild with new pointers")
	}
}

// ── structural integrity ──────────────────────────────────────────────────────

func TestDeepCopy_CountsArePreserved(t *testing.T) {
	gs := buildTestState()
	cp := gs.DeepCopy()

	if len(cp.Stations) != len(gs.Stations) {
		t.Errorf("station count: want %d got %d", len(gs.Stations), len(cp.Stations))
	}
	if len(cp.Lines) != len(gs.Lines) {
		t.Errorf("line count: want %d got %d", len(gs.Lines), len(cp.Lines))
	}
	if len(cp.Trains) != len(gs.Trains) {
		t.Errorf("train count: want %d got %d", len(gs.Trains), len(cp.Trains))
	}
	if len(cp.Passengers) != len(gs.Passengers) {
		t.Errorf("passenger count: want %d got %d", len(gs.Passengers), len(cp.Passengers))
	}
}

func TestDeepCopy_ScalarFieldsMatch(t *testing.T) {
	gs := buildTestState()
	gs.Score = 42
	gs.Week = 3
	gs.Bridges = 5
	gs.SimTimeMs = 12345.6

	cp := gs.DeepCopy()

	if cp.Score != 42 {
		t.Errorf("Score: want 42 got %d", cp.Score)
	}
	if cp.Week != 3 {
		t.Errorf("Week: want 3 got %d", cp.Week)
	}
	if cp.Bridges != 5 {
		t.Errorf("Bridges: want 5 got %d", cp.Bridges)
	}
	if cp.SimTimeMs != 12345.6 {
		t.Errorf("SimTimeMs: want 12345.6 got %f", cp.SimTimeMs)
	}
}

// ── dummy lines (trains on deleted lines) ────────────────────────────────────

func TestDeepCopy_DummyLineOnTrainIsPreserved(t *testing.T) {
	gs := NewGameState()
	gs.SelectedCity = "london"

	circle := components.NewStation(0, 100, 200, config.Circle)
	triangle := components.NewStation(1, 300, 200, config.Triangle)
	gs.Stations = []*components.Station{circle, triangle}

	// Create a "dummy" line that is not in gs.Lines (simulates post-deletion train).
	dummy := components.NewLine("#000000", 99)
	dummy.Active = false
	dummy.MarkedForDeletion = true
	dummy.Stations = []*components.Station{circle, triangle}

	train := components.NewTrain(0, dummy, 6, config.TrainMaxSpeed)
	gs.Trains = []*components.Train{train}

	// No lines in gs.Lines (only the dummy on the train).
	gs.Lines = []*components.Line{}

	cp := gs.DeepCopy()

	if len(cp.Trains) != 1 {
		t.Fatalf("expected 1 train in copy, got %d", len(cp.Trains))
	}
	if cp.Trains[0].Line == nil {
		t.Fatal("copy train.Line is nil; dummy line was not preserved")
	}
	if cp.Trains[0].Line == train.Line {
		t.Error("copy train.Line points to original dummy — not deep-copied")
	}
}
