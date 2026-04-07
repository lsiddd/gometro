package components

import "testing"

func makeSimpleLine(stations ...*Station) *Line {
	l := NewLine("#ff0000", 0)
	for _, s := range stations {
		l.AddStation(s, -1, nil)
	}
	return l
}

// --- Capacity ---

func TestTrain_TotalCapacity_NoCarriage(t *testing.T) {
	// Spec: base capacity 6 for London
	line := makeSimpleLine(NewStation(0, 0, 0, "circle"), NewStation(1, 100, 0, "triangle"))
	tr := NewTrain(0, line, 6, 1.2)
	if got := tr.TotalCapacity(); got != 6 {
		t.Errorf("total capacity without carriage: want 6, got %d", got)
	}
}

func TestTrain_TotalCapacity_WithOneCarriage(t *testing.T) {
	// Spec: each carriage doubles capacity slot — TotalCapacity = Capacity * (1 + carriages)
	line := makeSimpleLine(NewStation(0, 0, 0, "circle"), NewStation(1, 100, 0, "triangle"))
	tr := NewTrain(0, line, 6, 1.2)
	tr.CarriageCount = 1
	if got := tr.TotalCapacity(); got != 12 {
		t.Errorf("total capacity with 1 carriage: want 12, got %d", got)
	}
}

func TestTrain_TotalCapacity_SmallCityBase(t *testing.T) {
	// Cairo/Mumbai train capacity is 4 instead of 6
	line := makeSimpleLine(NewStation(0, 0, 0, "circle"), NewStation(1, 100, 0, "triangle"))
	tr := NewTrain(0, line, 4, 1.2)
	if got := tr.TotalCapacity(); got != 4 {
		t.Errorf("cairo train capacity: want 4, got %d", got)
	}
}

func TestTrain_HasCarriage(t *testing.T) {
	line := makeSimpleLine(NewStation(0, 0, 0, "circle"), NewStation(1, 100, 0, "triangle"))
	tr := NewTrain(0, line, 6, 1.2)
	if tr.HasCarriage() {
		t.Error("new train should not have carriage")
	}
	tr.CarriageCount = 1
	if !tr.HasCarriage() {
		t.Error("train with carriage should return HasCarriage=true")
	}
}

// --- Loop detection ---

func TestTrain_IsLoop_ThreeStationsClosedLoop(t *testing.T) {
	// Spec: loops allow trains to circulate without reversing
	a := NewStation(0, 0, 0, "circle")
	b := NewStation(1, 100, 0, "triangle")
	c := NewStation(2, 50, 100, "square")
	line := NewLine("#ff0000", 0)
	line.AddStation(a, -1, nil)
	line.AddStation(b, -1, nil)
	line.AddStation(c, -1, nil)
	line.AddStation(a, -1, nil) // close loop
	tr := NewTrain(0, line, 6, 1.2)
	if !tr.IsLoop {
		t.Error("closed 3-station line should be detected as loop")
	}
}

func TestTrain_IsLoop_TwoStations_NotLoop(t *testing.T) {
	a := NewStation(0, 0, 0, "circle")
	b := NewStation(1, 100, 0, "triangle")
	line := makeSimpleLine(a, b)
	tr := NewTrain(0, line, 6, 1.2)
	if tr.IsLoop {
		t.Error("two-station line cannot be a loop")
	}
}

func TestTrain_IsLoop_ThreeStations_Open(t *testing.T) {
	a := NewStation(0, 0, 0, "circle")
	b := NewStation(1, 100, 0, "triangle")
	c := NewStation(2, 200, 0, "square")
	line := makeSimpleLine(a, b, c)
	tr := NewTrain(0, line, 6, 1.2)
	if tr.IsLoop {
		t.Error("open three-station line must not be a loop")
	}
}

// --- Upcoming stops ---

func TestTrain_GetUpcomingStops_Linear_ForwardDirection(t *testing.T) {
	a := NewStation(0, 0, 0, "circle")
	b := NewStation(1, 100, 0, "triangle")
	c := NewStation(2, 200, 0, "square")
	line := makeSimpleLine(a, b, c)
	tr := NewTrain(0, line, 6, 1.2)
	tr.CurrentStationIndex = 0
	tr.Direction = 1

	stops := tr.GetUpcomingStops(a, false)
	if len(stops) == 0 {
		t.Fatal("expected upcoming stops for train at beginning of line")
	}
	found := func(target *Station) bool {
		for _, s := range stops {
			if s == target {
				return true
			}
		}
		return false
	}
	if !found(b) || !found(c) {
		t.Errorf("forward direction from A should include B and C, got %v", stops)
	}
}

func TestTrain_GetUpcomingStops_Loop_AllOtherStations(t *testing.T) {
	// In a loop A→B→C→A, from A all other unique stations should appear in upcoming stops
	a := NewStation(0, 0, 0, "circle")
	b := NewStation(1, 100, 0, "triangle")
	c := NewStation(2, 50, 100, "square")
	line := NewLine("#ff0000", 0)
	line.AddStation(a, -1, nil)
	line.AddStation(b, -1, nil)
	line.AddStation(c, -1, nil)
	line.AddStation(a, -1, nil) // close loop
	tr := NewTrain(0, line, 6, 1.2)
	tr.CurrentStationIndex = 0

	stops := tr.GetUpcomingStops(a, true)
	inStops := func(target *Station) bool {
		for _, s := range stops {
			if s == target {
				return true
			}
		}
		return false
	}
	if !inStops(b) || !inStops(c) {
		t.Errorf("loop from A should include B and C in upcoming stops, got %v", stops)
	}
}

func TestTrain_CheckLoopStatus_UpdatesOnStationChange(t *testing.T) {
	a := NewStation(0, 0, 0, "circle")
	b := NewStation(1, 100, 0, "triangle")
	line := makeSimpleLine(a, b)
	tr := NewTrain(0, line, 6, 1.2)

	// Add third station and close loop
	c := NewStation(2, 50, 100, "square")
	line.AddStation(c, -1, nil)
	line.AddStation(a, -1, nil)
	tr.CheckLoopStatus()

	if !tr.IsLoop {
		t.Error("CheckLoopStatus should detect newly closed loop")
	}
}
