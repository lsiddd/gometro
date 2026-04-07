package components

import (
	"math"
	"testing"
)

// --- AddStation ---

func TestLine_AddStation_FirstStation(t *testing.T) {
	l := NewLine("#ff0000", 0)
	s := NewStation(0, 0, 0, "circle")
	if !l.AddStation(s, -1, nil) {
		t.Error("AddStation should succeed for first station")
	}
	if len(l.Stations) != 1 {
		t.Errorf("expected 1 station, got %d", len(l.Stations))
	}
}

func TestLine_ActiveAfterTwoStations(t *testing.T) {
	l := NewLine("#ff0000", 0)
	a := NewStation(0, 0, 0, "circle")
	b := NewStation(1, 100, 0, "triangle")
	l.AddStation(a, -1, nil)
	if l.Active {
		t.Error("line should not be active with only one station")
	}
	l.AddStation(b, -1, nil)
	if !l.Active {
		t.Error("line must be active once it has two stations")
	}
}

func TestLine_AddStation_NoDuplicates(t *testing.T) {
	// Adding a non-terminal station that is already on the line must be rejected.
	// (Closing a loop by reconnecting the first station is handled separately.)
	l := NewLine("#ff0000", 0)
	a := NewStation(0, 0, 0, "circle")
	b := NewStation(1, 100, 0, "triangle")
	c := NewStation(2, 200, 0, "square")
	l.AddStation(a, -1, nil)
	l.AddStation(b, -1, nil)
	l.AddStation(c, -1, nil)
	// B is already on the line (not a loop-closing operation)
	added := l.AddStation(b, -1, nil)
	if added {
		t.Error("AddStation should reject a mid-line station already present")
	}
	if len(l.Stations) != 3 {
		t.Errorf("station count should remain 3, got %d", len(l.Stations))
	}
}

func TestLine_AddStation_LoopClosure(t *testing.T) {
	// Spec: players can close a line into a loop by connecting the last station back to the first
	l := NewLine("#ff0000", 0)
	a := NewStation(0, 0, 0, "circle")
	b := NewStation(1, 100, 0, "triangle")
	c := NewStation(2, 50, 100, "square")
	l.AddStation(a, -1, nil)
	l.AddStation(b, -1, nil)
	l.AddStation(c, -1, nil)
	ok := l.AddStation(a, -1, nil) // close loop
	if !ok {
		t.Error("closing a loop by re-adding first station should succeed")
	}
	if l.Stations[0] != l.Stations[len(l.Stations)-1] {
		t.Error("loop must have first == last station in the stations slice")
	}
}

func TestLine_AddStation_InsertAtIndex(t *testing.T) {
	l := NewLine("#ff0000", 0)
	a := NewStation(0, 0, 0, "circle")
	b := NewStation(1, 200, 0, "square")
	l.AddStation(a, -1, nil)
	l.AddStation(b, -1, nil)

	mid := NewStation(2, 100, 0, "triangle")
	l.AddStation(mid, 1, nil) // insert between a and b
	if l.Stations[1] != mid {
		t.Errorf("inserted station should be at index 1, got %v", l.Stations[1])
	}
}

func TestLine_AddStation_PrependAtIndex0(t *testing.T) {
	l := NewLine("#ff0000", 0)
	a := NewStation(0, 100, 0, "triangle")
	l.AddStation(a, -1, nil)
	b := NewStation(1, 0, 0, "circle")
	ok := l.AddStation(b, 0, nil)
	if !ok {
		t.Error("prepending a new station at index 0 should succeed")
	}
	if l.Stations[0] != b {
		t.Errorf("prepended station should be first, got %v", l.Stations[0])
	}
}

func TestLine_AddStation_PrependSameStationRejected(t *testing.T) {
	l := NewLine("#ff0000", 0)
	a := NewStation(0, 100, 0, "circle")
	l.AddStation(a, -1, nil)
	ok := l.AddStation(a, 0, nil) // same station at head
	if ok {
		t.Error("prepending the same station that is already at index 0 should fail")
	}
}

func TestLine_ClearLine(t *testing.T) {
	l := NewLine("#ff0000", 0)
	a := NewStation(0, 0, 0, "circle")
	b := NewStation(1, 100, 0, "triangle")
	l.AddStation(a, -1, nil)
	l.AddStation(b, -1, nil)
	l.ClearLine(nil)
	if l.Active {
		t.Error("cleared line should not be active")
	}
	if len(l.Stations) != 0 {
		t.Errorf("cleared line should have 0 stations, got %d", len(l.Stations))
	}
}

func TestLine_HasSegment(t *testing.T) {
	l := NewLine("#ff0000", 0)
	a := NewStation(0, 0, 0, "circle")
	b := NewStation(1, 100, 0, "triangle")
	c := NewStation(2, 200, 0, "square")
	l.AddStation(a, -1, nil)
	l.AddStation(b, -1, nil)
	l.AddStation(c, -1, nil)

	if !l.HasSegment(a, b) {
		t.Error("HasSegment should find A-B")
	}
	if !l.HasSegment(b, a) {
		t.Error("HasSegment should be bidirectional: B-A")
	}
	if l.HasSegment(a, c) {
		t.Error("HasSegment should not find A-C (not adjacent)")
	}
}

func TestLine_HasSegment_LoopWraparound(t *testing.T) {
	l := NewLine("#ff0000", 0)
	a := NewStation(0, 0, 0, "circle")
	b := NewStation(1, 100, 0, "triangle")
	c := NewStation(2, 50, 100, "square")
	l.AddStation(a, -1, nil)
	l.AddStation(b, -1, nil)
	l.AddStation(c, -1, nil)
	l.AddStation(a, -1, nil) // close loop: C→A segment exists
	if !l.HasSegment(c, a) {
		t.Error("HasSegment should find the loop-closing C-A segment")
	}
}

// --- ComputeMetroWaypoints (45° routing) ---

func TestComputeMetroWaypoints_Horizontal(t *testing.T) {
	// Purely horizontal: ady = 0 < 2 → direct two-point path
	pts := ComputeMetroWaypoints(0, 0, 100, 0)
	if len(pts) != 2 {
		t.Errorf("horizontal path: expected 2 waypoints, got %d", len(pts))
	}
}

func TestComputeMetroWaypoints_Vertical(t *testing.T) {
	pts := ComputeMetroWaypoints(0, 0, 0, 100)
	if len(pts) != 2 {
		t.Errorf("vertical path: expected 2 waypoints, got %d", len(pts))
	}
}

func TestComputeMetroWaypoints_PureDiagonal(t *testing.T) {
	// |adx - ady| < 2 → direct two-point path at 45°
	pts := ComputeMetroWaypoints(0, 0, 100, 100)
	if len(pts) != 2 {
		t.Errorf("45° diagonal: expected 2 waypoints, got %d", len(pts))
	}
}

func TestComputeMetroWaypoints_MixedAngle_ElbowPath(t *testing.T) {
	// dx=100, dy=50 → adx=100, ady=50, |100-50|=50 → 3-point elbow
	// Spec: lines snap to 0°/45°/90° creating L-shaped paths
	pts := ComputeMetroWaypoints(0, 0, 100, 50)
	if len(pts) != 3 {
		t.Errorf("mixed angle: expected 3-point elbow path, got %d points", len(pts))
	}
	// Elbow should be at (50, 50): diagonal component = min(100,50) = 50
	elbow := pts[1]
	if math.Abs(elbow[0]-50) > 0.01 || math.Abs(elbow[1]-50) > 0.01 {
		t.Errorf("elbow at wrong position: want (50,50), got (%.2f,%.2f)", elbow[0], elbow[1])
	}
}

func TestComputeMetroWaypoints_MixedAngle_NegativeDirection(t *testing.T) {
	// dx=-100, dy=-50 → elbow at (-50, -50)
	pts := ComputeMetroWaypoints(0, 0, -100, -50)
	if len(pts) != 3 {
		t.Errorf("negative mixed angle: expected 3-point elbow, got %d", len(pts))
	}
	elbow := pts[1]
	if math.Abs(elbow[0]-(-50)) > 0.01 || math.Abs(elbow[1]-(-50)) > 0.01 {
		t.Errorf("elbow at wrong position: want (-50,-50), got (%.2f,%.2f)", elbow[0], elbow[1])
	}
}

func TestComputeMetroWaypoints_StartEqualsEnd(t *testing.T) {
	// Zero-length segment should not panic
	pts := ComputeMetroWaypoints(50, 50, 50, 50)
	if len(pts) == 0 {
		t.Error("zero-length segment should return at least 2 waypoints")
	}
}

// --- OffsetPath ---

func TestOffsetPath_TwoPointPath(t *testing.T) {
	pts := [][2]float64{{0, 0}, {100, 0}}
	offset := 5.0
	result := OffsetPath(pts, offset)
	if len(result) != 2 {
		t.Errorf("offset of 2-point path should remain 2 points, got %d", len(result))
	}
	// Horizontal segment offset upward: y should shift by +offset (perp is (0,-1) flipped)
	// PerpUnit for (100,0) = (0/100, 100/100) but PerpUnit returns -dy/d, dx/d = (0, 1)
	// So y shift = +5
	if math.Abs(result[0][1]-5.0) > 0.01 {
		t.Errorf("offset y: want 5.0, got %.4f", result[0][1])
	}
}
