package systems

import (
	"math"
	"minimetro-go/components"
	"minimetro-go/config"
	"testing"
)

// ── AnglePenalty ─────────────────────────────────────────────────────────────

func TestAnglePenalty_NoInteriorNodes_Zero(t *testing.T) {
	a := components.NewStation(0, 0, 0, config.Circle)
	b := components.NewStation(1, 100, 0, config.Triangle)
	line := newConnectedLine(config.LineColors[0], 0, a, b)
	if got := AnglePenalty(line); got != 0 {
		t.Errorf("2-station line should have zero angle penalty, got %.2f", got)
	}
}

func TestAnglePenalty_StraightLine_Zero(t *testing.T) {
	// Collinear stations: the interior angle is 180° (cos = −1) — no penalty.
	a := components.NewStation(0, 0, 0, config.Circle)
	b := components.NewStation(1, 100, 0, config.Triangle)
	c := components.NewStation(2, 200, 0, config.Square)
	line := newConnectedLine(config.LineColors[0], 0, a, b, c)
	if got := AnglePenalty(line); got != 0 {
		t.Errorf("straight line should have zero angle penalty, got %.2f", got)
	}
}

func TestAnglePenalty_ObtuseAngle_Zero(t *testing.T) {
	// Obtuse angle (> 90°) → cos < 0 → no penalty.
	a := components.NewStation(0, 0, 0, config.Circle)
	b := components.NewStation(1, 100, 0, config.Triangle)
	c := components.NewStation(2, 110, 50, config.Square) // slight turn, obtuse
	line := newConnectedLine(config.LineColors[0], 0, a, b, c)
	if got := AnglePenalty(line); got != 0 {
		t.Errorf("obtuse turn should have zero angle penalty, got %.2f", got)
	}
}

func TestAnglePenalty_SharpUTurn_HighPenalty(t *testing.T) {
	// Sharp U-turn: A→B→A (cos = 1) should produce maximum penalty.
	a := components.NewStation(0, 0, 0, config.Circle)
	b := components.NewStation(1, 100, 0, config.Triangle)
	c := components.NewStation(2, 0, 0, config.Square) // back toward A
	line := newConnectedLine(config.LineColors[0], 0, a, b, c)
	got := AnglePenalty(line)
	// cos(180° reverse) ≈ 1 → penalty = 1 * 100 = 100
	if got < 90 {
		t.Errorf("sharp U-turn should yield high angle penalty, got %.2f", got)
	}
}

func TestAnglePenalty_RightAngle_SmallPenalty(t *testing.T) {
	// 90° turn: cos = 0 → no penalty (threshold exactly at 0).
	a := components.NewStation(0, 0, 0, config.Circle)
	b := components.NewStation(1, 100, 0, config.Triangle)
	c := components.NewStation(2, 100, 100, config.Square) // 90° turn
	line := newConnectedLine(config.LineColors[0], 0, a, b, c)
	got := AnglePenalty(line)
	if got > 1e-6 {
		t.Errorf("90° turn should have near-zero angle penalty, got %.6f", got)
	}
}

func TestAnglePenalty_SharpAngleHigherThanObtuse(t *testing.T) {
	a := components.NewStation(0, 0, 0, config.Circle)
	b := components.NewStation(1, 100, 0, config.Triangle)

	// Obtuse: slight outward turn
	cObtuse := components.NewStation(2, 110, 60, config.Square)
	lineObtuse := newConnectedLine(config.LineColors[0], 0, a, b, cObtuse)

	// Sharp: tight inward turn
	cSharp := components.NewStation(3, 50, 5, config.Square)
	lineSharp := newConnectedLine(config.LineColors[1], 1, a, b, cSharp)

	if AnglePenalty(lineSharp) <= AnglePenalty(lineObtuse) {
		t.Error("sharp angle should yield higher penalty than obtuse angle")
	}
}

// ── HygienePenalty ───────────────────────────────────────────────────────────

func TestHygienePenalty_AlternatingTypes_Zero(t *testing.T) {
	a := components.NewStation(0, 0, 0, config.Circle)
	b := components.NewStation(1, 100, 0, config.Triangle)
	c := components.NewStation(2, 200, 0, config.Circle)
	line := newConnectedLine(config.LineColors[0], 0, a, b, c)
	if got := HygienePenalty(line); got != 0 {
		t.Errorf("alternating types should give zero hygiene penalty, got %.2f", got)
	}
}

func TestHygienePenalty_TwoSameTypes_Penalised(t *testing.T) {
	a := components.NewStation(0, 0, 0, config.Circle)
	b := components.NewStation(1, 100, 0, config.Circle)
	line := newConnectedLine(config.LineColors[0], 0, a, b)
	if got := HygienePenalty(line); got <= 0 {
		t.Errorf("two consecutive same-type stations should yield positive penalty, got %.2f", got)
	}
}

func TestHygienePenalty_RunOfThreeWorsesThanRunOfTwo(t *testing.T) {
	a := components.NewStation(0, 0, 0, config.Circle)
	b := components.NewStation(1, 100, 0, config.Circle)

	twoLine := newConnectedLine(config.LineColors[0], 0, a, b)

	c := components.NewStation(2, 200, 0, config.Circle)
	threeLine := newConnectedLine(config.LineColors[1], 1, a, b, c)

	if HygienePenalty(threeLine) <= HygienePenalty(twoLine) {
		t.Error("run of 3 identical types should cost more than run of 2")
	}
}

func TestHygienePenalty_SingleStation_Zero(t *testing.T) {
	a := components.NewStation(0, 0, 0, config.Circle)
	line := components.NewLine(config.LineColors[0], 0)
	line.AddStation(a, -1, nil)
	if got := HygienePenalty(line); got != 0 {
		t.Errorf("single-station line should have zero hygiene penalty, got %.2f", got)
	}
}

// ── LoopBonus ────────────────────────────────────────────────────────────────

func TestLoopBonus_OpenLine_Zero(t *testing.T) {
	a := components.NewStation(0, 0, 0, config.Circle)
	b := components.NewStation(1, 100, 0, config.Triangle)
	c := components.NewStation(2, 200, 0, config.Square)
	line := newConnectedLine(config.LineColors[0], 0, a, b, c)
	if got := LoopBonus(line); got != 0 {
		t.Errorf("open line should have zero loop bonus, got %.2f", got)
	}
}

func TestLoopBonus_ClosedLoop_PositiveBonus(t *testing.T) {
	a := components.NewStation(0, 0, 0, config.Circle)
	b := components.NewStation(1, 100, 0, config.Triangle)
	c := components.NewStation(2, 50, 100, config.Square)
	line := newConnectedLine(config.LineColors[0], 0, a, b, c)
	line.AddStation(a, -1, nil) // close the loop
	if got := LoopBonus(line); got <= 0 {
		t.Errorf("closed loop should have positive bonus, got %.2f", got)
	}
}

func TestLoopBonus_ShortLoopHigherThanLongLoop(t *testing.T) {
	// A 3-station loop (triangle) should receive more bonus than a 6-station loop
	// because the short loop has a smaller cycle time — exactly what we optimise for.
	a := components.NewStation(0, 0, 0, config.Circle)
	b := components.NewStation(1, 100, 0, config.Triangle)
	c := components.NewStation(2, 50, 100, config.Square)
	shortLine := newConnectedLine(config.LineColors[0], 0, a, b, c)
	shortLine.AddStation(a, -1, nil)

	d := components.NewStation(3, 200, 0, config.Pentagon)
	e := components.NewStation(4, 200, 100, config.Circle)
	f := components.NewStation(5, 100, 150, config.Triangle)
	longLine := newConnectedLine(config.LineColors[1], 1, a, b, c, d, e, f)
	longLine.AddStation(a, -1, nil)

	if LoopBonus(shortLine) <= LoopBonus(longLine) {
		t.Errorf("short loop (%.2f) should have higher bonus than long loop (%.2f)",
			LoopBonus(shortLine), LoopBonus(longLine))
	}
}

// ── StationCountPenalty ──────────────────────────────────────────────────────

func TestStationCountPenalty_BelowCap_Zero(t *testing.T) {
	a := components.NewStation(0, 0, 0, config.Circle)
	b := components.NewStation(1, 100, 0, config.Triangle)
	line := newConnectedLine(config.LineColors[0], 0, a, b)
	if got := StationCountPenalty(line); got != 0 {
		t.Errorf("2-station line should have zero station-count penalty, got %.2f", got)
	}
}

func TestStationCountPenalty_AtCap_Zero(t *testing.T) {
	stations := make([]*components.Station, stationCountSoftCap)
	for i := range stations {
		stations[i] = components.NewStation(i, float64(i*100), 0, config.Circle)
	}
	args := make([]*components.Station, len(stations))
	copy(args, stations)
	line := newConnectedLine(config.LineColors[0], 0, args...)
	if got := StationCountPenalty(line); got != 0 {
		t.Errorf("line at soft cap should have zero penalty, got %.2f", got)
	}
}

func TestStationCountPenalty_AboveCap_Positive(t *testing.T) {
	n := stationCountSoftCap + 2
	stations := make([]*components.Station, n)
	for i := range stations {
		stations[i] = components.NewStation(i, float64(i*100), 0, config.Circle)
	}
	line := newConnectedLine(config.LineColors[0], 0, stations...)
	got := StationCountPenalty(line)
	want := float64(2) * stationCountPenaltyW
	if got != want {
		t.Errorf("station-count penalty for %d stations over cap: want %.2f, got %.2f", 2, want, got)
	}
}

func TestStationCountPenalty_LongerLineMorePenalty(t *testing.T) {
	makeN := func(n int) *components.Line {
		sts := make([]*components.Station, n)
		for i := range sts {
			sts[i] = components.NewStation(i, float64(i*100), 0, config.Circle)
		}
		return newConnectedLine(config.LineColors[0], 0, sts...)
	}
	short := makeN(stationCountSoftCap + 1)
	long := makeN(stationCountSoftCap + 3)
	if StationCountPenalty(long) <= StationCountPenalty(short) {
		t.Error("longer line should have larger station-count penalty")
	}
}

// ── NetworkCrossingPenalty ───────────────────────────────────────────────────

func TestNetworkCrossingPenalty_NoLines_Zero(t *testing.T) {
	gs := newTestGameState()
	if got := NetworkCrossingPenalty(gs); got != 0 {
		t.Errorf("empty network should have zero crossing penalty, got %.2f", got)
	}
}

func TestNetworkCrossingPenalty_ParallelLines_Zero(t *testing.T) {
	gs := newTestGameState()
	// Two parallel horizontal lines — no crossings.
	a1 := stationAt(0, 0, 0, config.Circle)
	b1 := stationAt(1, 200, 0, config.Triangle)
	a2 := stationAt(2, 0, 100, config.Square)
	b2 := stationAt(3, 200, 100, config.Pentagon)

	l1 := newConnectedLine(config.LineColors[0], 0, a1, b1)
	l1.Active = true
	l2 := newConnectedLine(config.LineColors[1], 1, a2, b2)
	l2.Active = true
	gs.Lines = []*components.Line{l1, l2}
	gs.AvailableLines = 2
	gs.Stations = []*components.Station{a1, b1, a2, b2}

	if got := NetworkCrossingPenalty(gs); got != 0 {
		t.Errorf("parallel lines should have zero crossing penalty, got %.2f", got)
	}
}

func TestNetworkCrossingPenalty_CrossingLines_Positive(t *testing.T) {
	gs := newTestGameState()
	// Two lines that cross in the middle (X pattern).
	a := stationAt(0, 0, 0, config.Circle)
	b := stationAt(1, 200, 200, config.Triangle)
	c := stationAt(2, 200, 0, config.Square)
	d := stationAt(3, 0, 200, config.Pentagon)

	l1 := newConnectedLine(config.LineColors[0], 0, a, b) // diagonal /
	l1.Active = true
	l2 := newConnectedLine(config.LineColors[1], 1, c, d) // diagonal \
	l2.Active = true
	gs.Lines = []*components.Line{l1, l2}
	gs.AvailableLines = 2
	gs.Stations = []*components.Station{a, b, c, d}

	if got := NetworkCrossingPenalty(gs); got <= 0 {
		t.Errorf("crossing lines should have positive penalty, got %.2f", got)
	}
}

// ── LineCost ─────────────────────────────────────────────────────────────────

func TestLineCost_InactiveLine_Zero(t *testing.T) {
	line := components.NewLine(config.LineColors[0], 0) // no stations → inactive
	if got := LineCost(line); got != 0 {
		t.Errorf("inactive line should have zero cost, got %.2f", got)
	}
}

func TestLineCost_LoopCheaperThanOpenWithSharpAngles(t *testing.T) {
	a := components.NewStation(0, 0, 0, config.Circle)
	b := components.NewStation(1, 100, 0, config.Triangle)
	c := components.NewStation(2, 50, 50, config.Square)
	d := components.NewStation(3, 0, 50, config.Pentagon)

	// Open line with a sharp interior angle.
	openLine := newConnectedLine(config.LineColors[0], 0, a, b, c, d)
	// Loop version of the same line.
	loopLine := newConnectedLine(config.LineColors[1], 1, a, b, c, d)
	loopLine.AddStation(a, -1, nil)

	if LineCost(loopLine) >= LineCost(openLine) {
		t.Error("loop line should be no more expensive than comparable open line (loop bonus not applied)")
	}
}

// ── NetworkCost ──────────────────────────────────────────────────────────────

func TestNetworkCost_IsolatedStationsIncreaseCost(t *testing.T) {
	gs := newTestGameState()
	circle := stationAt(0, 0, 0, config.Circle)
	triangle := stationAt(1, 100, 0, config.Triangle)
	isolated := stationAt(2, 500, 500, config.Square)
	gs.Stations = []*components.Station{circle, triangle, isolated}

	line := newConnectedLine(config.LineColors[0], 0, circle, triangle)
	line.Active = true
	gs.Lines = []*components.Line{line}
	gs.AvailableLines = 1

	costWith := NetworkCost(gs)

	// Remove the isolated station and re-measure.
	gs.Stations = []*components.Station{circle, triangle}
	costWithout := NetworkCost(gs)

	if costWith <= costWithout {
		t.Errorf("isolated station should increase network cost; with=%.2f without=%.2f",
			costWith, costWithout)
	}
}

func TestNetworkCost_OvercrowdingIncreaseCost(t *testing.T) {
	gs := newTestGameState()
	s := stationAt(0, 0, 0, config.Circle)
	gs.Stations = []*components.Station{s}
	gs.AvailableLines = 0

	baseCost := NetworkCost(gs)
	s.OvercrowdProgress = float64(config.OvercrowdTime) * 0.5
	overcrowdCost := NetworkCost(gs)

	if overcrowdCost <= baseCost {
		t.Errorf("overcrowding should increase network cost; base=%.2f overcrowded=%.2f",
			baseCost, overcrowdCost)
	}
}

func TestNetworkCost_GoodTopologyLowerThanBad(t *testing.T) {
	// A straight alternating-type line should cost less than a same-type line
	// with a sharp U-turn.
	goodGs := newTestGameState()
	a := stationAt(0, 0, 0, config.Circle)
	b := stationAt(1, 100, 0, config.Triangle)
	c := stationAt(2, 200, 0, config.Circle)
	goodLine := newConnectedLine(config.LineColors[0], 0, a, b, c)
	goodLine.Active = true
	goodGs.Stations = []*components.Station{a, b, c}
	goodGs.Lines = []*components.Line{goodLine}
	goodGs.AvailableLines = 1

	badGs := newTestGameState()
	x := stationAt(0, 0, 0, config.Circle)
	y := stationAt(1, 100, 0, config.Circle)    // same type
	z := stationAt(2, 0, 0, config.Circle)      // sharp U-turn, same type
	badLine := newConnectedLine(config.LineColors[0], 0, x, y, z)
	badLine.Active = true
	badGs.Stations = []*components.Station{x, y, z}
	badGs.Lines = []*components.Line{badLine}
	badGs.AvailableLines = 1

	good := NetworkCost(goodGs)
	bad := NetworkCost(badGs)

	if good >= bad {
		t.Errorf("good topology (%.2f) should cost less than bad topology (%.2f)", good, bad)
	}
}

func TestAnglePenalty_LoopClosureSkipped(t *testing.T) {
	// In a loop A→B→C→A, the repeated A at the end should not generate a
	// false angle between C and A (the repeated entry) and then A again.
	a := components.NewStation(0, 0, 0, config.Circle)
	b := components.NewStation(1, 100, 0, config.Triangle)
	c := components.NewStation(2, 50, 100, config.Square)
	line := newConnectedLine(config.LineColors[0], 0, a, b, c)
	line.AddStation(a, -1, nil)

	penalty := AnglePenalty(line)
	// The penalty at node c considers B→C→A (the original A, not the closure).
	// This is correct and expected — just verify it does not panic and is finite.
	if math.IsNaN(penalty) || math.IsInf(penalty, 0) {
		t.Errorf("loop line should produce finite angle penalty, got %f", penalty)
	}
}
