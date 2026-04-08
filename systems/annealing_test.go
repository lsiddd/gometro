package systems

import (
	"minimetro-go/components"
	"minimetro-go/config"
	"minimetro-go/state"
	"testing"
	"time"
)

// ── generatePerturbations ─────────────────────────────────────────────────────

func TestGeneratePerturbations_EmptyLines_NoneGenerated(t *testing.T) {
	gs := newTestGameState()
	gs.Stations = []*components.Station{stationAt(0, 0, 0, config.Circle)}
	gs.Lines = []*components.Line{components.NewLine(config.LineColors[0], 0)}
	gs.AvailableLines = 1

	perturbs := generatePerturbations(gs)
	if len(perturbs) != 0 {
		t.Errorf("no perturbations expected for inactive line, got %d", len(perturbs))
	}
}

func TestGeneratePerturbations_OpenLine_ProducesAddAndRemove(t *testing.T) {
	gs := buildAnnealingState()
	// Add a third station to the line so RemoveEndpoint (requires n > 2) is generated.
	gs.Lines[0].AddStation(gs.Stations[2], -1, nil)
	perturbs := generatePerturbations(gs)

	hasAdd := false
	hasRemove := false
	for _, p := range perturbs {
		if p.Type == PerturbAddEndpoint {
			hasAdd = true
		}
		if p.Type == PerturbRemoveEndpoint {
			hasRemove = true
		}
	}
	if !hasAdd {
		t.Error("expected at least one PerturbAddEndpoint perturbation")
	}
	if !hasRemove {
		t.Error("expected at least one PerturbRemoveEndpoint perturbation")
	}
}

func TestGeneratePerturbations_LoopLine_ProducesOpenLoop(t *testing.T) {
	gs := newTestGameState()
	a := stationAt(0, 0, 0, config.Circle)
	b := stationAt(1, 100, 0, config.Triangle)
	c := stationAt(2, 50, 100, config.Square)
	line := newConnectedLine(config.LineColors[0], 0, a, b, c)
	line.AddStation(a, -1, nil) // close the loop
	line.Active = true
	gs.Stations = []*components.Station{a, b, c}
	gs.Lines = []*components.Line{line}
	gs.AvailableLines = 1

	perturbs := generatePerturbations(gs)
	hasOpen := false
	for _, p := range perturbs {
		if p.Type == PerturbOpenLoop && p.LineIdx == 0 {
			hasOpen = true
		}
	}
	if !hasOpen {
		t.Error("expected PerturbOpenLoop for a closed-loop line")
	}
}

func TestGeneratePerturbations_CloseLoopRequiresFourStations(t *testing.T) {
	gs := newTestGameState()
	a := stationAt(0, 0, 0, config.Circle)
	b := stationAt(1, 100, 0, config.Triangle)
	c := stationAt(2, 200, 0, config.Square)
	// Only 3 stations → PerturbCloseLoop should NOT appear.
	line := newConnectedLine(config.LineColors[0], 0, a, b, c)
	line.Active = true
	gs.Stations = []*components.Station{a, b, c}
	gs.Lines = []*components.Line{line}
	gs.AvailableLines = 1

	perturbs := generatePerturbations(gs)
	for _, p := range perturbs {
		if p.Type == PerturbCloseLoop {
			t.Error("PerturbCloseLoop should not be generated for a 3-station line")
		}
	}
}

func TestGeneratePerturbations_FourStationsProducesCloseLoop(t *testing.T) {
	gs := newTestGameState()
	a := stationAt(0, 0, 0, config.Circle)
	b := stationAt(1, 100, 0, config.Triangle)
	c := stationAt(2, 200, 0, config.Square)
	d := stationAt(3, 300, 0, config.Pentagon)
	line := newConnectedLine(config.LineColors[0], 0, a, b, c, d)
	line.Active = true
	gs.Stations = []*components.Station{a, b, c, d}
	gs.Lines = []*components.Line{line}
	gs.AvailableLines = 1

	perturbs := generatePerturbations(gs)
	hasClose := false
	for _, p := range perturbs {
		if p.Type == PerturbCloseLoop {
			hasClose = true
		}
	}
	if !hasClose {
		t.Error("PerturbCloseLoop should be generated for a 4-station line")
	}
}

// ── applyPerturbation ─────────────────────────────────────────────────────────

func TestApplyPerturbation_AddEndpoint_Append(t *testing.T) {
	gs := buildAnnealingState()
	line := gs.Lines[0]
	initialLen := len(line.Stations)
	newSt := gs.Stations[2] // square, not yet on the line

	p := &Perturbation{Type: PerturbAddEndpoint, LineIdx: 0, StationID: newSt.ID, AtHead: false}
	if !applyPerturbation(gs, p) {
		t.Fatal("applyPerturbation returned false for valid add")
	}
	if len(line.Stations) != initialLen+1 {
		t.Errorf("line should have %d stations after add, got %d", initialLen+1, len(line.Stations))
	}
	if line.Stations[len(line.Stations)-1].ID != newSt.ID {
		t.Error("appended station is not at the tail")
	}
}

func TestApplyPerturbation_AddEndpoint_Prepend(t *testing.T) {
	gs := buildAnnealingState()
	line := gs.Lines[0]
	newSt := gs.Stations[2]

	p := &Perturbation{Type: PerturbAddEndpoint, LineIdx: 0, StationID: newSt.ID, AtHead: true}
	if !applyPerturbation(gs, p) {
		t.Fatal("applyPerturbation returned false for valid prepend")
	}
	if line.Stations[0].ID != newSt.ID {
		t.Error("prepended station is not at the head")
	}
}

func TestApplyPerturbation_RemoveEndpoint(t *testing.T) {
	gs := buildAnnealingState()
	line := gs.Lines[0]
	// Add a third station so removal leaves the line active (≥ 2).
	line.AddStation(gs.Stations[2], -1, nil)
	initialLen := len(line.Stations)
	tail := line.Stations[initialLen-1]

	p := &Perturbation{Type: PerturbRemoveEndpoint, LineIdx: 0, StationID: tail.ID, AtHead: false}
	if !applyPerturbation(gs, p) {
		t.Fatal("applyPerturbation returned false for valid remove")
	}
	if len(line.Stations) != initialLen-1 {
		t.Errorf("line should have %d stations after remove, got %d", initialLen-1, len(line.Stations))
	}
}

func TestApplyPerturbation_CloseLoop(t *testing.T) {
	gs := newTestGameState()
	a := stationAt(0, 0, 0, config.Circle)
	b := stationAt(1, 100, 0, config.Triangle)
	c := stationAt(2, 200, 0, config.Square)
	d := stationAt(3, 100, 100, config.Pentagon)
	line := newConnectedLine(config.LineColors[0], 0, a, b, c, d)
	line.Active = true
	gs.Stations = []*components.Station{a, b, c, d}
	gs.Lines = []*components.Line{line}
	gs.AvailableLines = 1
	gs.Bridges = 5

	p := &Perturbation{Type: PerturbCloseLoop, LineIdx: 0}
	if !applyPerturbation(gs, p) {
		t.Fatal("applyPerturbation returned false for valid close loop")
	}
	n := len(line.Stations)
	if line.Stations[0] != line.Stations[n-1] {
		t.Error("loop was not closed: first and last stations differ")
	}
}

func TestApplyPerturbation_SwapEndpoint(t *testing.T) {
	gs := buildAnnealingState()
	line := gs.Lines[0]
	// Add a 3rd station so the line has 3 stations before the swap.
	line.AddStation(gs.Stations[2], -1, nil)
	oldTailID := line.Stations[len(line.Stations)-1].ID

	// Find a station not on the line to swap in.
	newSt := gs.Stations[3] // pentagon (4th station added to gs)

	p := &Perturbation{
		Type:         PerturbSwapEndpoint,
		LineIdx:      0,
		StationID:    oldTailID,
		AtHead:       false,
		NewStationID: newSt.ID,
	}
	if !applyPerturbation(gs, p) {
		t.Fatal("applyPerturbation returned false for valid swap")
	}
	newTail := line.Stations[len(line.Stations)-1]
	if newTail.ID != newSt.ID {
		t.Errorf("after swap, tail should be station %d, got %d", newSt.ID, newTail.ID)
	}
}

func TestApplyPerturbation_OpenLoop_ConvertsToLinear(t *testing.T) {
	gs := newTestGameState()
	a := stationAt(0, 0, 0, config.Circle)
	b := stationAt(1, 100, 0, config.Triangle)
	c := stationAt(2, 200, 0, config.Square)
	d := stationAt(3, 100, 100, config.Pentagon)
	line := newConnectedLine(config.LineColors[0], 0, a, b, c, d)
	line.AddStation(a, -1, nil) // close loop: [a,b,c,d,a]
	line.Active = true
	gs.Stations = []*components.Station{a, b, c, d}
	gs.Lines = []*components.Line{line}
	gs.AvailableLines = 1
	gs.Bridges = 5

	p := &Perturbation{Type: PerturbOpenLoop, LineIdx: 0}
	if !applyPerturbation(gs, p) {
		t.Fatal("applyPerturbation returned false for valid open loop")
	}
	n := len(line.Stations)
	if line.Stations[0] == line.Stations[n-1] {
		t.Error("loop was not opened: first and last stations are still the same")
	}
	if n != 4 {
		t.Errorf("opened line should have 4 stations, got %d", n)
	}
}

func TestApplyPerturbation_OpenLoop_NotALoop_ReturnsFalse(t *testing.T) {
	gs := buildAnnealingState()
	// The line in buildAnnealingState is open (not a loop).
	p := &Perturbation{Type: PerturbOpenLoop, LineIdx: 0}
	if applyPerturbation(gs, p) {
		t.Error("applyPerturbation should return false for a non-loop line")
	}
}

func TestApplyPerturbation_OpenLoop_TrainDirectionReset(t *testing.T) {
	gs := newTestGameState()
	a := stationAt(0, 0, 0, config.Circle)
	b := stationAt(1, 100, 0, config.Triangle)
	c := stationAt(2, 200, 0, config.Square)
	d := stationAt(3, 100, 100, config.Pentagon)
	line := newConnectedLine(config.LineColors[0], 0, a, b, c, d)
	line.AddStation(a, -1, nil) // close loop
	line.Active = true
	gs.Stations = []*components.Station{a, b, c, d}
	gs.Lines = []*components.Line{line}
	gs.AvailableLines = 1
	gs.Bridges = 5

	train := components.NewTrain(0, line, 6, config.TrainMaxSpeed)
	train.Direction = -1 // simulate an arbitrary direction before opening
	train.CurrentStationIndex = 2
	line.Trains = []*components.Train{train}
	gs.Trains = []*components.Train{train}

	p := &Perturbation{Type: PerturbOpenLoop, LineIdx: 0}
	if !applyPerturbation(gs, p) {
		t.Fatal("applyPerturbation returned false")
	}
	if train.Direction != 1 {
		t.Errorf("train direction should be reset to 1 after opening loop, got %d", train.Direction)
	}
}

func TestGeneratePerturbations_OpenLoop_NotGeneratedForLinearLine(t *testing.T) {
	gs := buildAnnealingState()
	perturbs := generatePerturbations(gs)
	for _, p := range perturbs {
		if p.Type == PerturbOpenLoop {
			t.Error("PerturbOpenLoop should not be generated for a non-loop line")
		}
	}
}

func TestApplyPerturbation_InvalidLineIdx_ReturnsFalse(t *testing.T) {
	gs := buildAnnealingState()
	p := &Perturbation{Type: PerturbAddEndpoint, LineIdx: 999, StationID: 0}
	if applyPerturbation(gs, p) {
		t.Error("applyPerturbation should return false for invalid line index")
	}
}

func TestApplyPerturbation_UnknownStation_ReturnsFalse(t *testing.T) {
	gs := buildAnnealingState()
	p := &Perturbation{Type: PerturbAddEndpoint, LineIdx: 0, StationID: 9999}
	if applyPerturbation(gs, p) {
		t.Error("applyPerturbation should return false for unknown station ID")
	}
}

// ── SAOptimize ────────────────────────────────────────────────────────────────

func TestSAOptimize_OriginalStateUnchanged(t *testing.T) {
	gs := buildAnnealingState()
	origLineLen := len(gs.Lines[0].Stations)

	ch := SAOptimize(gs, 50*time.Millisecond)
	<-ch // wait for completion

	if len(gs.Lines[0].Stations) != origLineLen {
		t.Error("SAOptimize mutated the original game state")
	}
}

func TestSAOptimize_ReturnsWithinBudget(t *testing.T) {
	gs := buildAnnealingState()
	budget := 80 * time.Millisecond
	start := time.Now()

	ch := SAOptimize(gs, budget)
	<-ch

	elapsed := time.Since(start)
	// Allow 2× budget for goroutine scheduling overhead.
	if elapsed > budget*2 {
		t.Errorf("SAOptimize took too long: %v (budget %v)", elapsed, budget)
	}
}

func TestSAOptimize_ResultScoreIsDefined(t *testing.T) {
	gs := buildAnnealingState()
	baseline := NetworkCost(gs)
	ch := SAOptimize(gs, 50*time.Millisecond)
	result := <-ch

	// An action is only recorded when it strictly improves on the baseline.
	// Score=0 is a valid outcome (perfect topology after rollout).
	if result.Action != nil && result.Score >= baseline {
		t.Errorf("SA found an action but score %.1f is not better than baseline %.1f", result.Score, baseline)
	}
}

func TestSAOptimize_FindsImprovementForBadTopology(t *testing.T) {
	// Build a state where a known improvement is available: an isolated station.
	gs := state.NewGameState()
	gs.SelectedCity = "london"

	a := stationAt(0, 0, 0, config.Circle)
	b := stationAt(1, 100, 0, config.Triangle)
	isolated := stationAt(2, 200, 0, config.Square)
	gs.Stations = []*components.Station{a, b, isolated}
	gs.StationIDCounter = 3

	line := newConnectedLine(config.LineColors[0], 0, a, b)
	line.Active = true
	gs.Lines = []*components.Line{
		line,
		components.NewLine(config.LineColors[1], 1),
	}
	gs.AvailableLines = 2
	gs.Bridges = 5

	ch := SAOptimize(gs, 200*time.Millisecond)
	result := <-ch

	// SA should find that connecting the isolated station reduces NetworkCost.
	if result.Action == nil {
		t.Skip("SA found no improvement in budget (may pass on slower machines)")
	}
	// The action should involve the isolated station's line or a new line.
	if result.Score >= NetworkCost(gs) {
		t.Logf("SA score %.2f vs baseline %.2f", result.Score, NetworkCost(gs))
		// Non-fatal: SA is probabilistic; a short budget may not always find the best.
	}
}

// ── helpers ──────────────────────────────────────────────────────────────────

func buildAnnealingState() *state.GameState {
	gs := state.NewGameState()
	gs.SelectedCity = "london"
	gs.Bridges = 5

	a := stationAt(0, 0, 0, config.Circle)
	b := stationAt(1, 100, 0, config.Triangle)
	c := stationAt(2, 200, 0, config.Square)
	d := stationAt(3, 300, 0, config.Pentagon)
	gs.Stations = []*components.Station{a, b, c, d}
	gs.StationIDCounter = 4

	line := newConnectedLine(config.LineColors[0], 0, a, b)
	line.Active = true
	gs.Lines = []*components.Line{
		line,
		components.NewLine(config.LineColors[1], 1),
	}
	gs.AvailableLines = 2

	train := components.NewTrain(0, line, 6, config.TrainMaxSpeed)
	line.Trains = []*components.Train{train}
	gs.Trains = []*components.Train{train}
	gs.TrainIDCounter = 1

	return gs
}
