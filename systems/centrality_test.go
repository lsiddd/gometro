package systems

import (
	"minimetro-go/components"
	"minimetro-go/config"
	"minimetro-go/state"
	"testing"
)

// buildCentralityState builds a graph and returns a GameState + GraphManager
// ready for BetweennessCentrality.
func buildCentralityState(stations []*components.Station, lines ...*components.Line) (*state.GameState, *GraphManager) {
	gs := newTestGameState()
	gs.Stations = stations
	gs.Lines = lines
	gs.AvailableLines = len(lines)
	gm := NewGraphManager()
	return gs, gm
}

// ── Degenerate cases ─────────────────────────────────────────────────────────

func TestBetweennessCentrality_LessThanThreeNodes_AllZero(t *testing.T) {
	a := stationAt(0, 0, 0, config.Circle)
	b := stationAt(1, 100, 0, config.Triangle)
	line := newConnectedLine(config.LineColors[0], 0, a, b)
	line.Active = true

	gs, gm := buildCentralityState([]*components.Station{a, b}, line)
	c := BetweennessCentrality(gs, gm)

	for _, s := range gs.Stations {
		if c[s] != 0 {
			t.Errorf("2-node graph: expected 0 centrality for all, got %f for %s", c[s], s.Type)
		}
	}
}

func TestBetweennessCentrality_EmptyGraph_AllZero(t *testing.T) {
	gs := newTestGameState()
	gs.Stations = []*components.Station{}
	gm := NewGraphManager()
	c := BetweennessCentrality(gs, gm)
	if len(c) != 0 {
		t.Errorf("empty graph should return empty map, got %d entries", len(c))
	}
}

// ── Linear chain ─────────────────────────────────────────────────────────────

// For a path A–B–C, B lies on all paths between A and C.
// Its centrality should be strictly higher than A's and C's (which are endpoints).
func TestBetweennessCentrality_LinearChain_MiddleHighest(t *testing.T) {
	a := stationAt(0, 0, 0, config.Circle)
	b := stationAt(1, 100, 0, config.Triangle)
	c := stationAt(2, 200, 0, config.Square)
	line := newConnectedLine(config.LineColors[0], 0, a, b, c)
	line.Active = true

	gs, gm := buildCentralityState([]*components.Station{a, b, c}, line)
	cent := BetweennessCentrality(gs, gm)

	if cent[b] <= cent[a] || cent[b] <= cent[c] {
		t.Errorf("middle node B (%.4f) should have higher centrality than endpoints A (%.4f) and C (%.4f)",
			cent[b], cent[a], cent[c])
	}
}

// ── Star topology ─────────────────────────────────────────────────────────────

// Hub-and-spoke: B connects A–B and B–C and B–D on separate lines.
// B is a transfer hub — must have the highest centrality.
func TestBetweennessCentrality_StarTopology_HubHighest(t *testing.T) {
	hub := stationAt(0, 200, 200, config.Circle)
	a := stationAt(1, 0, 200, config.Triangle)
	b := stationAt(2, 400, 200, config.Square)
	c := stationAt(3, 200, 0, config.Pentagon)

	line1 := newConnectedLine(config.LineColors[0], 0, a, hub)
	line1.Active = true
	line2 := newConnectedLine(config.LineColors[1], 1, hub, b)
	line2.Active = true
	line3 := newConnectedLine(config.LineColors[2], 2, hub, c)
	line3.Active = true

	gs, gm := buildCentralityState(
		[]*components.Station{hub, a, b, c},
		line1, line2, line3,
	)
	cent := BetweennessCentrality(gs, gm)

	for _, leaf := range []*components.Station{a, b, c} {
		if cent[hub] <= cent[leaf] {
			t.Errorf("hub centrality (%.4f) should exceed leaf centrality (%.4f)", cent[hub], cent[leaf])
		}
	}
}

// ── Symmetric path ────────────────────────────────────────────────────────────

// On a symmetric 4-node path A–B–C–D, B and C should have equal centrality
// and both should exceed A and D.
func TestBetweennessCentrality_SymmetricPath_InnerNodesEqual(t *testing.T) {
	a := stationAt(0, 0, 0, config.Circle)
	b := stationAt(1, 100, 0, config.Triangle)
	c := stationAt(2, 200, 0, config.Square)
	d := stationAt(3, 300, 0, config.Pentagon)
	line := newConnectedLine(config.LineColors[0], 0, a, b, c, d)
	line.Active = true

	gs, gm := buildCentralityState([]*components.Station{a, b, c, d}, line)
	cent := BetweennessCentrality(gs, gm)

	const eps = 1e-9
	if abs(cent[b]-cent[c]) > eps {
		t.Errorf("inner nodes B (%.6f) and C (%.6f) should have equal centrality on symmetric path",
			cent[b], cent[c])
	}
	if cent[b] <= cent[a] || cent[b] <= cent[d] {
		t.Errorf("inner node B (%.4f) should have higher centrality than endpoints A (%.4f) / D (%.4f)",
			cent[b], cent[a], cent[d])
	}
}

// ── Normalisation ─────────────────────────────────────────────────────────────

func TestBetweennessCentrality_NormalisedRange(t *testing.T) {
	a := stationAt(0, 0, 0, config.Circle)
	b := stationAt(1, 100, 0, config.Triangle)
	c := stationAt(2, 200, 0, config.Square)
	d := stationAt(3, 300, 0, config.Pentagon)
	line := newConnectedLine(config.LineColors[0], 0, a, b, c, d)
	line.Active = true

	gs, gm := buildCentralityState([]*components.Station{a, b, c, d}, line)
	cent := BetweennessCentrality(gs, gm)

	for _, s := range gs.Stations {
		v := cent[s]
		if v < 0 || v > 1.0+1e-9 {
			t.Errorf("centrality for station %d = %.6f is outside [0,1]", s.ID, v)
		}
	}
}

// ── Disconnected nodes ────────────────────────────────────────────────────────

func TestBetweennessCentrality_IsolatedNode_Zero(t *testing.T) {
	// Node D is isolated (not on any line). It should have zero centrality.
	a := stationAt(0, 0, 0, config.Circle)
	b := stationAt(1, 100, 0, config.Triangle)
	c := stationAt(2, 200, 0, config.Square)
	isolated := stationAt(3, 999, 999, config.Pentagon)

	line := newConnectedLine(config.LineColors[0], 0, a, b, c)
	line.Active = true

	gs, gm := buildCentralityState([]*components.Station{a, b, c, isolated}, line)
	cent := BetweennessCentrality(gs, gm)

	if cent[isolated] != 0 {
		t.Errorf("isolated node should have 0 centrality, got %.6f", cent[isolated])
	}
}

// ── helpers ──────────────────────────────────────────────────────────────────

func abs(x float64) float64 {
	if x < 0 {
		return -x
	}
	return x
}
