package systems

import (
	"math"
	"math/rand"
	"minimetro-go/components"
	"minimetro-go/state"
	"time"
)

// PerturbType identifies the topology change a Perturbation represents.
type PerturbType int

const (
	PerturbAddEndpoint    PerturbType = iota // extend a line at one end
	PerturbRemoveEndpoint                    // shorten a line from one end
	PerturbCloseLoop                         // append first station to close a loop
	PerturbSwapEndpoint                      // remove one endpoint and append another
	PerturbInsertIntoLoop                    // insert a station into an interior segment of a loop
	PerturbOpenLoop                          // remove loop closure, restoring bidirectional bouncing
)

// Perturbation is a topology action that can be applied to a copy of the game
// state and evaluated via a forward rollout. Station references use IDs rather
// than pointers so the action remains valid after a deep copy.
type Perturbation struct {
	Type        PerturbType
	LineIdx     int // index into gs.Lines
	StationID   int // station to add or remove
	AtHead      bool
	NewStationID int // for PerturbSwapEndpoint: the replacement endpoint
}

// SAResult holds the best action found in a single SA run and its evaluated
// score. Action is nil when no perturbation improved on the baseline.
type SAResult struct {
	Action *Perturbation
	Score  float64
}

// SA tuning constants.
const (
	saRolloutFrames = 6000  // ~10 s of game time; enough for 2-3 full train route cycles
	saInitialTemp   = 50.0
	saCoolingRate   = 0.97
	saMinTemp       = 0.5
)

// SAOptimize launches a simulated-annealing search in a background goroutine.
// It operates on a deep copy of gs so the original is never touched.
// The goroutine terminates when budget elapses and sends its result on the
// returned channel (buffered 1 — the caller may drain it at leisure).
func SAOptimize(gs *state.GameState, budget time.Duration) <-chan SAResult {
	ch := make(chan SAResult, 1)
	snapshot := gs.DeepCopy()
	go func() {
		ch <- runSA(snapshot, budget)
	}()
	return ch
}

// runSA is the core SA loop. It mutates only its own copy of the state.
func runSA(gs *state.GameState, budget time.Duration) SAResult {
	deadline := time.Now().Add(budget)

	// Baseline: evaluate the current topology without any perturbation.
	baseScore := Evaluate(Rollout(gs, saRolloutFrames))

	current := gs
	currentScore := baseScore
	bestAction := (*Perturbation)(nil)
	bestScore := baseScore

	temp := saInitialTemp

	for time.Now().Before(deadline) && temp > saMinTemp {
		perturbs := generatePerturbations(current)
		if len(perturbs) == 0 {
			break
		}

		p := perturbs[rand.Intn(len(perturbs))]

		// Apply perturbation to an independent copy and evaluate via rollout.
		candidate := current.DeepCopy()
		if !applyPerturbation(candidate, p) {
			temp *= saCoolingRate
			continue
		}

		result := Rollout(candidate, saRolloutFrames)
		candidateScore := Evaluate(result)

		delta := candidateScore - currentScore
		// Accept if better, or probabilistically if worse (SA acceptance criterion).
		if delta < 0 || rand.Float64() < math.Exp(-delta/temp) {
			current = candidate
			currentScore = candidateScore
			if currentScore < bestScore {
				bestScore = currentScore
				bestAction = p
			}
		}

		temp *= saCoolingRate
	}

	return SAResult{Action: bestAction, Score: bestScore}
}

// generatePerturbations returns all feasible topology changes for gs.
func generatePerturbations(gs *state.GameState) []*Perturbation {
	var result []*Perturbation

	for i := 0; i < gs.AvailableLines && i < len(gs.Lines); i++ {
		line := gs.Lines[i]
		if !line.Active || line.MarkedForDeletion {
			continue
		}
		n := len(line.Stations)
		if n < 2 {
			continue
		}
		isLoop := line.Stations[0] == line.Stations[n-1]

		onLine := make(map[int]bool, n)
		for _, s := range line.Stations {
			onLine[s.ID] = true
		}

		if isLoop {
			// Open the loop, restoring bidirectional bouncing movement.
			result = append(result, &Perturbation{Type: PerturbOpenLoop, LineIdx: i})

			// Loops support interior insertion: sample a few positions to keep the
			// perturbation set tractable.
			samplePositions := []int{1, n / 2, n - 1}
			for _, pos := range samplePositions {
				if pos <= 0 || pos >= n {
					continue
				}
				for _, st := range gs.Stations {
					if onLine[st.ID] {
						continue
					}
					result = append(result, &Perturbation{
						Type:      PerturbInsertIntoLoop,
						LineIdx:   i,
						StationID: st.ID,
						AtHead:    false, // AtHead repurposed as insertion index via StationID
						// We encode the insertion position in the NewStationID field.
						NewStationID: pos,
					})
				}
			}
			continue
		}

		head := line.Stations[0]
		tail := line.Stations[n-1]

		// Add endpoint: any station not already on this line.
		for _, st := range gs.Stations {
			if onLine[st.ID] {
				continue
			}
			result = append(result,
				&Perturbation{Type: PerturbAddEndpoint, LineIdx: i, StationID: st.ID, AtHead: true},
				&Perturbation{Type: PerturbAddEndpoint, LineIdx: i, StationID: st.ID, AtHead: false},
			)
		}

		// Remove endpoint (only when ≥ 3 non-loop stations to keep line active).
		if n > 2 {
			result = append(result,
				&Perturbation{Type: PerturbRemoveEndpoint, LineIdx: i, StationID: head.ID, AtHead: true},
				&Perturbation{Type: PerturbRemoveEndpoint, LineIdx: i, StationID: tail.ID, AtHead: false},
			)
		}

		// Close loop (at least 4 stations required).
		if n >= 4 {
			result = append(result, &Perturbation{Type: PerturbCloseLoop, LineIdx: i})
		}

		// Swap tail: remove current tail, append another station.
		for _, st := range gs.Stations {
			if onLine[st.ID] {
				continue
			}
			result = append(result, &Perturbation{
				Type:         PerturbSwapEndpoint,
				LineIdx:      i,
				StationID:    tail.ID,
				AtHead:       false,
				NewStationID: st.ID,
			})
		}
	}

	return result
}

// applyPerturbation mutates gs in-place. Returns false when infeasible.
// Station lookups are performed by ID so this is safe to call on any copy
// of the state from which the Perturbation was originally generated.
func applyPerturbation(gs *state.GameState, p *Perturbation) bool {
	if p.LineIdx >= len(gs.Lines) {
		return false
	}
	line := gs.Lines[p.LineIdx]
	if line.MarkedForDeletion {
		return false
	}

	markDirty := func() { gs.GraphDirty = true }

	// Build a single ID→Station index up front so each lookup below is O(1)
	// rather than a separate linear scan.
	stIdx := make(map[int]*components.Station, len(gs.Stations))
	for _, s := range gs.Stations {
		stIdx[s.ID] = s
	}
	stByID := func(id int) *components.Station { return stIdx[id] }

	switch p.Type {
	case PerturbAddEndpoint:
		st := stByID(p.StationID)
		if st == nil {
			return false
		}
		insertIdx := -1
		if p.AtHead {
			insertIdx = 0
		}
		return line.AddStation(st, insertIdx, markDirty)

	case PerturbRemoveEndpoint:
		st := stByID(p.StationID)
		if st == nil {
			return false
		}
		line.RemoveEndStation(st, markDirty)
		clampTrainIndices(line)
		return true

	case PerturbCloseLoop:
		n := len(line.Stations)
		if n < 4 {
			return false
		}
		if CheckRiverCrossing(gs, line.Stations[0], line.Stations[n-1]) {
			if gs.Bridges == 0 {
				return false
			}
			gs.Bridges--
		}
		return line.AddStation(line.Stations[0], -1, markDirty)

	case PerturbSwapEndpoint:
		st := stByID(p.StationID)
		newSt := stByID(p.NewStationID)
		if st == nil || newSt == nil {
			return false
		}
		line.RemoveEndStation(st, markDirty)
		clampTrainIndices(line)
		insertIdx := -1
		if p.AtHead {
			insertIdx = 0
		}
		return line.AddStation(newSt, insertIdx, markDirty)

	case PerturbOpenLoop:
		n := len(line.Stations)
		if n < 4 || line.Stations[0] != line.Stations[n-1] {
			return false // not a valid loop
		}
		// Remove the closing duplicate so the line becomes a linear, bouncing route.
		line.RemoveEndStation(line.Stations[n-1], markDirty)
		clampTrainIndices(line)
		// Reset direction for all trains: the loop-circulation logic is no longer
		// active, and trains must start bouncing correctly from their current position.
		for _, t := range line.Trains {
			t.Direction = 1
		}
		return true

	case PerturbInsertIntoLoop:
		st := stByID(p.StationID)
		if st == nil {
			return false
		}
		// NewStationID encodes the insertion position within the loop.
		pos := p.NewStationID
		n := len(line.Stations)
		if pos <= 0 || pos >= n {
			return false
		}
		// Check net bridge cost.
		prev := line.Stations[pos-1]
		next := line.Stations[pos]
		netCost := 0
		if CheckRiverCrossing(gs, prev, st) {
			netCost++
		}
		if CheckRiverCrossing(gs, st, next) {
			netCost++
		}
		if CheckRiverCrossing(gs, prev, next) {
			netCost--
		}
		if netCost < 0 {
			netCost = 0
		}
		if gs.Bridges < netCost {
			return false
		}
		gs.Bridges -= netCost
		return line.AddStation(st, pos, markDirty)
	}
	return false
}

// ApplyPerturbation is the exported entry point for topology mutations used by
// the RL environment. It delegates to the internal applyPerturbation helper.
func ApplyPerturbation(gs *state.GameState, p *Perturbation) bool {
	return applyPerturbation(gs, p)
}

// clampTrainIndices ensures that all trains on line have CurrentStationIndex
// and NextStationIndex within the current station slice bounds. Called after
// any operation that reduces the number of stations on a line.
func clampTrainIndices(line *components.Line) {
	n := len(line.Stations)
	if n == 0 {
		return
	}
	for _, t := range line.Trains {
		if t.CurrentStationIndex >= n {
			t.CurrentStationIndex = n - 1
			t.Progress = 0
		}
		if t.NextStationIndex >= n {
			t.NextStationIndex = t.CurrentStationIndex
		}
	}
}

