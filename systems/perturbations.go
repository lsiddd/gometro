package systems

import (
	"minimetro-go/components"
	"minimetro-go/state"
)

// PerturbType identifies the topology change a Perturbation represents.
type PerturbType int

const (
	PerturbAddEndpoint    PerturbType = iota
	PerturbRemoveEndpoint
	PerturbCloseLoop
	PerturbSwapEndpoint
	PerturbInsertIntoLoop
	PerturbOpenLoop
)

// Perturbation is a topology action applied to the game state. Station
// references use IDs rather than pointers so the action remains valid after
// a deep copy.
type Perturbation struct {
	Type         PerturbType
	LineIdx      int
	StationID    int
	AtHead       bool
	NewStationID int // for PerturbSwapEndpoint: replacement endpoint; for PerturbInsertIntoLoop: insertion index
}

// ApplyPerturbation mutates gs according to p. Returns false when the
// perturbation is invalid or cannot be applied (e.g. no bridges available).
func ApplyPerturbation(gs *state.GameState, p *Perturbation) bool {
	if p.LineIdx >= len(gs.Lines) {
		return false
	}
	line := gs.Lines[p.LineIdx]
	if line.MarkedForDeletion {
		return false
	}

	markDirty := func() { gs.GraphDirty = true }

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
		n := len(line.Stations)
		if n > 2 && line.Stations[0] == line.Stations[n-1] {
			return false
		}
		insertIdx := -1
		if p.AtHead {
			insertIdx = 0
		}
		if n > 0 {
			var neighbor *components.Station
			if p.AtHead {
				neighbor = line.Stations[0]
			} else {
				neighbor = line.Stations[n-1]
			}
			if CheckRiverCrossing(gs, neighbor, st) {
				if gs.Bridges <= 0 {
					return false
				}
				gs.Bridges--
			}
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
		if n < 3 {
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
			return false
		}
		line.RemoveEndStation(line.Stations[n-1], markDirty)
		clampTrainIndices(line)
		for _, t := range line.Trains {
			t.Direction = 1
		}
		return true

	case PerturbInsertIntoLoop:
		st := stByID(p.StationID)
		if st == nil {
			return false
		}
		pos := p.NewStationID
		n := len(line.Stations)
		if pos <= 0 || pos >= n {
			return false
		}
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
