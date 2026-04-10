package rl

// Action space layout (flat discrete, total = NumActions):
//
//	[0]          NoOp
//	[1..700]     AddEndpoint:       lineIdx*100 + stSlot*2 + atHead  + offAddEndpoint
//	[701..714]   RemoveEndpoint:    lineIdx*2   + atHead              + offRemoveEndpoint
//	[715..721]   CloseLoop:         lineIdx                            + offCloseLoop
//	[722..728]   OpenLoop:          lineIdx                            + offOpenLoop
//	[729..1428]  SwapEndpoint:      lineIdx*100 + stSlot*2 + atHead  + offSwapEndpoint
//	[1429..1778] InsertIntoLoop:    lineIdx*50  + stSlot               + offInsertLoop
//	[1779..1785] DeployTrain:       lineIdx                            + offDeployTrain
//	[1786..1792] AddCarriage:       lineIdx                            + offAddCarriage
//	[1793..1842] UpgradeInterchange stSlot                             + offUpgradeInterchange
//	[1843]       ChooseUpgrade option 0
//	[1844]       ChooseUpgrade option 1

import (
	"minimetro-go/components"
	"minimetro-go/config"
	"minimetro-go/state"
	"minimetro-go/systems"
)

const (
	MaxLineSlots    = 7
	MaxStationSlots = 50
	NumActions      = 1845

	offAddEndpoint        = 1    // 7 × 50 × 2 = 700
	offRemoveEndpoint     = 701  // 7 × 2      = 14
	offCloseLoop          = 715  // 7
	offOpenLoop           = 722  // 7
	offSwapEndpoint       = 729  // 7 × 50 × 2 = 700
	offInsertLoop         = 1429 // 7 × 50     = 350
	offDeployTrain        = 1779 // 7
	offAddCarriage        = 1786 // 7
	offUpgradeInterchange = 1793 // 50
	offChooseUpgrade      = 1843 // 2
)

// BuildActionMask returns a boolean slice of length NumActions where true
// means the action is currently legal.
func BuildActionMask(env *RLEnv) []bool {
	gs := env.gs
	mask := make([]bool, NumActions)

	// NoOp is always legal.
	mask[0] = true

	if env.inUpgradeModal {
		// During upgrade modal only the two upgrade-choice actions are valid.
		for i, choice := range env.upgradeChoices {
			if i < 2 && choice != "" {
				mask[offChooseUpgrade+i] = true
			}
		}
		return mask
	}

	nStations := len(gs.Stations)
	nLines := gs.AvailableLines
	if nLines > len(gs.Lines) {
		nLines = len(gs.Lines)
	}

	// Per-station lookup: how many active lines pass through this station.
	lineCount := make([]int, nStations)
	for l := 0; l < nLines; l++ {
		line := gs.Lines[l]
		if !line.Active || line.MarkedForDeletion {
			continue
		}
		seen := make(map[int]bool)
		for _, s := range line.Stations {
			if !seen[s.ID] {
				seen[s.ID] = true
				if s.ID < nStations {
					lineCount[s.ID]++
				}
			}
		}
	}

	for l := 0; l < nLines; l++ {
		line := gs.Lines[l]
		isDel := line.MarkedForDeletion
		isActive := line.Active
		n := len(line.Stations)
		isLoop := isActive && n > 2 && line.Stations[0] == line.Stations[n-1]

		// Build set of station IDs on this line.
		onLine := make(map[int]bool, n)
		for _, s := range line.Stations {
			onLine[s.ID] = true
		}

		if !isDel {
			// ── AddEndpoint ──────────────────────────────────────────────────────
			// Valid for any non-loop line (active or not) with < MaxStationSlots.
			if !isLoop {
				for s := 0; s < nStations; s++ {
					if onLine[gs.Stations[s].ID] {
						continue
					}
					// Head (atHead=1)
					bridgeOK := bridgeCostOK(gs, gs.Stations[s], line, true)
					if bridgeOK {
						mask[offAddEndpoint+l*100+s*2+1] = true
					}
					// Tail (atHead=0)
					bridgeOK = bridgeCostOK(gs, gs.Stations[s], line, false)
					if bridgeOK {
						mask[offAddEndpoint+l*100+s*2+0] = true
					}
				}
			}

			// ── RemoveEndpoint ───────────────────────────────────────────────────
			// Requires active non-loop with ≥ 3 stations so the line stays alive.
			if isActive && !isLoop && n >= 3 {
				mask[offRemoveEndpoint+l*2+0] = true // tail
				mask[offRemoveEndpoint+l*2+1] = true // head
			}

			// ── CloseLoop ────────────────────────────────────────────────────────
			if isActive && !isLoop && n >= 4 {
				// Bridge check: last station → first station.
				last := line.Stations[n-1]
				first := line.Stations[0]
				if !systems.CheckRiverCrossing(gs, last, first) || gs.Bridges > 0 {
					mask[offCloseLoop+l] = true
				}
			}

			// ── OpenLoop ─────────────────────────────────────────────────────────
			if isLoop && n >= 4 {
				mask[offOpenLoop+l] = true
			}

			// ── SwapEndpoint ─────────────────────────────────────────────────────
			// Replace tail (atHead=0) or head (atHead=1) with a different station.
			if isActive && !isLoop && n >= 2 {
				for s := 0; s < nStations; s++ {
					st := gs.Stations[s]
					if onLine[st.ID] {
						continue
					}
					// For tail swap: from line.Stations[n-2] to new station.
					prev := line.Stations[n-2]
					if !systems.CheckRiverCrossing(gs, prev, st) || gs.Bridges > 0 {
						mask[offSwapEndpoint+l*100+s*2+0] = true
					}
					// For head swap: from line.Stations[1] to new station.
					next := line.Stations[1]
					if !systems.CheckRiverCrossing(gs, next, st) || gs.Bridges > 0 {
						mask[offSwapEndpoint+l*100+s*2+1] = true
					}
				}
			}

			// ── InsertIntoLoop ───────────────────────────────────────────────────
			if isLoop && n >= 4 {
				pos := n / 2
				prev := line.Stations[pos-1]
				next := line.Stations[pos]
				for s := 0; s < nStations; s++ {
					st := gs.Stations[s]
					if onLine[st.ID] {
						continue
					}
					netCost := riverInsertCost(gs, prev, st, next)
					if gs.Bridges >= netCost {
						mask[offInsertLoop+l*MaxStationSlots+s] = true
					}
				}
			}
		}

		// ── DeployTrain ──────────────────────────────────────────────────────
		if isActive && !isDel && gs.AvailableTrains > 0 &&
			len(line.Trains) < config.MaxTrainsPerLine {
			mask[offDeployTrain+l] = true
		}

		// ── AddCarriage ──────────────────────────────────────────────────────
		if isActive && !isDel && gs.Carriages > 0 {
			for _, t := range line.Trains {
				if t.CarriageCount < config.MaxCarriagesPerTrain {
					mask[offAddCarriage+l] = true
					break
				}
			}
		}
	}

	// ── UpgradeInterchange ───────────────────────────────────────────────────
	if gs.Interchanges > 0 {
		for s := 0; s < nStations; s++ {
			if !gs.Stations[s].IsInterchange {
				mask[offUpgradeInterchange+s] = true
			}
		}
	}

	return mask
}

// bridgeCostOK returns true if adding station st to line at the given end does
// not require more bridges than available.
func bridgeCostOK(gs *state.GameState, st *components.Station, line *components.Line, atHead bool) bool {
	n := len(line.Stations)
	if n == 0 {
		return true // first station — no crossing possible
	}
	var neighbor *components.Station
	if atHead {
		neighbor = line.Stations[0]
	} else {
		neighbor = line.Stations[n-1]
	}
	if systems.CheckRiverCrossing(gs, neighbor, st) {
		return gs.Bridges > 0
	}
	return true
}

// riverInsertCost computes the net bridge cost of inserting st between prev and
// next: bridges needed for (prev→st) + (st→next) minus (prev→next).
func riverInsertCost(gs *state.GameState, prev, st, next *components.Station) int {
	cost := 0
	if systems.CheckRiverCrossing(gs, prev, st) {
		cost++
	}
	if systems.CheckRiverCrossing(gs, st, next) {
		cost++
	}
	if systems.CheckRiverCrossing(gs, prev, next) {
		cost--
	}
	if cost < 0 {
		cost = 0
	}
	return cost
}

// ApplyRLAction applies the flat action index to env.gs. Returns true when
// successfully applied. Invalid or masked actions are treated as NoOp.
func ApplyRLAction(env *RLEnv, actionIdx int) bool {
	gs := env.gs
	markDirty := func() { gs.GraphDirty = true }

	switch {
	case actionIdx == 0:
		// NoOp — nothing to do.
		return true

	case actionIdx >= offAddEndpoint && actionIdx < offRemoveEndpoint:
		idx := actionIdx - offAddEndpoint
		atHead := (idx & 1) == 1
		stSlot := (idx >> 1) % MaxStationSlots
		lineIdx := (idx >> 1) / MaxStationSlots
		if lineIdx >= len(gs.Lines) || stSlot >= len(gs.Stations) {
			return false
		}
		line := gs.Lines[lineIdx]
		st := gs.Stations[stSlot]
		if line.MarkedForDeletion {
			return false
		}
		insertIdx := -1
		if atHead {
			insertIdx = 0
		}
		// Bridge deduction for endpoint addition.
		n := len(line.Stations)
		if n > 0 {
			var neighbor *components.Station
			if atHead {
				neighbor = line.Stations[0]
			} else {
				neighbor = line.Stations[n-1]
			}
			if systems.CheckRiverCrossing(gs, neighbor, st) {
				if gs.Bridges <= 0 {
					return false
				}
				gs.Bridges--
			}
		}
		return line.AddStation(st, insertIdx, markDirty)

	case actionIdx >= offRemoveEndpoint && actionIdx < offCloseLoop:
		idx := actionIdx - offRemoveEndpoint
		atHead := (idx & 1) == 1
		lineIdx := idx >> 1
		if lineIdx >= len(gs.Lines) {
			return false
		}
		p := &systems.Perturbation{
			Type:    systems.PerturbRemoveEndpoint,
			LineIdx: lineIdx,
			AtHead:  atHead,
		}
		line := gs.Lines[lineIdx]
		n := len(line.Stations)
		if n == 0 {
			return false
		}
		if atHead {
			p.StationID = line.Stations[0].ID
		} else {
			p.StationID = line.Stations[n-1].ID
		}
		return systems.ApplyPerturbation(gs, p)

	case actionIdx >= offCloseLoop && actionIdx < offOpenLoop:
		lineIdx := actionIdx - offCloseLoop
		return systems.ApplyPerturbation(gs, &systems.Perturbation{
			Type:    systems.PerturbCloseLoop,
			LineIdx: lineIdx,
		})

	case actionIdx >= offOpenLoop && actionIdx < offSwapEndpoint:
		lineIdx := actionIdx - offOpenLoop
		return systems.ApplyPerturbation(gs, &systems.Perturbation{
			Type:    systems.PerturbOpenLoop,
			LineIdx: lineIdx,
		})

	case actionIdx >= offSwapEndpoint && actionIdx < offInsertLoop:
		idx := actionIdx - offSwapEndpoint
		atHead := (idx & 1) == 1
		stSlot := (idx >> 1) % MaxStationSlots
		lineIdx := (idx >> 1) / MaxStationSlots
		if lineIdx >= len(gs.Lines) || stSlot >= len(gs.Stations) {
			return false
		}
		line := gs.Lines[lineIdx]
		n := len(line.Stations)
		if n == 0 {
			return false
		}
		var endpointID int
		if atHead {
			endpointID = line.Stations[0].ID
		} else {
			endpointID = line.Stations[n-1].ID
		}
		return systems.ApplyPerturbation(gs, &systems.Perturbation{
			Type:         systems.PerturbSwapEndpoint,
			LineIdx:      lineIdx,
			StationID:    endpointID,
			AtHead:       atHead,
			NewStationID: gs.Stations[stSlot].ID,
		})

	case actionIdx >= offInsertLoop && actionIdx < offDeployTrain:
		idx := actionIdx - offInsertLoop
		stSlot := idx % MaxStationSlots
		lineIdx := idx / MaxStationSlots
		if lineIdx >= len(gs.Lines) || stSlot >= len(gs.Stations) {
			return false
		}
		line := gs.Lines[lineIdx]
		n := len(line.Stations)
		pos := n / 2
		if pos <= 0 {
			pos = 1
		}
		return systems.ApplyPerturbation(gs, &systems.Perturbation{
			Type:         systems.PerturbInsertIntoLoop,
			LineIdx:      lineIdx,
			StationID:    gs.Stations[stSlot].ID,
			NewStationID: pos,
		})

	case actionIdx >= offDeployTrain && actionIdx < offAddCarriage:
		lineIdx := actionIdx - offDeployTrain
		if lineIdx >= len(gs.Lines) {
			return false
		}
		line := gs.Lines[lineIdx]
		if !line.Active || line.MarkedForDeletion {
			return false
		}
		if gs.AvailableTrains <= 0 || len(line.Trains) >= config.MaxTrainsPerLine {
			return false
		}
		gs.AvailableTrains--
		cityCfg := config.Cities[gs.SelectedCity]
		train := components.NewTrain(gs.TrainIDCounter, line, cityCfg.TrainCapacity, config.TrainMaxSpeed)
		gs.AddTrain(train)
		line.Trains = append(line.Trains, train)
		return true

	case actionIdx >= offAddCarriage && actionIdx < offUpgradeInterchange:
		lineIdx := actionIdx - offAddCarriage
		if lineIdx >= len(gs.Lines) {
			return false
		}
		if gs.Carriages <= 0 {
			return false
		}
		line := gs.Lines[lineIdx]
		// Add carriage to the most-loaded eligible train on this line.
		var best *components.Train
		bestRatio := -1.0
		for _, t := range line.Trains {
			if t.CarriageCount >= config.MaxCarriagesPerTrain {
				continue
			}
			cap := t.TotalCapacity()
			ratio := float64(len(t.Passengers)) / float64(cap)
			if ratio > bestRatio {
				bestRatio = ratio
				best = t
			}
		}
		if best == nil {
			return false
		}
		best.CarriageCount++
		gs.Carriages--
		return true

	case actionIdx >= offUpgradeInterchange && actionIdx < offChooseUpgrade:
		stSlot := actionIdx - offUpgradeInterchange
		if stSlot >= len(gs.Stations) {
			return false
		}
		st := gs.Stations[stSlot]
		if st.IsInterchange || gs.Interchanges <= 0 {
			return false
		}
		st.IsInterchange = true
		gs.Interchanges--
		gs.GraphDirty = true
		return true

	case actionIdx >= offChooseUpgrade && actionIdx <= offChooseUpgrade+1:
		// Handled in RLEnv.Step — should not reach here normally.
		return false
	}

	return false
}
