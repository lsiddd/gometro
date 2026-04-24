package rl

import (
	"minimetro-go/components"
	"minimetro-go/config"
	"minimetro-go/state"
	"minimetro-go/systems"
)

const (
	MaxLineSlots    = 7
	MaxStationSlots = 50
	NumActionCats   = 14 // action-category dimension of the MultiDiscrete space
	NumOptions      = 2  // head/tail option dimension

	BaseMaskSize = NumActionCats + MaxLineSlots + MaxStationSlots + NumOptions
	// Mask layout:
	//   [base categorical masks]
	//   [per-action line masks:    NumActionCats * MaxLineSlots]
	//   [per-action+line station masks: NumActionCats * MaxLineSlots * MaxStationSlots]
	//   [per-action option masks:  NumActionCats * NumOptions]
	//
	// The base section keeps backwards compatibility with consumers that only
	// understand independent MultiDiscrete masks. The conditional sections are
	// used by the autoregressive Python policy after it samples the action
	// category, so later heads only see parameters that are valid for that
	// category.
	CondLineOffset    = BaseMaskSize
	CondStationOffset = CondLineOffset + NumActionCats*MaxLineSlots
	CondOptionOffset  = CondStationOffset + NumActionCats*MaxLineSlots*MaxStationSlots
	MaskSize          = CondOptionOffset + NumActionCats*NumOptions

	// Action Categories
	ActNoOp               = 0
	ActAddEndpoint        = 1
	ActRemoveEndpoint     = 2
	ActCloseLoop          = 3
	ActOpenLoop           = 4
	ActSwapEndpoint       = 5
	ActInsertIntoLoop     = 6
	ActDeployTrain        = 7
	ActAddCarriage        = 8
	ActUpgradeInterchange = 9
	ActChooseUpgrade      = 10
	// 11-13 reserved padding
)

func BuildActionMaskMulti(env *RLEnv) []bool {
	gs := env.gs
	mask := make([]bool, MaskSize)

	if env.inUpgradeModal {
		mask[ActChooseUpgrade] = true
		for i, choice := range env.upgradeChoices {
			if i < 2 && choice != "" {
				setOption(mask, ActChooseUpgrade, i)
			}
		}
		setLine(mask, ActChooseUpgrade, 0)
		setStation(mask, ActChooseUpgrade, 0, 0)
		projectConditionalMask(mask)
		return mask
	}

	mask[ActNoOp] = true
	setLine(mask, ActNoOp, 0)
	setStation(mask, ActNoOp, 0, 0)
	setOption(mask, ActNoOp, 0)

	for l := 0; l < gs.AvailableLines && l < len(gs.Lines); l++ {
		line := gs.Lines[l]
		if line.MarkedForDeletion {
			continue
		}
		n := len(line.Stations)
		isActive := line.Active

		if !isActive {
			mask[ActAddEndpoint] = true
			setLine(mask, ActAddEndpoint, l)
			continue
		}

		isLoop := isActive && n > 2 && line.Stations[0] == line.Stations[n-1]
		if !isLoop {
			mask[ActAddEndpoint] = true
			setLine(mask, ActAddEndpoint, l)
		}
		if n >= 3 && !isLoop {
			mask[ActRemoveEndpoint] = true
			setLine(mask, ActRemoveEndpoint, l)
		}
		if n >= 3 && !isLoop {
			last := line.Stations[n-1]
			first := line.Stations[0]
			if !systems.CheckRiverCrossing(gs, last, first) || gs.Bridges > 0 {
				mask[ActCloseLoop] = true
				setLine(mask, ActCloseLoop, l)
			}
		}
		if isLoop {
			mask[ActOpenLoop] = true
			setLine(mask, ActOpenLoop, l)
		}
		if n >= 2 && !isLoop {
			mask[ActSwapEndpoint] = true
			setLine(mask, ActSwapEndpoint, l)
		}
		if isLoop && n >= 4 {
			mask[ActInsertIntoLoop] = true
			setLine(mask, ActInsertIntoLoop, l)
		}
	}

	for l := 0; l < gs.AvailableLines && l < len(gs.Lines); l++ {
		line := gs.Lines[l]
		if line.MarkedForDeletion {
			continue
		}
		setStation(mask, ActRemoveEndpoint, l, 0)
		for s := 0; s < len(gs.Stations) && s < MaxStationSlots; s++ {
			st := gs.Stations[s]
			if !line.Active {
				setStation(mask, ActAddEndpoint, l, s)
				continue
			}
			n := len(line.Stations)
			isLoop := n > 2 && line.Stations[0] == line.Stations[n-1]
			if !isLoop && !lineContainsStation(line, st) {
				setStation(mask, ActAddEndpoint, l, s)
			}
			if n >= 2 && !isLoop && !lineContainsStation(line, st) {
				setStation(mask, ActSwapEndpoint, l, s)
			}
			if isLoop && n >= 4 && !lineContainsStation(line, st) && canInsertStationIntoLoop(gs, line, st) {
				setStation(mask, ActInsertIntoLoop, l, s)
			}
		}
	}

	for s := 0; s < len(gs.Stations) && s < MaxStationSlots; s++ {
		st := gs.Stations[s]
		if !st.IsInterchange {
			setStation(mask, ActUpgradeInterchange, 0, s)
		}
	}

	if gs.AvailableTrains > 0 && firstMatchingLine(gs, 0, canDeployTrainOnLine) != nil {
		mask[ActDeployTrain] = true
		for l := 0; l < gs.AvailableLines && l < len(gs.Lines); l++ {
			if canDeployTrainOnLine(gs.Lines[l]) {
				setLine(mask, ActDeployTrain, l)
			}
		}
	}
	if gs.Carriages > 0 && firstMatchingLine(gs, 0, canAddCarriageToLine) != nil {
		mask[ActAddCarriage] = true
		for l := 0; l < gs.AvailableLines && l < len(gs.Lines); l++ {
			if canAddCarriageToLine(gs.Lines[l]) {
				setLine(mask, ActAddCarriage, l)
			}
		}
	}
	if gs.Interchanges > 0 && firstMatchingStation(gs, 0, func(st *components.Station) bool {
		return !st.IsInterchange
	}) != nil {
		mask[ActUpgradeInterchange] = true
	}

	for l := 0; l < MaxLineSlots; l++ {
		setStation(mask, ActCloseLoop, l, 0)
		setStation(mask, ActOpenLoop, l, 0)
		setStation(mask, ActDeployTrain, l, 0)
		setStation(mask, ActAddCarriage, l, 0)
	}

	setLine(mask, ActUpgradeInterchange, 0)

	for _, act := range []int{ActAddEndpoint, ActRemoveEndpoint, ActSwapEndpoint} {
		setOption(mask, act, 0)
		setOption(mask, act, 1)
	}
	for _, act := range []int{ActCloseLoop, ActOpenLoop, ActInsertIntoLoop, ActDeployTrain, ActAddCarriage, ActUpgradeInterchange} {
		setOption(mask, act, 0)
	}

	clearUnavailableConditionalActions(mask)
	projectConditionalMask(mask)
	return mask
}

func setLine(mask []bool, act, line int) {
	if act >= 0 && act < NumActionCats && line >= 0 && line < MaxLineSlots {
		mask[CondLineOffset+act*MaxLineSlots+line] = true
	}
}

func setStation(mask []bool, act, line, station int) {
	if act >= 0 && act < NumActionCats && line >= 0 && line < MaxLineSlots && station >= 0 && station < MaxStationSlots {
		mask[CondStationOffset+((act*MaxLineSlots+line)*MaxStationSlots)+station] = true
	}
}

func setOption(mask []bool, act, option int) {
	if act >= 0 && act < NumActionCats && option >= 0 && option < NumOptions {
		mask[CondOptionOffset+act*NumOptions+option] = true
	}
}

func clearUnavailableConditionalActions(mask []bool) {
	for act := 0; act < NumActionCats; act++ {
		if !mask[act] {
			for l := 0; l < MaxLineSlots; l++ {
				mask[CondLineOffset+act*MaxLineSlots+l] = false
			}
			for l := 0; l < MaxLineSlots; l++ {
				for s := 0; s < MaxStationSlots; s++ {
					mask[CondStationOffset+((act*MaxLineSlots+l)*MaxStationSlots)+s] = false
				}
			}
			for o := 0; o < NumOptions; o++ {
				mask[CondOptionOffset+act*NumOptions+o] = false
			}
		}
	}
}

func projectConditionalMask(mask []bool) {
	for act := 0; act < NumActionCats; act++ {
		for l := 0; l < MaxLineSlots; l++ {
			if mask[CondLineOffset+act*MaxLineSlots+l] {
				mask[NumActionCats+l] = true
			}
		}
		for l := 0; l < MaxLineSlots; l++ {
			for s := 0; s < MaxStationSlots; s++ {
				if mask[CondStationOffset+((act*MaxLineSlots+l)*MaxStationSlots)+s] {
					mask[NumActionCats+MaxLineSlots+s] = true
				}
			}
		}
		for o := 0; o < NumOptions; o++ {
			if mask[CondOptionOffset+act*NumOptions+o] {
				mask[NumActionCats+MaxLineSlots+MaxStationSlots+o] = true
			}
		}
	}
}

func lineContainsStation(line *components.Line, st *components.Station) bool {
	for _, cur := range line.Stations {
		if cur == st {
			return true
		}
	}
	return false
}

func canInsertStationIntoLoop(gs *state.GameState, line *components.Line, st *components.Station) bool {
	n := len(line.Stations)
	pos := n / 2
	if pos <= 0 {
		pos = 1
	}
	prev := line.Stations[pos-1]
	next := line.Stations[pos]
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
	return gs.Bridges >= cost
}

func firstMatchingLine(gs *state.GameState, preferred int, valid func(*components.Line) bool) *components.Line {
	if preferred >= 0 && preferred < gs.AvailableLines && preferred < len(gs.Lines) && valid(gs.Lines[preferred]) {
		return gs.Lines[preferred]
	}
	for i := 0; i < gs.AvailableLines && i < len(gs.Lines); i++ {
		if i != preferred && valid(gs.Lines[i]) {
			return gs.Lines[i]
		}
	}
	return nil
}

func firstMatchingStation(gs *state.GameState, preferred int, valid func(*components.Station) bool) *components.Station {
	if preferred >= 0 && preferred < len(gs.Stations) && valid(gs.Stations[preferred]) {
		return gs.Stations[preferred]
	}
	for i, st := range gs.Stations {
		if i != preferred && valid(st) {
			return st
		}
	}
	return nil
}

func canDeployTrainOnLine(line *components.Line) bool {
	return line != nil && line.Active && !line.MarkedForDeletion && len(line.Trains) < config.MaxTrainsPerLine
}

func canAddCarriageToLine(line *components.Line) bool {
	if line == nil || !line.Active || line.MarkedForDeletion {
		return false
	}
	for _, t := range line.Trains {
		if t.CarriageCount < config.MaxCarriagesPerTrain {
			return true
		}
	}
	return false
}

func ApplyRLAction(env *RLEnv, action []int) bool {
	if len(action) != 4 {
		return false
	}
	actCat := action[0]
	lIdx := action[1]
	sIdx := action[2]
	opt := action[3]

	gs := env.gs

	// Ensure indices are in bounds
	if (lIdx < 0 || lIdx >= len(gs.Lines)) && actCat != ActNoOp && actCat != ActUpgradeInterchange && actCat != ActChooseUpgrade {
		return false
	}
	if (sIdx < 0 || sIdx >= len(gs.Stations)) && actCat != ActNoOp && actCat != ActCloseLoop && actCat != ActOpenLoop && actCat != ActDeployTrain && actCat != ActAddCarriage {
		return false
	}

	var line *components.Line
	if lIdx >= 0 && lIdx < len(gs.Lines) {
		line = gs.Lines[lIdx]
	}
	var st *components.Station
	if sIdx >= 0 && sIdx < len(gs.Stations) {
		st = gs.Stations[sIdx]
	}

	switch actCat {
	case ActNoOp:
		return true

	case ActAddEndpoint:
		if line == nil || st == nil || line.MarkedForDeletion {
			return false
		}
		return systems.ApplyPerturbation(gs, &systems.Perturbation{
			Type:      systems.PerturbAddEndpoint,
			LineIdx:   lIdx,
			StationID: st.ID,
			AtHead:    opt == 1,
		})

	case ActRemoveEndpoint:
		if line == nil || line.MarkedForDeletion || !line.Active {
			return false
		}
		n := len(line.Stations)
		if n < 3 { // don't kill line implicitly
			return false
		}
		isLoop := line.Active && n > 2 && line.Stations[0] == line.Stations[n-1]
		if isLoop {
			return false
		}
		atHead := opt == 1
		p := &systems.Perturbation{
			Type:    systems.PerturbRemoveEndpoint,
			LineIdx: lIdx,
			AtHead:  atHead,
		}
		if atHead {
			p.StationID = line.Stations[0].ID
		} else {
			p.StationID = line.Stations[n-1].ID
		}
		return systems.ApplyPerturbation(gs, p)

	case ActCloseLoop:
		if line == nil || !line.Active {
			return false
		}
		n := len(line.Stations)
		if n < 3 {
			return false
		}
		isLoop := line.Active && n > 2 && line.Stations[0] == line.Stations[n-1]
		if isLoop {
			return false
		}
		last := line.Stations[n-1]
		first := line.Stations[0]
		if systems.CheckRiverCrossing(gs, last, first) && gs.Bridges <= 0 {
			return false
		}
		return systems.ApplyPerturbation(gs, &systems.Perturbation{
			Type:    systems.PerturbCloseLoop,
			LineIdx: lIdx,
		})

	case ActOpenLoop:
		if line == nil || !line.Active {
			return false
		}
		n := len(line.Stations)
		isLoop := line.Active && n > 2 && line.Stations[0] == line.Stations[n-1]
		if !isLoop {
			return false
		}
		return systems.ApplyPerturbation(gs, &systems.Perturbation{
			Type:    systems.PerturbOpenLoop,
			LineIdx: lIdx,
		})

	case ActSwapEndpoint:
		if line == nil || st == nil || !line.Active {
			return false
		}
		n := len(line.Stations)
		if n < 2 {
			return false
		}
		isLoop := line.Active && n > 2 && line.Stations[0] == line.Stations[n-1]
		if isLoop {
			return false
		}
		for _, cur := range line.Stations {
			if cur == st {
				return false
			}
		}
		atHead := opt == 1
		var endpointID int
		if atHead {
			endpointID = line.Stations[0].ID
		} else {
			endpointID = line.Stations[n-1].ID
		}
		return systems.ApplyPerturbation(gs, &systems.Perturbation{
			Type:         systems.PerturbSwapEndpoint,
			LineIdx:      lIdx,
			StationID:    endpointID,
			AtHead:       atHead,
			NewStationID: st.ID,
		})

	case ActInsertIntoLoop:
		if line == nil || st == nil || !line.Active {
			return false
		}
		n := len(line.Stations)
		isLoop := line.Active && n > 2 && line.Stations[0] == line.Stations[n-1]
		if !isLoop || n < 4 {
			return false
		}
		for _, cur := range line.Stations {
			if cur == st {
				return false
			}
		}
		pos := n / 2
		if pos <= 0 {
			pos = 1
		}
		// check bridges
		prev := line.Stations[pos-1]
		next := line.Stations[pos]
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
		if gs.Bridges < cost {
			return false
		}
		return systems.ApplyPerturbation(gs, &systems.Perturbation{
			Type:         systems.PerturbInsertIntoLoop,
			LineIdx:      lIdx,
			StationID:    st.ID,
			NewStationID: pos,
		})

	case ActDeployTrain:
		line = firstMatchingLine(gs, lIdx, canDeployTrainOnLine)
		if line == nil {
			return false
		}
		if gs.AvailableTrains <= 0 {
			return false
		}
		gs.AvailableTrains--
		cityCfg := config.Cities[gs.SelectedCity]
		train := components.NewTrain(gs.TrainIDCounter, line, cityCfg.TrainCapacity, config.TrainMaxSpeed)
		gs.AddTrain(train)
		line.Trains = append(line.Trains, train)
		return true

	case ActAddCarriage:
		if gs.Carriages <= 0 {
			return false
		}
		line = firstMatchingLine(gs, lIdx, canAddCarriageToLine)
		if line == nil {
			return false
		}
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

	case ActUpgradeInterchange:
		st = firstMatchingStation(gs, sIdx, func(st *components.Station) bool {
			return !st.IsInterchange
		})
		if st == nil {
			return false
		}
		if st.IsInterchange || gs.Interchanges <= 0 {
			return false
		}
		st.IsInterchange = true
		gs.Interchanges--
		gs.GraphDirty = true
		return true

	case ActChooseUpgrade:
		return false
	}

	return false
}
