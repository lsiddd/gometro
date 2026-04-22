package rl

import (
	"minimetro-go/components"
	"minimetro-go/config"
	"minimetro-go/systems"
)

const (
	MaxLineSlots    = 7
	MaxStationSlots = 50
	NumActionCats   = 14 // action-category dimension of the MultiDiscrete space
	NumOptions      = 2  // head/tail option dimension

	MaskSize = NumActionCats + MaxLineSlots + MaxStationSlots + NumOptions

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

	// In margin-based masking for MultiDiscrete, we simply allow
	// any independent dimension slot that could hypothetically be valid.
	if env.inUpgradeModal {
		mask[ActChooseUpgrade] = true
		// Keep irrelevant parameter heads non-empty. The environment ignores
		// line/station during upgrade selection, but all-masked categoricals add
		// artificial log-probability/entropy noise in the policy.
		mask[NumActionCats] = true
		mask[NumActionCats+MaxLineSlots] = true
		for i, choice := range env.upgradeChoices {
			if i < 2 && choice != "" {
				mask[14+7+50+i] = true // unmask option
			}
		}
		return mask
	}

	// Always allow NoOp
	mask[ActNoOp] = true

	canAddEndpoint := false
	canRemoveEndpoint := false
	canCloseLoop := false
	canOpenLoop := false
	canSwapEndpoint := false
	canInsertIntoLoop := false

	for l := 0; l < gs.AvailableLines && l < len(gs.Lines); l++ {
		line := gs.Lines[l]
		if line.MarkedForDeletion {
			continue
		}
		n := len(line.Stations)
		isActive := line.Active

		if !isActive {
			canAddEndpoint = true
			continue
		}

		isLoop := isActive && n > 2 && line.Stations[0] == line.Stations[n-1]
		if !isLoop {
			canAddEndpoint = true
		}
		if n >= 3 && !isLoop {
			canRemoveEndpoint = true
		}
		if n >= 3 && !isLoop {
			canCloseLoop = true
		}
		if isLoop {
			canOpenLoop = true
		}
		if n >= 2 && !isLoop {
			canSwapEndpoint = true
		}
		if isLoop && n >= 4 {
			canInsertIntoLoop = true
		}
	}

	if canAddEndpoint {
		mask[ActAddEndpoint] = true
	}
	if canRemoveEndpoint {
		mask[ActRemoveEndpoint] = true
	}
	if canCloseLoop {
		mask[ActCloseLoop] = true
	}
	if canOpenLoop {
		mask[ActOpenLoop] = true
	}
	if canSwapEndpoint {
		mask[ActSwapEndpoint] = true
	}
	if canInsertIntoLoop {
		mask[ActInsertIntoLoop] = true
	}

	if gs.AvailableTrains > 0 {
		mask[ActDeployTrain] = true
	}
	if gs.Carriages > 0 {
		mask[ActAddCarriage] = true
	}
	if gs.Interchanges > 0 {
		mask[ActUpgradeInterchange] = true
	}

	// Mask Lines
	nLines := gs.AvailableLines
	if nLines > len(gs.Lines) {
		nLines = len(gs.Lines)
	}
	for l := 0; l < MaxLineSlots; l++ {
		if l < nLines {
			mask[14+l] = true
		}
	}

	// Mask Stations
	for s := 0; s < MaxStationSlots; s++ {
		if s < len(gs.Stations) {
			mask[14+7+s] = true
		}
	}

	// Mask options (Head/Tail)
	mask[14+7+50+0] = true
	mask[14+7+50+1] = true

	return mask
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
	if lIdx >= len(gs.Lines) && actCat != ActNoOp && actCat != ActUpgradeInterchange && actCat != ActChooseUpgrade {
		return false
	}
	if sIdx >= len(gs.Stations) && actCat != ActNoOp && actCat != ActCloseLoop && actCat != ActOpenLoop && actCat != ActDeployTrain && actCat != ActAddCarriage {
		return false
	}

	var line *components.Line
	if lIdx < len(gs.Lines) {
		line = gs.Lines[lIdx]
	}
	var st *components.Station
	if sIdx < len(gs.Stations) {
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
		if line == nil || !line.Active || line.MarkedForDeletion {
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

	case ActAddCarriage:
		if line == nil || !line.Active || line.MarkedForDeletion {
			return false
		}
		if gs.Carriages <= 0 {
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
