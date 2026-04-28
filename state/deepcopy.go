package state

import "minimetro-go/components"

// DeepCopy returns a fully independent copy of gs. All internal pointer
// relationships (Station ↔ Passenger ↔ Train ↔ Line) are preserved within
// the copy — a Passenger whose CurrentStation points to a Station in the
// original will, in the copy, point to the corresponding new Station.
//
// Rivers are not deep-copied because they are immutable geographic data.
//
// The returned copy is configured for forward simulation:
//   - SpawnStationsEnabled = false  (no new stations during evaluation)
//   - Paused = false
//   - GraphDirty = true             (forces graph rebuild with new pointers)
//
// Implementation uses a strict two-phase approach:
//
//  1. Allocation phase (Steps 1–4): allocate every new object and populate the
//     four identity maps (stMap, lineMap, trainMap, paxMap) that translate old
//     pointers to new ones.  No cross-references are set here.
//
//  2. Wiring phase (Steps 5–8): iterate over the maps and fill every pointer
//     field (CurrentStation, OnTrain, Line.Stations, …) using the maps built
//     in phase 1.
//
// This ordering guarantees that every destination pointer already exists when
// it is assigned — there is no partial initialisation window where a pointer
// could be set to a not-yet-constructed object.
func (gs *GameState) DeepCopy() *GameState {
	// ── Step 1: Stations ────────────────────────────────────────────────────
	stMap := make(map[*components.Station]*components.Station, len(gs.Stations))
	newStations := make([]*components.Station, len(gs.Stations))
	for i, st := range gs.Stations {
		nst := &components.Station{
			ID:                st.ID,
			X:                 st.X,
			Y:                 st.Y,
			Type:              st.Type,
			IsInterchange:     st.IsInterchange,
			OvercrowdProgress: st.OvercrowdProgress,
			OvercrowdIsGrace:  st.OvercrowdIsGrace,
			Passengers:        make([]*components.Passenger, 0, len(st.Passengers)),
			// DeliveryAnimation is rendering-only; not needed in simulation.
		}
		stMap[st] = nst
		newStations[i] = nst
	}

	// ── Step 2: Lines ────────────────────────────────────────────────────────
	// Collect ALL lines: both the official gs.Lines and any dummy lines that
	// trains may reference after a line deletion (cleanupDeletedLines).
	//
	// Dummy lines are ephemeral: when a line is deleted, cleanupDeletedLines
	// replaces the gs.Lines slot with a fresh empty line but keeps the old line
	// pointer alive on each train so it can finish evacuating passengers.  The
	// dummy line is never stored in gs.Lines, so we discover it by scanning
	// gs.Trains and copying any line pointer absent from lineMap.
	lineMap := make(map[*components.Line]*components.Line)
	newLines := make([]*components.Line, len(gs.Lines))

	copyLine := func(ln *components.Line) *components.Line {
		return &components.Line{
			Index:             ln.Index,
			Color:             ln.Color,
			Active:            ln.Active,
			MarkedForDeletion: ln.MarkedForDeletion,
			Stations:          make([]*components.Station, 0, len(ln.Stations)),
			Trains:            make([]*components.Train, 0, len(ln.Trains)),
		}
	}

	for i, ln := range gs.Lines {
		nln := copyLine(ln)
		lineMap[ln] = nln
		newLines[i] = nln
	}
	// Dummy lines referenced by trains but absent from gs.Lines.
	for _, tr := range gs.Trains {
		if tr.Line != nil {
			if _, exists := lineMap[tr.Line]; !exists {
				lineMap[tr.Line] = copyLine(tr.Line)
			}
		}
	}

	// ── Step 3: Trains ───────────────────────────────────────────────────────
	trainMap := make(map[*components.Train]*components.Train, len(gs.Trains))
	newTrains := make([]*components.Train, len(gs.Trains))
	for i, tr := range gs.Trains {
		var pathPts [][2]float64
		if len(tr.PathPts) > 0 {
			pathPts = make([][2]float64, len(tr.PathPts))
			copy(pathPts, tr.PathPts)
		}
		ntr := &components.Train{
			ID:                  tr.ID,
			Capacity:            tr.Capacity,
			CurrentStationIndex: tr.CurrentStationIndex,
			NextStationIndex:    tr.NextStationIndex,
			Direction:           tr.Direction,
			X:                   tr.X,
			Y:                   tr.Y,
			Progress:            tr.Progress,
			State:               tr.State,
			WaitTimer:           tr.WaitTimer,
			Speed:               tr.Speed,
			MaxSpeed:            tr.MaxSpeed,
			PathPts:             pathPts,
			PathLength:          tr.PathLength,
			CarriageCount:       tr.CarriageCount,
			ReservedSeats:       tr.ReservedSeats,
			Passengers:          make([]*components.Passenger, 0, len(tr.Passengers)),
		}
		trainMap[tr] = ntr
		newTrains[i] = ntr
	}

	// ── Step 4: Passengers ───────────────────────────────────────────────────
	paxMap := make(map[*components.Passenger]*components.Passenger, len(gs.Passengers))
	newPassengers := make([]*components.Passenger, len(gs.Passengers))
	for i, p := range gs.Passengers {
		np := &components.Passenger{
			Destination:          p.Destination,
			PathIndex:            p.PathIndex,
			WaitStartTime:        p.WaitStartTime,
			LastRouteCalculation: p.LastRouteCalculation,
		}
		paxMap[p] = np
		newPassengers[i] = np
	}

	// ── Step 5: Wire Passenger cross-refs ────────────────────────────────────
	for oldP, newP := range paxMap {
		if oldP.CurrentStation != nil {
			newP.CurrentStation = stMap[oldP.CurrentStation]
		}
		if oldP.OnTrain != nil {
			newP.OnTrain = trainMap[oldP.OnTrain]
		}
		if oldP.ReservedTrain != nil {
			newP.ReservedTrain = trainMap[oldP.ReservedTrain]
		}
		if len(oldP.Path) > 0 {
			newP.Path = make([]*components.Station, len(oldP.Path))
			for j, s := range oldP.Path {
				if s != nil {
					newP.Path[j] = stMap[s]
				}
			}
		}
	}

	// ── Step 6: Wire Station.Passengers ──────────────────────────────────────
	for oldSt, newSt := range stMap {
		for _, p := range oldSt.Passengers {
			if np, ok := paxMap[p]; ok {
				newSt.Passengers = append(newSt.Passengers, np)
			}
		}
	}

	// ── Step 7: Wire Train cross-refs ────────────────────────────────────────
	for oldTr, newTr := range trainMap {
		if oldTr.Line != nil {
			newTr.Line = lineMap[oldTr.Line]
		}
		for _, p := range oldTr.Passengers {
			if np, ok := paxMap[p]; ok {
				newTr.Passengers = append(newTr.Passengers, np)
			}
		}
	}

	// ── Step 8: Wire Line cross-refs ─────────────────────────────────────────
	for oldLn, newLn := range lineMap {
		for _, st := range oldLn.Stations {
			if nst, ok := stMap[st]; ok {
				newLn.Stations = append(newLn.Stations, nst)
			}
		}
		for _, tr := range oldLn.Trains {
			if ntr, ok := trainMap[tr]; ok {
				newLn.Trains = append(newLn.Trains, ntr)
			}
		}
		if oldLn.OriginalStart != nil {
			newLn.OriginalStart = stMap[oldLn.OriginalStart]
		}
		if oldLn.OriginalEnd != nil {
			newLn.OriginalEnd = stMap[oldLn.OriginalEnd]
		}
	}

	// ── Step 9: Assemble the copy ────────────────────────────────────────────
	cp := &GameState{
		Paused:               false, // never pause during simulation
		SpawnStationsEnabled: false, // keep station count fixed during evaluation
		UpgradesEnabled:      gs.UpgradesEnabled,
		StationSpawnLimit:    gs.StationSpawnLimit,
		Speed:                gs.Speed,
		FastForward:          0,
		Score:                gs.Score,
		Week:                 gs.Week,
		Day:                  gs.Day,
		SelectedLine:         gs.SelectedLine,
		MaxLines:             gs.MaxLines,
		AvailableLines:       gs.AvailableLines,
		Bridges:              gs.Bridges,
		Carriages:            gs.Carriages,
		Interchanges:         gs.Interchanges,
		AvailableTrains:      gs.AvailableTrains,
		GameOver:             gs.GameOver,
		SelectedCity:         gs.SelectedCity,
		LastSpawnTime:        gs.LastSpawnTime,
		LastStationSpawnTime: gs.LastStationSpawnTime,
		WeekStartTime:        gs.WeekStartTime,
		StationIDCounter:     gs.StationIDCounter,
		TrainIDCounter:       gs.TrainIDCounter,
		PassengersDelivered:  gs.PassengersDelivered,
		GameStartTime:        gs.GameStartTime,
		CameraZoom:           gs.CameraZoom,
		SimTimeMs:            gs.SimTimeMs,
		SpawnRateFactor:      gs.SpawnRateFactor,
		GraphDirty:           true,      // force rebuild with new pointer set
		Rivers:               gs.Rivers, // immutable — safe to share
		Stations:             newStations,
		Lines:                newLines,
		Trains:               newTrains,
		Passengers:           newPassengers,
	}
	return cp
}
