package rl

import (
	"minimetro-go/config"
	"minimetro-go/systems"
)

// ObsDim is the total length of the observation vector returned by BuildObservation.
//
//	Global:   15
//	Stations: MaxStationSlots × 9 = 450
//	Lines:    MaxLineSlots    × 7 = 49
//	Total:    514
const ObsDim = 15 + MaxStationSlots*9 + MaxLineSlots*7

// upgradeTypeIndex maps an upgrade key to an integer label for the observation.
func upgradeTypeIndex(key string) float32 {
	switch key {
	case systems.UpgradeNewLine:
		return 0
	case systems.UpgradeCarriage:
		return 1
	case systems.UpgradeBridge:
		return 2
	case systems.UpgradeInterchange:
		return 3
	}
	return -1
}

// stationTypeIndex maps a StationType to an integer label.
func stationTypeIndex(t config.StationType) float32 {
	switch t {
	case config.Circle:
		return 0
	case config.Triangle:
		return 1
	case config.Square:
		return 2
	case config.Pentagon:
		return 3
	case config.Diamond:
		return 4
	case config.Star:
		return 5
	case config.Cross:
		return 6
	}
	return 0
}

// BuildObservation serialises the current game state into a fixed-size float32
// vector of length ObsDim (514). All values are normalised to roughly [0, 1].
func BuildObservation(env *RLEnv) []float32 {
	gs := env.gs
	obs := make([]float32, ObsDim)
	i := 0

	// ── Global features ──────────────────────────────────────────────────────
	obs[i] = float32(gs.Week) / 10.0
	i++
	obs[i] = float32(gs.Day) / 7.0
	i++
	obs[i] = float32(gs.Score) / 10000.0
	i++
	obs[i] = float32(gs.PassengersDelivered) / 1000.0
	i++
	obs[i] = float32(gs.AvailableTrains) / 10.0
	i++
	obs[i] = float32(gs.Carriages) / 10.0
	i++
	obs[i] = float32(gs.Interchanges) / 10.0
	i++
	obs[i] = float32(gs.AvailableLines) / float32(MaxLineSlots)
	i++
	obs[i] = float32(gs.MaxLines) / float32(MaxLineSlots)
	i++
	obs[i] = float32(gs.Bridges) / 10.0
	i++
	// Fraction of current week elapsed.
	weekElapsed := gs.SimTimeMs - gs.WeekStartTime
	obs[i] = float32(weekElapsed) / config.WeekDuration
	i++
	// Upgrade modal state.
	if env.inUpgradeModal {
		obs[i] = 1
	}
	i++
	// Upgrade choice types (0..3, or -1 when no modal).
	if env.inUpgradeModal && len(env.upgradeChoices) >= 2 {
		obs[i] = upgradeTypeIndex(env.upgradeChoices[0])
		i++
		obs[i] = upgradeTypeIndex(env.upgradeChoices[1])
		i++
	} else {
		obs[i] = -1
		i++
		obs[i] = -1
		i++
	}
	obs[i] = 0 // reserved
	i++

	// ── Centrality ───────────────────────────────────────────────────────────
	// Rebuild graph if needed so centrality uses current topology.
	env.game.GraphManager.GetGraph(gs)
	centrality := systems.BetweennessCentrality(gs, env.game.GraphManager)

	// ── Station features (50 × 9) ────────────────────────────────────────────
	cityCfg := config.Cities[gs.SelectedCity]
	for s := 0; s < MaxStationSlots; s++ {
		base := i + s*9
		if s >= len(gs.Stations) {
			// Padding: all zeros except is_valid already zero.
			continue
		}
		st := gs.Stations[s]
		cap := st.Capacity(cityCfg.StationCapacity)

		// lines through this station
		linesThr := 0
		for l := 0; l < gs.AvailableLines && l < len(gs.Lines); l++ {
			line := gs.Lines[l]
			if !line.Active || line.MarkedForDeletion {
				continue
			}
			for _, ls := range line.Stations {
				if ls == st {
					linesThr++
					break
				}
			}
		}

		obs[base+0] = float32(st.X) / systems.SimScreenWidth
		obs[base+1] = float32(st.Y) / systems.SimScreenHeight
		obs[base+2] = stationTypeIndex(st.Type) / 6.0
		obs[base+3] = float32(len(st.Passengers)) / float32(cap)
		obs[base+4] = float32(st.OvercrowdProgress) / config.OvercrowdTime
		if st.IsInterchange {
			obs[base+5] = 1
		}
		obs[base+6] = 1 // is_valid
		obs[base+7] = float32(linesThr) / float32(MaxLineSlots)
		obs[base+8] = float32(centrality[st])
	}
	i += MaxStationSlots * 9

	// ── Line features (7 × 7) ────────────────────────────────────────────────
	for l := 0; l < MaxLineSlots; l++ {
		base := i + l*7
		if l >= len(gs.Lines) {
			continue
		}
		line := gs.Lines[l]

		// arc length: sum of Euclidean distances between consecutive stations.
		arcLen := 0.0
		stations := line.Stations
		for k := 0; k+1 < len(stations); k++ {
			dx := stations[k+1].X - stations[k].X
			dy := stations[k+1].Y - stations[k].Y
			arcLen += float64(dx*dx+dy*dy) // approximate; will sqrt at end
		}
		if arcLen > 0 {
			arcLen = sqrtApprox(arcLen)
		}

		// Passenger load ratio across all trains on this line.
		totalPax, totalCap := 0, 0
		for _, t := range line.Trains {
			totalPax += len(t.Passengers)
			totalCap += t.TotalCapacity()
		}
		loadRatio := float32(0)
		if totalCap > 0 {
			loadRatio = float32(totalPax) / float32(totalCap)
		}

		nSt := len(stations)
		isLoop := line.Active && nSt > 2 && stations[0] == stations[nSt-1]

		if line.Active {
			obs[base+0] = 1
		}
		if l < gs.AvailableLines {
			obs[base+1] = 1 // is_valid
		}
		obs[base+2] = float32(nSt) / float32(MaxStationSlots)
		obs[base+3] = float32(len(line.Trains)) / float32(config.MaxTrainsPerLine)
		obs[base+4] = loadRatio
		if isLoop {
			obs[base+5] = 1
		}
		obs[base+6] = float32(arcLen) / 1000.0
	}

	return obs
}

// sqrtApprox computes an integer square root via Newton's method, avoiding
// importing "math" just for this helper.
func sqrtApprox(x float64) float64 {
	if x <= 0 {
		return 0
	}
	g := x / 2
	for i := 0; i < 20; i++ {
		g = (g + x/g) / 2
	}
	return g
}
