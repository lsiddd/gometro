package rl

import (
	"math"
	"minimetro-go/components"
	"minimetro-go/config"
	"minimetro-go/systems"
	"minimetro-go/systems/graph"
)

// Observation-vector layout constants. Python reads these via GET /info and
// validates them at startup; keep them in sync with python/constants.py.
const (
	GlobalDim  = 15 // global scalar features
	StationDim = 16 // per-station features (9 base + 7 passenger-demand dims)
	LineDim    = 7  // per-line features
)

// ObsDim is the total length of the observation vector returned by BuildObservation.
//
//	Global:   GlobalDim                        = 15
//	Stations: MaxStationSlots × StationDim     = 800
//	Lines:    MaxLineSlots    × LineDim        = 49
//	Topology: MaxLineSlots    × MaxStationSlots = 350
//	Total:    1214
const ObsDim = GlobalDim + MaxStationSlots*StationDim + MaxLineSlots*LineDim + MaxLineSlots*MaxStationSlots

// numStationTypes is the number of distinct station/passenger types.
const numStationTypes = 7

// upgradeTypeIndex maps an upgrade key to an integer label for the observation.
// Returns values in [1, 4]; 0 is reserved for "no choice" (modal inactive).
func upgradeTypeIndex(key string) float32 {
	switch key {
	case systems.UpgradeNewLine:
		return 1
	case systems.UpgradeCarriage:
		return 2
	case systems.UpgradeBridge:
		return 3
	case systems.UpgradeInterchange:
		return 4
	}
	return 0
}

// stationTypeIndex maps a StationType to an integer label in [0, 6].
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

// clamp01 clamps v to [0, 1].
func clamp01(v float32) float32 {
	if v < 0 {
		return 0
	}
	if v > 1 {
		return 1
	}
	return v
}

// BuildObservation serialises the current game state into a fixed-size float32
// vector of length ObsDim. All values are normalised to [0, 1].
func BuildObservation(env *RLEnv) []float32 {
	gs := env.gs
	obs := make([]float32, ObsDim)
	i := 0

	// ── Global features (15) ─────────────────────────────────────────────────
	obs[i] = clamp01(float32(gs.Week) / 10.0)
	i++
	obs[i] = clamp01(float32(gs.Day) / 7.0)
	i++
	obs[i] = clamp01(float32(gs.Score) / 10000.0)
	i++
	obs[i] = clamp01(float32(gs.PassengersDelivered) / 1000.0)
	i++
	obs[i] = clamp01(float32(gs.AvailableTrains) / 10.0)
	i++
	obs[i] = clamp01(float32(gs.Carriages) / 10.0)
	i++
	obs[i] = clamp01(float32(gs.Interchanges) / 10.0)
	i++
	obs[i] = float32(gs.AvailableLines) / float32(MaxLineSlots)
	i++
	obs[i] = float32(gs.MaxLines) / float32(MaxLineSlots)
	i++
	obs[i] = clamp01(float32(gs.Bridges) / 10.0)
	i++
	// Fraction of current week elapsed.
	weekElapsed := gs.SimTimeMs - gs.WeekStartTime
	obs[i] = clamp01(float32(weekElapsed) / config.WeekDuration)
	i++
	// Upgrade modal flag.
	if env.inUpgradeModal {
		obs[i] = 1
	}
	i++
	// Upgrade choice types: 0 = no modal / unknown, 1-4 = upgrade type.
	// When the modal is inactive both slots are 0; the flag above disambiguates.
	if env.inUpgradeModal && len(env.upgradeChoices) >= 2 {
		obs[i] = upgradeTypeIndex(env.upgradeChoices[0]) / 4.0
		i++
		obs[i] = upgradeTypeIndex(env.upgradeChoices[1]) / 4.0
		i++
	} else {
		obs[i] = 0
		i++
		obs[i] = 0
		i++
	}
	obs[i] = 0 // reserved
	i++

	// ── Centrality ───────────────────────────────────────────────────────────
	env.game.GraphManager.GetGraph(gs)
	centrality := graph.BetweennessCentrality(gs, env.game.GraphManager)

	// Normalise centrality to [0, 1] by dividing by the max observed value.
	maxCentrality := float32(0)
	for _, v := range centrality {
		fv := float32(v)
		if fv > maxCentrality {
			maxCentrality = fv
		}
	}

	// ── Station features (50 × 16) ───────────────────────────────────────────
	// Per-station layout (base+0 .. base+15):
	//   0: x            1: y             2: type
	//   3: pax ratio    4: overcrowd     5: is_interchange
	//   6: is_valid     7: lines through 8: centrality
	//   9-15: passenger demand by type (Circle..Cross) / capacity
	cityCfg := config.Cities[gs.SelectedCity]
	for s := 0; s < MaxStationSlots; s++ {
		base := i + s*16
		if s >= len(gs.Stations) {
			continue // padding zeros
		}
		st := gs.Stations[s]
		cap := st.Capacity(cityCfg.StationCapacity)
		fcap := float32(cap)

		// Lines through this station.
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

		// Passenger demand breakdown by destination type.
		var demand [numStationTypes]float32
		for _, p := range st.Passengers {
			idx := int(stationTypeIndex(p.Destination))
			if idx >= 0 && idx < numStationTypes {
				demand[idx]++
			}
		}

		centVal := float32(0)
		if maxCentrality > 0 {
			centVal = float32(centrality[st]) / maxCentrality
		}

		obs[base+0] = float32(st.X) / systems.SimScreenWidth
		obs[base+1] = float32(st.Y) / systems.SimScreenHeight
		obs[base+2] = stationTypeIndex(st.Type) / float32(numStationTypes-1)
		obs[base+3] = clamp01(float32(len(st.Passengers)) / fcap)
		obs[base+4] = clamp01(float32(st.OvercrowdProgress) / config.OvercrowdTime)
		if st.IsInterchange {
			obs[base+5] = 1
		}
		obs[base+6] = 1 // is_valid
		obs[base+7] = float32(linesThr) / float32(MaxLineSlots)
		obs[base+8] = clamp01(centVal)
		for t := 0; t < numStationTypes; t++ {
			obs[base+9+t] = clamp01(demand[t] / fcap)
		}
	}
	i += MaxStationSlots * 16

	// ── Line features (7 × 7) ────────────────────────────────────────────────
	for l := 0; l < MaxLineSlots; l++ {
		base := i + l*7
		if l >= len(gs.Lines) {
			continue
		}
		line := gs.Lines[l]

		arcLen := 0.0
		stations := line.Stations
		for k := 0; k+1 < len(stations); k++ {
			dx := stations[k+1].X - stations[k].X
			dy := stations[k+1].Y - stations[k].Y
			arcLen += float64(dx*dx + dy*dy)
		}
		if arcLen > 0 {
			arcLen = math.Sqrt(arcLen)
		}

		totalPax, totalCap := 0, 0
		for _, t := range line.Trains {
			totalPax += len(t.Passengers)
			totalCap += t.TotalCapacity()
		}
		loadRatio := float32(0)
		if totalCap > 0 {
			loadRatio = clamp01(float32(totalPax) / float32(totalCap))
		}

		nSt := len(stations)
		isLoop := line.Active && nSt > 2 && stations[0] == stations[nSt-1]

		if line.Active {
			obs[base+0] = 1
		}
		if l < gs.AvailableLines {
			obs[base+1] = 1
		}
		obs[base+2] = float32(nSt) / float32(MaxStationSlots)
		obs[base+3] = float32(len(line.Trains)) / float32(config.MaxTrainsPerLine)
		obs[base+4] = loadRatio
		if isLoop {
			obs[base+5] = 1
		}
		obs[base+6] = clamp01(float32(arcLen) / 1000.0)
	}
	i += MaxLineSlots * 7

	// ── Topology (7 × 50) ────────────────────────────────────────────────────
	for l := 0; l < MaxLineSlots; l++ {
		var stations []*components.Station
		if l < len(gs.Lines) && gs.Lines[l].Active {
			stations = gs.Lines[l].Stations
		}

		for s := 0; s < MaxStationSlots; s++ {
			var st *components.Station
			if s < len(gs.Stations) {
				st = gs.Stations[s]
			}

			val := float32(0.0)

			if st != nil && len(stations) > 0 {
				isHead, isTail, isMiddle := false, false, false
				for idx, ls := range stations {
					if ls == st {
						if idx == 0 {
							isHead = true
						} else if idx == len(stations)-1 {
							isTail = true
						} else {
							isMiddle = true
						}
					}
				}
				if isHead && isTail {
					val = 0.75 // loop point
				} else if isHead {
					val = 0.5 // head
				} else if isTail {
					val = 1.0 // tail
				} else if isMiddle {
					val = 0.25 // middle
				}
			}

			obs[i] = val
			i++
		}
	}

	return obs
}

